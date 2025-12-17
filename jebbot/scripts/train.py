#!/usr/bin/env python3
"""Train the StyleSelector model."""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from jebbot.data.encode import ChessPositionDataset, index_to_move
from jebbot.model.style_selector import StyleSelector, get_model_size
from jebbot.training.trainer import get_device, validate


def split_dataset(
    dataset: ChessPositionDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train, validation, and test sets."""
    torch.manual_seed(seed)
    total_size = len(dataset)
    indices = torch.randperm(total_size).tolist()

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def sample_positions_for_visualization(
    dataset: ChessPositionDataset,
    model: nn.Module,
    positions_batch: torch.Tensor,
    move_indices_batch: torch.Tensor,
    targets_batch: torch.Tensor,
    device: torch.device,
    num_samples: int = 6,
) -> list[dict]:
    """Sample positions and get model predictions for visualization.

    Args:
        dataset: The dataset (to get FEN strings)
        model: The trained model
        positions_batch: Batch of position tensors
        move_indices_batch: Batch of move indices
        targets_batch: Batch of targets
        device: Device to run on
        num_samples: Number of positions to sample

    Returns:
        List of dicts with fen, actual_move, predicted_move, confidence
    """
    model.eval()

    # Get batch size and sample indices
    batch_size = positions_batch.size(0)
    if batch_size < num_samples:
        sample_indices = list(range(batch_size))
    else:
        sample_indices = random.sample(range(batch_size), num_samples)

    results = []

    with torch.no_grad():
        for idx in sample_indices:
            position = positions_batch[idx:idx+1].to(device)
            actual_move_idx = move_indices_batch[idx].item()
            target = targets_batch[idx].item()

            # Get model's confidence for this move
            move_tensor = move_indices_batch[idx:idx+1].to(device)
            confidence = model(position, move_tensor).item()

            # For visualization, we need to find the model's best move
            # Test a sample of legal moves to find highest confidence
            # (Full search over 4096 moves is too slow)
            best_pred_idx = actual_move_idx
            best_conf = confidence

            # Sample some alternative moves to compare
            for _ in range(20):
                alt_move = random.randint(0, 4095)
                alt_tensor = torch.tensor([alt_move], device=device)
                alt_conf = model(position, alt_tensor).item()
                if alt_conf > best_conf:
                    best_conf = alt_conf
                    best_pred_idx = alt_move

            # Convert to FEN (reconstruct from tensor - simplified)
            # For now, use a placeholder FEN since we don't have direct access
            # In practice, we'd need to store FEN or pass dataset indices
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

            results.append({
                "fen": fen,
                "actual_move": index_to_move(actual_move_idx),
                "predicted_move": index_to_move(best_pred_idx),
                "confidence": best_conf,
            })

    model.train()
    return results


def train_with_visualization(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset: ChessPositionDataset,
    epochs: int,
    lr: float,
    device: torch.device,
    save_dir: Path,
    visualize: bool = False,
    vis_every: int = 100,
) -> dict:
    """Training loop with optional visualization support.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        dataset: Original dataset (for FEN access)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save model
        visualize: Whether to send visualization updates
        vis_every: Send visualization update every N batches

    Returns:
        Training history dict
    """
    # Import visualization client if needed
    send_update = None
    if visualize:
        try:
            from jebbot.visualization.client import send_update as _send_update, is_server_running
            if is_server_running():
                send_update = _send_update
                print("Visualization server connected!")
            else:
                print("Warning: Visualization server not running. Start it with:")
                print("  uv run python scripts/start_visualizer.py")
                print("Continuing without visualization...")
        except ImportError:
            print("Warning: Visualization module not found")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_epoch": 0,
    }

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    total_batches = len(train_loader)
    print(f"Training on {device}")
    print(f"Train batches: {total_batches}, Val batches: {len(val_loader)}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (positions, move_indices, targets) in enumerate(train_loader):
            # Move data to device
            positions = positions.to(device)
            move_indices = move_indices.to(device)
            targets = targets.to(torch.float32).unsqueeze(1).to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(positions, move_indices)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Visualization update every N batches
            if send_update and batch_idx > 0 and batch_idx % vis_every == 0:
                # Sample positions for visualization
                vis_positions = []
                for i in range(min(6, positions.size(0))):
                    move_idx = move_indices[i].item()

                    # Get model's prediction confidence
                    with torch.no_grad():
                        conf = outputs[i].item()

                    # Find a contrasting move for visualization
                    pred_move_idx = move_idx  # Default to actual

                    vis_positions.append({
                        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                        "actual_move": index_to_move(move_idx),
                        "predicted_move": index_to_move(pred_move_idx),
                        "confidence": conf,
                    })

                send_update(
                    epoch=epoch,
                    total_epochs=epochs,
                    batch=batch_idx,
                    total_batches=total_batches,
                    train_loss=total_loss / num_batches,
                    val_loss=history["val_loss"][-1] if history["val_loss"] else 0,
                    val_accuracy=history["val_accuracy"][-1] if history["val_accuracy"] else 0,
                    positions=vis_positions,
                )

            # Clear MPS cache periodically
            if num_batches % 1000 == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

        # Sync at end of epoch
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        train_loss = total_loss / num_batches

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Clear memory
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Send epoch-end visualization update
        if send_update:
            send_update(
                epoch=epoch,
                total_epochs=epochs,
                batch=total_batches,
                total_batches=total_batches,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                positions=[],  # No positions at epoch end
            )

        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            patience_counter = 0

            save_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print("-" * 60)
    print(f"Training complete. Best epoch: {history['best_epoch']}")
    print(f"Best val loss: {best_val_loss:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train the StyleSelector model")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable training visualization (requires visualization server)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    positions_file = data_dir / "processed" / "training_positions.json"
    models_dir = data_dir / "models"

    # Check if data exists
    if not positions_file.exists():
        print(f"Error: {positions_file} not found")
        print("Run build_dataset.py first to create training data")
        return

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset from {positions_file}...")
    dataset = ChessPositionDataset(positions_file, cache_tensors=False)
    print(f"Original positions: {dataset.get_num_positions():,}")
    print(f"With 1:1 pos/neg (2x): {len(dataset):,}")

    # Split dataset
    print("\nSplitting dataset (80% train, 10% val, 10% test)...")
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"  Train: {len(train_set):,}")
    print(f"  Val:   {len(val_set):,}")
    print(f"  Test:  {len(test_set):,}")

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nCreating StyleSelector model...")
    model = StyleSelector()
    param_count = get_model_size(model)
    print(f"Total parameters: {param_count:,}")

    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Early stopping patience: 10 epochs")
    print(f"  Visualization: {'enabled' if args.visualize else 'disabled'}")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60 + "\n")

    start_time = time.time()
    history = train_with_visualization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=models_dir,
        visualize=args.visualize,
    )
    elapsed = time.time() - start_time

    # Save training history
    history_file = models_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Best val accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Model saved to: {models_dir / 'best_model.pt'}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(models_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.BCELoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
