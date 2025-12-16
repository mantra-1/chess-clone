#!/usr/bin/env python3
"""Quick test of training loop (2 epochs only)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from jebbot.data.encode import ChessPositionDataset
from jebbot.model.style_selector import StyleSelector, get_model_size
from jebbot.training.trainer import get_device, train_epoch, validate


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


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    positions_file = data_dir / "processed" / "training_positions.json"
    models_dir = data_dir / "models" / "test"

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
    dataset = ChessPositionDataset(positions_file)
    num_positions = dataset.get_num_positions()
    print(f"Original positions: {num_positions:,}")
    print(f"With negatives (5x): {len(dataset):,}")

    # DEBUG: Check a few examples
    print("\n" + "=" * 60)
    print("DEBUG: Sample examples from dataset")
    print("=" * 60)
    for i in range(10):
        pos, move_idx, target = dataset[i]
        position_idx = i // 5
        variant = i % 5
        print(f"  idx={i} (pos={position_idx}, var={variant}) → target={target}, move_idx={move_idx}")

    # Split dataset
    print("\nSplitting dataset (80% train, 10% val, 10% test)...")
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"  Train: {len(train_set):,}")
    print(f"  Val:   {len(val_set):,}")
    print(f"  Test:  {len(test_set):,}")

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nCreating StyleSelector model...")
    model = StyleSelector()
    model = model.to(device)
    param_count = get_model_size(model)
    print(f"Total parameters: {param_count:,}")

    # Setup
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    models_dir.mkdir(parents=True, exist_ok=True)

    # DEBUG: Check initial model outputs
    print("\n" + "=" * 60)
    print("DEBUG: Initial model outputs (before training)")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        for i, (positions, move_indices, targets) in enumerate(train_loader):
            if i >= 1:
                break
            positions = positions.to(device)
            move_indices = move_indices.to(device)
            outputs = model(positions, move_indices)
            print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}], mean: {outputs.mean():.3f}")
            print(f"  Target range: [{targets.min():.1f}, {targets.max():.1f}], mean: {targets.mean():.3f}")
            print(f"  Positive examples in batch: {(targets == 1.0).sum().item()}")
            print(f"  Negative examples in batch: {(targets == 0.0).sum().item()}")

    # Train with timing
    print("\n" + "=" * 60)
    print("TRAINING (TEST - 2 EPOCHS)")
    print("=" * 60)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("-" * 60)

    total_start = time.time()
    epochs = 2

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - epoch_start

        # Validate
        val_start = time.time()
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_time = time.time() - val_start

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s (train: {train_time:.1f}s, val: {val_time:.1f}s)"
        )

    total_time = time.time() - total_start

    # DEBUG: Check final model outputs
    print("\n" + "=" * 60)
    print("DEBUG: Final model outputs (after training)")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        for i, (positions, move_indices, targets) in enumerate(val_loader):
            if i >= 1:
                break
            positions = positions.to(device)
            move_indices = move_indices.to(device)
            outputs = model(positions, move_indices)
            print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}], mean: {outputs.mean():.3f}")

            # Show some predictions vs targets
            print("  Sample predictions:")
            for j in range(min(5, len(outputs))):
                print(f"    pred={outputs[j].item():.3f}, target={targets[j].item():.1f}")

    # Save test model
    save_path = models_dir / "test_model.pt"
    torch.save(model.state_dict(), save_path)

    print("-" * 60)
    print(f"\n✅ Training test complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg time per epoch: {total_time/epochs:.1f}s")
    print(f"Estimated full training (50 epochs): {total_time/epochs * 50 / 60:.1f} min")
    print(f"Test model saved to: {save_path}")


if __name__ == "__main__":
    main()
