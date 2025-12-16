#!/usr/bin/env python3
"""Train the StyleSelector model."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from jebbot.data.encode import ChessPositionDataset
from jebbot.model.style_selector import StyleSelector, get_model_size
from jebbot.training.trainer import get_device, train


def split_dataset(
    dataset: ChessPositionDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train, validation, and test sets.

    Args:
        dataset: Full dataset
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    # Set seed for reproducibility
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
    dataset = ChessPositionDataset(positions_file)
    print(f"Total positions: {len(dataset):,}")

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
        num_workers=0,  # MPS doesn't support multiprocessing well
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nCreating StyleSelector model...")
    model = StyleSelector()
    param_count = get_model_size(model)
    print(f"Total parameters: {param_count:,}")

    # Training config
    epochs = 50
    lr = 1e-4

    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Early stopping patience: 10 epochs")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60 + "\n")

    start_time = time.time()
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_dir=models_dir,
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

    from jebbot.training.trainer import validate
    import torch.nn as nn

    criterion = nn.BCELoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
