#!/usr/bin/env python3
"""Quick test of training loop (2 epochs only) - DEBUG VERSION."""

import json
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
from jebbot.training.trainer import get_device


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
    print(f"Total positions: {len(dataset):,}")

    # Split dataset
    print("\nSplitting dataset (80% train, 10% val, 10% test)...")
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"  Train: {len(train_set):,}")
    print(f"  Val:   {len(val_set):,}")
    print(f"  Test:  {len(test_set):,}")

    # DEBUG: Print first 5 FENs from train and val to confirm different data
    print("\n" + "=" * 60)
    print("DEBUG: Checking train/val split")
    print("=" * 60)
    print("\nFirst 5 TRAIN positions (index, FEN):")
    for i in range(5):
        idx = train_set.indices[i]
        meta = dataset.get_position_metadata(idx)
        print(f"  [{idx}] {meta['fen'][:50]}...")

    print("\nFirst 5 VAL positions (index, FEN):")
    for i in range(5):
        idx = val_set.indices[i]
        meta = dataset.get_position_metadata(idx)
        print(f"  [{idx}] {meta['fen'][:50]}...")

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

    # DEBUG: Test raw model outputs BEFORE any training
    print("\n" + "=" * 60)
    print("DEBUG: Raw model outputs BEFORE training")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        for i, (positions, move_indices) in enumerate(train_loader):
            if i >= 1:  # Just first batch
                break
            positions = positions.to(device)
            move_indices = move_indices.to(device)
            outputs = model(positions, move_indices)
            print(f"Batch 0 - First 10 raw outputs: {outputs[:10].squeeze().tolist()}")
            print(f"Output min: {outputs.min():.4f}, max: {outputs.max():.4f}, mean: {outputs.mean():.4f}")

    # DEBUG: Manual training loop with per-batch loss
    print("\n" + "=" * 60)
    print("DEBUG: Training with per-batch loss (first 10 batches)")
    print("=" * 60)

    model.train()
    for batch_idx, (positions, move_indices) in enumerate(train_loader):
        if batch_idx >= 10:
            break

        positions = positions.to(device)
        move_indices = move_indices.to(device)

        # Target is always 1.0 - THIS IS THE BUG!
        targets = torch.ones(positions.size(0), 1, device=device)

        optimizer.zero_grad()
        outputs = model(positions, move_indices)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx:2d} | Loss: {loss.item():.6f} | "
              f"Output mean: {outputs.mean().item():.4f} | "
              f"Output min: {outputs.min().item():.4f} | "
              f"Output max: {outputs.max().item():.4f} | "
              f"Target: always 1.0")

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("""
THE BUG: Every training example has target=1.0!

We're only showing the model moves the player DID make, so:
- Input: position + move_that_was_played
- Target: 1.0 (yes, player made this move)

The model quickly learns: "just output 1.0 for everything"
This gives perfect accuracy and near-zero loss.

FIX NEEDED: Add NEGATIVE examples where:
- Input: position + move_player_did_NOT_make
- Target: 0.0

For each real move, sample 3-5 random legal moves as negatives.
This makes the model actually learn which moves are "Jeb-like".
""")


if __name__ == "__main__":
    main()
