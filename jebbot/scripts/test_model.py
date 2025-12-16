#!/usr/bin/env python3
"""Test the StyleSelector model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from jebbot.model.style_selector import StyleSelector, get_model_size


def main():
    print("Instantiating StyleSelector...")
    model = StyleSelector()

    # Print parameter count
    param_count = get_model_size(model)
    print(f"Total parameters: {param_count:,}")

    # Create dummy input: batch of 4 positions and 4 move indices
    batch_size = 4
    positions = torch.randn(batch_size, 12, 8, 8)
    move_indices = torch.randint(0, 4096, (batch_size,))

    print(f"\nInput shapes:")
    print(f"  positions: {positions.shape}")
    print(f"  move_indices: {move_indices.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    output = model(positions, move_indices)

    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().tolist()}")

    # Verify output shape
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"

    print("\n✅ Model works!")


if __name__ == "__main__":
    main()
