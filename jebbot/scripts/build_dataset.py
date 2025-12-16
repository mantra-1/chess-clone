#!/usr/bin/env python3
"""Build training dataset from downloaded chess games."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jebbot.data.parse import build_training_dataset


def main():
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_file = output_dir / "training_positions.json"

    print(f"Processing games from {raw_dir}...")

    if not raw_dir.exists() or not list(raw_dir.glob("*.json")):
        print("No game files found. Run fetch_games.py first.")
        return

    # Build dataset
    positions, stats = build_training_dataset(raw_dir)

    if not positions:
        print("No positions extracted. Check if games exist and username is correct.")
        return

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(positions, f, indent=2)

    print(f"\nDataset saved to {output_file}")

    # Print stats
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total games processed:    {stats['total_games']}")
    print(f"Games with positions:     {stats['games_with_positions']}")
    print(f"Skipped games:            {stats['skipped_games']}")
    print(f"Total positions:          {stats['total_positions']}")
    print(f"Avg positions per game:   {stats['avg_positions_per_game']:.1f}")

    # Show sample positions
    print("\n" + "=" * 50)
    print("SAMPLE POSITIONS (3 random)")
    print("=" * 50)

    samples = random.sample(positions, min(3, len(positions)))
    for i, pos in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"  FEN:          {pos['fen']}")
        print(f"  Move:         {pos['move_uci']}")
        print(f"  Game ID:      {pos['game_id']}")
        print(f"  Color:        {pos['color']}")
        print(f"  Move number:  {pos['move_number']}")
        print(f"  Time control: {pos['time_control']}")


if __name__ == "__main__":
    main()
