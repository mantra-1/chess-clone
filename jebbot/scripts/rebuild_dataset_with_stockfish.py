#!/usr/bin/env python3
"""Add Stockfish analysis to existing training dataset.

This script loads an existing training_positions.json file and adds
stockfish_top_moves field to each position. This is useful for upgrading
an existing dataset without re-parsing all games.

Usage:
    # First install stockfish: brew install stockfish
    uv run python scripts/rebuild_dataset_with_stockfish.py

The script will:
1. Load existing data/processed/training_positions.json
2. Analyze each position with Stockfish at 1500 ELO
3. Add stockfish_top_moves (top 10 moves) to each position
4. Save the updated file (backing up the original)
"""

import argparse
import json
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jebbot.data.parse import add_stockfish_to_positions, init_stockfish


def format_eta(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return str(timedelta(seconds=int(seconds)))


def progress_callback(current: int, total: int, message: str):
    """Print progress with ETA."""
    if total > 0:
        pct = current / total * 100
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Calculate ETA
        if current > 0:
            elapsed = time.time() - progress_callback.start_time
            rate = current / elapsed  # positions per second
            remaining = total - current
            eta = remaining / rate if rate > 0 else 0
            eta_str = f"ETA: {format_eta(eta)}"
        else:
            eta_str = "ETA: calculating..."

        print(f"\r[{bar}] {pct:5.1f}% ({current:,}/{total:,}) {eta_str}", end="", flush=True)

        if current == total:
            elapsed = time.time() - progress_callback.start_time
            print(f"\nCompleted in {format_eta(elapsed)}")


def main():
    parser = argparse.ArgumentParser(
        description="Add Stockfish analysis to existing training dataset"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary (auto-detected if not provided)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file (default: data/processed/training_positions.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: same as input)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process first N positions (for testing)",
    )
    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    input_file = Path(args.input) if args.input else data_dir / "training_positions.json"
    output_file = Path(args.output) if args.output else input_file

    # Check input exists
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        print("Run build_dataset.py first to create training data")
        return

    # Test Stockfish initialization
    print("Initializing Stockfish...")
    stockfish = init_stockfish(args.stockfish_path)
    if not stockfish:
        print("\nError: Could not initialize Stockfish")
        print("Install it with: brew install stockfish")
        return
    print(f"Stockfish initialized (ELO: 1500)")

    # Load existing positions
    print(f"\nLoading positions from {input_file}...")
    with open(input_file) as f:
        positions = json.load(f)

    total_positions = len(positions)
    print(f"Loaded {total_positions:,} positions")

    # Check if already has Stockfish analysis
    if positions and "stockfish_top_moves" in positions[0]:
        print("\nWarning: Dataset already has Stockfish analysis!")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # Sample for testing
    if args.sample:
        positions = positions[:args.sample]
        print(f"Processing sample of {len(positions):,} positions")

    # Create backup
    if not args.no_backup and output_file == input_file:
        backup_file = input_file.with_suffix(".json.bak")
        print(f"\nBacking up to {backup_file}...")
        shutil.copy(input_file, backup_file)

    # Estimate time
    print(f"\nAnalyzing {len(positions):,} positions with Stockfish...")
    print("(This will take a while - ~0.05s per position)")
    estimated_time = len(positions) * 0.05
    print(f"Estimated time: {format_eta(estimated_time)}")
    print()

    # Add Stockfish analysis
    progress_callback.start_time = time.time()
    positions = add_stockfish_to_positions(
        positions,
        stockfish_path=args.stockfish_path,
        progress_callback=progress_callback,
    )

    # Save updated file
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(positions, f, indent=2)

    # Show sample output
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    for i, pos in enumerate(positions[:3]):
        print(f"\n--- Position {i+1} ---")
        print(f"  FEN:              {pos['fen'][:50]}...")
        print(f"  Jeb's move:       {pos['move_uci']}")
        print(f"  Stockfish top 10: {pos.get('stockfish_top_moves', [])}")

        # Check if Jeb's move was in Stockfish's top moves
        if "stockfish_top_moves" in pos:
            if pos["move_uci"] in pos["stockfish_top_moves"]:
                rank = pos["stockfish_top_moves"].index(pos["move_uci"]) + 1
                print(f"  Jeb's move rank:  #{rank} in Stockfish top 10")
            else:
                print(f"  Jeb's move rank:  Not in top 10 (unique style!)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Updated {len(positions):,} positions with Stockfish analysis")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
