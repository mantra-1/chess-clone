#!/usr/bin/env python3
"""Add Stockfish analysis to existing training dataset with multiprocessing.

This script loads an existing training_positions.json file and adds
stockfish_top_moves field to each position using parallel processing.

Usage:
    # First install stockfish: brew install stockfish
    uv run python scripts/rebuild_dataset_with_stockfish.py

    # Quick test with first 1000 positions
    uv run python scripts/rebuild_dataset_with_stockfish.py --test

    # Custom settings
    uv run python scripts/rebuild_dataset_with_stockfish.py --workers 8 --depth 10

The script will:
1. Load existing data/processed/training_positions.json
2. Analyze each position with Stockfish at 1500 ELO (depth 8)
3. Add stockfish_top_moves (top 6 moves) to each position
4. Save the updated file (backing up the original)

Performance: ~20-30 minutes for 219K positions with 4-6 workers (vs ~1 day single-threaded)
"""

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jebbot.data.parse import (
    STOCKFISH_DEPTH,
    STOCKFISH_TOP_N_MOVES,
    get_stockfish_top_moves,
    init_stockfish,
)


# Global stockfish instance per worker process
_worker_stockfish = None
_worker_stockfish_path = None


def init_worker(stockfish_path: str | None, depth: int):
    """Initialize Stockfish for this worker process."""
    global _worker_stockfish, _worker_stockfish_path
    _worker_stockfish_path = stockfish_path
    _worker_stockfish = init_stockfish(stockfish_path, depth=depth)
    if _worker_stockfish is None:
        raise RuntimeError("Could not initialize Stockfish in worker")


def analyze_position(args: tuple) -> tuple[int, list[str]]:
    """Analyze a single position with Stockfish.

    Args:
        args: Tuple of (index, fen, num_moves)

    Returns:
        Tuple of (index, list of top moves)
    """
    idx, fen, num_moves = args
    global _worker_stockfish

    if _worker_stockfish is None:
        return idx, []

    top_moves = get_stockfish_top_moves(fen, _worker_stockfish, num_moves)
    return idx, top_moves


def format_eta(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return str(timedelta(seconds=int(seconds)))


def main():
    parser = argparse.ArgumentParser(
        description="Add Stockfish analysis to existing training dataset (multiprocessed)"
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
        "--test",
        action="store_true",
        help="Only process first 1000 positions (for testing)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process first N positions (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 2, min 2)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=STOCKFISH_DEPTH,
        help=f"Stockfish search depth (default: {STOCKFISH_DEPTH})",
    )
    parser.add_argument(
        "--top-moves",
        type=int,
        default=STOCKFISH_TOP_N_MOVES,
        help=f"Number of top moves to retrieve (default: {STOCKFISH_TOP_N_MOVES})",
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

    # Determine number of workers
    if args.workers:
        num_workers = args.workers
    else:
        num_workers = max(2, mp.cpu_count() - 2)

    print(f"Using {num_workers} parallel workers")

    # Test Stockfish initialization (single instance to verify it works)
    print("Testing Stockfish initialization...")
    stockfish = init_stockfish(args.stockfish_path, depth=args.depth)
    if not stockfish:
        print("\nError: Could not initialize Stockfish")
        print("Install it with: brew install stockfish")
        return
    print(f"Stockfish initialized (ELO: 1500, depth: {args.depth})")
    del stockfish  # Close test instance

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

    # Handle --test flag (1000 positions)
    if args.test:
        positions = positions[:1000]
        print(f"TEST MODE: Processing only {len(positions):,} positions")
    elif args.sample:
        positions = positions[:args.sample]
        print(f"Processing sample of {len(positions):,} positions")

    # Create backup
    if not args.no_backup and output_file == input_file:
        backup_file = input_file.with_suffix(".json.bak")
        print(f"\nBacking up to {backup_file}...")
        shutil.copy(input_file, backup_file)

    # Prepare work items
    work_items = [
        (i, pos["fen"], args.top_moves)
        for i, pos in enumerate(positions)
    ]

    total = len(work_items)

    # Estimate time (with multiprocessing ~0.01-0.02s per position)
    print(f"\nAnalyzing {total:,} positions with Stockfish...")
    print(f"Settings: depth={args.depth}, top_moves={args.top_moves}, workers={num_workers}")
    estimated_per_position = 0.02  # Conservative estimate with parallelism
    estimated_time = (total * estimated_per_position) / num_workers
    print(f"Estimated time: {format_eta(estimated_time)}")
    print()

    # Process with multiprocessing
    start_time = time.time()
    completed = 0
    results = {}

    # Use imap_unordered for better progress tracking
    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.stockfish_path, args.depth),
    ) as pool:
        for idx, top_moves in pool.imap_unordered(analyze_position, work_items, chunksize=100):
            results[idx] = top_moves
            completed += 1

            # Progress update every 1000 positions or at completion
            if completed % 1000 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = total - completed
                eta = remaining / rate if rate > 0 else 0

                pct = completed / total * 100
                bar_width = 30
                filled = int(bar_width * completed / total)
                bar = "█" * filled + "░" * (bar_width - filled)

                print(
                    f"\r[{bar}] {pct:5.1f}% ({completed:,}/{total:,}) "
                    f"{rate:.1f} pos/s, ETA: {format_eta(eta)}",
                    end="",
                    flush=True,
                )

    # Final timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {format_eta(elapsed)} ({elapsed/total*1000:.1f}ms/position)")

    # Apply results to positions
    for idx, top_moves in results.items():
        positions[idx]["stockfish_top_moves"] = top_moves

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
        print(f"  Stockfish top {args.top_moves}:  {pos.get('stockfish_top_moves', [])}")

        # Check if Jeb's move was in Stockfish's top moves
        if "stockfish_top_moves" in pos:
            if pos["move_uci"] in pos["stockfish_top_moves"]:
                rank = pos["stockfish_top_moves"].index(pos["move_uci"]) + 1
                print(f"  Jeb's move rank:  #{rank} in Stockfish top {args.top_moves}")
            else:
                print(f"  Jeb's move rank:  Not in top {args.top_moves} (unique style!)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Updated {len(positions):,} positions with Stockfish analysis")
    print(f"Output saved to: {output_file}")
    print(f"Total time: {format_eta(elapsed)}")
    print(f"Throughput: {len(positions)/elapsed:.1f} positions/second")


if __name__ == "__main__":
    main()
