#!/usr/bin/env python3
"""Fetch all chess games for a user from chess.com."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jebbot.data.download import download_all_games, get_game_archives


def main():
    parser = argparse.ArgumentParser(description="Download chess.com games for a user")
    parser.add_argument(
        "username",
        help="Chess.com username to download games for",
    )
    args = parser.parse_args()

    username = args.username
    output_dir = Path(__file__).parent.parent / "data"

    print(f"Fetching games for: {username}")
    print(f"Output directory: {output_dir}")
    print("-" * 40)

    start_time = time.time()

    # Get archives to determine date range
    archives = get_game_archives(username)
    if not archives:
        print(f"No games found for user: {username}")
        return

    # Extract date range from archive URLs
    first_archive = archives[0].rstrip("/").split("/")
    last_archive = archives[-1].rstrip("/").split("/")
    first_date = f"{first_archive[-2]}-{first_archive[-1]}"
    last_date = f"{last_archive[-2]}-{last_archive[-1]}"

    # Download all games
    total_games = download_all_games(username, output_dir)

    elapsed = time.time() - start_time

    print("-" * 40)
    print(f"Summary:")
    print(f"  Username:    {username}")
    print(f"  Total games: {total_games}")
    print(f"  Date range:  {first_date} to {last_date}")
    print(f"  Time elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
