#!/usr/bin/env python3
"""Explore and analyze downloaded chess games."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess.pgn
import io


USERNAME = "jebhead"


def load_all_games(raw_dir: Path) -> list[dict]:
    """Load all games from JSON files in raw directory."""
    games = []
    json_files = sorted(raw_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {raw_dir}")
        return games

    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)
            games.extend(data.get("games", []))

    return games


def get_time_control_category(time_class: str) -> str:
    """Normalize time control category."""
    return time_class.lower() if time_class else "unknown"


def get_result_for_player(game: dict, username: str) -> tuple[str, str]:
    """Get result and color for a specific player.

    Returns:
        Tuple of (color, result) where result is 'win', 'loss', or 'draw'
    """
    username_lower = username.lower()
    white = game.get("white", {})
    black = game.get("black", {})

    white_username = white.get("username", "").lower()
    black_username = black.get("username", "").lower()

    if white_username == username_lower:
        color = "white"
        result = white.get("result", "")
    elif black_username == username_lower:
        color = "black"
        result = black.get("result", "")
    else:
        return ("unknown", "unknown")

    # Chess.com result values
    if result == "win":
        return (color, "win")
    elif result in ("checkmated", "timeout", "resigned", "abandoned", "lose"):
        return (color, "loss")
    elif result in ("agreed", "stalemate", "repetition", "insufficient",
                    "50move", "timevsinsufficient", "draw"):
        return (color, "draw")
    else:
        return (color, "unknown")


def parse_opening_from_pgn(pgn_string: str) -> str:
    """Extract opening name from PGN using python-chess."""
    try:
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return "Unknown"

        # Try ECOUrl first (more descriptive), then Opening header
        eco_url = game.headers.get("ECOUrl", "")
        if eco_url:
            # Extract opening name from URL like:
            # https://www.chess.com/openings/Queens-Gambit-Declined-3.Nc3-Be7
            match = re.search(r"/openings/(.+)$", eco_url)
            if match:
                opening = match.group(1).replace("-", " ")
                # Take just the main opening name (before variations get too specific)
                parts = opening.split()
                # Keep first 3-4 meaningful words
                return " ".join(parts[:4])

        opening = game.headers.get("Opening", "")
        if opening:
            return opening

        return "Unknown"
    except Exception:
        return "Unknown"


def extract_year_month(game: dict) -> str:
    """Extract year-month from game end_time."""
    end_time = game.get("end_time", 0)
    if end_time:
        from datetime import datetime
        dt = datetime.fromtimestamp(end_time)
        return dt.strftime("%Y-%m")
    return "Unknown"


def print_bar(count: int, max_count: int, width: int = 30) -> str:
    """Create a simple text bar chart."""
    if max_count == 0:
        return ""
    bar_length = int((count / max_count) * width)
    return "█" * bar_length


def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    print(f"Loading games from {data_dir}...")
    games = load_all_games(data_dir)

    if not games:
        print("No games found. Run fetch_games.py first.")
        return

    print(f"\n{'='*50}")
    print(f"CHESS.COM GAME ANALYSIS FOR: {USERNAME}")
    print(f"{'='*50}\n")

    # Total games
    print(f"Total games: {len(games)}\n")

    # Games by time control
    print("Games by Time Control:")
    print("-" * 30)
    time_controls = Counter(get_time_control_category(g.get("time_class", "")) for g in games)
    for tc, count in time_controls.most_common():
        pct = count / len(games) * 100
        print(f"  {tc:12} {count:5} ({pct:5.1f}%)")
    print()

    # Win/loss/draw record
    print("Win/Loss/Draw Record:")
    print("-" * 30)

    results_white = Counter()
    results_black = Counter()

    for game in games:
        color, result = get_result_for_player(game, USERNAME)
        if color == "white":
            results_white[result] += 1
        elif color == "black":
            results_black[result] += 1

    total_white = sum(results_white.values())
    total_black = sum(results_black.values())

    print(f"  As White ({total_white} games):")
    for result in ["win", "loss", "draw"]:
        count = results_white[result]
        pct = count / total_white * 100 if total_white > 0 else 0
        print(f"    {result:6} {count:5} ({pct:5.1f}%)")

    print(f"  As Black ({total_black} games):")
    for result in ["win", "loss", "draw"]:
        count = results_black[result]
        pct = count / total_black * 100 if total_black > 0 else 0
        print(f"    {result:6} {count:5} ({pct:5.1f}%)")

    # Overall record
    total_wins = results_white["win"] + results_black["win"]
    total_losses = results_white["loss"] + results_black["loss"]
    total_draws = results_white["draw"] + results_black["draw"]
    print(f"  Overall: +{total_wins} -{total_losses} ={total_draws}")
    print()

    # Most played openings
    print("Top 10 Most Played Openings:")
    print("-" * 30)

    openings = Counter()
    for game in games:
        pgn = game.get("pgn", "")
        if pgn:
            opening = parse_opening_from_pgn(pgn)
            openings[opening] += 1

    for opening, count in openings.most_common(10):
        pct = count / len(games) * 100
        print(f"  {count:4} ({pct:4.1f}%) {opening}")
    print()

    # Monthly activity
    print("Monthly Game Count:")
    print("-" * 30)

    monthly = Counter(extract_year_month(g) for g in games)
    monthly_sorted = sorted(monthly.items())

    if monthly_sorted:
        max_count = max(monthly.values())
        for month, count in monthly_sorted:
            bar = print_bar(count, max_count, 20)
            print(f"  {month}: {bar} {count}")
    print()


if __name__ == "__main__":
    main()
