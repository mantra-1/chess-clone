"""Parse chess games into training examples."""

import io
import json
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn


USERNAME = "jebhead"
SKIP_FIRST_N_MOVES = 5
# Results that indicate the game didn't end normally
SKIP_RESULTS = {"timeout", "abandoned", "timevsinsufficient"}


def parse_game_to_positions(game_data: dict, username: str = USERNAME) -> list[dict]:
    """Parse a single game into training positions.

    Args:
        game_data: Game dict from chess.com API
        username: Username to extract positions for (case-insensitive)

    Returns:
        List of training position dicts
    """
    username_lower = username.lower()

    # Determine player's color
    white_user = game_data.get("white", {}).get("username", "").lower()
    black_user = game_data.get("black", {}).get("username", "").lower()

    if white_user == username_lower:
        player_color = chess.WHITE
        color_str = "white"
        result = game_data.get("white", {}).get("result", "")
    elif black_user == username_lower:
        player_color = chess.BLACK
        color_str = "black"
        result = game_data.get("black", {}).get("result", "")
    else:
        return []  # User not in this game

    # Skip games that ended abnormally
    if result in SKIP_RESULTS:
        return []

    # Get metadata
    game_id = game_data.get("url", "").split("/")[-1] if game_data.get("url") else ""
    time_control = game_data.get("time_class", "unknown")

    # Parse PGN
    pgn_string = game_data.get("pgn", "")
    if not pgn_string:
        return []

    try:
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return []
    except Exception:
        return []

    positions = []
    board = game.board()
    move_number = 0

    for node in game.mainline():
        move = node.move
        # move_number counts full moves (increments after black moves)
        if board.turn == chess.WHITE:
            move_number += 1

        # Check if it's the player's turn BEFORE the move is made
        if board.turn == player_color:
            # Skip opening moves
            if move_number > SKIP_FIRST_N_MOVES:
                position = {
                    "fen": board.fen(),
                    "move": move.uci(),
                    "move_uci": move.uci(),
                    "game_id": game_id,
                    "color": color_str,
                    "move_number": move_number,
                    "time_control": time_control,
                }
                positions.append(position)

        # Make the move on the board
        board.push(move)

    return positions


def load_games_from_file(filepath: Path) -> list[dict]:
    """Load games from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("games", [])


def process_all_games(
    raw_dir: Path,
    username: str = USERNAME,
) -> Iterator[dict]:
    """Process all games from raw JSON files.

    Args:
        raw_dir: Directory containing raw JSON files
        username: Username to extract positions for

    Yields:
        Training position dicts
    """
    json_files = sorted(raw_dir.glob("*.json"))

    for filepath in json_files:
        games = load_games_from_file(filepath)
        for game_data in games:
            positions = parse_game_to_positions(game_data, username)
            yield from positions


def build_training_dataset(
    raw_dir: Path,
    username: str = USERNAME,
) -> tuple[list[dict], dict]:
    """Build complete training dataset from raw games.

    Args:
        raw_dir: Directory containing raw JSON files
        username: Username to extract positions for

    Returns:
        Tuple of (positions list, stats dict)
    """
    json_files = sorted(raw_dir.glob("*.json"))

    all_positions = []
    total_games = 0
    skipped_games = 0

    for filepath in json_files:
        games = load_games_from_file(filepath)
        for game_data in games:
            total_games += 1
            positions = parse_game_to_positions(game_data, username)
            if positions:
                all_positions.extend(positions)
            else:
                skipped_games += 1

    stats = {
        "total_games": total_games,
        "games_with_positions": total_games - skipped_games,
        "skipped_games": skipped_games,
        "total_positions": len(all_positions),
        "avg_positions_per_game": (
            len(all_positions) / (total_games - skipped_games)
            if total_games > skipped_games
            else 0
        ),
    }

    return all_positions, stats
