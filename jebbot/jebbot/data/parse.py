"""Parse chess games into training examples."""

import io
import json
from pathlib import Path
from typing import Iterator, Optional

import chess
import chess.pgn

try:
    from stockfish import Stockfish
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False


USERNAME = "jebhead"
SKIP_FIRST_N_MOVES = 5
# Results that indicate the game didn't end normally
SKIP_RESULTS = {"timeout", "abandoned", "timevsinsufficient"}

# Stockfish settings
STOCKFISH_ELO = 1500
STOCKFISH_TOP_N_MOVES = 6  # Reduced from 10 for faster analysis
STOCKFISH_DEPTH = 8  # Reduced from default ~15 for faster analysis


def get_stockfish_top_moves(
    fen: str,
    stockfish: "Stockfish",
    num_moves: int = STOCKFISH_TOP_N_MOVES,
) -> list[str]:
    """Get top N moves from Stockfish for a position.

    Args:
        fen: FEN string of the position
        stockfish: Initialized Stockfish instance
        num_moves: Number of top moves to return

    Returns:
        List of UCI move strings (e.g., ["e2e4", "d2d4", ...])
    """
    try:
        stockfish.set_fen_position(fen)
        top_moves = stockfish.get_top_moves(num_moves)
        return [move["Move"] for move in top_moves]
    except Exception:
        return []


def init_stockfish(
    path: Optional[str] = None,
    elo: int = STOCKFISH_ELO,
    depth: int = STOCKFISH_DEPTH,
) -> Optional["Stockfish"]:
    """Initialize Stockfish engine with ELO limit and depth.

    Args:
        path: Path to stockfish binary (auto-detected if None)
        elo: ELO rating limit for the engine
        depth: Search depth for analysis (lower = faster)

    Returns:
        Stockfish instance or None if not available
    """
    if not STOCKFISH_AVAILABLE:
        print("Warning: stockfish package not installed")
        return None

    try:
        # Common paths for stockfish binary
        paths_to_try = [
            path,
            "/opt/homebrew/bin/stockfish",  # Mac M1/M2/M3 homebrew
            "/usr/local/bin/stockfish",      # Mac Intel homebrew
            "/usr/bin/stockfish",            # Linux
            "stockfish",                      # In PATH
        ]

        stockfish = None
        for p in paths_to_try:
            if p is None:
                continue
            try:
                stockfish = Stockfish(path=p)
                break
            except Exception:
                continue

        if stockfish is None:
            stockfish = Stockfish()  # Let it auto-detect

        stockfish.set_elo_rating(elo)
        stockfish.set_depth(depth)
        return stockfish
    except Exception as e:
        print(f"Warning: Could not initialize Stockfish: {e}")
        return None


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
    use_stockfish: bool = False,
    stockfish_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> tuple[list[dict], dict]:
    """Build complete training dataset from raw games.

    Args:
        raw_dir: Directory containing raw JSON files
        username: Username to extract positions for
        use_stockfish: Whether to add Stockfish analysis
        stockfish_path: Path to stockfish binary
        progress_callback: Optional callback(current, total, message) for progress

    Returns:
        Tuple of (positions list, stats dict)
    """
    json_files = sorted(raw_dir.glob("*.json"))

    all_positions = []
    total_games = 0
    skipped_games = 0

    # First pass: collect all positions
    for filepath in json_files:
        games = load_games_from_file(filepath)
        for game_data in games:
            total_games += 1
            positions = parse_game_to_positions(game_data, username)
            if positions:
                all_positions.extend(positions)
            else:
                skipped_games += 1

    # Second pass: add Stockfish analysis if requested
    if use_stockfish:
        stockfish = init_stockfish(stockfish_path)
        if stockfish:
            total = len(all_positions)
            for i, position in enumerate(all_positions):
                if progress_callback and i % 100 == 0:
                    progress_callback(i, total, f"Analyzing position {i}/{total}")

                top_moves = get_stockfish_top_moves(position["fen"], stockfish)
                position["stockfish_top_moves"] = top_moves

            if progress_callback:
                progress_callback(total, total, "Stockfish analysis complete")
        else:
            print("Warning: Stockfish not available, skipping analysis")

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
        "has_stockfish_analysis": use_stockfish and STOCKFISH_AVAILABLE,
    }

    return all_positions, stats


def add_stockfish_to_positions(
    positions: list[dict],
    stockfish_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> list[dict]:
    """Add Stockfish top moves to existing positions.

    Args:
        positions: List of position dicts
        stockfish_path: Path to stockfish binary
        progress_callback: Optional callback(current, total, message) for progress

    Returns:
        Updated positions list with stockfish_top_moves field
    """
    stockfish = init_stockfish(stockfish_path)
    if not stockfish:
        print("Error: Could not initialize Stockfish")
        return positions

    total = len(positions)
    for i, position in enumerate(positions):
        if progress_callback and i % 100 == 0:
            progress_callback(i, total, f"Analyzing position {i}/{total}")

        top_moves = get_stockfish_top_moves(position["fen"], stockfish)
        position["stockfish_top_moves"] = top_moves

    if progress_callback:
        progress_callback(total, total, "Complete")

    return positions
