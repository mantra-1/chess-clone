"""Opening book for JebBot - weighted random opening choices."""

import random
from typing import Optional

import chess


# Opening book for WHITE
# Key: tuple of moves played so far (as UCI strings)
# Value: list of (move, weight) tuples
WHITE_BOOK = {
    # Move 1: Always e4
    (): [("e2e4", 100)],

    # After 1...e5
    ("e2e4", "e7e5"): [
        ("d2d4", 30),   # Danish Gambit
        ("g1f3", 40),   # Italian/Spanish setup
        ("f1c4", 30),   # Italian
    ],

    # After 1...c5 (Sicilian)
    ("e2e4", "c7c5"): [
        ("g1f3", 60),   # Open Sicilian
        ("b1c3", 25),   # Closed Sicilian
        ("c2c3", 15),   # Alapin
    ],

    # After 1...c6 (Caro-Kann)
    ("e2e4", "c7c6"): [
        ("d2d4", 80),
        ("b1c3", 20),
    ],

    # After 1...e6 (French)
    ("e2e4", "e7e6"): [
        ("d2d4", 90),
        ("d2d3", 10),
    ],

    # After 1...d5 (Scandinavian)
    ("e2e4", "d7d5"): [
        ("e4d5", 100),  # Always capture
    ],

    # After 1...d6 (Pirc)
    ("e2e4", "d7d6"): [
        ("d2d4", 80),
        ("g1f3", 20),
    ],

    # After 1...g6 (Modern)
    ("e2e4", "g7g6"): [
        ("d2d4", 85),
        ("g1f3", 15),
    ],
}


# Opening book for BLACK
# Key: tuple of moves played so far (as UCI strings)
# Value: list of (move, weight) tuples
BLACK_BOOK = {
    # vs 1.e4
    ("e2e4",): [
        ("c7c6", 45),   # Caro-Kann
        ("c7c5", 45),   # Sicilian
        ("d7d5", 10),   # Scandinavian
    ],

    # vs 1.d4
    ("d2d4",): [
        ("d7d5", 50),   # Queen's Pawn / Slav setup
        ("g7g6", 35),   # Modern Defense
        ("g8f6", 15),   # Indian setups
    ],

    # vs 1.c4 (English)
    ("c2c4",): [
        ("e7e5", 60),   # Reversed Sicilian
        ("c7c5", 40),   # Symmetrical
    ],

    # vs 1.Nf3
    ("g1f3",): [
        ("d7d5", 50),
        ("g7g6", 30),
        ("c7c5", 20),
    ],

    # vs 1.b3 (Larsen)
    ("b2b3",): [
        ("e7e5", 50),
        ("d7d5", 50),
    ],

    # vs 1.g3 (Benko)
    ("g2g3",): [
        ("d7d5", 60),
        ("e7e5", 40),
    ],

    # Caro-Kann continuations after 1.e4 c6 2.d4
    ("e2e4", "c7c6", "d2d4"): [
        ("d7d5", 100),  # Main line
    ],

    # Sicilian continuations after 1.e4 c5 2.Nf3
    ("e2e4", "c7c5", "g1f3"): [
        ("d7d6", 50),   # Najdorf/Dragon setup
        ("b8c6", 35),   # Classical
        ("e7e6", 15),   # Scheveningen
    ],
}


def get_book_move(fen: str, move_history: list[str]) -> Optional[str]:
    """Get a book move if available.

    Args:
        fen: Current position FEN
        move_history: List of UCI moves played so far

    Returns:
        UCI move string if in book, None otherwise
    """
    board = chess.Board(fen)
    history_tuple = tuple(move_history)

    # Determine which book to use based on side to move
    if board.turn == chess.WHITE:
        book = WHITE_BOOK
    else:
        book = BLACK_BOOK

    # Look up position in book
    if history_tuple not in book:
        return None

    options = book[history_tuple]
    moves = [m for m, w in options]
    weights = [w for m, w in options]

    # Weighted random selection
    selected = random.choices(moves, weights=weights, k=1)[0]

    # Verify the move is legal
    try:
        move = chess.Move.from_uci(selected)
        if move in board.legal_moves:
            return selected
    except ValueError:
        pass

    return None
