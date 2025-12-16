"""Encode chess positions as tensors for neural network training."""

import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset


# Piece to channel index mapping
# Channels 0-5: White pieces (P, N, B, R, Q, K)
# Channels 6-11: Black pieces (p, n, b, r, q, k)
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def fen_to_tensor(fen: str) -> np.ndarray:
    """Convert FEN string to a (12, 8, 8) numpy array.

    Args:
        fen: FEN string representing board position

    Returns:
        numpy array of shape (12, 8, 8) where:
        - Channels 0-5: White P, N, B, R, Q, K
        - Channels 6-11: Black p, n, b, r, q, k
        - Each channel is 8x8 binary (1 if piece present, 0 otherwise)
    """
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get channel index
            channel = PIECE_TO_CHANNEL[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6  # Black pieces in channels 6-11

            # Convert square to row, col (rank 1 = row 0, file a = col 0)
            row = chess.square_rank(square)
            col = chess.square_file(square)

            tensor[channel, row, col] = 1.0

    return tensor


def move_to_index(move_uci: str) -> int:
    """Convert UCI move string to an index 0-4095.

    Args:
        move_uci: UCI move string (e.g., "e2e4")

    Returns:
        Index in range [0, 4095] computed as from_square * 64 + to_square
    """
    move = chess.Move.from_uci(move_uci)
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square


def index_to_move(index: int) -> str:
    """Convert index 0-4095 back to UCI move string.

    Args:
        index: Index in range [0, 4095]

    Returns:
        UCI move string (e.g., "e2e4")
    """
    from_square = index // 64
    to_square = index % 64
    move = chess.Move(from_square, to_square)
    return move.uci()


class ChessPositionDataset(Dataset):
    """PyTorch Dataset for chess positions.

    Loads training positions from JSON file and provides (position_tensor, move_index) pairs.
    """

    def __init__(self, positions_file: Path):
        """Initialize dataset from positions JSON file.

        Args:
            positions_file: Path to training_positions.json
        """
        self.positions_file = Path(positions_file)

        with open(self.positions_file) as f:
            self.positions = json.load(f)

    def __len__(self) -> int:
        """Return number of positions in dataset."""
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single training example.

        Args:
            idx: Index of position to retrieve

        Returns:
            Tuple of (position_tensor, move_index) where:
            - position_tensor: torch.Tensor of shape (12, 8, 8)
            - move_index: int in range [0, 4095]
        """
        position = self.positions[idx]

        # Convert FEN to tensor
        fen = position["fen"]
        tensor = fen_to_tensor(fen)

        # Convert move to index
        move_uci = position["move_uci"]
        move_idx = move_to_index(move_uci)

        return torch.from_numpy(tensor), move_idx

    def get_position_metadata(self, idx: int) -> dict:
        """Get full position metadata for inspection.

        Args:
            idx: Index of position

        Returns:
            Full position dict with fen, move, game_id, etc.
        """
        return self.positions[idx]
