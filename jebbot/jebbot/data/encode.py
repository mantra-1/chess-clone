"""Encode chess positions as tensors for neural network training."""

import json
import random
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

# Number of negative examples per positive example
NUM_NEGATIVES = 4


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
    """PyTorch Dataset for chess positions with positive and negative examples.

    For each position, returns:
    - 1 positive example: the move that was actually played (target = 1.0)
    - 4 negative examples: random legal moves NOT played (target = 0.0)

    This creates a balanced training signal where the model learns to
    distinguish "Jeb-like" moves from other legal moves.
    """

    def __init__(self, positions_file: Path, seed: int = 42):
        """Initialize dataset from positions JSON file.

        Args:
            positions_file: Path to training_positions.json
            seed: Random seed for reproducible negative sampling
        """
        self.positions_file = Path(positions_file)
        self.seed = seed
        self.rng = random.Random(seed)

        with open(self.positions_file) as f:
            self.positions = json.load(f)

        # Cache for legal moves per position (computed lazily)
        self._legal_moves_cache: dict[int, list[int]] = {}

        # Cache for position tensors (computed lazily)
        self._tensor_cache: dict[int, np.ndarray] = {}

    def _get_legal_moves(self, position_idx: int) -> list[int]:
        """Get list of legal move indices for a position (cached).

        Args:
            position_idx: Index into self.positions

        Returns:
            List of move indices (0-4095) for all legal moves
        """
        if position_idx not in self._legal_moves_cache:
            position = self.positions[position_idx]
            fen = position["fen"]
            board = chess.Board(fen)

            legal_indices = []
            for move in board.legal_moves:
                idx = move.from_square * 64 + move.to_square
                legal_indices.append(idx)

            self._legal_moves_cache[position_idx] = legal_indices

        return self._legal_moves_cache[position_idx]

    def _get_position_tensor(self, position_idx: int) -> np.ndarray:
        """Get position tensor (cached).

        Args:
            position_idx: Index into self.positions

        Returns:
            numpy array of shape (12, 8, 8)
        """
        if position_idx not in self._tensor_cache:
            position = self.positions[position_idx]
            fen = position["fen"]
            self._tensor_cache[position_idx] = fen_to_tensor(fen)

        return self._tensor_cache[position_idx]

    def __len__(self) -> int:
        """Return total number of examples (1 positive + 4 negatives per position)."""
        return len(self.positions) * (1 + NUM_NEGATIVES)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        """Get a single training example.

        Args:
            idx: Index in range [0, len(positions) * 5)

        Returns:
            Tuple of (position_tensor, move_index, target) where:
            - position_tensor: torch.Tensor of shape (12, 8, 8)
            - move_index: int in range [0, 4095]
            - target: float, 1.0 for positive (actual move), 0.0 for negative
        """
        # Decode index: which position and which variant (0=positive, 1-4=negative)
        position_idx = idx // (1 + NUM_NEGATIVES)
        variant = idx % (1 + NUM_NEGATIVES)

        position = self.positions[position_idx]
        actual_move_uci = position["move_uci"]
        actual_move_idx = move_to_index(actual_move_uci)

        # Get cached position tensor
        tensor = self._get_position_tensor(position_idx)

        if variant == 0:
            # Positive example: the move that was actually played
            return torch.from_numpy(tensor), actual_move_idx, 1.0
        else:
            # Negative example: a random legal move that was NOT played
            legal_moves = self._get_legal_moves(position_idx)

            # Filter out the actual move
            negative_moves = [m for m in legal_moves if m != actual_move_idx]

            if not negative_moves:
                # Edge case: only one legal move (very rare)
                # Return the actual move but with target 0.0 to maintain dataset size
                # This is a bit hacky but keeps things simple
                return torch.from_numpy(tensor), actual_move_idx, 0.0

            # Use deterministic sampling based on idx for reproducibility
            # This ensures same negative is returned for same idx
            rng = random.Random(self.seed + idx)
            negative_move_idx = rng.choice(negative_moves)

            return torch.from_numpy(tensor), negative_move_idx, 0.0

    def get_position_metadata(self, idx: int) -> dict:
        """Get full position metadata for inspection.

        Args:
            idx: Index of original position (NOT the expanded idx)

        Returns:
            Full position dict with fen, move, game_id, etc.
        """
        return self.positions[idx]

    def get_num_positions(self) -> int:
        """Get number of original positions (before expansion)."""
        return len(self.positions)
