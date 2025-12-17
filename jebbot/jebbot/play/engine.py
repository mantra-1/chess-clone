"""JebBot chess engine - uses trained model to select moves."""

import shutil
from pathlib import Path
from typing import Optional

import chess
import torch
from stockfish import Stockfish

from jebbot.data.encode import fen_to_tensor, move_to_index
from jebbot.model.style_selector import StyleSelector

# Common Stockfish paths to try
STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",  # Mac Homebrew (Apple Silicon)
    "/usr/local/bin/stockfish",  # Mac Homebrew (Intel) / Linux manual install
    "/usr/bin/stockfish",  # Linux package manager
]


def find_stockfish() -> str:
    """Find Stockfish binary path.

    Returns:
        Path to Stockfish binary

    Raises:
        FileNotFoundError: If Stockfish cannot be found
    """
    # First try to find in PATH
    path_stockfish = shutil.which("stockfish")
    if path_stockfish:
        return path_stockfish

    # Try common installation paths
    for path in STOCKFISH_PATHS:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "Stockfish not found. Please install it:\n"
        "  Mac: brew install stockfish\n"
        "  Ubuntu/Debian: sudo apt install stockfish\n"
        "  Or download from https://stockfishchess.org/download/"
    )


class JebBotEngine:
    """Chess engine that plays in Jeb's style using trained model."""

    def __init__(
        self,
        model_path: str | Path,
        stockfish_path: Optional[str] = None,
        elo: int = 1500,
    ):
        """Initialize JebBot engine.

        Args:
            model_path: Path to trained model checkpoint
            stockfish_path: Path to Stockfish binary (auto-detect if None)
            elo: Stockfish ELO rating for candidate moves
        """
        self.device = self._get_device()

        # Load model
        self.model = StyleSelector()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Find Stockfish path if not provided
        if stockfish_path is None:
            stockfish_path = find_stockfish()

        # Initialize Stockfish
        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.set_elo_rating(elo)

    def _get_device(self) -> torch.device:
        """Get best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_candidates(self, fen: str, n: int = 5) -> list[str]:
        """Get top N candidate moves from Stockfish.

        Args:
            fen: FEN string of current position
            n: Number of candidate moves to return

        Returns:
            List of UCI move strings
        """
        self.stockfish.set_fen_position(fen)
        top_moves = self.stockfish.get_top_moves(n)

        moves = []
        for move_info in top_moves:
            move_uci = move_info["Move"]
            # Convert castling notation if needed
            move_uci = self._normalize_castling(fen, move_uci)
            moves.append(move_uci)

        return moves

    def _normalize_castling(self, fen: str, move_uci: str) -> str:
        """Normalize castling moves to standard UCI format.

        Stockfish sometimes returns O-O notation or different formats.
        This ensures e1g1/e1c1 for white, e8g8/e8c8 for black.

        Args:
            fen: Current position FEN
            move_uci: Move in UCI format

        Returns:
            Normalized UCI move string
        """
        board = chess.Board(fen)

        # Try to parse as UCI
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move_uci
        except ValueError:
            pass

        # Handle potential castling notation variations
        if move_uci.upper() in ("O-O", "0-0"):
            if board.turn == chess.WHITE:
                return "e1g1"
            else:
                return "e8g8"
        elif move_uci.upper() in ("O-O-O", "0-0-0"):
            if board.turn == chess.WHITE:
                return "e1c1"
            else:
                return "e8c8"

        return move_uci

    def score_moves(self, fen: str, moves: list[str]) -> list[float]:
        """Score moves using the trained model.

        Args:
            fen: FEN string of current position
            moves: List of UCI move strings to score

        Returns:
            List of confidence scores (0-1) for each move
        """
        if not moves:
            return []

        # Encode position
        position_tensor = torch.tensor(fen_to_tensor(fen)).unsqueeze(0)
        position_tensor = position_tensor.to(self.device)

        scores = []
        with torch.no_grad():
            for move_uci in moves:
                move_idx = move_to_index(move_uci)
                move_tensor = torch.tensor([move_idx]).to(self.device)

                # Get model prediction
                confidence = self.model(position_tensor, move_tensor)
                scores.append(confidence.item())

        return scores

    def select_move(self, fen: str, n_candidates: int = 5) -> dict:
        """Select a move balancing Stockfish quality with Jeb's style.

        Selection logic:
        1. Find first Stockfish move with >50% Jeb score ("safe pick")
        2. If another move has Jeb score 15%+ higher, play that instead
        3. If no move is >50% Jeb, fall back to highest Jeb score

        Args:
            fen: FEN string of current position
            n_candidates: Number of Stockfish candidates to consider

        Returns:
            Dict with:
                - selected_move: UCI string of chosen move
                - candidates: List of {move, score, algebraic} dicts
                - selection_reason: "safe_pick", "strong_jeb_preference", or "fallback"
        """
        board = chess.Board(fen)

        # Get candidate moves from Stockfish (ranked by quality)
        candidates = self.get_candidates(fen, n_candidates)

        if not candidates:
            # Fallback: get any legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                candidates = [legal_moves[0].uci()]
            else:
                return {"selected_move": None, "candidates": [], "selection_reason": "no_moves"}

        # Score all candidates
        scores = self.score_moves(fen, candidates)

        # Build results with algebraic notation
        results = []
        for move_uci, score in zip(candidates, scores):
            try:
                move = chess.Move.from_uci(move_uci)
                algebraic = board.san(move)
            except (ValueError, chess.InvalidMoveError):
                algebraic = move_uci

            results.append({
                "move": move_uci,
                "score": score,
                "algebraic": algebraic,
            })

        # New selection logic
        JEB_THRESHOLD = 0.50  # Minimum score to be considered "Jeb-like"
        PREFERENCE_MARGIN = 0.15  # How much higher to override safe pick

        # Find safe pick: first Stockfish move (best quality) with >50% Jeb score
        safe_pick_idx = None
        for i, score in enumerate(scores):
            if score > JEB_THRESHOLD:
                safe_pick_idx = i
                break

        if safe_pick_idx is not None:
            safe_pick_score = scores[safe_pick_idx]

            # Check if any move has significantly higher Jeb score
            best_jeb_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_jeb_score = scores[best_jeb_idx]

            if best_jeb_score >= safe_pick_score + PREFERENCE_MARGIN:
                # Strong Jeb preference - play the more Jeb-like move
                selected_move = candidates[best_jeb_idx]
                selection_reason = "strong_jeb_preference"
            else:
                # Play the safe pick (good chess + acceptable Jeb score)
                selected_move = candidates[safe_pick_idx]
                selection_reason = "safe_pick"
        else:
            # No move is >50% Jeb, fall back to highest Jeb score
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            selected_move = candidates[best_idx]
            selection_reason = "fallback"

        return {
            "selected_move": selected_move,
            "candidates": results,
            "selection_reason": selection_reason,
        }

    def get_stockfish_move(self, fen: str) -> str:
        """Get Stockfish's best move for the position.

        Args:
            fen: FEN string of current position

        Returns:
            UCI move string
        """
        self.stockfish.set_fen_position(fen)
        move = self.stockfish.get_best_move()
        return self._normalize_castling(fen, move)
