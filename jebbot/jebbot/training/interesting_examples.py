"""Log interesting examples during training for later analysis."""

import json
from pathlib import Path
from typing import Any


class InterestingExamplesLogger:
    """Collects and stores interesting training examples.

    Tracks four categories of examples:
    - high_confidence_correct: Model confidently recognizes Jeb's style
    - high_confidence_wrong: Model fooled by Stockfish moves
    - missed_jeb_move: Model failed to recognize Jeb's move
    - confident_not_jeb: Model correctly rejects non-Jeb moves
    """

    # Thresholds for each category
    HIGH_CONFIDENCE_CORRECT_THRESHOLD = 0.90  # confidence > 0.90 and label == 1.0
    HIGH_CONFIDENCE_WRONG_THRESHOLD = 0.85  # confidence > 0.85 and label == 0.0
    MISSED_JEB_THRESHOLD = 0.15  # confidence < 0.15 and label == 1.0
    CONFIDENT_NOT_JEB_THRESHOLD = 0.10  # confidence < 0.10 and label == 0.0

    MAX_EXAMPLES_PER_CATEGORY = 100

    def __init__(self) -> None:
        """Initialize empty example lists."""
        self.high_confidence_correct: list[dict[str, Any]] = []
        self.high_confidence_wrong: list[dict[str, Any]] = []
        self.missed_jeb_move: list[dict[str, Any]] = []
        self.confident_not_jeb: list[dict[str, Any]] = []

    def log_batch(
        self,
        fens: list[str],
        moves: list[str],
        confidences: list[float],
        labels: list[float],
        epoch: int,
        batch: int,
    ) -> None:
        """Check batch for interesting examples and store them.

        Args:
            fens: List of FEN strings for each position
            moves: List of UCI move strings
            confidences: Model confidence (0-1) for each position
            labels: Actual labels (1.0 for Jeb, 0.0 for not Jeb)
            epoch: Current epoch number
            batch: Current batch number
        """
        for fen, move, conf, label in zip(fens, moves, confidences, labels):
            entry = {
                "fen": fen,
                "move": move,
                "confidence": conf,
                "label": label,
                "epoch": epoch,
                "batch": batch,
            }

            is_jeb = label >= 0.5

            if is_jeb and conf > self.HIGH_CONFIDENCE_CORRECT_THRESHOLD:
                self._add_example(self.high_confidence_correct, entry, key="confidence", keep_highest=True)
            elif not is_jeb and conf > self.HIGH_CONFIDENCE_WRONG_THRESHOLD:
                self._add_example(self.high_confidence_wrong, entry, key="confidence", keep_highest=True)
            elif is_jeb and conf < self.MISSED_JEB_THRESHOLD:
                self._add_example(self.missed_jeb_move, entry, key="confidence", keep_highest=False)
            elif not is_jeb and conf < self.CONFIDENT_NOT_JEB_THRESHOLD:
                self._add_example(self.confident_not_jeb, entry, key="confidence", keep_highest=False)

    def _add_example(
        self,
        category: list[dict[str, Any]],
        entry: dict[str, Any],
        key: str,
        keep_highest: bool,
    ) -> None:
        """Add example to category, keeping only the most extreme ones.

        Args:
            category: The category list to add to
            entry: The example entry to add
            key: The key to sort by (e.g., "confidence")
            keep_highest: If True, keep highest values; if False, keep lowest
        """
        category.append(entry)

        if len(category) > self.MAX_EXAMPLES_PER_CATEGORY:
            # Sort and trim to keep most extreme examples
            category.sort(key=lambda x: x[key], reverse=keep_highest)
            del category[self.MAX_EXAMPLES_PER_CATEGORY :]

    def save(self, path: Path | str) -> None:
        """Save all examples to a JSON file.

        Args:
            path: Path to save the JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "high_confidence_correct": self.high_confidence_correct,
            "high_confidence_wrong": self.high_confidence_wrong,
            "missed_jeb_move": self.missed_jeb_move,
            "confident_not_jeb": self.confident_not_jeb,
            "summary": {
                "high_confidence_correct_count": len(self.high_confidence_correct),
                "high_confidence_wrong_count": len(self.high_confidence_wrong),
                "missed_jeb_move_count": len(self.missed_jeb_move),
                "confident_not_jeb_count": len(self.confident_not_jeb),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> dict[str, int]:
        """Get counts for each category."""
        return {
            "high_confidence_correct": len(self.high_confidence_correct),
            "high_confidence_wrong": len(self.high_confidence_wrong),
            "missed_jeb_move": len(self.missed_jeb_move),
            "confident_not_jeb": len(self.confident_not_jeb),
        }
