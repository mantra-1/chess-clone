"""Client for sending training updates to visualization server."""

import json
import threading
from urllib.error import URLError
from urllib.request import Request, urlopen

# Default server URL
DEFAULT_URL = "http://localhost:8765/update"


def send_update(
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    positions: list[dict],
    url: str = DEFAULT_URL,
    timeout: float = 0.5,
) -> None:
    """Send training update to visualization server (non-blocking).

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        batch: Current batch number
        total_batches: Total batches per epoch
        train_loss: Current training loss
        val_loss: Current validation loss (0 if not yet computed)
        val_accuracy: Current validation accuracy (0 if not yet computed)
        positions: List of position dicts for visualization. Each dict has:
            - fen: FEN string of the board position
            - move: UCI notation of the move being evaluated (e.g., "e2e4")
            - confidence: Model's output probability (0.0 to 1.0)
            - label: Actual label (1.0 = Jeb's move, 0.0 = not Jeb's move)
            - correct: Boolean, whether prediction matches label
        url: Server URL to POST to
        timeout: Request timeout in seconds

    Note:
        This function is non-blocking and silently fails if the server
        is not running. It will not break training.
    """

    def _send():
        try:
            data = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "batch": batch,
                "total_batches": total_batches,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "positions": positions,
            }
            body = json.dumps(data).encode("utf-8")

            req = Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(req, timeout=timeout)
        except (URLError, TimeoutError, OSError):
            # Silently fail - don't break training if visualization server is down
            pass
        except Exception:
            # Catch any other unexpected errors silently
            pass

    # Run in background thread to avoid blocking training
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()


def is_server_running(url: str = "http://localhost:8765/status", timeout: float = 0.5) -> bool:
    """Check if visualization server is running.

    Args:
        url: Server status URL to check
        timeout: Request timeout in seconds

    Returns:
        True if server is responding, False otherwise
    """
    try:
        req = Request(url, method="GET")
        urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False
