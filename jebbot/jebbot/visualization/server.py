"""HTTP server for training visualization."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Global state for training updates (thread-safe)
_state_lock = threading.Lock()
_current_state = {
    "epoch": 0,
    "total_epochs": 50,
    "batch": 0,
    "total_batches": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "val_accuracy": 0.0,
    "positions": [],
    "timestamp": 0,
}


def update_state(
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    positions: list[dict],
) -> None:
    """Update the global training state (thread-safe)."""
    import time

    with _state_lock:
        _current_state["epoch"] = epoch
        _current_state["total_epochs"] = total_epochs
        _current_state["batch"] = batch
        _current_state["total_batches"] = total_batches
        _current_state["train_loss"] = train_loss
        _current_state["val_loss"] = val_loss
        _current_state["val_accuracy"] = val_accuracy
        _current_state["positions"] = positions
        _current_state["timestamp"] = time.time()


def get_state() -> dict:
    """Get current training state (thread-safe)."""
    with _state_lock:
        return dict(_current_state)


class VisualizationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for visualization server."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_response(self, status: int, content_type: str, body: bytes) -> None:
        """Send HTTP response."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            # Serve index.html
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                content = html_path.read_bytes()
                self._send_response(200, "text/html", content)
            else:
                self._send_response(404, "text/plain", b"index.html not found")

        elif self.path == "/status":
            # Return current training state as JSON
            state = get_state()
            content = json.dumps(state).encode("utf-8")
            self._send_response(200, "application/json", content)

        else:
            self._send_response(404, "text/plain", b"Not found")

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/update":
            # Receive training update
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode("utf-8"))
                update_state(
                    epoch=data.get("epoch", 0),
                    total_epochs=data.get("total_epochs", 50),
                    batch=data.get("batch", 0),
                    total_batches=data.get("total_batches", 0),
                    train_loss=data.get("train_loss", 0.0),
                    val_loss=data.get("val_loss", 0.0),
                    val_accuracy=data.get("val_accuracy", 0.0),
                    positions=data.get("positions", []),
                )
                self._send_response(200, "application/json", b'{"status": "ok"}')
            except Exception as e:
                error = {"status": "error", "message": str(e)}
                self._send_response(400, "application/json", json.dumps(error).encode())

        else:
            self._send_response(404, "text/plain", b"Not found")


def run_server(host: str = "localhost", port: int = 8765) -> None:
    """Run the visualization server."""
    server = HTTPServer((host, port), VisualizationHandler)
    print(f"Visualization server running at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_server()
