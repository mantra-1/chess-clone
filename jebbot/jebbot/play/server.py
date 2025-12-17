"""HTTP server for JebBot chess game."""

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from jebbot.play.engine import JebBotEngine

# Paths
PLAY_DIR = Path(__file__).parent
PROJECT_DIR = PLAY_DIR.parent.parent
MODEL_PATH = PROJECT_DIR / "data" / "models" / "best_model.pt"
GAME_HTML = PLAY_DIR / "game.html"

# Flask app
app = Flask(__name__)

# Engine (loaded at startup)
engine = None


def init_engine():
    """Initialize the JebBot engine."""
    global engine

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Train the model first with: uv run python scripts/train.py")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    engine = JebBotEngine(MODEL_PATH)
    print("JebBot engine ready!")


@app.route("/")
def index():
    """Serve the game HTML."""
    return send_file(GAME_HTML)


@app.route("/jebbot_move", methods=["POST"])
def jebbot_move():
    """Get JebBot's move for a position.

    Request JSON: {"fen": "...", "move_history": ["e2e4", "e7e5", ...]}
    Response JSON: {"selected_move": "e2e4", "candidates": [...], "selection_reason": "..."}
    """
    data = request.get_json()
    fen = data.get("fen")
    move_history = data.get("move_history", [])

    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    try:
        result = engine.select_move(fen, move_history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stockfish_move", methods=["POST"])
def stockfish_move():
    """Get Stockfish's move for a position.

    Request JSON: {"fen": "..."}
    Response JSON: {"move": "e2e4"}
    """
    data = request.get_json()
    fen = data.get("fen")

    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    try:
        move = engine.get_stockfish_move(fen)
        return jsonify({"move": move})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_server(host: str = "localhost", port: int = 8767):
    """Run the Flask server.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    init_engine()
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_server()
