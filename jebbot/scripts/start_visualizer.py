#!/usr/bin/env python3
"""Start the training visualization server.

Usage:
    Terminal 1: uv run python scripts/start_visualizer.py
    Terminal 2: uv run python scripts/train.py --visualize

Then open http://localhost:8765 in your browser to watch training progress.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jebbot.visualization.server import run_server


def main():
    print("=" * 60)
    print("JebBot Training Visualizer")
    print("=" * 60)
    print()
    print("Open http://localhost:8765 in your browser")
    print()
    print("Then start training with:")
    print("  uv run python scripts/train.py --visualize")
    print()
    print("=" * 60)
    print()

    run_server(host="localhost", port=8765)


if __name__ == "__main__":
    main()
