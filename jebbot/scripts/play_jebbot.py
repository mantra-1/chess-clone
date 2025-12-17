#!/usr/bin/env python3
"""Start the JebBot chess game server."""

import sys
import webbrowser
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jebbot.play.server import run_server

PORT = 8767


def main():
    url = f"http://localhost:{PORT}"
    print(f"Play against JebBot at {url}")
    print("Press Ctrl+C to stop\n")

    # Open browser
    webbrowser.open(url)

    # Start server
    try:
        run_server(port=PORT)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
