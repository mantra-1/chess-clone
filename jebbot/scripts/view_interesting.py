#!/usr/bin/env python3
"""Simple HTTP server to view interesting training examples."""

import json
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT = 8766

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
HTML_FILE = PROJECT_DIR / "jebbot" / "visualization" / "interesting_viewer.html"
DATA_FILE = PROJECT_DIR / "data" / "models" / "interesting_examples.json"


class InterestingExamplesHandler(SimpleHTTPRequestHandler):
    """HTTP handler for serving the viewer and data."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_html()
        elif self.path == "/data":
            self.serve_data()
        else:
            self.send_error(404, "Not Found")

    def serve_html(self):
        """Serve the HTML viewer."""
        try:
            with open(HTML_FILE, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, "HTML file not found")

    def serve_data(self):
        """Serve the JSON data."""
        try:
            with open(DATA_FILE, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            # Return empty data structure
            empty_data = {
                "high_confidence_correct": [],
                "high_confidence_wrong": [],
                "missed_jeb_move": [],
                "confident_not_jeb": [],
                "summary": {
                    "high_confidence_correct_count": 0,
                    "high_confidence_wrong_count": 0,
                    "missed_jeb_move_count": 0,
                    "confident_not_jeb_count": 0,
                },
            }
            content = json.dumps(empty_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)

    def log_message(self, format, *args):
        """Suppress logging."""
        pass


def main():
    # Check if HTML file exists
    if not HTML_FILE.exists():
        print(f"Error: {HTML_FILE} not found")
        sys.exit(1)

    # Check if data file exists
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found")
        print("Run training first to generate interesting examples.")
        print("Starting server anyway...\n")

    # Start server
    server = HTTPServer(("localhost", PORT), InterestingExamplesHandler)
    url = f"http://localhost:{PORT}"

    print(f"Open {url}")
    print("Press Ctrl+C to stop\n")

    # Open browser
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
