#!/usr/bin/env python3
"""Simple dev server for ollama-search.

Usage: python3 serve.py [port]
  Defaults to port 8000.
Serves ~/ollama-search/public/ at http://localhost:{PORT}/
"""

import http.server
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIRECTORY = Path(__file__).resolve().parent / "public"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def do_GET(self):
        # Redirect /search to /search/ (trailing slash only)
        # / serves index.html directly (no redirect)
        if self.path == "/search":
            self.send_response(301)
            self.send_header("Location", "/search/")
            self.end_headers()
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    print(f"Serving {DIRECTORY} at http://localhost:{PORT}/")
    http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
