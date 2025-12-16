"""
HTTP server for remote interaction with the consciousness loop.
Runs the loop as a subprocess and provides endpoints for input/output.
"""

import subprocess
import sys
import threading
import os
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import json

INPUT_FILE = Path(__file__).parent / "input.txt"  # Same dir as run.py
START_TIME = None
output_buffer = []
MAX_LINES = 1000
buffer_lock = threading.Lock()
loop_process = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logging

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/output"):
            # Get line count from query
            lines = 50
            if "?" in self.path:
                try:
                    lines = int(self.path.split("lines=")[1].split("&")[0])
                except:
                    pass
            with buffer_lock:
                recent = output_buffer[-lines:]
            self._send_json({"lines": recent, "total": len(output_buffer)})

        elif self.path == "/status":
            running = loop_process and loop_process.poll() is None
            uptime = str(datetime.now() - START_TIME).split('.')[0] if START_TIME else "0:00:00"
            self._send_json({
                "running": running,
                "uptime": uptime,
                "buffer_size": len(output_buffer),
                "pid": loop_process.pid if loop_process else None
            })

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/input":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()

            # Try to parse as JSON, otherwise use raw body
            try:
                data = json.loads(body)
                message = data.get("message", body)
            except:
                message = body

            if message.strip():
                with open(INPUT_FILE, "a") as f:
                    f.write(message.strip() + "\n")
                self._send_json({"sent": message.strip()})
            else:
                self._send_json({"error": "empty message"}, 400)
        else:
            self._send_json({"error": "not found"}, 404)


def capture_output(proc):
    """Capture loop output to buffer."""
    for line in iter(proc.stdout.readline, ""):
        if line:
            line = line.rstrip()
            print(line, flush=True)
            with buffer_lock:
                output_buffer.append(line)
                if len(output_buffer) > MAX_LINES:
                    output_buffer.pop(0)


def log(msg):
    """Print with timestamp."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def main():
    global loop_process, START_TIME

    INPUT_FILE.touch()
    START_TIME = datetime.now()

    log("=== Consciousness Loop Server ===")
    log(f"Started at {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")

    # Start consciousness loop (run.py is in same dir as server.py)
    script_dir = Path(__file__).parent
    loop_process = subprocess.Popen(
        [sys.executable, "-u", str(script_dir / "run.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(script_dir),
    )
    log(f"Loop process started (PID {loop_process.pid})")

    # Capture output in background
    threading.Thread(target=capture_output, args=(loop_process,), daemon=True).start()

    # Start HTTP server
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"API listening on port {port}")
    print("  GET  /output?lines=N  - get recent output")
    print("  POST /input           - send message")
    print("  GET  /status          - check status")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        loop_process.terminate()


if __name__ == "__main__":
    main()
