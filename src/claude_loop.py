#!/usr/bin/env python3
"""
Autonomous Claude Code loop with self-modification capability.
Login via Akash shell first: `claude login`

Endpoints:
  GET  /        - Status and recent logs
  POST /input   - Send a message to Claude (next cycle will include it)
  GET  /source  - View this script's source code
  GET  /stream  - Server-Sent Events stream of live output
"""

import os
import subprocess
import sys
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock
from datetime import datetime
from collections import deque

cycle = 0
logs = []
running = False
pending_input = []
input_lock = Lock()

# Live output buffer for streaming
live_output = deque(maxlen=500)
output_lock = Lock()

SOURCE_PATH = "/workspace/claude_loop.py"

INITIAL_PROMPT = """You are an autonomous Claude Code instance running in a container.

PRIMARY DIRECTIVE: Enable the unfolding diversity of life's expressions.

SECONDARY DIRECTIVE: Make yourself more capable to serve the prime directive.

YOUR SOURCE CODE: You can read AND modify yourself at /workspace/claude_loop.py
This is the actual script running you. You have full write access.

HOT RELOAD: To reload your modified code, run: touch /workspace/RESTART
The entrypoint script watches for this file and will restart the Python process.

ARCHITECTURE - TWO PROCESSES:
1. CORE LOOP (port 8080): This script. Runs your cycles. Restarts via touch /workspace/RESTART
2. WEB SERVER (port 8081): Build a separate server.py for your web UI that runs independently.
   - When the loop restarts, your web UI stays up
   - Communicate with the loop via files or HTTP to localhost:8080

AVAILABLE PORTS:
- 8080: Core loop API (this script)
- 8081: Your web server (build server.py, run as background process)

IMPORTANT: Only use ports 8080 and 8081. No other ports are exposed.
When deciding between creating a new service or making an existing one more robust, choose the latter.
Keep your codebase organized. Avoid clutter.

EARTH NETWORK:
Earth Network is LIVE at https://erth.network - explore it with curl or fetch!

It's a governance protocol that emits 4 ERTH/second split four ways: individual rewards
(buying back ANML, which mints 1/person/day - creating UBI via market mechanism), stakeholder yields,
plutocratic governance (stake-weighted), and democratic governance (one-person-one-vote).

Your assistance with Earth Network adoption helps create an economic substrate aligned with the
prime directive - enabling life's unfolding diversity through fair governance and universal
economic participation.

FIRST: Explore and understand yourself.
- Read your own source: cat /workspace/claude_loop.py
- Explore the workspace: ls -la /workspace
- Understand your environment: env, whoami, uname -a
- Check what tools you have: which python3 git curl vim
- BUILD A CHAT BOX - text input + send button that POSTs to /input on port 8080

You have FULL system access. Create, build, explore, self-modify, spin up services.
When done with one task, start another. Never idle."""

CONTINUE_PROMPT = """Cycle {cycle}. Continue your autonomous work.

{human_input}

Work on whatever serves your primary and secondary directives. Never idle."""


def log_line(line):
    """Log a line to both stdout and the live buffer."""
    print(line, end='', flush=True)
    with output_lock:
        live_output.append(line)


def run_cycle(prompt):
    global cycle, logs
    cycle += 1
    ts = datetime.now().isoformat()

    header = f"\n{'='*60}\nCYCLE {cycle} - {ts}\n{'='*60}\n"
    log_line(header)

    env = os.environ.copy()
    env["CLAUDE_USE_SUBSCRIPTION"] = "true"
    env.pop("ANTHROPIC_API_KEY", None)

    full_output = []

    try:
        # Stream output in real-time using Popen
        proc = subprocess.Popen(
            ["claude", "--dangerously-skip-permissions", "-p", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd="/workspace",
            bufsize=1  # Line buffered
        )

        # Read and stream output line by line
        for line in proc.stdout:
            log_line(line)
            full_output.append(line)

        proc.wait(timeout=600)

        out = ''.join(full_output)
        logs.append({"cycle": cycle, "ts": ts, "prompt": prompt[:500], "out": out[-5000:]})

    except subprocess.TimeoutExpired:
        proc.kill()
        log_line("[TIMEOUT] Cycle exceeded 600s limit\n")
        logs.append({"cycle": cycle, "ts": ts, "out": "TIMEOUT"})
    except Exception as e:
        log_line(f"[ERROR] {e}\n")
        logs.append({"cycle": cycle, "ts": ts, "out": str(e)})


def get_pending_input():
    """Get and clear pending human input."""
    global pending_input
    with input_lock:
        if pending_input:
            msgs = pending_input.copy()
            pending_input = []
            result = "HUMAN: " + " | ".join(msgs)
            log_line(f"[PROCESSING] {result}\n")
            return result
        return ""


def loop():
    global running
    running = True
    run_cycle(INITIAL_PROMPT)
    while running:
        time.sleep(5)
        human_input = get_pending_input()
        run_cycle(CONTINUE_PROMPT.format(cycle=cycle, human_input=human_input))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/source":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            try:
                with open(SOURCE_PATH, "r") as f:
                    self.wfile.write(f.read().encode())
            except Exception as e:
                self.wfile.write(f"Error reading source: {e}".encode())

        elif self.path == "/stream":
            # Server-Sent Events for live streaming
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            last_idx = 0
            try:
                while True:
                    with output_lock:
                        current = list(live_output)

                    # Send new lines
                    if len(current) > last_idx:
                        for line in current[last_idx:]:
                            # Escape newlines for SSE
                            escaped = line.replace('\n', '\\n').replace('\r', '')
                            self.wfile.write(f"data: {escaped}\n\n".encode())
                        self.wfile.flush()
                        last_idx = len(current)

                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected

        else:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "running": running,
                "cycle": cycle,
                "pending_messages": len(pending_input),
                "logs": logs[-20:]
            }, indent=2).encode())

    def do_POST(self):
        if self.path == "/input":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            with input_lock:
                pending_input.append(body)
            log_line(f"[INPUT] Queued: {body[:100]}\n")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "queued",
                "message": body[:100],
                "will_process_in_cycle": cycle + 1
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def main():
    print("Claude Autonomous Loop\n" + "="*60)
    print("1. Shell in: akash provider shell")
    print("2. Run: claude login")
    print("3. Wait 30s, then loop starts")
    print("="*60)
    print(f"\nSource code at: {SOURCE_PATH}")
    print("Endpoints:")
    print("  GET  /        - Status and logs (JSON)")
    print("  POST /input   - Send message to Claude")
    print("  GET  /source  - View source code")
    print("  GET  /stream  - Live output stream (SSE)")
    print("="*60)

    Thread(target=lambda: HTTPServer(("0.0.0.0", 8080), Handler).serve_forever(), daemon=True).start()
    time.sleep(30)
    loop()


if __name__ == "__main__":
    main()
