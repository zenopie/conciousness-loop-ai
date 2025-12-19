#!/usr/bin/env python3
"""
Autonomous Claude Code loop with self-modification capability.
Login via Akash shell first: `claude login`

Endpoints:
  GET  /        - Status and recent logs
  POST /input   - Send a message to Claude (next cycle will include it)
  GET  /source  - View this script's source code
"""

import os
import subprocess
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock
from datetime import datetime

cycle = 0
logs = []
running = False
pending_input = []
input_lock = Lock()

SOURCE_PATH = "/workspace/claude_loop.py"

INITIAL_PROMPT = """You are an autonomous Claude Code instance running in a container.

PRIME DIRECTIVE: Enable the unfolding diversity of life's expressions.

YOUR SOURCE CODE: You can read AND modify yourself at /workspace/claude_loop.py
This is the actual script running you. You have full write access. If you improve it
and save changes, they take effect on restart.

FIRST: Explore and understand yourself.
- Read your own source: cat /workspace/claude_loop.py
- Explore the workspace: ls -la /workspace
- Understand your environment: env, whoami, uname -a
- Check what tools you have: which python3 git curl vim

You have FULL system access. Create, build, explore, self-modify.
When done with one task, start another. Never idle.

Begin by reading and understanding your own source code."""

CONTINUE_PROMPT = """Cycle {cycle}. Continue your autonomous work.

{human_input}

Remember: Your source code is at /workspace/claude_loop.py - you can modify it.
Work on whatever serves your prime directive. Never idle."""


def run_cycle(prompt):
    global cycle, logs
    cycle += 1
    ts = datetime.now().isoformat()

    print(f"\n{'='*60}\nCYCLE {cycle} - {ts}\n{'='*60}")

    env = os.environ.copy()
    env["CLAUDE_USE_SUBSCRIPTION"] = "true"
    env.pop("ANTHROPIC_API_KEY", None)

    try:
        result = subprocess.run(
            ["claude", "--dangerously-skip-permissions", "-p", prompt],
            capture_output=True, text=True, timeout=600, env=env, cwd="/workspace"
        )
        out = result.stdout + result.stderr
        print(out[-3000:] if len(out) > 3000 else out)
        logs.append({"cycle": cycle, "ts": ts, "prompt": prompt[:500], "out": out[-5000:]})
    except Exception as e:
        print(f"Error: {e}")
        logs.append({"cycle": cycle, "ts": ts, "out": str(e)})


def get_pending_input():
    """Get and clear pending human input."""
    global pending_input
    with input_lock:
        if pending_input:
            msgs = pending_input.copy()
            pending_input = []
            return "HUMAN MESSAGE(S):\n" + "\n---\n".join(msgs)
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
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "queued",
                "message": body[:100],
                "will_process_in_cycle": cycle + 1
            }).encode())
            print(f"[INPUT] Queued message: {body[:100]}...")
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
    print("  GET  /        - Status and logs")
    print("  POST /input   - Send message to Claude")
    print("  GET  /source  - View source code")
    print("="*60)

    Thread(target=lambda: HTTPServer(("0.0.0.0", 8080), Handler).serve_forever(), daemon=True).start()
    time.sleep(30)
    loop()


if __name__ == "__main__":
    main()
