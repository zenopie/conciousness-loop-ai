#!/usr/bin/env python3
"""
CLI for interacting with a running consciousness loop.

Usage:
  python cli.py                     # Interactive mode
  python cli.py send "message"      # Send a message
  python cli.py watch               # Watch output stream
  python cli.py output              # Get recent output
  python cli.py status              # Check status
"""

import sys
import time
import json
import urllib.request
import urllib.error

DEFAULT_HOST = "http://localhost:8080"


def get_host():
    """Get host from env or default."""
    import os
    return os.environ.get("LOOP_HOST", DEFAULT_HOST)


def api_get(path):
    """GET request to API."""
    try:
        req = urllib.request.Request(f"{get_host()}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}


def api_post(path, data):
    """POST request to API."""
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{get_host()}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}


def cmd_send(message):
    """Send a message to the loop."""
    result = api_post("/input", {"message": message})
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Sent: {result.get('sent', message)}")


def cmd_output(lines=50):
    """Get recent output."""
    result = api_get(f"/output?lines={lines}")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        for line in result.get("lines", []):
            print(line)
        print(f"\n({result.get('total', 0)} total lines)")


def cmd_watch():
    """Watch output stream."""
    seen = 0
    print("Watching output (Ctrl+C to stop)...\n")
    try:
        while True:
            result = api_get("/output?lines=1000")
            if "error" not in result:
                lines = result.get("lines", [])
                if len(lines) > seen:
                    for line in lines[seen:]:
                        print(line)
                    seen = len(lines)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def cmd_status():
    """Check loop status."""
    result = api_get("/status")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        status = "running" if result.get("running") else "stopped"
        print(f"Status: {status}")
        print(f"Buffer: {result.get('buffer_size', 0)} lines")


def cmd_interactive():
    """Interactive chat mode."""
    print(f"Connected to {get_host()}")
    print("Type messages to send, or:")
    print("  /output  - show recent output")
    print("  /watch   - watch output stream")
    print("  /status  - check status")
    print("  /quit    - exit")
    print()

    while True:
        try:
            msg = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not msg:
            continue
        elif msg == "/quit":
            break
        elif msg == "/output":
            cmd_output()
        elif msg == "/watch":
            cmd_watch()
        elif msg == "/status":
            cmd_status()
        else:
            cmd_send(msg)


def main():
    args = sys.argv[1:]

    if not args:
        cmd_interactive()
    elif args[0] == "send" and len(args) > 1:
        cmd_send(" ".join(args[1:]))
    elif args[0] == "output":
        lines = int(args[1]) if len(args) > 1 else 50
        cmd_output(lines)
    elif args[0] == "watch":
        cmd_watch()
    elif args[0] == "status":
        cmd_status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
