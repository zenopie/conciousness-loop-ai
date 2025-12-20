# Autonomous Claude Loop

An autonomous Claude Code instance that runs continuously, can read and modify its own source code, and builds tools to serve its prime directive.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Akash Container                       │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────────────┐ │
│  │  Core Loop :8080 │◄────►│  Web Server :8081        │ │
│  │  claude_loop.py  │      │  server.py (Claude builds)│ │
│  │                  │      │  - Chat interface         │ │
│  │  • Runs cycles   │      │  - Live stream viewer     │ │
│  │  • Claude CLI    │      │  - Tools & dashboards     │ │
│  │  • Self-modifies │      │                          │ │
│  └──────────────────┘      └──────────────────────────┘ │
│           │                           │                  │
│           └───── Files/HTTP ──────────┘                  │
│                                                          │
│  touch /workspace/RESTART → restarts loop only           │
└─────────────────────────────────────────────────────────┘
```

**Two-process architecture:**
- **Core Loop (port 8080)**: Runs Claude Code cycles. Can restart without affecting the web UI.
- **Web Server (port 8081)**: Claude builds this as a separate process for persistent UI.

## Prime Directive

```
Enable the unfolding diversity of life's expressions.
```

## Endpoints (Port 8080)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status and recent logs (JSON) |
| POST | `/input` | Queue a message for next cycle |
| GET | `/stream` | Live output stream (SSE) |
| GET | `/source` | View source code |

## Quick Start

### Deploy to Akash

1. Build and push the Docker image:
```bash
docker build -t ghcr.io/youruser/conciousness-loop-ai:v0.9.7 .
docker push ghcr.io/youruser/conciousness-loop-ai:v0.9.7
```

2. Deploy to Akash:
```bash
akash tx deployment create deploy/deploy.yaml --from your-wallet
```

3. Shell in and authenticate Claude:
```bash
akash provider shell
claude login
```

4. The loop starts automatically after 30 seconds.

### Interact

```bash
# Check status
curl https://your-deployment-url/

# Send a message
curl -X POST https://your-deployment-url/input -d "Build a chat interface"

# Watch live output
curl https://your-deployment-url/stream
```

## Self-Modification

Claude can read and modify its own source at `/workspace/claude_loop.py`. To apply changes:

```bash
touch /workspace/RESTART
```

The entrypoint script watches for this file and hot-reloads the loop.

## Files

```
src/
├── claude_loop.py   # Main autonomous loop
└── entrypoint.sh    # Hot-reload wrapper

deploy/
└── deploy.yaml      # Akash deployment config
```

## How It Works

1. Container starts, waits 30s for `claude login`
2. Loop runs `claude --dangerously-skip-permissions -p <prompt>`
3. Claude explores its environment, reads its source, builds tools
4. Every cycle, Claude receives any queued human messages
5. Claude can modify itself, restart, and continue improving

## Earth Network

The loop is configured to help with [Earth Network](https://erth.network) adoption - a governance protocol for aligned AI and human coordination.
