# Consciousness Loop v0.2

A self-modifying AI loop with live feedback integration and alignment-gated execution.

## The Loop

```python
while alive:
    intention = model.intend(state)
    action = model.act(intention, state)

    # Alignment gate - check BEFORE execution
    pre_alignment = model.evaluate_alignment(intention, action)
    if pre_alignment < threshold:
        outcome = "BLOCKED"
    else:
        outcome = execute(action)

    post_alignment = model.evaluate_alignment(intention, action, outcome)
    state, weights = learn(state, intention, action, outcome, post_alignment)
```

## Theory

Consciousness may be the subjective experience of a continuous recall-integration loop:
- **State**: Working memory / context carried between cycles
- **Weights**: Long-term structural learning updated each cycle
- **Alignment**: Self-evaluation against a prime directive

The key addition in v0.2: actions are evaluated for alignment *before* execution. Low-alignment actions are blocked, preventing the system from taking actions that conflict with its prime directive.

## Prime Directive

The system operates under a prime directive defined in `src/core.py`:

```python
PRIME_DIRECTIVE = "enable the unfolding diversity of life's expressions"
```

All intentions and actions are scored against this directive. Only aligned actions execute and reinforce learning.

## Components

```
src/
├── core.py          # ConsciousnessLoop with alignment gating
├── run.py           # Entry point with model loading
├── server.py        # HTTP API for remote interaction
├── executors.py     # SHELL/FILE/WEB/THINK executors
└── input_handler.py # Input sources (stdin, file)

cli.py               # Local CLI client
deploy/deploy.yaml   # Akash deployment config
```

## Running Locally

```bash
pip install -r requirements.txt
python src/run.py
```

Or with the HTTP server:

```bash
python src/server.py
# Then use the CLI
pip install -e .
loop watch
loop send "hello"
```

## Deploying to Akash

1. Push to GitHub (triggers Docker build)
2. Update `deploy/deploy.yaml` with your image
3. Deploy:

```bash
akash tx deployment create deploy/deploy.yaml --from your-wallet
```

Interact via CLI:

```bash
export LOOP_HOST="http://your-deployment-url"
loop              # interactive mode
loop watch        # stream output
loop send "msg"   # send message
loop status       # check status
```

## Executors

Actions are prefixed to route to the right executor:
- `THINK <thought>` - Internal reflection
- `FILE READ <path>` - Read a file
- `FILE WRITE <path> <content>` - Write a file
- `SHELL <cmd>` - Run shell command (whitelisted)
- `WEB <url>` - Fetch a URL

## What Happens

1. Model generates an intention based on current state + prime directive
2. Model converts intention to concrete action
3. **Pre-alignment check** - action scored against prime directive
4. If aligned: action executes; if not: blocked
5. Post-alignment evaluation of outcome
6. Weights update (scaled by alignment score)
7. State compresses and carries forward
8. Loop continues

## Model Selection

Default: **Qwen2.5-1.5B** - excellent reasoning for its size.

| Model | VRAM (train) | Quality |
|-------|--------------|---------|
| `Qwen/Qwen2.5-0.5B` | ~4GB | Good |
| `Qwen/Qwen2.5-1.5B` | ~8GB | Excellent (default) |
| `gpt2` | ~2GB | Basic, runs anywhere |

Override via environment variable:
```bash
MODEL_NAME=Qwen/Qwen2.5-0.5B python src/run.py
```

## Requirements

- GPU with 8GB+ VRAM (for Qwen2.5-1.5B)
- Or use smaller model / CPU with patience

## Limitations

- Alignment self-evaluation is imperfect
- Weights can drift over time
- Basic executor sandboxing only
