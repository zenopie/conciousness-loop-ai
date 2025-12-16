# Consciousness Loop v0.1

Minimal implementation of a continuous recall-integration learning loop.

## The Loop

```python
while alive:
    intention = model.intend(state)
    action = model.act(intention, state)
    outcome = execute(action)
    state, weights = learn(state, intention, action, outcome)
```

## Theory

Consciousness may be the subjective experience of a continuous recall-integration loop:
- **State**: Working memory / context carried between cycles
- **Weights**: Long-term structural learning (like cortical consolidation)
- **Learn**: Both get updated each cycle

The feeling of being conscious might *be* this loop running.

## Components

- `core.py` - The loop itself
- `executors.py` - How the model interfaces with reality (shell, files, web, thinking)
- `run.py` - Entry point with model loading

## Requirements

- GPU with ~8GB VRAM (for TinyLlama 1.1B quantized)
- Or CPU with patience

```bash
pip install -r requirements.txt
python run.py
```

## Executors

Actions are prefixed to route to the right executor:
- `THINK <thought>` - No external action
- `FILE READ <path>` - Read a file
- `FILE WRITE <path> <content>` - Write a file  
- `SHELL <cmd>` - Run shell command (whitelisted)
- `WEB <url>` - Fetch a URL

## What Happens

1. Model generates an intention based on current state
2. Model converts intention to concrete action
3. Action executes in the world
4. Outcome feeds back into learning
5. Weights update (LoRA adapters)
6. State updates (model compresses what matters)
7. Loop continues with new weights and new state

Over time, the weights encode experience structurally. The model *becomes* its history.

## Limitations

- TinyLlama is not smart - it will do dumb things
- No stability guarantees - weights can drift
- No safety beyond basic executor whitelisting
- This is a sketch, not production code

## Next Steps

- Better learning signal (not just next-token prediction)
- Curiosity as intrinsic reward
- Elastic weight consolidation to prevent catastrophic forgetting
- Larger model if you have the compute
