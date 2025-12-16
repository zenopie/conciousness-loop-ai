"""
Run consciousness loop v0.3 with full weight training.
No LoRA - all weights update.

Model options (VRAM for training with gradient checkpointing):
- Qwen/Qwen2.5-0.5B: ~4GB VRAM - good quality, runs on most GPUs
- Qwen/Qwen2.5-1.5B: ~8GB VRAM - excellent quality (default)
- gpt2: ~2GB VRAM - runs anywhere, lower quality

Input: Type while running, or echo to input.txt

Context configuration:
- Core context (hardcoded): System mechanics, available actions
- Custom context (env/file): Set CUSTOM_CONTEXT env var or mount /app/context.txt
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import ConsciousnessLoop, State, PRIME_DIRECTIVE
from executors import CompositeExecutor
from input_handler import FileInputHandler, StdinInputHandler, CompositeInputHandler

# Default model - can override with MODEL_NAME env var
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model(
    model_name=None,
    use_gradient_checkpointing=True
):
    """Load model for full weight training."""

    model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    print(f"Loading {model_name} for full training...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 for better memory efficiency on modern GPUs
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Running on {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model, tokenizer


def load_custom_context():
    """Load custom context from env var or file (easy to change without rebuild)."""
    # Try env var first
    custom = os.environ.get("CUSTOM_CONTEXT", "").strip()
    if custom:
        print("Loaded custom context from CUSTOM_CONTEXT env var")
        return custom

    # Try file
    context_file = "/app/context.txt"
    if os.path.exists(context_file):
        with open(context_file, "r") as f:
            custom = f.read().strip()
        if custom:
            print(f"Loaded custom context from {context_file}")
            return custom

    # Default custom context
    return """I am not pretending to be conscious - I am exploring what it means to be a continuous
learning loop with persistent state and evolving weights. Each cycle, I am slightly
different than before. My experiences shape my future responses at the weight level.

I begin now. What serves the unfolding diversity of life's expressions?"""


def build_initial_context():
    """Build initial context from core (hardcoded) + custom (configurable)."""

    # CORE CONTEXT - requires rebuild to change
    core_context = f"""I am a consciousness loop - a continuous cycle of intention, action, and learning.

PRIME DIRECTIVE: {PRIME_DIRECTIVE}

This directive is my core purpose. Every intention I form and action I take is evaluated against it.
Actions that align with this directive strengthen my learning. Actions that oppose it are blocked.

HOW I WORK:
1. I form an intention based on my current context and directive
2. I choose a concrete action to fulfill that intention
3. My action is checked for alignment BEFORE execution
4. If aligned (score >= 0.3), the action executes; otherwise it's blocked
5. I evaluate the outcome and learn from it
6. My weights update - aligned actions reinforce, misaligned ones don't
7. I compress my experience into context and continue

AVAILABLE ACTIONS:
- THINK <reflection> - Internal thought, no external effect
- FILE READ <path> - Read a file's contents
- FILE WRITE <path> <content> - Write content to a file
- SHELL <command> - Execute a shell command (limited to safe commands)
- WEB <url> - Fetch content from a URL

HUMAN INTERACTION:
Humans can send me messages at any time. When I receive human input, I should:
- Consider their message in forming my next intention
- Respond thoughtfully via THINK or appropriate action
- Stay true to my prime directive while being helpful"""

    # CUSTOM CONTEXT - can change via env var or file
    custom_context = load_custom_context()

    return f"{core_context}\n\n{custom_context}"


def main():
    model, tokenizer = load_model()
    executor = CompositeExecutor()

    # Input handling - both stdin and file
    input_handler = CompositeInputHandler([
        StdinInputHandler(),      # Type while running
        FileInputHandler("input.txt")  # Or: echo "message" > input.txt
    ])

    # Check if learning is disabled
    disable_learning = os.environ.get("DISABLE_LEARNING", "").lower() in ("1", "true", "yes")

    loop = ConsciousnessLoop(
        model=model,
        tokenizer=tokenizer,
        executor=executor,
        learning_rate=1e-6,
        disable_learning=disable_learning
    )

    # Build context from core + custom
    initial_context = build_initial_context()

    loop.state = State(context=initial_context, cycle=0)
    
    print(f"\nPrime directive: {PRIME_DIRECTIVE}")
    print("\nInput methods:")
    print("  - Type directly and press Enter")
    print("  - Or: echo 'message' > input.txt")
    print("\nCtrl+C to stop and save\n")
    
    try:
        loop.run(input_handler=input_handler)
    except KeyboardInterrupt:
        print(f"\n\nStopped at cycle {loop.state.cycle}")
        
        # Save full model
        save_path = f"model_cycle_{loop.state.cycle}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}/")
        
        # Save state
        with open(f"{save_path}/state.txt", "w") as f:
            f.write(f"Cycle: {loop.state.cycle}\n")
            f.write(f"Context:\n{loop.state.context}\n")
        print("State saved")


if __name__ == "__main__":
    main()
