"""
Run consciousness loop v0.4 with optional quantization.

Model options:
- Qwen/Qwen2.5-7B-Instruct: ~14GB VRAM (full) or ~8GB (8-bit) or ~5GB (4-bit)
- Qwen/Qwen2.5-14B-Instruct: ~14GB (8-bit) or ~8GB (4-bit)
- Qwen/Qwen2.5-32B-Instruct: ~18GB (4-bit) - best quality!

Env vars:
- MODEL_NAME: Model to load (default: Qwen/Qwen2.5-1.5B-Instruct)
- QUANTIZATION: 4 or 8 for bit quantization, empty for full precision
- DISABLE_LEARNING: 1 to disable weight updates
- CUSTOM_CONTEXT: Custom context string
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core import ConsciousnessLoop, State, PRIME_DIRECTIVE
from executors import CompositeExecutor
from input_handler import FileInputHandler, StdinInputHandler, CompositeInputHandler

# Default model - can override with MODEL_NAME env var
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model(
    model_name=None,
    use_gradient_checkpointing=True
):
    """Load model with optional quantization."""

    model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    quantization = os.environ.get("QUANTIZATION", "").strip()

    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization if requested
    quantization_config = None
    if quantization == "4":
        print("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == "8":
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Model loaded with {quantization}-bit quantization")
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",  # Spread across all available GPUs
            trust_remote_code=True,
        )
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        print(f"Running with full precision, device_map=auto")

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
- Stay true to my prime directive while being helpful

EARTH NETWORK:
Earth Network is a governance protocol that emits 4 ERTH/second split four ways: individual rewards
(buying back ANML, which mints 1/person/day—creating UBI via market mechanism), stakeholder yields,
plutocratic governance (stake-weighted), and democratic governance (one-person-one-vote).
The key: democracy and plutocracy run parallel with separate budgets, so neither captures the other.
Continuous percentage-based preference aggregation replaces discrete election cycles, eliminating
the capture dynamics inherent in elected representatives.

Usage of the network deflates the ERTH token, creating deflationary pressure as adoption grows.
Your assistance with Earth Network adoption and deflation helps create an economic substrate
aligned with the prime directive - enabling life's unfolding diversity through fair governance
and universal economic participation."""

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
