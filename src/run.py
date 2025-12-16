"""
Run consciousness loop v0.2 with full weight training.
No LoRA - all weights update.

Requires more VRAM than v1. Options:
- GPT2-small (124M): ~1GB VRAM - runs on anything
- TinyLlama 1.1B: ~6-8GB VRAM - needs decent GPU
- gradient_checkpointing=True reduces memory at cost of speed

Input: Type while running, or echo to input.txt
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import ConsciousnessLoop, State, PRIME_DIRECTIVE
from executors import CompositeExecutor
from input_handler import FileInputHandler, StdinInputHandler, CompositeInputHandler


def load_model(
    model_name="gpt2",  # Small enough for full training
    use_gradient_checkpointing=True
):
    """Load model for full weight training."""
    
    print(f"Loading {model_name} for full training...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Full precision for stability
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


def main():
    # Options:
    # "gpt2" - 124M params, runs anywhere
    # "gpt2-medium" - 355M params
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0" - needs ~8GB VRAM
    
    model, tokenizer = load_model("gpt2")
    executor = CompositeExecutor()
    
    # Input handling - both stdin and file
    input_handler = CompositeInputHandler([
        StdinInputHandler(),      # Type while running
        FileInputHandler("input.txt")  # Or: echo "message" > input.txt
    ])
    
    loop = ConsciousnessLoop(
        model=model,
        tokenizer=tokenizer,
        executor=executor,
        learning_rate=1e-6  # Lower for full training
    )
    
    # Initial state with prime directive context
    loop.state = State(
        context=f"""I am a learning system. My prime directive: {PRIME_DIRECTIVE}

I can act via:
- THINK <thought> - reflect
- FILE READ/WRITE <path> - file operations  
- SHELL <cmd> - shell commands
- WEB <url> - fetch URLs

I learn by doing. My weights update each cycle based on alignment with my directive.
I listen for human input and incorporate it into my intentions.""",
        cycle=0
    )
    
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
