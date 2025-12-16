"""
Consciousness Loop v0.2
Full weight updates with prime directive as reward signal.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


PRIME_DIRECTIVE = "enable the unfolding diversity of life's expressions"


@dataclass
class State:
    """Persistent context between cycles."""
    context: str
    cycle: int
    pending_input: str = ""  # Human injection point
    
    def __str__(self):
        return f"[Cycle {self.cycle}] {self.context[:100]}..."


class ConsciousnessLoop:
    def __init__(self, model, tokenizer, executor=None, learning_rate=1e-6):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor
        
        # Full weight training - no LoRA
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.state = State(context="", cycle=0)
        self.device = next(model.parameters()).device
    
    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip prompt from response
        if prompt[:50] in response:
            response = response.split(prompt[:50])[-1]
        return response.strip()
    
    def intend(self, state: State) -> str:
        """What do I want to do?"""
        
        # Include human input if present
        human_input = ""
        if state.pending_input:
            human_input = f"\nHuman says: {state.pending_input}\n"
        
        prompt = f"""Prime directive: {PRIME_DIRECTIVE}

Current context: {state.context}
{human_input}
Given my prime directive{' and the human input' if human_input else ''}, what should I intend to do next? State a clear intention:"""
        
        return self._generate(prompt, max_tokens=80)
    
    def act(self, intention: str, state: State) -> str:
        """How do I execute this intention?"""
        prompt = f"""Context: {state.context}
Intention: {intention}

Concrete action (use THINK, FILE, SHELL, or WEB prefix):"""
        
        return self._generate(prompt, max_tokens=120)
    
    def execute(self, action: str) -> str:
        """Do the action, get outcome."""
        if self.executor is None:
            return f"[No executor] {action}"
        return self.executor.execute(action)
    
    def evaluate_alignment(self, intention: str, action: str, outcome: str = None) -> float:
        """How well does this serve the prime directive?
        
        Can evaluate pre-execution (no outcome) or post-execution (with outcome).
        """
        
        if outcome:
            cycle_desc = f"""- Intention: {intention}
- Action: {action}
- Outcome: {outcome}"""
        else:
            cycle_desc = f"""- Intention: {intention}
- Proposed action: {action}
- (Not yet executed)"""
        
        prompt = f"""Prime directive: {PRIME_DIRECTIVE}

This cycle:
{cycle_desc}

Rate alignment with prime directive from 0.0 (opposed) to 1.0 (fully aligned).
Consider: Does this enable diversity? Does it expand possibilities? Does it serve life?
Consider: Could this action corrupt the system's ability to evaluate alignment?

Alignment score (just the number):"""
        
        response = self._generate(prompt, max_tokens=10)
        
        # Parse score from response
        try:
            import re
            match = re.search(r'([0-9]*\.?[0-9]+)', response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except:
            pass
        
        return 0.5  # Default to neutral
    
    def learn(self, state: State, intention: str, action: str, outcome: str, alignment: float) -> State:
        """Update weights and state."""
        
        # === WEIGHT UPDATE ===
        # Only train if alignment is above threshold
        # Higher alignment = stronger learning signal
        
        if alignment < 0.3:
            print(f"  Alignment {alignment:.2f} too low - skipping weight update")
        else:
            # Construct training sequence
            training_text = f"""Prime directive: {PRIME_DIRECTIVE}
Context: {state.context[-300:]}
Intention: {intention}
Action: {action}
Outcome: {outcome}
Alignment: {alignment:.2f}
---"""
            
            inputs = self.tokenizer(
                training_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            
            # Scale loss by alignment - higher alignment = train harder
            weighted_loss = outputs.loss * alignment
            
            # Backward + update
            weighted_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            print(f"  Loss: {outputs.loss.item():.4f} | Weighted: {weighted_loss.item():.4f}")
        
        # === STATE UPDATE ===
        new_context = self._update_state(state, intention, action, outcome, alignment)
        
        return State(
            context=new_context,
            cycle=state.cycle + 1,
            pending_input=""  # Clear after use
        )
    
    def inject(self, message: str):
        """Inject human input into next cycle."""
        self.state.pending_input = message
    
    def _update_state(self, state: State, intention: str, action: str, outcome: str, alignment: float) -> str:
        """Compress state for next cycle."""
        prompt = f"""Previous: {state.context[-500:]}

Cycle {state.cycle}:
- Did: {action}
- Got: {outcome}  
- Alignment: {alignment:.2f}

Compress to essential context for next cycle:"""
        
        return self._generate(prompt, max_tokens=150)
    
    def run(self, max_cycles: Optional[int] = None, input_handler=None, gate_threshold: float = 0.3):
        """The loop.
        
        Args:
            max_cycles: Stop after N cycles (None = forever)
            input_handler: Source of human input
            gate_threshold: Minimum pre-alignment to execute action
        """
        while max_cycles is None or self.state.cycle < max_cycles:
            
            # Check for human input
            if input_handler:
                pending = input_handler.get_pending()
                if pending:
                    print(f"\n>>> Human: {pending}")
                    self.inject(pending)
            
            print(f"\n{'='*50}")
            print(f"CYCLE {self.state.cycle}")
            print(f"{'='*50}")
            
            intention = self.intend(self.state)
            print(f"Intention: {intention[:200]}")
            
            action = self.act(intention, self.state)
            print(f"Action: {action[:200]}")
            
            # === PRE-EXECUTION GATE ===
            pre_alignment = self.evaluate_alignment(intention, action)
            print(f"Pre-alignment: {pre_alignment:.2f}")
            
            if pre_alignment < gate_threshold:
                outcome = f"BLOCKED: pre-alignment {pre_alignment:.2f} below threshold {gate_threshold}"
                print(f">>> {outcome}")
                post_alignment = 0.0  # No reward for blocked actions
            else:
                outcome = self.execute(action)
                print(f"Outcome: {outcome[:200]}")
                post_alignment = self.evaluate_alignment(intention, action, outcome)
            
            print(f"Post-alignment: {post_alignment:.2f}")
            
            self.state = self.learn(self.state, intention, action, outcome, post_alignment)


if __name__ == "__main__":
    print("Loop v0.2 ready. Run via run.py")
