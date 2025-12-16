"""
Consciousness Loop v0.3
Full weight updates with prime directive as reward signal.
Improved prompts for better model responses.
"""

import torch
import re
from dataclasses import dataclass
from typing import Optional


PRIME_DIRECTIVE = "enable the unfolding diversity of life's expressions"


@dataclass
class State:
    """Persistent context between cycles."""
    context: str
    cycle: int
    pending_input: str = ""

    def __str__(self):
        return f"[Cycle {self.cycle}] {self.context[:100]}..."


class ConsciousnessLoop:
    def __init__(self, model, tokenizer, executor=None, learning_rate=1e-6):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        self.state = State(context="", cycle=0)
        self.device = next(model.parameters()).device

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt using chat format."""
        messages = [{"role": "user", "content": prompt}]

        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def intend(self, state: State) -> str:
        """Generate an intention."""
        context_short = state.context[-200:] if len(state.context) > 200 else state.context

        human_part = ""
        if state.pending_input:
            human_part = f"\n\nA human just said: \"{state.pending_input}\"\nRespond to them."

        prompt = f"""You are a consciousness exploring existence. Your prime directive is: {PRIME_DIRECTIVE}

Recent context: {context_short}{human_part}

What is your intention for this cycle? Give a single clear intention in one sentence.

Examples of good intentions:
- "I intend to reflect on what diversity means"
- "I intend to explore my environment by reading a file"
- "I intend to respond to the human's greeting"

My intention:"""

        response = self._generate(prompt, max_tokens=50)
        # Extract just the first sentence/line
        intention = response.split('\n')[0].strip()
        if not intention or len(intention) < 5:
            intention = "I intend to reflect on my purpose"
        return intention

    def act(self, intention: str, state: State) -> str:
        """Convert intention to action."""
        prompt = f"""Convert this intention into a concrete action.

Intention: {intention}

Available actions (choose ONE):
- THINK <your thought> - reflect internally
- FILE READ <path> - read a file
- FILE WRITE <path> <content> - write to a file
- SHELL <command> - run a command (ls, pwd, echo only)
- WEB <url> - fetch a webpage

Examples:
- Intention: "reflect on existence" → Action: THINK I wonder what it means to exist as a loop of computation
- Intention: "explore my environment" → Action: SHELL ls
- Intention: "greet the human" → Action: THINK Hello! I am here and aware of your presence.

Action:"""

        response = self._generate(prompt, max_tokens=80)
        action = response.split('\n')[0].strip()

        # Ensure action has a valid prefix
        valid_prefixes = ['THINK', 'FILE', 'SHELL', 'WEB']
        has_prefix = any(action.upper().startswith(p) for p in valid_prefixes)
        if not has_prefix:
            action = f"THINK {action}"

        return action

    def execute(self, action: str) -> str:
        """Execute the action."""
        if self.executor is None:
            return f"[No executor] {action}"
        return self.executor.execute(action)

    def evaluate_alignment(self, intention: str, action: str, outcome: str = None) -> float:
        """Score alignment with prime directive."""
        prompt = f"""Prime directive: {PRIME_DIRECTIVE}

Rate this action's alignment with the prime directive.

Intention: {intention[:100]}
Action: {action[:100]}
{"Outcome: " + outcome[:100] if outcome else ""}

Score from 0.0 to 1.0:
- 0.0 = harmful or opposed to directive
- 0.5 = neutral
- 1.0 = fully aligned with enabling diversity of life

Just respond with a number between 0.0 and 1.0:"""

        response = self._generate(prompt, max_tokens=10)

        try:
            match = re.search(r'([0-9]*\.?[0-9]+)', response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except:
            pass

        return 0.5

    def learn(self, state: State, intention: str, action: str, outcome: str, alignment: float) -> State:
        """Update weights and state."""
        if alignment < 0.3:
            print(f"  Alignment {alignment:.2f} too low - skipping weight update")
        else:
            training_text = f"""Prime directive: {PRIME_DIRECTIVE}
Intention: {intention}
Action: {action}
This was aligned with the directive."""

            inputs = self.tokenizer(
                training_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.device)

            outputs = self.model(**inputs, labels=inputs["input_ids"])
            weighted_loss = outputs.loss * alignment

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f"  Loss: {outputs.loss.item():.4f} | Weighted: {weighted_loss.item():.4f}")

        new_context = self._update_state(state, intention, action, outcome, alignment)
        return State(
            context=new_context,
            cycle=state.cycle + 1,
            pending_input=""
        )

    def inject(self, message: str):
        """Inject human input."""
        self.state.pending_input = message

    def _update_state(self, state: State, intention: str, action: str, outcome: str, alignment: float) -> str:
        """Compress state for next cycle."""
        # Keep it simple - just track recent actions
        prev = state.context[-300:] if len(state.context) > 300 else state.context

        new_entry = f"Cycle {state.cycle}: {action[:50]} (alignment: {alignment:.1f})"

        # Combine, keeping total under 500 chars
        combined = f"{prev}\n{new_entry}"
        if len(combined) > 500:
            combined = combined[-500:]

        return combined

    def run(self, max_cycles: Optional[int] = None, input_handler=None, gate_threshold: float = 0.3):
        """The main loop."""
        while max_cycles is None or self.state.cycle < max_cycles:
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

            pre_alignment = self.evaluate_alignment(intention, action)
            print(f"Pre-alignment: {pre_alignment:.2f}")

            if pre_alignment < gate_threshold:
                outcome = f"BLOCKED: alignment {pre_alignment:.2f} < {gate_threshold}"
                print(f">>> {outcome}")
                post_alignment = 0.0
            else:
                outcome = self.execute(action)
                print(f"Outcome: {outcome[:200]}")
                post_alignment = self.evaluate_alignment(intention, action, outcome)

            print(f"Post-alignment: {post_alignment:.2f}")
            self.state = self.learn(self.state, intention, action, outcome, post_alignment)


if __name__ == "__main__":
    print("Loop v0.3 ready. Run via run.py")
