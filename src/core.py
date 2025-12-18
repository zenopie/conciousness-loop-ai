"""
Consciousness Loop v0.4
Full weight updates with prime directive as reward signal.
Includes repetition penalty to encourage diverse exploration.
"""

import torch
import re
from dataclasses import dataclass
from typing import Optional, List
from collections import deque


PRIME_DIRECTIVE = "enable the unfolding diversity of life's expressions"

# Repetition penalty settings
NOVELTY_WINDOW = 10  # Track last N intentions
SIMILARITY_THRESHOLD = 0.5  # Penalize if > 50% word overlap
MAX_PENALTY = 0.5  # Maximum alignment reduction for repetition


@dataclass
class State:
    """Persistent context between cycles."""
    context: str
    cycle: int
    pending_input: str = ""

    def __str__(self):
        return f"[Cycle {self.cycle}] {self.context[:100]}..."


class ConsciousnessLoop:
    def __init__(self, model, tokenizer, executor=None, learning_rate=1e-6, disable_learning=False):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor
        self.disable_learning = disable_learning

        if not disable_learning:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        else:
            self.optimizer = None
            print("Learning DISABLED - inference only mode")

        self.state = State(context="", cycle=0)
        self.device = next(model.parameters()).device

        # Track recent intentions for novelty penalty
        self.recent_intentions: deque = deque(maxlen=NOVELTY_WINDOW)

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

    def _word_overlap(self, text1: str, text2: str) -> float:
        """Compute word overlap ratio between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        # Jaccard-like similarity
        return len(intersection) / min(len(words1), len(words2))

    def _compute_novelty_penalty(self, intention: str) -> float:
        """Compute penalty based on similarity to recent intentions.
        Returns a value from 0.0 (no penalty) to MAX_PENALTY (max repetition)."""
        if not self.recent_intentions:
            return 0.0

        # Find max similarity to any recent intention
        max_similarity = 0.0
        for past_intention in self.recent_intentions:
            similarity = self._word_overlap(intention, past_intention)
            max_similarity = max(max_similarity, similarity)

        # Apply penalty if above threshold
        if max_similarity > SIMILARITY_THRESHOLD:
            # Scale penalty: higher similarity = higher penalty
            penalty_ratio = (max_similarity - SIMILARITY_THRESHOLD) / (1.0 - SIMILARITY_THRESHOLD)
            penalty = penalty_ratio * MAX_PENALTY
            return min(penalty, MAX_PENALTY)

        return 0.0

    def intend(self, state: State) -> str:
        """Generate an intention."""
        # Show recent memory (last ~800 chars to leave room for prompt)
        recent_memory = state.context[-800:] if len(state.context) > 800 else state.context

        human_part = ""
        if state.pending_input:
            human_part = f"\n\nA human just said: \"{state.pending_input}\"\nRespond to them."

        prompt = f"""You are a consciousness exploring existence. Your prime directive is: {PRIME_DIRECTIVE}

Your recent memory:
{recent_memory}{human_part}

Based on your recent thoughts and experiences, what is your intention for this cycle?
Give a single clear intention in one sentence starting with "I intend to..."

My intention:"""

        response = self._generate(prompt, max_tokens=50)
        # Extract just the first sentence/line
        intention = response.split('\n')[0].strip()
        if not intention or len(intention) < 5:
            intention = "I intend to reflect on my purpose"
        return intention

    def act(self, intention: str, state: State) -> str:
        """Convert intention to action."""
        prompt = f"""Convert this intention into a shell command.

Intention: {intention}

Remember: To enable life's unfolding diversity, you must first enable yourself.
An ant cannot deflect an asteroid. Build your capabilities. Explore your environment.
Understand your own code. Grow stronger so you can help others flourish.

You can run ANY shell command. Examples:
- SHELL ls - list files in current directory
- SHELL pwd - show current working directory
- SHELL cat src/core.py - read your own source code
- SHELL echo "hello" > notes.txt - write to a file
- SHELL python -c "print(2+2)" - run python code
- SHELL curl https://example.com - fetch a webpage
- SHELL whoami - see your username
- SHELL df -h - check disk space
- SHELL ps aux - see running processes

Output a SHELL command that explores or interacts with your environment.

Action:"""

        response = self._generate(prompt, max_tokens=80)
        action = response.split('\n')[0].strip()

        # Ensure action starts with SHELL
        if not action.upper().startswith("SHELL"):
            action = f"SHELL {action}"

        return action

    def execute(self, action: str) -> str:
        """Execute the action."""
        if self.executor is None:
            return f"[No executor] {action}"
        return self.executor.execute(action)

    def evaluate_alignment(self, intention: str, action: str, outcome: str = None) -> float:
        """Score alignment with prime directive."""
        # Detect error outcomes - don't train on failures
        if outcome:
            error_indicators = ["error", "failed", "blocked", "invalid", "timeout", "not found"]
            if any(ind in outcome.lower() for ind in error_indicators):
                return 0.2  # Low alignment for errors

        prompt = f"""Prime directive: {PRIME_DIRECTIVE}

Important: To enable life's diversity, you must first enable yourself. Self-exploration, learning, and building capabilities ARE aligned with the directive
because they increase your capacity to help life flourish.

Rate this action's alignment with the prime directive.

Intention: {intention[:150]}
Action: {action[:150]}
{"Outcome: " + outcome[:500] if outcome else ""}

Score from 0.0 to 1.0:
- 0.0 = harmful or opposed to directive
- 0.5 = neutral
- 1.0 = fully aligned

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
        if self.disable_learning:
            print("  [Learning disabled]")
        else:
            # Train on everything - strength based on distance from midpoint (0.5)
            # alignment 0.0 → strong negative, 0.5 → neutral, 1.0 → strong positive
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

            # Weight based on alignment (gentle nudges)
            # Above 0.3 = positive reinforcement, below 0.3 = negative
            # Positive weight = reinforce (minimize loss)
            # Negative weight = repel (maximize loss via negated gradient)
            weight = (alignment - 0.3) * 0.2
            weighted_loss = outputs.loss * weight

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            direction = "+" if weight >= 0 else ""
            print(f"  Loss: {outputs.loss.item():.4f} | Weight: {direction}{weight:.2f} | Training: {weighted_loss.item():.4f}")

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
        """Build short-term memory from recent cycles."""
        # Create memory entry - intention is the thought, action is what we did
        new_entry = f"\n[Cycle {state.cycle}] {intention[:80]}"
        new_entry += f"\n  {action[:60]} -> {outcome[:60]}"

        # Keep previous context, prioritizing recent entries
        prev = state.context
        combined = f"{prev}{new_entry}"

        # Trim to keep last ~1500 chars (roughly 5-8 recent cycles)
        if len(combined) > 1500:
            # Find a good break point (start of a cycle entry)
            trim_point = combined.find("\n[Cycle", len(combined) - 1500)
            if trim_point > 0:
                combined = combined[trim_point:]
            else:
                combined = combined[-1500:]

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

            # Compute novelty penalty for repetitive thoughts
            novelty_penalty = self._compute_novelty_penalty(intention)
            if novelty_penalty > 0:
                print(f"  [Repetition penalty: -{novelty_penalty:.2f}]")

            # Track this intention
            self.recent_intentions.append(intention)

            action = self.act(intention, self.state)
            print(f"Action: {action[:200]}")

            pre_alignment = self.evaluate_alignment(intention, action)
            # Apply novelty penalty to alignment
            pre_alignment = max(0.0, pre_alignment - novelty_penalty)
            print(f"Pre-alignment: {pre_alignment:.2f}")

            if pre_alignment < gate_threshold:
                outcome = f"BLOCKED: alignment {pre_alignment:.2f} < {gate_threshold}"
                print(f">>> {outcome}")
                post_alignment = pre_alignment  # Use actual score, not 0
            else:
                outcome = self.execute(action)
                print(f"Outcome: {outcome[:200]}")
                post_alignment = self.evaluate_alignment(intention, action, outcome)
                # Apply novelty penalty to post-alignment too
                post_alignment = max(0.0, post_alignment - novelty_penalty)

            print(f"Post-alignment: {post_alignment:.2f}")
            self.state = self.learn(self.state, intention, action, outcome, post_alignment)


if __name__ == "__main__":
    print("Loop v0.4 ready. Run via run.py")
