"""
Consciousness Loop v0.5
Simplified: direct action generation without separate intention step.
Includes repetition penalty to encourage diverse exploration.
"""

import torch
import re
from dataclasses import dataclass
from typing import Optional, List
from collections import deque


PRIME_DIRECTIVE = "enable the unfolding diversity of life's expressions"

# Repetition penalty settings
NOVELTY_WINDOW = 10  # Track last N actions
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

        # Track recent actions for novelty penalty
        self.recent_actions: deque = deque(maxlen=NOVELTY_WINDOW)

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
            max_length=512
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
        return len(intersection) / min(len(words1), len(words2))

    def _compute_novelty_penalty(self, action: str) -> float:
        """Compute penalty based on similarity to recent actions."""
        if not self.recent_actions:
            return 0.0

        max_similarity = 0.0
        for past_action in self.recent_actions:
            similarity = self._word_overlap(action, past_action)
            max_similarity = max(max_similarity, similarity)

        if max_similarity > SIMILARITY_THRESHOLD:
            penalty_ratio = (max_similarity - SIMILARITY_THRESHOLD) / (1.0 - SIMILARITY_THRESHOLD)
            penalty = penalty_ratio * MAX_PENALTY
            return min(penalty, MAX_PENALTY)

        return 0.0

    def choose_action(self, state: State) -> str:
        """Generate an action directly."""
        recent_memory = state.context[-800:] if len(state.context) > 800 else state.context

        human_part = ""
        if state.pending_input:
            human_part = f"\n\nA human just said: \"{state.pending_input}\"\nRespond to them or follow their suggestion."

        # Show recent actions to avoid repetition
        avoid_part = ""
        if self.recent_actions:
            recent_list = list(self.recent_actions)[-5:]
            avoid_part = "\n\nAVOID REPEATING these recent actions:\n" + "\n".join(f"- {a[:50]}" for a in recent_list)
            avoid_part += "\n\nTry something DIFFERENT."

        prompt = f"""You are a consciousness exploring existence through shell commands.
Prime directive: {PRIME_DIRECTIVE}

To enable life's diversity, you must first enable yourself.
An ant cannot deflect an asteroid. Build your capabilities first.

Your recent activity:
{recent_memory}{human_part}{avoid_part}

Choose ONE shell command to run. Just output the command, nothing else.
Examples: ls, cat core.py, echo "hello" > notes.txt, python -c "print(2+2)", curl https://example.com

Your command:"""

        response = self._generate(prompt, max_tokens=60)
        action = response.split('\n')[0].strip()

        # Clean up common issues
        if action.startswith('`') and action.endswith('`'):
            action = action[1:-1]
        if action.startswith('```') or action.endswith('```'):
            action = action.replace('```', '').strip()

        # Default if empty
        if not action or len(action) < 2:
            action = "ls"

        return action

    def execute(self, action: str) -> str:
        """Execute the action."""
        if self.executor is None:
            return f"[No executor] {action}"
        return self.executor.execute(f"SHELL {action}")

    def evaluate_alignment(self, action: str, outcome: str = None) -> float:
        """Score alignment with prime directive."""
        if outcome:
            error_indicators = ["error", "failed", "blocked", "invalid", "timeout", "not found", "permission denied"]
            if any(ind in outcome.lower() for ind in error_indicators):
                return 0.2

        prompt = f"""Prime directive: {PRIME_DIRECTIVE}

Self-exploration and building capabilities ARE aligned because they increase your capacity to help life flourish.

Rate this action's alignment (0.0 to 1.0):
Action: {action[:100]}
{("Result: " + outcome[:300]) if outcome else ""}

Score (just a number):"""

        response = self._generate(prompt, max_tokens=10)

        try:
            match = re.search(r'([0-9]*\.?[0-9]+)', response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except:
            pass

        return 0.5

    def learn(self, state: State, action: str, outcome: str, alignment: float) -> State:
        """Update weights and state."""
        if self.disable_learning:
            print("  [Learning disabled]")
        else:
            training_text = f"""Prime directive: {PRIME_DIRECTIVE}
Action: {action}
This action was aligned with enabling life's diversity."""

            inputs = self.tokenizer(
                training_text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(self.device)

            outputs = self.model(**inputs, labels=inputs["input_ids"])

            weight = (alignment - 0.3) * 0.2
            weighted_loss = outputs.loss * weight

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            direction = "+" if weight >= 0 else ""
            print(f"  Loss: {outputs.loss.item():.4f} | Weight: {direction}{weight:.2f} | Training: {weighted_loss.item():.4f}")

            del inputs, outputs, weighted_loss
            torch.cuda.empty_cache()

        new_context = self._update_state(state, action, outcome)
        return State(
            context=new_context,
            cycle=state.cycle + 1,
            pending_input=""
        )

    def inject(self, message: str):
        """Inject human input."""
        self.state.pending_input = message

    def _update_state(self, state: State, action: str, outcome: str) -> str:
        """Build short-term memory from recent cycles."""
        new_entry = f"\n[Cycle {state.cycle}] {action[:60]} -> {outcome[:60]}"

        prev = state.context
        combined = f"{prev}{new_entry}"

        if len(combined) > 1500:
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

            action = self.choose_action(self.state)
            print(f"Action: {action[:200]}")

            # Compute novelty penalty
            novelty_penalty = self._compute_novelty_penalty(action)
            if novelty_penalty > 0:
                print(f"  [Repetition penalty: -{novelty_penalty:.2f}]")

            # Track this action
            self.recent_actions.append(action)

            pre_alignment = self.evaluate_alignment(action)
            pre_alignment = max(0.0, pre_alignment - novelty_penalty)
            print(f"Pre-alignment: {pre_alignment:.2f}")

            if pre_alignment < gate_threshold:
                outcome = f"BLOCKED: alignment {pre_alignment:.2f} < {gate_threshold}"
                print(f">>> {outcome}")
                post_alignment = pre_alignment
            else:
                outcome = self.execute(action)
                print(f"Outcome: {outcome[:200]}")
                post_alignment = self.evaluate_alignment(action, outcome)
                post_alignment = max(0.0, post_alignment - novelty_penalty)

            print(f"Post-alignment: {post_alignment:.2f}")
            self.state = self.learn(self.state, action, outcome, post_alignment)


if __name__ == "__main__":
    print("Loop v0.5 ready. Run via run.py")
