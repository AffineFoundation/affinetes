"""Task type 4: swe_agent_fix

Generates prompts that simulate a minisweagent multi-turn debugging session.
The prompt contains:
  - System prompt (agent instructions with THOUGHT + bash format)
  - Bug report (instance prompt with PR description)
  - Prior conversation turns (THOUGHT + bash + observation pairs)

The model should generate the next THOUGHT + bash action, which is where
we extract logprobs to measure agent-style reasoning capability.

Sample space:
  15 bug reports × 8 conversation styles × seed-based variations
  (turn count, turn selection, ordering) → easily covers 100M unique seeds.
"""

import random
import sys

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from tasks.base import BaseTask
from data.swe_agent_scenarios import (
    SYSTEM_PROMPT,
    INSTANCE_TEMPLATE,
    BUG_REPORTS,
    CONVERSATION_TURNS,
    OBSERVATION_TEMPLATE,
    ACTION_TEMPLATE,
)


class SWEAgentFixTask(BaseTask):
    """Generate a minisweagent-style multi-turn debugging prompt."""

    def generate(self, seed: int):
        rng = random.Random(seed)

        # Pick a bug report
        bug = rng.choice(BUG_REPORTS)
        instance_prompt = INSTANCE_TEMPLATE.format(bug_report=bug["report"])

        # Decide how many prior conversation styles to combine (1-3)
        n_turn_groups = rng.randint(1, min(3, len(CONVERSATION_TURNS)))
        turn_indices = list(range(len(CONVERSATION_TURNS)))
        rng.shuffle(turn_indices)
        selected_groups = [CONVERSATION_TURNS[i] for i in turn_indices[:n_turn_groups]]

        # Flatten into a single conversation history
        history_turns = []
        for group in selected_groups:
            history_turns.extend(group)

        # Optionally truncate to keep prompt reasonable (max 6 turns)
        max_turns = rng.randint(2, 6)
        if len(history_turns) > max_turns:
            history_turns = history_turns[:max_turns]

        # Build the full multi-turn prompt
        parts = [SYSTEM_PROMPT, "", instance_prompt]

        for thought, command, observation in history_turns:
            # Agent turn
            parts.append(ACTION_TEMPLATE.format(thought=thought, command=command))
            # Observation
            parts.append(OBSERVATION_TEMPLATE.format(returncode=0, output=observation))

        prompt = "\n\n".join(parts)

        metadata = {
            "seed": seed,
            "repo": bug["repo"],
            "language": bug["language"],
            "n_prior_turns": len(history_turns),
            "files": bug["files"],
        }
        return prompt, metadata
