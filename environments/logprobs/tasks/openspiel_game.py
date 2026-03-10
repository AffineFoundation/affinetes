"""Task type 3: openspiel_game

Generates prompts that match the exact format LLMs see during OpenSpiel game
evaluation.  The prompt combines a system section (game name, rules, output
format) with a user section (current state, player id, legal actions).

This is useful for measuring logprob distributions of models that were
specifically trained on OpenSpiel game environments.

Sample space:
  15 games × 2-4 scenarios each ≈ 40 base scenarios.
  Seed selects scenario and applies minor variations (player_id swap,
  action-list shuffling) to expand coverage.
"""

import random
import sys

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from tasks.base import BaseTask
from data.openspiel_scenarios import GAMES, SYSTEM_TEMPLATE, USER_TEMPLATE


class OpenSpielGameTask(BaseTask):
    """Generate an OpenSpiel-style game prompt from a seed."""

    def __init__(self):
        # Flatten all scenarios into a list for easy indexing
        self._scenarios = []
        for game in GAMES:
            for scenario in game["scenarios"]:
                self._scenarios.append((game, scenario))

    def generate(self, seed: int):
        rng = random.Random(seed)

        # Pick a scenario deterministically
        idx = rng.randint(0, len(self._scenarios) - 1)
        game, (state_desc, player_id, legal_actions) = self._scenarios[idx]

        # Optionally shuffle the legal actions order (preserves content)
        actions = list(legal_actions)
        if rng.random() < 0.5:
            rng.shuffle(actions)

        # Build the system part
        system_part = SYSTEM_TEMPLATE.format(
            game_name=game["name"],
            rules=game["rules"],
        )

        # Build the user part
        actions_str = "\n".join(actions)
        user_part = USER_TEMPLATE.format(
            state_desc=state_desc,
            player_id=player_id,
            legal_actions=actions_str,
        )

        # Combine into a single prompt (for /v1/completions)
        prompt = system_part + "\n\n" + user_part

        metadata = {
            "seed": seed,
            "game_name": game["name"],
            "player_id": player_id,
            "n_legal_actions": len(actions),
            "scenario_index": idx,
        }
        return prompt, metadata
