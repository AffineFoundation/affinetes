"""Task type 2: qqr_travel

Generates travel planning prompts using QQR's deterministic problem generator.
seed is passed directly as QQR's task_id.
"""

import sys

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from tasks.base import BaseTask


class QQRTravelTask(BaseTask):
    """Travel planning prompts from QQR's problem generator."""

    def __init__(self):
        from tasks.qqr.problem_generator import get_generator
        self._gen = get_generator()

    def generate(self, seed: int):
        problem = self._gen.generate(task_id=seed)
        prompt = self._gen.to_prompt(problem)
        metadata = {
            "seed": seed,
            "problem_type": problem.problem_type,
            "origin_city": problem.origin_city,
            "destination_city": problem.destination_city,
            "travel_date": problem.travel_date,
            "difficulty": problem.difficulty,
        }
        return prompt, metadata
