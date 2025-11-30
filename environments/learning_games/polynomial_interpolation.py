"""Entry point for polynomial interpolation game."""

from __future__ import annotations

import random
import time
from typing import Dict

from . import _poly


class Actor:
    def __init__(self, difficulty: str = "easy", seed: int | None = None):
        self.session = _poly.preset(difficulty, seed)

    async def reset(
        self,
        difficulty: str = "easy",
        seed: int | None = None,
        max_degree: int | None = None,
        coeff_min: int = -10,
        coeff_max: int = 10,
        query_budget: int | None = None,
    ) -> dict:
        if max_degree:
            rng = random.Random(seed)
            poly = _poly.random_poly(max_degree, range(coeff_min, coeff_max + 1), rng)
            budget = query_budget or (max_degree + 6)
            self.session = _poly.Session(poly=poly, query_budget=budget, rng=rng)
        else:
            self.session = _poly.preset(difficulty, seed)
        return await self.spec()

    async def spec(self) -> dict:
        return {
            "task": "Discover polynomial f(x)",
            "max_degree": self.session.poly.degree,
            "coefficient_range": [-10, 10],
            "query_budget": self.session.query_budget,
        }

    async def query(self, x: float) -> float:
        return self.session.query(x)

    async def submit(self, submission) -> dict:
        start = time.time()
        stats = self.session.score_submission(submission)
        return {
            "task_name": "polynomial_interpolation",
            "score": stats["score"],
            "success": stats["accuracy"] == 1.0,
            "time_taken": time.time() - start,
            "extra": {
                "accuracy": stats["accuracy"],
                "queries_used": self.session.queries_used,
                "query_budget": self.session.query_budget,
                "degree": self.session.poly.degree,
            },
        }

    async def evaluate(
        self,
        submission=None,
        difficulty: str = "easy",
        seed: int | None = None,
    ) -> dict:
        if submission is None:
            await self.reset(difficulty=difficulty, seed=seed)
            spec = await self.spec()
            spec["message"] = "Use query(x=...) to sample points, then submit(coeffs or expr) to score."
            return spec
        return await self.submit(submission)

