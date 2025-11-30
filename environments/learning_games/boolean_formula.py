"""Entry point for boolean formula discovery game."""

from __future__ import annotations

import random
import time
from typing import Dict, Sequence

from . import _boolean


class Actor:
    def __init__(self, difficulty: str = "easy", seed: int | None = None):
        self.session = _boolean.preset(difficulty, seed)

    async def reset(
        self,
        difficulty: str = "easy",
        seed: int | None = None,
        variables: Sequence[str] | None = None,
        depth: int = 3,
        query_budget: int | None = None,
    ) -> Dict[str, object]:
        if variables:
            form = _boolean.random_formula(variables, depth=depth, seed=seed)
            budget = query_budget or max(4, len(variables) + 2)
            self.session = _boolean.Session(formula=form, query_budget=budget, rng=random.Random(seed))
        else:
            self.session = _boolean.preset(difficulty, seed)
        return await self.spec()

    async def spec(self) -> Dict[str, object]:
        form = self.session.formula
        return {
            "task": "Discover boolean function",
            "variables": list(form.variables),
            "output": "bool",
            "query_budget": self.session.query_budget,
        }

    async def query(self, **assignments: bool) -> bool:
        return self.session.query(assignments)

    async def submit(self, expression: str) -> Dict[str, object]:
        start = time.time()
        stats = self.session.score_submission(expression)
        return {
            "task_name": "boolean_formula",
            "score": stats["score"],
            "success": stats["accuracy"] == 1.0,
            "time_taken": time.time() - start,
            "extra": {
                "accuracy": stats["accuracy"],
                "queries_used": self.session.queries_used,
                "query_budget": self.session.query_budget,
                "hidden_expr": self.session.formula.expr,
            },
        }

    async def evaluate(
        self,
        expression: str | None = None,
        difficulty: str = "easy",
        seed: int | None = None,
    ) -> Dict[str, object]:
        if expression is None:
            await self.reset(difficulty=difficulty, seed=seed)
            spec = await self.spec()
            spec["message"] = "Use query(...) to probe the function, then submit(expression) to score."
            return spec
        return await self.submit(expression)

