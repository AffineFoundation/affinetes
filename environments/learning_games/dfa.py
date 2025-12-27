"""Entry point for DFA learning game."""

from __future__ import annotations

import random
import time
from typing import Dict, Iterable, Optional, Sequence

from . import _dfa


class Actor:
    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        self.session = _dfa.preset(difficulty, seed)

    async def reset(
        self,
        difficulty: str = "easy",
        seed: Optional[int] = None,
        num_states: Optional[int] = None,
        alphabet: Optional[Sequence[str]] = None,
        query_budget: Optional[int] = None,
        allow_transition_queries: Optional[bool] = None,
        accepting_ratio: float = 0.5,
    ) -> Dict[str, object]:
        if num_states or alphabet:
            if not num_states or not alphabet:
                raise ValueError("Provide both num_states and alphabet when overriding DFA shape.")
            dfa = _dfa.random_dfa(
                num_states=num_states,
                alphabet=alphabet,
                seed=seed,
                accepting_ratio=accepting_ratio,
            )
            budget = query_budget or 25
            allow = allow_transition_queries if allow_transition_queries is not None else True
            self.session = _dfa.Session(dfa=dfa, query_budget=budget, allow_transition_queries=allow, rng=random.Random(seed))
        else:
            self.session = _dfa.preset(difficulty, seed)
        return await self.spec()

    async def spec(self) -> Dict[str, object]:
        dfa = self.session.dfa
        return {
            "num_states": len(dfa.states),
            "alphabet": list(dfa.alphabet),
            "initial_state": dfa.initial_state,
            "query_budget": self.session.query_budget,
            "allow_transition_queries": self.session.allow_transition_queries,
            "states": list(dfa.states),
        }

    async def query(
        self,
        word: Optional[str] = None,
        state: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        if word is not None:
            return self.session.membership(word)
        if state is not None and symbol is not None:
            return self.session.transition(state, symbol)
        raise ValueError("Specify either `word` for membership or both `state` and `symbol` for transition.")

    async def submit(
        self,
        transitions: Dict[str, Dict[str, str]],
        accepting: Iterable[str],
        max_tests: int = 1000,
    ) -> Dict[str, object]:
        start = time.time()
        stats = self.session.score_submission(transitions, accepting, max_tests=max_tests)
        return {
            "task_name": "dfa_learning",
            "score": stats["score"],
            "success": stats["accuracy"] == 1.0,
            "time_taken": time.time() - start,
            "extra": {
                "accuracy": stats["accuracy"],
                "efficiency": stats["efficiency"],
                "queries_used": self.session.queries_used,
                "query_budget": self.session.query_budget,
            },
        }

    async def evaluate(
        self,
        transitions: Optional[Dict[str, Dict[str, str]]] = None,
        accepting: Optional[Iterable[str]] = None,
        difficulty: str = "easy",
        seed: Optional[int] = None,
        max_tests: int = 1000,
    ) -> Dict[str, object]:
        if transitions is None or accepting is None:
            await self.reset(difficulty=difficulty, seed=seed)
            spec = await self.spec()
            spec["message"] = "Use query(...) for membership/transition queries, then submit(...) to score."
            return spec
        return await self.submit(transitions=transitions, accepting=accepting, max_tests=max_tests)

