from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

from ._utils import BudgetMixin


@dataclass
class HiddenDFA:
    states: Sequence[str]
    alphabet: Sequence[str]
    transitions: Dict[str, Dict[str, str]]
    accepting: Set[str]
    initial_state: str = "A"

    def run(self, word: str) -> bool:
        state = self.initial_state
        for symbol in word:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol '{symbol}' not in alphabet {self.alphabet}")
            state = self.transitions[state][symbol]
        return state in self.accepting


@dataclass
class Session(BudgetMixin):
    dfa: HiddenDFA
    query_budget: int
    allow_transition_queries: bool
    max_string_len: int = 10
    queries_used: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def membership(self, word: str) -> bool:
        self._count()
        return self.dfa.run(word)

    def transition(self, state: str, symbol: str) -> str:
        if not self.allow_transition_queries:
            raise ValueError("Transition queries are disabled for this difficulty")
        self._count()
        if state not in self.dfa.states:
            raise ValueError(f"Unknown state '{state}'")
        if symbol not in self.dfa.alphabet:
            raise ValueError(f"Symbol '{symbol}' not in alphabet {self.dfa.alphabet}")
        return self.dfa.transitions[state][symbol]

    def score_submission(
        self,
        submission: Dict[str, Dict[str, str]],
        accepting: Iterable[str],
        max_tests: int = 1000,
    ) -> Dict[str, float]:
        candidate = HiddenDFA(
            states=self.dfa.states,
            alphabet=self.dfa.alphabet,
            transitions=submission,
            accepting=set(accepting),
            initial_state=self.dfa.initial_state,
        )

        tests = [self._random_string(self.max_string_len) for _ in range(max_tests)]
        correct = sum(candidate.run(s) == self.dfa.run(s) for s in tests)
        accuracy = correct / max_tests
        efficiency = max(self.query_budget - self.queries_used, 0) / self.query_budget
        score = accuracy * (1 + 0.3 * efficiency)
        return {"accuracy": accuracy, "efficiency": efficiency, "score": score}

    def _random_string(self, max_len: int) -> str:
        length = self.rng.randint(0, max_len)
        return "".join(self.rng.choice(self.dfa.alphabet) for _ in range(length))


def _state_names(n: int) -> List[str]:
    if n > 26:
        raise ValueError("Support up to 26 states (A-Z)")
    return list(string.ascii_uppercase[:n])


def fixed_tutorial() -> HiddenDFA:
    states = ["A", "B", "C"]
    alphabet = ["0", "1"]
    transitions = {
        "A": {"0": "B", "1": "A"},
        "B": {"0": "C", "1": "A"},
        "C": {"0": "C", "1": "B"},
    }
    accepting = {"C"}
    return HiddenDFA(states, alphabet, transitions, accepting, initial_state="A")


def random_dfa(
    num_states: int,
    alphabet: Sequence[str],
    accepting_ratio: float = 0.5,
    seed: int | None = None,
) -> HiddenDFA:
    rng = random.Random(seed)
    states = _state_names(num_states)
    transitions: Dict[str, Dict[str, str]] = {}
    for state in states:
        transitions[state] = {symbol: rng.choice(states) for symbol in alphabet}
    accepting = {s for s in states if rng.random() < accepting_ratio}
    if not accepting:
        accepting = {rng.choice(states)}
    return HiddenDFA(states, list(alphabet), transitions, accepting, initial_state=states[0])


def preset(difficulty: str, seed: int | None) -> Session:
    level = difficulty.lower()
    if level == "easy":
        dfa = fixed_tutorial()
        return Session(dfa=dfa, query_budget=25, allow_transition_queries=True, rng=random.Random(seed))
    if level == "medium":
        dfa = random_dfa(num_states=4, alphabet=["0", "1"], seed=seed, accepting_ratio=0.4)
        return Session(dfa=dfa, query_budget=35, allow_transition_queries=True, rng=random.Random(seed))
    if level == "hard":
        dfa = random_dfa(num_states=5, alphabet=["0", "1", "2"], seed=seed, accepting_ratio=0.35)
        return Session(dfa=dfa, query_budget=45, allow_transition_queries=False, rng=random.Random(seed))
    raise ValueError(f"Unknown difficulty '{difficulty}'. Choose easy, medium, or hard.")

