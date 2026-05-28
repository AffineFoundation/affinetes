"""Scoring strategy ports.

Two parallel families of pure functions:

  * ``ScoringStrategy`` — pair-based. Consumes ``(PairCEResult, Pair)`` from
    the legacy pair-mining flow. Kept around for back-compat but no longer
    the default; see ``adapters/scoring/ce_diff.py``.

  * ``RolloutScoringStrategy`` — per-rollout. Consumes one CE + reward +
    pre-computed per-task advantage. This is the GRPO/REINFORCE-style flow
    we use now: every rollout contributes signal, no pair mining required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from distill_v2_eval.domain.models import Pair, PairCEResult


@runtime_checkable
class ScoringStrategy(Protocol):
    """Pair-based scoring. Legacy — superseded by ``RolloutScoringStrategy``."""

    name: str
    version: str

    def score_pair(self, ce: PairCEResult, pair: Pair) -> float:
        """Return per-pair score. Higher = student more aligned with win over lose."""


@runtime_checkable
class RolloutScoringStrategy(Protocol):
    """Per-rollout scoring.

    The caller is responsible for pre-computing the per-task ``advantage``
    (typically a z-score of reward within the rollout's task bucket) — the
    strategy itself stays pure: CE + tokens + raw reward + advantage in,
    a single score out.
    """

    name: str
    version: str

    def score_rollout(
        self, *,
        ce: float,
        n_tokens: int,
        reward: float,
        advantage: float,
    ) -> float:
        """Return per-rollout score. Higher = student more aligned with high-reward trajectories."""
