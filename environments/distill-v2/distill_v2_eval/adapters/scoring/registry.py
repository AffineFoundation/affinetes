"""Scoring strategy registry.

Two families:

  * ``ce_diff`` variants — legacy pair-based scoring; superseded by
    rollout-level scoring but kept buildable for back-compat with old
    ``student_scores.config`` snapshots.
  * ``reward_weighted_ce`` — current default. Per-rollout REINFORCE-style
    score, consumed by ``evaluate_student_rollout_level``.
"""

from __future__ import annotations

from typing import Any

from distill_v2_eval.adapters.scoring.ce_diff import CeDiffScoring
from distill_v2_eval.adapters.scoring.reward_weighted import RewardWeightedScoring
from distill_v2_eval.domain.errors import EvalError
from distill_v2_eval.domain.ports.scoring import RolloutScoringStrategy, ScoringStrategy


# Pair-based (legacy). Still buildable so we can replay old runs.
_PAIR_BUILDERS: dict[str, Any] = {
    "ce_diff":                  lambda **kw: CeDiffScoring(**kw),
    "ce_diff_lnorm":            lambda **kw: CeDiffScoring(length_normalize=True, **kw),
    "weighted_ce_diff":         lambda **kw: CeDiffScoring(reward_weighted=True, **kw),
    "weighted_ce_diff_lnorm":   lambda **kw: CeDiffScoring(
        reward_weighted=True, length_normalize=True, **kw,
    ),
    # historical alias
    "ce_diff_length_normalized": lambda **kw: CeDiffScoring(length_normalize=True, **kw),
}


# Rollout-based (current).
_ROLLOUT_BUILDERS: dict[str, Any] = {
    "reward_weighted_ce":       lambda **kw: RewardWeightedScoring(**kw),
    "reward_weighted_ce_total": lambda **kw: RewardWeightedScoring(length_normalize=False, **kw),
}


def build_scoring(name: str, **kwargs: Any) -> ScoringStrategy:
    """Build a pair-based scoring strategy (legacy path)."""
    try:
        return _PAIR_BUILDERS[name](**kwargs)
    except KeyError as e:
        raise EvalError(
            f"unknown scoring strategy (pair-based): {name!r}. "
            f"Known: {', '.join(sorted(_PAIR_BUILDERS))}"
        ) from e


def build_rollout_scoring(name: str, **kwargs: Any) -> RolloutScoringStrategy:
    """Build a rollout-based scoring strategy."""
    try:
        return _ROLLOUT_BUILDERS[name](**kwargs)
    except KeyError as e:
        raise EvalError(
            f"unknown scoring strategy (rollout-based): {name!r}. "
            f"Known: {', '.join(sorted(_ROLLOUT_BUILDERS))}"
        ) from e


def known_scorings() -> list[str]:
    """All scoring strategy names — pair + rollout — sorted."""
    return sorted({*_PAIR_BUILDERS, *_ROLLOUT_BUILDERS})
