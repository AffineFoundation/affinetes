"""Scoring strategy registry.

Three families:

  * ``ce_diff`` variants — legacy pair-based scoring; superseded by
    rollout-level scoring but kept buildable for back-compat with old
    ``student_scores.config`` snapshots.
  * ``reward_weighted_ce`` — per-rollout REINFORCE-style score. Unbounded
    in CE; kept for replay and ablation.
  * ``softmax_advantage`` — current default. Whole-cell, softmax-normalized
    and therefore *bounded*; see ``adapters/scoring/softmax_advantage.py``.
"""

from __future__ import annotations

from typing import Any

from distill_v2_eval.adapters.scoring.ce_diff import CeDiffScoring
from distill_v2_eval.adapters.scoring.reward_weighted import RewardWeightedScoring
from distill_v2_eval.adapters.scoring.softmax_advantage import SoftmaxAdvantageScoring
from distill_v2_eval.domain.errors import EvalError
from distill_v2_eval.domain.ports.scoring import (
    CellScoringStrategy,
    RolloutScoringStrategy,
    ScoringStrategy,
)


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


# Rollout-based (per-rollout, unbounded in CE). Kept for replay/ablation.
_ROLLOUT_BUILDERS: dict[str, Any] = {
    "reward_weighted_ce":       lambda **kw: RewardWeightedScoring(**kw),
    "reward_weighted_ce_total": lambda **kw: RewardWeightedScoring(length_normalize=False, **kw),
}


# Cell-based (whole-cell, softmax-normalized, bounded). Current default.
_CELL_BUILDERS: dict[str, Any] = {
    "softmax_advantage":       lambda **kw: SoftmaxAdvantageScoring(**kw),
    "softmax_advantage_total": lambda **kw: SoftmaxAdvantageScoring(length_normalize=False, **kw),
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


def build_cell_scoring(name: str, **kwargs: Any) -> CellScoringStrategy:
    """Build a whole-cell scoring strategy."""
    try:
        return _CELL_BUILDERS[name](**kwargs)
    except KeyError as e:
        raise EvalError(
            f"unknown scoring strategy (cell-based): {name!r}. "
            f"Known: {', '.join(sorted(_CELL_BUILDERS))}"
        ) from e


def is_cell_scoring(name: str) -> bool:
    """True if ``name`` is a whole-cell scorer (vs per-rollout/pair)."""
    return name in _CELL_BUILDERS


def known_scorings() -> list[str]:
    """All scoring strategy names — pair + rollout + cell — sorted."""
    return sorted({*_PAIR_BUILDERS, *_ROLLOUT_BUILDERS, *_CELL_BUILDERS})
