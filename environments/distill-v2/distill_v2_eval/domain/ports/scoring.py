"""Scoring strategy ports.

Three parallel families of pure functions:

  * ``ScoringStrategy`` — pair-based. Consumes ``(PairCEResult, Pair)`` from
    the legacy pair-mining flow. Kept around for back-compat but no longer
    the default; see ``adapters/scoring/ce_diff.py``.

  * ``RolloutScoringStrategy`` — per-rollout. Consumes one CE + reward +
    pre-computed per-task advantage. GRPO/REINFORCE-style: every rollout
    contributes signal independently, no pair mining required. The
    aggregate is a linear function of CE and therefore *unbounded* — the
    reported score grows with the NLL spread, which itself grows as the
    student sharpens with training.

  * ``CellScoringStrategy`` — whole-cell. Sees all N rollouts of a cell at
    once so it can normalize across them (e.g. a softmax over per-token
    NLL). This is what lets the score be *bounded*: the cell score is an
    expectation of the (bounded) advantage under a probability distribution
    the student implicitly assigns to the rollouts, so it saturates instead
    of exploding. See ``adapters/scoring/softmax_advantage.py``.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

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


class RolloutMeasurement(NamedTuple):
    """One rollout's forward result, fed to a ``CellScoringStrategy``.

    ``ce`` is the summed cross-entropy (NLL) over the rollout's unmasked
    assistant tokens; ``advantage`` is its within-cell reward z-score.
    """

    ce: float
    n_tokens: int
    advantage: float


class CellScore(NamedTuple):
    """A ``CellScoringStrategy`` verdict for one cell.

    ``per_rollout_scores`` is aligned with the input measurement order so
    the driver can attach diagnostics; for the softmax scorer these are the
    per-rollout advantage *contributions* and sum to ``mean_score``.
    """

    mean_score: float
    win_rate: float
    per_rollout_scores: list[float]


@runtime_checkable
class CellScoringStrategy(Protocol):
    """Whole-cell scoring.

    Unlike ``RolloutScoringStrategy`` the strategy sees every rollout of the
    cell together, so it can normalize the absolute CE scale away (softmax
    over per-token NLL) and return a *bounded* score. The driver still owns
    advantage computation; the strategy only maps CE + advantage → score.
    """

    name: str
    version: str

    def score_cell(self, measurements: list[RolloutMeasurement]) -> CellScore:
        """Return the cell's bounded score. Higher = student implicitly
        prefers the higher-advantage trajectories."""
