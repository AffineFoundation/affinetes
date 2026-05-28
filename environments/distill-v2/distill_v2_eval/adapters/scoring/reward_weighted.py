"""``reward_weighted_ce`` — per-rollout REINFORCE-style scoring.

Replaces the pair-based scoring family for the rollout-level eval flow.

Score formula (length-normalized by default since real trajectories have
wildly different lengths):

    score(rollout) = - (ce_sum / n_tokens) * advantage

where ``advantage`` is a pre-computed per-task z-score of reward — see
:func:`distill_v2_eval.application.advantage.compute_advantages`. The
strategy itself stays pure; it does not know about the rollout's task
or peers.

Sign convention: higher score ⇔ student model is *more aligned with
high-advantage trajectories*. A trajectory whose advantage is large and
positive contributes more if the student's NLL on it is small.
"""

from __future__ import annotations


class RewardWeightedScoring:
    """Per-rollout REINFORCE-style score: ``- ce/n × advantage``.

    ``length_normalize=True`` (the default) divides CE by the rollout's
    masked token count so long trajectories don't dominate. Flip off only
    for ablations.
    """

    version = "1.0"

    def __init__(self, *, length_normalize: bool = True) -> None:
        self._length_normalize = length_normalize

    @property
    def name(self) -> str:
        return "reward_weighted_ce" if self._length_normalize else "reward_weighted_ce_total"

    def score_rollout(
        self, *,
        ce: float,
        n_tokens: int,
        reward: float,  # noqa: ARG002 — kept in signature for diagnostic adapters
        advantage: float,
    ) -> float:
        if self._length_normalize:
            ce_term = ce / max(n_tokens, 1)
        else:
            ce_term = ce
        return -ce_term * advantage
