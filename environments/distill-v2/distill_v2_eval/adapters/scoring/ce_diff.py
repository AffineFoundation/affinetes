"""``ce_diff`` family of scoring strategies. **Legacy — pair-based.**

Superseded by ``adapters.scoring.reward_weighted.RewardWeightedScoring``
for the rollout-level flow. Kept buildable so old runs and back-compat
configs still load.

All four variants share one body parametrised by two orthogonal flags:

- ``length_normalize``: divide each side's CE by its assistant-token count.
  Removes the bias from long-vs-short trajectories so a pair's contribution
  is per-token NLL diff instead of total NLL diff.
- ``reward_weighted``: multiply the final margin by ``pair.reward_gap``.
  Down-weights weak-signal pairs (small partial-reward gap) and lets
  binary-style 1.0-vs-0.0 pairs carry full weight.

Combining the two yields the recommended default for partial-reward
environments (e.g. SWE-rebench):

    score(pair) = reward_gap * (ce_lose / n_lose − ce_win / n_win)

Both flags off is the baseline `ce_lose − ce_win` (DPO-style margin) and
remains useful for binary pass/fail comparisons where reward_gap ∈ {0, 1}.
"""

from __future__ import annotations

from distill_v2_eval.domain.models import Pair, PairCEResult


class CeDiffScoring:
    """Configurable ce_diff scoring; ``name`` reflects the active flag combo."""

    version = "1.0"

    def __init__(
        self,
        *,
        length_normalize: bool = False,
        reward_weighted: bool = False,
    ) -> None:
        self._length_normalize = length_normalize
        self._reward_weighted = reward_weighted

    @property
    def name(self) -> str:
        parts = ["ce_diff"]
        if self._reward_weighted:
            parts.insert(0, "weighted")
        if self._length_normalize:
            parts.append("lnorm")
        return "_".join(parts)

    def score_pair(self, ce: PairCEResult, pair: Pair) -> float:
        if self._length_normalize:
            ce_win = ce.ce_win / max(ce.tokens_win, 1)
            ce_lose = ce.ce_lose / max(ce.tokens_lose, 1)
        else:
            ce_win = ce.ce_win
            ce_lose = ce.ce_lose
        margin = ce_lose - ce_win
        if self._reward_weighted:
            margin *= pair.reward_gap
        return margin
