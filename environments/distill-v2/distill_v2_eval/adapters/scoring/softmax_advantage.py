"""``softmax_advantage`` — bounded, whole-cell rollout scoring.

Motivation
----------

The rollout-level ``reward_weighted_ce`` score is ``-(ce/n)·advantage``
summed over a cell. That is *linear and unbounded* in the per-token NLL
``c = ce/n``: since NLL has no upper bound (a model can assign a token
probability arbitrarily close to 0), the aggregate can be driven to
``+∞`` simply by inflating the NLL of the low-advantage rollouts. Worse,
it drifts upward with *honest* training too — as the student sharpens,
the NLL spread ``std(c)`` grows and the score grows with it, so the
number is not comparable across time and has no natural ceiling.

Formula
-------

Treat the student's per-token NLL as an (un-normalized) log-preference
over the cell's rollouts and softmax it into a distribution, then take
the expected advantage under that distribution::

    p_i = softmax(-c_i / τ)_i          # student's implicit trajectory preference
    S   = Σ_i p_i · a_i                 # expected advantage under p

``S`` is an expectation of the bounded advantage ``a`` under a probability
distribution, so ``S ∈ [min_i a_i, max_i a_i]`` — bounded by construction.
A model with no preference (uniform ``p``) scores ``Σ a_i / N = 0`` because
advantages are within-cell z-scores. Preferring high-advantage rollouts
pushes ``S`` toward ``max a``; preferring low-advantage ones pushes it
negative. As the student sharpens, ``p`` concentrates on its favourite
rollout and ``S`` *saturates* instead of exploding.

Temperature
-----------

``temperature=None`` (default) uses an **adaptive** ``τ = std(c)`` — i.e.
softmax over the *z-scored* NLL. This makes the score fully scale-free:
multiplying every NLL by a constant leaves ``S`` unchanged, so a model
cannot farm score by spiking NLL on bad trajectories, and the number
cannot drift with training. Only the *shape* of the NLL ranking matters.

A fixed ``temperature`` (in per-token-NLL units) keeps some sensitivity to
absolute confidence: as gaps grow, ``p`` concentrates and ``S`` climbs
toward ``max a`` — still bounded, but no longer scale-free. Use it only
for ablations; the adaptive default is the anti-gaming choice.
"""

from __future__ import annotations

import math

from distill_v2_eval.domain.ports.scoring import CellScore, RolloutMeasurement


class SoftmaxAdvantageScoring:
    """Bounded whole-cell scorer: ``S = Σ softmax(-c/τ)·a``."""

    version = "1.0"

    def __init__(
        self,
        *,
        temperature: float | None = None,
        length_normalize: bool = True,
    ) -> None:
        # temperature=None → adaptive τ = std(c) (scale-free). A positive
        # float pins τ in per-token-NLL units (confidence-sensitive).
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be > 0 or None, got {temperature!r}")
        self._temperature = temperature
        self._length_normalize = length_normalize

    @property
    def name(self) -> str:
        base = "softmax_advantage" if self._length_normalize else "softmax_advantage_total"
        if self._temperature is None:
            return base
        return f"{base}_t{self._temperature:g}"

    def score_cell(self, measurements: list[RolloutMeasurement]) -> CellScore:
        n = len(measurements)
        if n == 0:
            return CellScore(mean_score=0.0, win_rate=0.0, per_rollout_scores=[])

        # Per-token NLL (length-normalized) is the default: real
        # trajectories differ wildly in length, and total NLL would just
        # rank by brevity. Flip off only for ablations.
        if self._length_normalize:
            c = [m.ce / max(m.n_tokens, 1) for m in measurements]
        else:
            c = [m.ce for m in measurements]
        a = [m.advantage for m in measurements]

        tau = self._resolve_tau(c)
        if tau <= 0.0:
            # Every rollout has identical NLL (or a single rollout): the
            # student expresses no preference → uniform → S = mean(a) = 0.
            p = [1.0 / n] * n
        else:
            logits = [-x / tau for x in c]
            hi = max(logits)  # subtract max for numerical stability
            exps = [math.exp(v - hi) for v in logits]
            z = sum(exps)
            p = [e / z for e in exps]

        contributions = [p_i * a_i for p_i, a_i in zip(p, a)]
        mean_score = sum(contributions)
        # Probability mass the student places on above-baseline (a>0)
        # rollouts — the bounded analogue of the old win_rate ∈ [0, 1].
        win_rate = sum(p_i for p_i, a_i in zip(p, a) if a_i > 0.0)
        return CellScore(
            mean_score=mean_score,
            win_rate=win_rate,
            per_rollout_scores=contributions,
        )

    def _resolve_tau(self, c: list[float]) -> float:
        if self._temperature is not None:
            return self._temperature
        # Adaptive: population std of the NLLs. softmax(-c/std(c)) is
        # exactly softmax over the z-scored NLL, hence scale-free.
        n = len(c)
        mean_c = sum(c) / n
        var_c = sum((x - mean_c) ** 2 for x in c) / n
        return math.sqrt(var_c)
