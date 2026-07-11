"""End-to-end evaluator driver, in-memory only.

One ``evaluate`` call = one cell (one ``task_idx``) = one
``(env, task, teacher)`` group of N rollouts.

Sequence:
    1. Fetch the cell's parquet from R2 (direct by task_idx — the shard
       key is ``tasks/{task_idx:08d}.parquet`` by construction, so the
       manifest is not needed; downloading + parsing it per call was a
       per-request memory/latency tax that grows with the corpus).
    2. Spin up the ``vllm_logprob`` forward engine against the
       caller-supplied vLLM endpoint and load the student tokenizer.
    3. Within the cell, z-score rewards → per-rollout advantage. (All
       rollouts share the same teacher, so this isolates the
       reward-vs-baseline signal from teacher-style differences.)
    4. Render + forward each rollout, compute CE, apply
       ``RolloutScoringStrategy``.
    5. Aggregate into a single per-cell score (mean over the N
       rollouts) — that's the value cortex consumes.

No PG, no S3 writes, no global state. The score is fully determined
by (task_idx, miner model, base_url) since the cell shard is
immutable.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import Any

from distill_v2_eval.adapters.forward_engines.render import (
    RenderConfig,
    render_rollout,
)
from distill_v2_eval.adapters.forward_engines.render_registry import build_renderer
from distill_v2_eval.adapters.forward_engines.vllm_logprob import VLLMLogprobEngine
from distill_v2_eval.adapters.scoring.registry import (
    build_cell_scoring,
    build_rollout_scoring,
    is_cell_scoring,
)
from distill_v2_eval.application.advantage import compute_advantages
from distill_v2_eval.domain.errors import EvalError
from distill_v2_eval.domain.ids import Revision, StudentName
from distill_v2_eval.domain.models import StudentSubmission
from distill_v2_eval.domain.ports.scoring import RolloutMeasurement
from distill_v2_eval.infrastructure.logging import get_logger
from distill_v2_eval.parquet_source import fetch_task_rollouts_direct

log = get_logger("driver")


@dataclass(frozen=True)
class _Scored:
    rollout_id: str
    score: float
    advantage: float
    ce: float
    n_tokens: int


class DistillV2Evaluator:
    """Stateless per-cell evaluator."""

    async def evaluate(
        self,
        *,
        student_name: str,
        revision: str,
        hf_repo: str,
        arch: str,
        base_url: str,
        served_model_name: str,
        dataset_base_url: str,
        task_idx: int,
        scoring_name: str = "softmax_advantage",
        dtype: str = "bfloat16",
        max_seq_len: int = 32768,
        mask_reasoning: bool = True,
    ) -> dict[str, Any]:
        rollouts = await asyncio.to_thread(
            fetch_task_rollouts_direct, dataset_base_url, task_idx,
        )
        if not rollouts:
            log.warning("driver.no_rollouts", task_idx=task_idx)
            return _empty_snapshot(task_idx)
        cell = _cell_info(task_idx, rollouts)

        student = StudentSubmission(
            student_name=StudentName(student_name),
            revision=Revision(revision),
            hf_repo=hf_repo,
            arch=arch,
            model_hash=f"distill-v2:{student_name[:8]}:{revision[:8]}",
            submitted_by="affinetes:distill-v2",
        )

        engine = VLLMLogprobEngine(
            base_url=base_url, model_name=served_model_name,
        )
        # vllm_logprob.load is sync (transformers tokenizer fetch +
        # httpx.Client); offload so the actor's event loop stays free.
        await asyncio.to_thread(
            engine.load, student, dtype=dtype, max_seq_len=max_seq_len,
        )
        try:
            renderer = build_renderer(
                student.arch, engine.tokenizer,  # type: ignore[attr-defined]
                cfg=RenderConfig(
                    mask_reasoning=mask_reasoning, max_seq_len=max_seq_len,
                ),
            )
            scoring = (
                build_cell_scoring(scoring_name)
                if is_cell_scoring(scoring_name)
                else build_rollout_scoring(scoring_name)
            )
            return await asyncio.to_thread(
                _score_cell,
                rollouts=rollouts,
                student=student,
                engine=engine,
                renderer=renderer,
                scoring=scoring,
                cell=cell,
            )
        finally:
            engine.unload()


def _cell_info(task_idx: int, rollouts) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Cell coordinates for the result snapshot, from the shard itself.

    Every row in a shard shares (env, task, teacher) by construction.
    reward_mean/std are recomputed from the rows — same data the publisher
    aggregated into the manifest, which we no longer download.
    """
    rewards = [float(r.reward or 0.0) for r in rollouts]
    return {
        "task_idx": task_idx,
        "env_name": str(rollouts[0].env_name),
        "task_id": int(rollouts[0].task_id),
        "teacher_name": str(rollouts[0].teacher_name),
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
    }


def _score_cell(
    *, rollouts, student, engine, renderer, scoring, cell: dict[str, Any],
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    # All rollouts in a cell share (env, task, teacher); both grouping
    # modes collapse to the same single z-score bucket. We pass
    # "task_teacher" to keep the contract explicit.
    try:
        advantages, _group_stats = compute_advantages(
            rollouts, grouping="task_teacher",
        )
    except Exception as exc:  # noqa: BLE001
        raise EvalError(f"advantage computation failed: {exc}") from exc

    # Single forward pass over the cell: collect CE + advantage per rollout.
    # Aggregation differs by scorer family (below) but the measurements are
    # the same, so we gather them once.
    measured: list[tuple[Any, float, int, float]] = []  # (rollout, ce, n_tokens, advantage)
    total_tokens = 0
    with engine.session():
        for rollout in rollouts:
            rendered = render_rollout(rollout, student, renderer=renderer)
            m = engine.compute_ce([rendered])[0]
            advantage = advantages.get(rollout.rollout_id, 0.0)
            measured.append((rollout, m.ce_sum, m.n_tokens, advantage))
            total_tokens += m.n_tokens

    if hasattr(scoring, "score_cell"):
        # Whole-cell scorer (softmax_advantage): softmax over the rollouts'
        # per-token NLL, then expected advantage. Bounded by construction.
        cell = scoring.score_cell([
            RolloutMeasurement(ce=ce, n_tokens=n, advantage=adv)
            for (_r, ce, n, adv) in measured
        ])
        mean_score, win_rate = cell.mean_score, cell.win_rate
        per_scores = cell.per_rollout_scores
    else:
        # Per-rollout scorer (reward_weighted_ce): score each rollout, then
        # take the |advantage|-weighted mean — baseline rollouts (adv≈0)
        # contribute nothing, high-|advantage| outliers carry the signal.
        per_scores = [
            scoring.score_rollout(
                ce=ce, n_tokens=n,
                reward=float(r.reward or 0.0),
                advantage=adv,
            )
            for (r, ce, n, adv) in measured
        ]
        weights = [abs(adv) for (_r, _ce, _n, adv) in measured]
        total_w = sum(weights)
        if total_w > 0:
            mean_score = sum(s * w for s, w in zip(per_scores, weights)) / total_w
            win_rate = sum(w for s, w in zip(per_scores, weights) if s > 0) / total_w
        else:
            mean_score = statistics.fmean(per_scores) if per_scores else 0.0
            win_rate = (
                sum(1 for s in per_scores if s > 0) / len(per_scores)
                if per_scores else 0.0
            )

    scored: list[_Scored] = [
        _Scored(
            rollout_id=str(r.rollout_id),
            score=s, advantage=adv,
            ce=ce, n_tokens=n,
        )
        for (r, ce, n, adv), s in zip(measured, per_scores)
    ]

    return {
        "mean_score": mean_score,
        "win_rate": win_rate,
        "n_rollouts": len(scored),
        "total_forward_tokens": total_tokens,
        "cell": cell,
        "per_rollout": [
            {
                "rollout_id": s.rollout_id,
                "score": s.score, "advantage": s.advantage,
                "ce": s.ce, "n_tokens": s.n_tokens,
            }
            for s in scored
        ],
    }


def _empty_snapshot(task_idx: int) -> dict[str, Any]:
    return {
        "mean_score": 0.0,
        "win_rate": 0.0,
        "n_rollouts": 0,
        "total_forward_tokens": 0,
        "cell": {
            "task_idx": task_idx,
            "env_name": None,
            "task_id": None,
            "teacher_name": None,
            "reward_mean": None,
            "reward_std": None,
        },
        "per_rollout": [],
    }
