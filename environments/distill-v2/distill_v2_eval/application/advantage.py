"""Advantage computation for rollout-level scoring.

GRPO-style z-score of reward within a *group*. Two grouping modes:

  * ``"task"`` (default): one group per task; baseline is computed across
    every teacher's rollouts on that task. This is what we use by default
    — reward is verifiable (pytest pass rate) so cross-teacher pooling
    boosts signal without much bias, and a strong claude trajectory
    legitimately deserves higher advantage than a weak minimax one.

  * ``"task_teacher"``: one group per ``(task, teacher)``. Baseline is
    per-teacher, so the score reflects "this rollout vs same teacher's
    other attempts" — robust to systematic per-teacher reward shifts
    (e.g. reward-model bias, or one teacher's typical reward floor
    differing from another's). At the cost of single-cell groups with
    only K=4 samples for std estimation.

Why subtract a group baseline at all
------------------------------------

Rewards have wildly different scales across tasks. Easy tasks sit at
reward≈1, hard tasks sit at ≈0. Subtracting the per-group mean is what
makes a per-rollout score informative — without it the easy tasks
dominate the signal regardless of model quality.

Why divide by std
-----------------

Groups where every rollout scored the same (std≈0) carry no information
— we zero out their advantage so they don't pollute the aggregate.
Groups with high variance get full weight per unit of reward delta.

Degenerate cases handled
------------------------

  * ``len(group) < 2``     → advantage = 0 (no within-group contrast).
  * ``std == 0``           → advantage = 0 (everyone scored the same).
  * ``reward is None``     → rollout dropped before grouping.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, NamedTuple

from distill_v2_eval.domain.ids import RolloutId, TaskId, TeacherName
from distill_v2_eval.domain.models import Rollout


AdvantageGrouping = Literal["task", "task_teacher"]


class GroupAdvantageStats(NamedTuple):
    """Per-group summary, persisted under ``StudentScore.config["advantage_groups"]``
    for traceability. ``group_key`` is the JSON-friendly string we use as the
    dict key (``str(task_id)`` for task grouping; ``"task:teacher"`` for
    per-teacher grouping)."""

    group_key: str
    task_id: TaskId
    teacher_name: TeacherName | None     # None when grouping by task only
    n: int
    mean_reward: float
    std_reward: float
    nonzero: bool        # whether the group contributed any signal


def compute_advantages(
    rollouts: Iterable[Rollout],
    *,
    grouping: AdvantageGrouping = "task",
) -> tuple[dict[RolloutId, float], dict[str, GroupAdvantageStats]]:
    """Return ``({rollout_id: advantage}, {group_key: stats})``.

    ``grouping`` decides what counts as a peer group for the z-score; see
    module docstring for the semantics of the two modes.
    """
    if grouping not in ("task", "task_teacher"):
        raise ValueError(
            f"unknown advantage grouping: {grouping!r}; expected 'task' or 'task_teacher'"
        )

    buckets: dict[tuple, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        if r.reward is None:
            continue
        key = _bucket_key(r, grouping)
        buckets[key].append(r)

    advantages: dict[RolloutId, float] = {}
    stats: dict[str, GroupAdvantageStats] = {}

    for key, group in buckets.items():
        rewards = [float(r.reward) for r in group if r.reward is not None]
        n = len(rewards)
        if n < 2:
            mean_r = rewards[0] if rewards else 0.0
            std_r = 0.0
            nonzero = False
            for r in group:
                advantages[r.rollout_id] = 0.0
        else:
            mean_r = statistics.fmean(rewards)
            std_r = statistics.pstdev(rewards)
            if std_r == 0:
                nonzero = False
                for r in group:
                    advantages[r.rollout_id] = 0.0
            else:
                nonzero = True
                for r in group:
                    advantages[r.rollout_id] = (float(r.reward) - mean_r) / std_r  # type: ignore[arg-type]
        stats[_group_key_string(key, grouping)] = GroupAdvantageStats(
            group_key=_group_key_string(key, grouping),
            task_id=group[0].task_id,
            teacher_name=(group[0].teacher_name if grouping == "task_teacher" else None),
            n=n,
            mean_reward=mean_r,
            std_reward=std_r,
            nonzero=nonzero,
        )

    return advantages, stats


# --------------------------------------------------------------------------- #


def _bucket_key(rollout: Rollout, grouping: AdvantageGrouping) -> tuple:
    if grouping == "task":
        return (rollout.task_id,)
    return (rollout.task_id, rollout.teacher_name)


def _group_key_string(key: tuple, grouping: AdvantageGrouping) -> str:
    if grouping == "task":
        return str(int(key[0]))
    return f"{int(key[0])}:{key[1]}"


# Back-compat alias — older modules may still import this name.
TaskAdvantageStats = GroupAdvantageStats
