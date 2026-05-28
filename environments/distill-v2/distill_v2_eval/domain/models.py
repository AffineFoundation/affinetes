"""Domain models.

Pydantic v2 models — frozen by default so they behave like value objects. Field
names mirror the database schema for low cognitive overhead; small helpers
(serialize_for_db, from_db_row) live in the SQLAlchemy adapter, not here.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from distill_v2_eval.domain.ids import (
    EnvName,
    PairId,
    PairSetName,
    Revision,
    RolloutId,
    RunId,
    SamplingListName,
    StudentName,
    TaskId,
    TeacherName,
)

_FROZEN = ConfigDict(frozen=True, extra="forbid")
_MUT = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Trajectory normalized form (stored zstd-compressed in rollouts.extra_*)
# --------------------------------------------------------------------------- #

class ToolCall(BaseModel):
    model_config = _FROZEN
    id: str
    name: str
    arguments: dict[str, Any]
    malformed: bool = False


class NormalizedMessage(BaseModel):
    """Single message in the normalized trajectory.

    role:
        - system / user / tool: ``content`` is text.
        - assistant: may carry ``content`` (text reply), ``reasoning`` (thinking
          trace, optionally masked at eval time), and/or ``tool_calls``.
    """

    model_config = _FROZEN
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # for role='tool'


class NormalizedTrajectory(BaseModel):
    """The canonical, teacher-agnostic form of a rollout.

    Stored as zstd(JSON(...)) in ``rollouts.extra_compressed``. Students
    re-tokenize from this object via their own chat template.
    """

    model_config = _FROZEN
    schema_version: str
    messages: list[NormalizedMessage]
    reward_breakdown: dict[str, Any] = Field(default_factory=dict)
    agent_trace: list[dict[str, Any]] | None = None
    teacher_meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core entities (1:1 with tables)
# --------------------------------------------------------------------------- #

class Environment(BaseModel):
    model_config = _FROZEN
    env_name: EnvName
    dataset: str
    dataset_version: str
    task_id_min: int
    task_id_max: int
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class Task(BaseModel):
    model_config = _FROZEN
    env_name: EnvName
    task_id: TaskId
    repo: str
    base_commit: str
    problem: str
    hidden_tests: dict[str, Any]
    difficulty: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Teacher(BaseModel):
    model_config = _FROZEN
    teacher_name: TeacherName
    model_family: str
    provider: str
    endpoint: str
    api_key_env: str
    tool_format: Literal["xml", "openai_json", "hermes", "glm_function"]
    reasoning_format: Literal["none", "thinking_tag", "think_tag", "reasoning_field"]
    context_window: int
    price_per_mtoken_in: Decimal | None = None
    price_per_mtoken_out: Decimal | None = None
    active: bool = True
    meta: dict[str, Any] = Field(default_factory=dict)


class RolloutStatus(str, Enum):
    ok = "ok"
    parse_failed = "parse_failed"
    truncated = "truncated"
    env_error = "env_error"
    teacher_error = "teacher_error"


class GroupLabel(str, Enum):
    paired = "paired"
    all_win = "all_win"
    all_lose = "all_lose"
    degenerate = "degenerate"
    singleton = "singleton"


class Rollout(BaseModel):
    """One agent loop episode for ``(env, task, teacher, sample_idx)``.

    Mirrors the cortex ``sample_results`` row layout: business key first,
    sampling params next, compressed body last. ``extra_compressed`` is bytes
    here so the model stays in the domain layer; SQLAlchemy adapter handles
    the round-trip transparently.
    """

    model_config = _MUT  # group_label is back-filled by miner
    rollout_id: RolloutId
    env_name: EnvName
    task_id: TaskId
    teacher_name: TeacherName
    sample_idx: int

    temperature: float
    top_p: float | None = None
    seed: int | None = None

    status: RolloutStatus
    reward: float | None = None
    reward_breakdown: dict[str, Any] | None = None
    steps: int | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: Decimal | None = None

    schema_version: str
    extra_compressed: bytes  # zstd(JSON(NormalizedTrajectory.model_dump()))
    extra_sha256: str
    blob_uri: str | None = None

    group_label: GroupLabel | None = None
    producer_id: str
    trace_id: str | None = None
    created_at: datetime | None = None


class PairCandidate(BaseModel):
    """Selected by a PairStrategy, not yet persisted."""

    model_config = _FROZEN
    env_name: EnvName
    task_id: TaskId
    teacher_name: TeacherName
    win_rollout: RolloutId
    lose_rollout: RolloutId
    reward_win: float
    reward_lose: float
    reward_gap: float
    selection_meta: dict[str, Any] = Field(default_factory=dict)


class Pair(PairCandidate):
    """Persisted pair."""

    pair_id: PairId
    pair_set_name: PairSetName


class PairSetStatus(str, Enum):
    building = "building"
    ready = "ready"
    archived = "archived"


class PairSet(BaseModel):
    model_config = _MUT
    pair_set_name: PairSetName
    source_filter: dict[str, Any]
    strategy: dict[str, Any]
    rollouts_scanned: int | None = None
    pairs_created: int | None = None
    groups_summary: dict[str, Any] | None = None
    status: PairSetStatus
    triggered_by: str
    started_at: datetime
    finished_at: datetime | None = None
    trace_id: str | None = None


class SamplingList(BaseModel):
    model_config = _FROZEN
    list_name: SamplingListName
    config: dict[str, Any]
    description: str | None = None
    created_by: str
    created_at: datetime | None = None


class SamplingProgress(BaseModel):
    model_config = _FROZEN
    list_name: SamplingListName
    env_name: EnvName
    task_id: TaskId
    teacher_name: TeacherName
    target_samples: int
    collected: int


class StudentSubmission(BaseModel):
    model_config = _MUT
    student_name: StudentName
    revision: Revision
    hf_repo: str | None = None
    arch: str
    param_b: float | None = None
    model_hash: str
    template_check: str | None = None
    is_valid: bool = True
    invalid_reason: str | None = None
    submitted_by: str
    submitted_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class AntiCopyResult(BaseModel):
    model_config = _FROZEN
    student_name: StudentName
    revision: Revision
    round_ts: datetime
    is_copy: bool
    copied_from: list[dict[str, Any]] | None = None
    detection_meta: dict[str, Any] | None = None


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class StudentScore(BaseModel):
    """A student's full snapshot for one ``(run, pair_set)``.

    Mirrors cortex ``scores`` row layout: overall scalars + nested breakdowns
    + run config snapshot + a ``latest_marker`` for partial-unique latest lookup.
    """

    model_config = _MUT
    run_id: RunId
    student_name: StudentName
    revision: Revision
    pair_set_name: PairSetName
    mean_score: float
    win_rate: float
    scores_by_env: dict[str, Any]
    scores_by_teacher: dict[str, Any]
    scores_by_difficulty: dict[str, Any] | None = None
    pairs_evaluated: int
    config: dict[str, Any]
    forward_engine: str
    total_forward_tokens: int | None = None
    total_compute_seconds: float | None = None
    status: RunStatus
    latest_marker: str | None = None
    triggered_by: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    trace_id: str | None = None


class PairCEResult(BaseModel):
    """Per-pair CE measurement, intermediate output of the forward engine."""

    model_config = _FROZEN
    pair_id: PairId
    ce_win: float
    ce_lose: float
    tokens_win: int
    tokens_lose: int
    ce_per_message_win: list[float] | None = None
    ce_per_message_lose: list[float] | None = None


class DeploymentStatus(str, Enum):
    deploying = "deploying"
    active = "active"
    draining = "draining"
    failed = "failed"


class StudentDeployment(BaseModel):
    model_config = _MUT
    deployment_id: str
    student_name: StudentName
    revision: Revision
    engine: str
    host: str
    gpu_indices: list[int]
    base_url: str | None = None
    instance_count: int = 0
    status: DeploymentStatus
    consecutive_failures: int = 0
    next_retry_at: datetime | None = None
    last_health_check_at: datetime | None = None
    meta: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# --------------------------------------------------------------------------- #
# Helper request / value objects
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    """Inputs to a single teacher chat call.

    Carries normalized message form so per-provider adapters serialize to
    their own native shape.
    """

    model_config = _FROZEN
    messages: list[NormalizedMessage]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    extra_provider_args: dict[str, Any] = Field(default_factory=dict)


class RawAssistantMessage(BaseModel):
    """Provider-native assistant response. Not normalized yet."""

    model_config = _FROZEN
    teacher_name: TeacherName
    raw: dict[str, Any]                   # entire provider payload
    text: str | None = None               # convenience
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int


class RawTrajectory(BaseModel):
    """Raw, teacher-specific agent loop trace. Input to ``TrajectoryNormalizer``."""

    model_config = _FROZEN
    teacher_name: TeacherName
    family: str
    steps: list[dict[str, Any]]
    reward_breakdown: dict[str, Any]
    agent_meta: dict[str, Any] = Field(default_factory=dict)


class RenderedSample(BaseModel):
    """Rolled-out, tokenized form ready for student forward.

    ``input_ids`` / ``loss_mask`` are kept as plain lists at the domain
    boundary; the adapter converts to torch tensors inside the engine.
    """

    model_config = _FROZEN
    rollout_id: RolloutId
    input_ids: list[int]
    loss_mask: list[int]
    message_spans: list[tuple[int, int]]  # per assistant message (start, end) in input_ids
    meta: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AntiCopyResult",
    "ChatRequest",
    "DeploymentStatus",
    "Environment",
    "GroupLabel",
    "NormalizedMessage",
    "NormalizedTrajectory",
    "Pair",
    "PairCandidate",
    "PairCEResult",
    "PairSet",
    "PairSetStatus",
    "RawAssistantMessage",
    "RawTrajectory",
    "RenderedSample",
    "Rollout",
    "RolloutStatus",
    "RunStatus",
    "SamplingList",
    "SamplingProgress",
    "StudentDeployment",
    "StudentScore",
    "StudentSubmission",
    "Task",
    "Teacher",
    "ToolCall",
]


def utc_now() -> datetime:
    """Tiny convenience to avoid importing ``datetime`` everywhere."""
    from datetime import timezone

    return datetime.now(timezone.utc)
