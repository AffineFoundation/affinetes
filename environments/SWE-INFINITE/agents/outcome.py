"""Structured execution outcomes shared by SWE-INFINITE agent adapters."""

from enum import Enum
from typing import Optional

from .affent_protocol import (
    AffentDisposition,
    classify_affent_run,
    failure_kind_from_provider_error as _failure_kind_from_provider_error,
)


_MODEL_FAILURE_KINDS = frozenset({
    "context_overflow",
    "context_window_exceeded",
    "contextWindowExceeded",
    "ContextWindowExceeded",
})


class AgentOutcome(str, Enum):
    """Whether an agent execution produced a trustworthy scoring attempt."""

    COMPLETED = "completed"
    MODEL_FAILURE = "model_failure"
    INFRA_FAILURE = "infra_failure"

    @property
    def valid_for_scoring(self) -> bool:
        return self is not AgentOutcome.INFRA_FAILURE


def failure_kind_from_provider_error(message: object) -> Optional[str]:
    """Expose the shared provider error contract to all SWE agents."""
    return _failure_kind_from_provider_error(message)


def outcome_from_process_exit_code(
    exit_code: int,
    *,
    failure_kind: Optional[str] = None,
) -> AgentOutcome:
    """Map an exit code plus a structured failure kind to a disposition."""
    if exit_code == 0:
        return AgentOutcome.COMPLETED
    if failure_kind in _MODEL_FAILURE_KINDS:
        return AgentOutcome.MODEL_FAILURE
    return AgentOutcome.INFRA_FAILURE


def outcome_from_affent_protocol(
    exit_code: int,
    *,
    turn_end_reason: Optional[str],
    failure_kind: Optional[str] = None,
) -> AgentOutcome:
    """Map affentctl's versioned exit/trace contract to a disposition."""
    disposition = classify_affent_run(
        exit_code,
        turn_end_reason=turn_end_reason,
        failure_kind=failure_kind,
    )
    if disposition is AffentDisposition.GRADABLE:
        return AgentOutcome.COMPLETED
    if disposition is AffentDisposition.MODEL_FAILURE:
        return AgentOutcome.MODEL_FAILURE
    return AgentOutcome.INFRA_FAILURE
