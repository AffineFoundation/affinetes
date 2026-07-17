"""Structured execution outcomes shared by SWE-INFINITE agent adapters."""

from enum import Enum


class AgentOutcome(str, Enum):
    """Whether an agent execution produced a trustworthy scoring attempt."""

    COMPLETED = "completed"
    MODEL_FAILURE = "model_failure"
    INFRA_FAILURE = "infra_failure"

    @property
    def valid_for_scoring(self) -> bool:
        return self is not AgentOutcome.INFRA_FAILURE


def outcome_from_process_exit_code(exit_code: int) -> AgentOutcome:
    """Map the native process protocol to a scoring disposition."""
    if exit_code == 0:
        return AgentOutcome.COMPLETED
    return AgentOutcome.INFRA_FAILURE
