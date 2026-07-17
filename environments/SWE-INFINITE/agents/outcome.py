"""Structured execution outcomes shared by SWE-INFINITE agent adapters."""

import json
import re
from enum import Enum
from typing import Optional


_MODEL_FAILURE_KINDS = frozenset({
    "context_overflow",
    "context_window_exceeded",
    "contextWindowExceeded",
    "ContextWindowExceeded",
})
_CONTEXT_ERROR_CODES = frozenset({
    "context_length_exceeded",
    "context_window_exceeded",
})
_AFFENT_HTTP_ERROR = re.compile(
    r"chat http (?P<status>\d{3}): (?P<body>\{.*\})",
    re.DOTALL,
)
_SGLANG_CONTEXT_ERROR = re.compile(
    r"Input length \((?P<input_tokens>\d+) tokens\) exceeds the maximum "
    r"allowed length \((?P<max_tokens>\d+) tokens\)\. Use a shorter input "
    r"or enable --allow-auto-truncate\."
)


class AgentOutcome(str, Enum):
    """Whether an agent execution produced a trustworthy scoring attempt."""

    COMPLETED = "completed"
    MODEL_FAILURE = "model_failure"
    INFRA_FAILURE = "infra_failure"

    @property
    def valid_for_scoring(self) -> bool:
        return self is not AgentOutcome.INFRA_FAILURE


def failure_kind_from_provider_error(message: object) -> Optional[str]:
    """Classify only machine-readable provider error contracts."""
    if not isinstance(message, str):
        return None

    raw_body = message.strip()
    http_status: Optional[int] = None
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        match = _AFFENT_HTTP_ERROR.fullmatch(raw_body)
        if match is None:
            return None
        http_status = int(match.group("status"))
        try:
            body = json.loads(match.group("body"))
        except json.JSONDecodeError:
            return None

    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None

    code = error.get("code")
    if code in _CONTEXT_ERROR_CODES:
        return "context_overflow"

    # SGLang currently uses a generic 400 code for context overflow. Require
    # its complete JSON contract and numeric invariant; other 400s remain
    # infrastructure failures and are retried.
    if http_status is None and code in (400, "400"):
        http_status = 400
    if http_status != 400 or error.get("type") != "BadRequestError":
        return None
    detail = error.get("message")
    if not isinstance(detail, str):
        return None
    match = _SGLANG_CONTEXT_ERROR.fullmatch(detail)
    if match is None:
        return None
    if int(match.group("input_tokens")) <= int(match.group("max_tokens")):
        return None
    return "context_overflow"


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
