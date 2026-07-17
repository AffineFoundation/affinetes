"""Tests for SWE-INFINITE's structured scoring disposition contract."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.miniswe import outcome_from_exception
from agents.affent import (
    AffentAgent,
    AffentConfig,
    _is_gradable_turn_budget_exit,
)
from agents.codex import CodexAgent, CodexConfig, CodexResult
from agents.outcome import (
    AgentOutcome,
    failure_kind_from_provider_error,
    outcome_from_process_exit_code,
)
from env import InfiniteActor
from utils import is_container_running


def test_process_exit_code_controls_disposition_without_stderr() -> None:
    assert outcome_from_process_exit_code(0) is AgentOutcome.COMPLETED
    for exit_code in (1, 2, 3, 17, 130):
        assert outcome_from_process_exit_code(exit_code) is AgentOutcome.INFRA_FAILURE


def _sglang_context_error(*, input_tokens: int = 57719) -> str:
    return json.dumps({
        "object": "error",
        "message": (
            f"Input length ({input_tokens} tokens) exceeds the maximum allowed "
            "length (55539 tokens). Use a shorter input or enable "
            "--allow-auto-truncate."
        ),
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    })


def test_structured_context_failure_is_a_gradable_model_failure() -> None:
    assert outcome_from_process_exit_code(
        1,
        failure_kind="context_overflow",
    ) is AgentOutcome.MODEL_FAILURE
    assert outcome_from_process_exit_code(
        1,
        failure_kind="llm_timeout",
    ) is AgentOutcome.INFRA_FAILURE


def test_provider_context_error_uses_exact_json_contract() -> None:
    provider_error = _sglang_context_error()
    assert failure_kind_from_provider_error(provider_error) == "context_overflow"
    assert failure_kind_from_provider_error(
        f"chat http 400: {provider_error}"
    ) == "context_overflow"
    assert failure_kind_from_provider_error(json.dumps({
        "error": {
            "message": "request too large",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        },
    })) == "context_overflow"


def test_unstructured_or_unrelated_bad_requests_remain_infra_failures() -> None:
    assert failure_kind_from_provider_error(
        "Input length exceeds the maximum allowed length"
    ) is None
    assert failure_kind_from_provider_error(json.dumps({
        "object": "error",
        "message": "developer role is not supported",
        "type": "BadRequestError",
        "code": 400,
    })) is None
    assert failure_kind_from_provider_error(
        _sglang_context_error(input_tokens=100)
    ) is None


def test_codex_jsonl_preserves_structured_provider_failure() -> None:
    agent = CodexAgent(CodexConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    provider_error = _sglang_context_error()
    output = "\n".join((
        json.dumps({"type": "error", "message": provider_error}),
        json.dumps({
            "type": "turn.failed",
            "error": {"message": provider_error},
        }),
    ))

    _, _, _, error, failure_kind = agent._parse_json_output(output)

    assert error == provider_error
    assert failure_kind == "context_overflow"


def test_affent_trace_preserves_structured_provider_failure() -> None:
    agent = AffentAgent(AffentConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    provider_error = f"chat http 400: {_sglang_context_error()}"
    trace = json.dumps({
        "type": "error",
        "data": {
            "code": "llm_request",
            "message": provider_error,
            "recoverable": False,
        },
    })

    _, _, _, error, failure_kind, _ = agent._parse_jsonl_trace(trace)

    assert error == provider_error
    assert failure_kind == "context_overflow"


def test_affent_recoverable_context_error_is_not_terminal_model_failure() -> None:
    agent = AffentAgent(AffentConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    trace = json.dumps({
        "type": "error",
        "data": {
            "code": "llm_request",
            "message": f"chat http 400: {_sglang_context_error()}",
            "recoverable": True,
        },
    })

    _, _, _, _, failure_kind, _ = agent._parse_jsonl_trace(trace)

    assert failure_kind is None


def test_affent_turn_budget_exhaustion_is_gradable_without_retry() -> None:
    agent = AffentAgent(AffentConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    for reason in ("max_turns", "length"):
        trace = json.dumps({
            "type": "turn.end",
            "data": {"reason": reason},
        })

        _, _, _, _, _, turn_end_reason = agent._parse_jsonl_trace(trace)

        assert turn_end_reason == reason
        assert _is_gradable_turn_budget_exit(2, turn_end_reason) is True


def test_affent_exit_two_without_budget_contract_remains_infra() -> None:
    assert _is_gradable_turn_budget_exit(2, None) is False
    assert _is_gradable_turn_budget_exit(2, "error") is False
    assert outcome_from_process_exit_code(2) is AgentOutcome.INFRA_FAILURE


def test_miniswe_uses_exception_types_not_messages() -> None:
    model_error = litellm.exceptions.ContextWindowExceededError(
        message="arbitrary diagnostic",
        model="test",
        llm_provider="openai",
    )
    infra_error = ConnectionError("looks like a context length error")
    untyped_bad_request = litellm.exceptions.BadRequestError(
        message="input exceeds the maximum context length",
        model="test",
        llm_provider="openai",
    )

    assert outcome_from_exception(model_error) is AgentOutcome.MODEL_FAILURE
    assert outcome_from_exception(infra_error) is AgentOutcome.INFRA_FAILURE
    assert outcome_from_exception(untyped_bad_request) is AgentOutcome.INFRA_FAILURE


def test_container_state_is_read_from_docker_inspect_json() -> None:
    state = {
        "Running": True,
        "Paused": False,
        "Restarting": False,
        "Dead": False,
    }
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps([{"State": state}]),
    )

    with patch.object(subprocess, "run", return_value=completed):
        assert is_container_running("task-container") is True


def test_container_state_fails_closed_on_invalid_inspect_payload() -> None:
    completed = SimpleNamespace(returncode=0, stdout="not-json")

    with patch.object(subprocess, "run", return_value=completed):
        assert is_container_running("task-container") is False


def test_invalid_agent_execution_cannot_score_a_partial_patch() -> None:
    actor = InfiniteActor.__new__(InfiniteActor)
    actor.api_key = "test-key"
    actor._load_task = lambda _task_id: {
        "instance_id": "task-1",
        "dockerhub_tag": "example/task:latest",
        "problem_statement": "fix it",
    }
    actor._verify = lambda *_args: (_ for _ in ()).throw(
        AssertionError("invalid execution must not be verified")
    )
    actor._cleanup_docker_resources = lambda **_kwargs: None

    class FakeAgent:
        async def solve(self, **_kwargs):
            return CodexResult(
                patch="diff --git a/a b/a\n",
                error="diagnostic text that looks like a model error",
                outcome=AgentOutcome.INFRA_FAILURE,
                process_exit_code=1,
            )

        def cleanup(self):
            pass

    with patch("env.CodexAgent", return_value=FakeAgent()):
        result = asyncio.run(
            actor.evaluate(
                task_id=1,
                model="model",
                base_url="http://inference/v1",
                agent="codex",
            )
        )

    assert result["score"] == 0.0
    assert result["error"] == "agent_execution_invalid"
    assert result["extra"]["valid_for_scoring"] is False
    assert result["extra"]["agent_process_exit_code"] == 1


def test_model_failure_scores_zero_without_verifying_partial_patch() -> None:
    actor = InfiniteActor.__new__(InfiniteActor)
    actor.api_key = "test-key"
    actor._load_task = lambda _task_id: {
        "instance_id": "task-1",
        "dockerhub_tag": "example/task:latest",
        "problem_statement": "fix it",
    }
    actor._verify = lambda *_args: (_ for _ in ()).throw(
        AssertionError("partial patch from a failed process must not be verified")
    )
    actor._cleanup_docker_resources = lambda **_kwargs: None

    class FakeAgent:
        async def solve(self, **_kwargs):
            return CodexResult(
                patch="diff --git a/a b/a\n",
                error="request exceeded context window",
                outcome=AgentOutcome.MODEL_FAILURE,
                process_exit_code=1,
                failure_kind="context_overflow",
            )

        def cleanup(self):
            pass

    with patch("env.CodexAgent", return_value=FakeAgent()):
        result = asyncio.run(
            actor.evaluate(
                task_id=1,
                model="model",
                base_url="http://inference/v1",
                agent="codex",
            )
        )

    assert result["score"] == 0.0
    assert result["error"] is None
    assert result["extra"]["valid_for_scoring"] is True
    assert result["extra"]["agent_outcome"] == "model_failure"
    assert result["extra"]["agent_failure_kind"] == "context_overflow"
