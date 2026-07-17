"""Tests for SWE-INFINITE's structured scoring disposition contract."""

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.miniswe import outcome_from_exception
from agents.affent import AffentAgent, AffentConfig
from agents.codex import CodexAgent, CodexConfig, CodexResult
from agents.outcome import (
    AgentOutcome,
    failure_kind_from_provider_error,
    outcome_from_affent_protocol,
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


def test_native_failure_kind_can_identify_model_context_exhaustion() -> None:
    for failure_kind in (
        "context_overflow",
        "context_window_exceeded",
        "contextWindowExceeded",
        "ContextWindowExceeded",
    ):
        assert outcome_from_process_exit_code(
            1,
            failure_kind=failure_kind,
        ) is AgentOutcome.MODEL_FAILURE

    for failure_kind in (
        None,
        "llm_timeout",
        "llm_incomplete_stream",
        "context window exceeded",
    ):
        assert outcome_from_process_exit_code(
            1,
            failure_kind=failure_kind,
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


def test_affent_trace_exposes_native_failure_kind() -> None:
    agent = AffentAgent(AffentConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    trace = json.dumps({
        "type": "error",
        "data": {
            "message": "provider rejected request",
            "failure_kind": "context_overflow",
            "recoverable": False,
        },
    })

    _, _, _, error, failure_kind, turn_end_reason = agent._parse_jsonl_trace(trace)

    assert error == "provider rejected request"
    assert failure_kind == "context_overflow"
    assert turn_end_reason is None


def test_affent_trace_recovers_nonrecoverable_provider_failure() -> None:
    agent = AffentAgent(AffentConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    provider_error = f"chat http 400: {_sglang_context_error()}"
    trace = json.dumps({
        "type": "error",
        "data": {
            "message": provider_error,
            "recoverable": False,
        },
    })

    _, _, _, error, failure_kind, _ = agent._parse_jsonl_trace(trace)

    assert error == provider_error
    assert failure_kind == "context_overflow"


def test_affent_recoverable_provider_failure_is_not_terminal() -> None:
    agent = AffentAgent(AffentConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    trace = json.dumps({
        "type": "error",
        "data": {
            "message": f"chat http 400: {_sglang_context_error()}",
            "recoverable": True,
        },
    })

    _, _, _, _, failure_kind, _ = agent._parse_jsonl_trace(trace)

    assert failure_kind is None


def test_affent_protocol_accepts_old_and_new_action_limit_exit_codes() -> None:
    assert outcome_from_affent_protocol(
        0,
        turn_end_reason="max_turns",
    ) is AgentOutcome.COMPLETED
    assert outcome_from_affent_protocol(
        2,
        turn_end_reason="max_turns",
    ) is AgentOutcome.COMPLETED
    assert outcome_from_affent_protocol(
        2,
        turn_end_reason="length",
    ) is AgentOutcome.COMPLETED


def test_affent_protocol_rejects_unstructured_or_contradictory_success() -> None:
    for exit_code, reason in (
        (0, None),
        (2, None),
        (2, "completed"),
        (2, "error"),
        (0, "error"),
        (3, "max_turns"),
        (0, "future_reason"),
    ):
        assert outcome_from_affent_protocol(
            exit_code,
            turn_end_reason=reason,
        ) is AgentOutcome.INFRA_FAILURE


def test_affent_protocol_uses_structured_model_failure_only_at_error_end() -> None:
    assert outcome_from_affent_protocol(
        3,
        turn_end_reason="error",
        failure_kind="context_overflow",
    ) is AgentOutcome.MODEL_FAILURE
    assert outcome_from_affent_protocol(
        3,
        turn_end_reason=None,
        failure_kind="context_overflow",
    ) is AgentOutcome.INFRA_FAILURE


def test_affent_trace_exposes_terminal_reason() -> None:
    agent = AffentAgent(AffentConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    trace = "\n".join((
        json.dumps({"type": "turn.start", "data": {}}),
        json.dumps({"type": "turn.end", "data": {"reason": "max_turns"}}),
    ))

    _, _, _, _, _, turn_end_reason = agent._parse_jsonl_trace(trace)

    assert turn_end_reason == "max_turns"


def test_codex_parser_uses_native_error_enum_when_available() -> None:
    agent = CodexAgent(CodexConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    output = json.dumps({
        "method": "turn/completed",
        "params": {
            "turn": {
                "status": "failed",
                "error": {
                    "message": "request rejected",
                    "codexErrorInfo": "contextWindowExceeded",
                },
            },
        },
    })

    _, _, _, error, failure_kind = agent._parse_json_output(output)

    assert error == "request rejected"
    assert failure_kind == "contextWindowExceeded"


def test_codex_legacy_jsonl_preserves_structured_provider_failure() -> None:
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


def test_codex_parser_recovers_sglang_context_kind_from_error_json() -> None:
    agent = CodexAgent(CodexConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    provider_error = json.dumps({
        "object": "error",
        "message": (
            "Input length (57719 tokens) exceeds the maximum allowed length "
            "(55539 tokens). Use a shorter input or enable --allow-auto-truncate."
        ),
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    })
    output = json.dumps({
        "method": "turn/completed",
        "params": {
            "turn": {
                "status": "failed",
                "error": {
                    "message": provider_error,
                    "codexErrorInfo": "other",
                },
            },
        },
    })

    _, _, _, _, failure_kind = agent._parse_json_output(output)

    assert failure_kind == "context_overflow"
    assert outcome_from_process_exit_code(
        1,
        failure_kind=failure_kind,
    ) is AgentOutcome.MODEL_FAILURE


def test_codex_parser_keeps_other_bad_requests_retryable() -> None:
    agent = CodexAgent(CodexConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    provider_error = json.dumps({
        "object": "error",
        "message": "developer role is not supported",
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    })
    output = json.dumps({
        "method": "turn/completed",
        "params": {
            "turn": {
                "status": "failed",
                "error": {
                    "message": provider_error,
                    "codexErrorInfo": "other",
                },
            },
        },
    })

    _, _, _, _, failure_kind = agent._parse_json_output(output)

    assert failure_kind == "other"
    assert outcome_from_process_exit_code(
        1,
        failure_kind=failure_kind,
    ) is AgentOutcome.INFRA_FAILURE


def test_codex_app_server_run_preserves_terminal_failure_kind() -> None:
    agent = CodexAgent(CodexConfig(
        model="model",
        api_base="http://inference/v1",
        api_key="key",
    ))
    agent._container_name = "task-container"
    fake_server = textwrap.dedent("""
        import json
        import sys

        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            request_id = request.get("id")
            if request_id == 0:
                assert method == "initialize"
                print(json.dumps({"id": 0, "result": {"userAgent": "fake"}}), flush=True)
            elif method == "initialized":
                continue
            elif request_id == 1:
                assert request["params"]["approvalPolicy"] == "never"
                assert request["params"]["sandbox"] == "danger-full-access"
                response = {
                    "id": 1,
                    "result": {"thread": {"id": "thread-1"}},
                }
                print(json.dumps(response), flush=True)
            elif request_id == 2:
                assert request["params"]["threadId"] == "thread-1"
                assert request["params"]["sandboxPolicy"] == {
                    "type": "dangerFullAccess",
                }
                print(json.dumps({
                    "id": 2,
                    "result": {"turn": {"status": "inProgress"}},
                }), flush=True)
                print(json.dumps({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {
                            "status": "failed",
                            "error": {
                                "message": "request rejected",
                                "codexErrorInfo": "contextWindowExceeded",
                            },
                        },
                    },
                }), flush=True)
    """)
    real_popen = subprocess.Popen

    def start_fake_server(_command, **kwargs):
        return real_popen(
            [sys.executable, "-u", "-c", fake_server],
            **kwargs,
        )

    with patch("agents.codex.subprocess.Popen", side_effect=start_fake_server):
        completed = agent._run_codex_app_server("fix it", timeout=3)

    _, _, _, error, failure_kind = agent._parse_json_output(completed.stdout)
    assert completed.returncode == 1
    assert error == "request rejected"
    assert failure_kind == "contextWindowExceeded"


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
                failure_kind="context_window_exceeded",
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
    assert result["extra"]["agent_failure_kind"] == "context_window_exceeded"
