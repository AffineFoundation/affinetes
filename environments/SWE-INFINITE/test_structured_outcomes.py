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
from agents.codex import CodexResult
from agents.outcome import AgentOutcome, outcome_from_process_exit_code
from env import InfiniteActor
from utils import is_container_running


def test_process_exit_code_controls_disposition_without_stderr() -> None:
    assert outcome_from_process_exit_code(0) is AgentOutcome.COMPLETED
    for exit_code in (1, 2, 3, 17, 130):
        assert outcome_from_process_exit_code(exit_code) is AgentOutcome.INFRA_FAILURE


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
