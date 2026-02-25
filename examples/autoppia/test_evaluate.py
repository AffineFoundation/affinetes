"""Tests for the Autoppia evaluation script (pure logic, no Docker)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure we import the local evaluate module from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    build_env_vars,
    build_eval_kwargs,
    get_output_dir,
    parse_args,
    print_result,
    save_result,
    validate_env,
)


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.image == "autoppia-affine-env:latest"
    assert args.model == "test-model"
    assert args.task_id is None
    assert args.max_steps == 30
    assert args.timeout == 600
    assert args.output_dir is None
    assert getattr(args, "require_chutes", False) is False


def test_parse_args_overrides() -> None:
    args = parse_args(
        [
            "--task-id",
            "task-1",
            "--max-steps",
            "50",
            "--output-dir",
            "/tmp/out",
            "--require-chutes",
        ]
    )
    assert args.task_id == "task-1"
    assert args.max_steps == 50
    assert args.output_dir == "/tmp/out"
    assert args.require_chutes is True


def test_build_eval_kwargs_all_tasks() -> None:
    args = parse_args([])
    kwargs = build_eval_kwargs(args)
    assert kwargs["model"] == "test-model"
    assert "base_url" in kwargs
    assert kwargs["max_steps"] == 30
    assert "task_id" not in kwargs


def test_build_eval_kwargs_single_task() -> None:
    args = parse_args(["--task-id", "autobooks-demo-task-1"])
    kwargs = build_eval_kwargs(args)
    assert kwargs["task_id"] == "autobooks-demo-task-1"


def test_build_env_vars_empty_without_chutes() -> None:
    with patch.dict("os.environ", {}, clear=True):
        env_vars = build_env_vars()
    assert env_vars == {}


def test_build_env_vars_includes_chutes() -> None:
    with patch.dict("os.environ", {"CHUTES_API_KEY": "secret"}, clear=False):
        env_vars = build_env_vars()
    assert env_vars.get("CHUTES_API_KEY") == "secret"


def test_validate_env_exits_when_chutes_missing_and_required() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            validate_env(require_chutes=True)
    assert exc_info.value.code == 1


def test_validate_env_passes_when_chutes_set() -> None:
    with patch.dict("os.environ", {"CHUTES_API_KEY": "x"}, clear=False):
        validate_env(require_chutes=True)


def test_validate_env_optional_does_not_exit() -> None:
    with patch.dict("os.environ", {}, clear=True):
        validate_env(require_chutes=False)


def test_print_result_no_details(capsys: pytest.CaptureFixture[str]) -> None:
    result = {
        "total_score": 1.5,
        "success_rate": 0.5,
        "evaluated": 2,
    }
    print_result(result)
    out = capsys.readouterr().out
    assert "EVALUATION RESULT" in out
    assert "1.50" in out
    assert "50.0%" in out
    assert "2 task(s)" in out


def test_print_result_with_details(capsys: pytest.CaptureFixture[str]) -> None:
    result = {
        "total_score": 1.0,
        "success_rate": 1.0,
        "evaluated": 1,
        "details": [
            {
                "task_id": "task-1",
                "success": True,
                "score": 1.0,
                "steps": 10,
                "tests_passed": 3,
                "total_tests": 3,
            }
        ],
    }
    print_result(result)
    out = capsys.readouterr().out
    assert "Task Details" in out
    assert "task-1" in out
    assert "PASS" in out


def test_print_result_with_error(capsys: pytest.CaptureFixture[str]) -> None:
    result = {"error": "Something failed", "evaluated": 0}
    print_result(result)
    out = capsys.readouterr().out
    assert "Something failed" in out


def test_get_output_dir_default() -> None:
    args = parse_args([])
    out_dir = get_output_dir(args)
    assert out_dir.name == "eval"
    assert "autoppia" in str(out_dir).lower()


def test_get_output_dir_custom() -> None:
    args = parse_args(["--output-dir", "/tmp/custom"])
    out_dir = get_output_dir(args)
    assert out_dir == Path("/tmp/custom").resolve()


def test_save_result(tmp_path: Path) -> None:
    result = {
        "total_score": 2.0,
        "success_rate": 1.0,
        "evaluated": 2,
        "details": [],
    }
    path = save_result(result, tmp_path)
    assert path.parent == tmp_path
    assert path.suffix == ".json"
    assert "autoppia_" in path.name
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["total_score"] == 2.0
    assert content["evaluated"] == 2
