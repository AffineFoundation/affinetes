"""Verify FailFastLitellmModel: 4xx aborts immediately, 5xx still retries.

Run: python3 environments/SWE-INFINITE/test_failfast_litellm.py
"""
import os
import sys
import time
from pathlib import Path

# Cap retries at 3 so the 5xx case finishes quickly enough.
os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "3"

sys.path.insert(0, str(Path(__file__).resolve().parent / "agents"))

from unittest.mock import patch

import litellm

from miniswe import FailFastLitellmModel


def make_model():
    # cost_tracking=ignore_errors so we don't fail on cost calc.
    return FailFastLitellmModel(
        model_name="openai/test-model",
        model_kwargs={"api_base": "http://invalid", "api_key": "x"},
        cost_tracking="ignore_errors",
    )


class _FakeResponse:
    """Minimal stand-in for litellm.completion's return value."""

    def __init__(self):
        class _Choice:
            class _Msg:
                content = "ok"
            message = _Msg()
        self.choices = [_Choice()]

    def model_dump(self):
        return {}


def run_case(label, exc, expect_call_count, expect_exc_type):
    """Patch litellm.completion to raise `exc` every call, then count calls.

    expect_call_count = 1: blacklisted exception (no retry).
    expect_call_count > 1: retry up to MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT.
    """
    calls = []

    def _fake_completion(*args, **kwargs):
        calls.append(time.time())
        raise exc

    model = make_model()
    raised = None
    t0 = time.time()
    with patch.object(litellm, "completion", side_effect=_fake_completion):
        try:
            model.query([{"role": "user", "content": "hi"}])
        except BaseException as e:
            raised = e
    elapsed = time.time() - t0

    ok = (
        len(calls) == expect_call_count
        and isinstance(raised, expect_exc_type)
    )
    mark = "PASS" if ok else "FAIL"
    print(
        f"[{mark}] {label:<55} calls={len(calls):>2} (expected {expect_call_count}) "
        f"raised={type(raised).__name__} elapsed={elapsed:>5.1f}s"
    )
    return ok


def main():
    failed = 0
    cases = [
        # 4xx — must abort on first try (no retry, instant)
        (
            "BadRequestError 'Input length exceeds maximum allowed length'",
            litellm.exceptions.BadRequestError(
                message="Input length (57719 tokens) exceeds the maximum allowed length (55539 tokens).",
                model="test", llm_provider="openai",
            ),
            1, litellm.exceptions.BadRequestError,
        ),
        (
            "ContextWindowExceededError (already in default blacklist)",
            litellm.exceptions.ContextWindowExceededError(
                message="ctx exceeded", model="test", llm_provider="openai",
            ),
            1, litellm.exceptions.ContextWindowExceededError,
        ),
        (
            "AuthenticationError 401",
            litellm.exceptions.AuthenticationError(
                message="bad key", model="test", llm_provider="openai",
            ),
            1, litellm.exceptions.AuthenticationError,
        ),

        # 5xx-like — must retry up to stop_after_attempt(3)
        (
            "InternalServerError 500 (must retry)",
            litellm.exceptions.InternalServerError(
                message="server boom", model="test", llm_provider="openai",
            ),
            3, litellm.exceptions.InternalServerError,
        ),
        (
            "ServiceUnavailableError 503 (must retry)",
            litellm.exceptions.ServiceUnavailableError(
                message="503", model="test", llm_provider="openai",
            ),
            3, litellm.exceptions.ServiceUnavailableError,
        ),
        (
            "RateLimitError 429 (must retry)",
            litellm.exceptions.RateLimitError(
                message="rate limited", model="test", llm_provider="openai",
            ),
            3, litellm.exceptions.RateLimitError,
        ),
    ]

    for label, exc, n, t in cases:
        if not run_case(label, exc, n, t):
            failed += 1

    print()
    print(f"{len(cases) - failed}/{len(cases)} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
