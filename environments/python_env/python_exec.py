"""Predict-the-output task for generated Python snippets."""

from __future__ import annotations

import os
import random
import time
from typing import Any, Sequence

from _llm import chat
from _program import (
    DEFAULT_ALLOWED_OPS,
    build_program,
    execute_program,
    extract_answer,
    format_prompt,
)


class PythonExecTask:
    """Generate a Python program and score the model's predicted output."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

    async def evaluate(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 120,
        temperature: float = 0.0,
        api_key: str | None = None,
        task_id: int | None = None,
        seed: int | None = None,
        op_count: int = 8,
        allowed_ops: Sequence[str] | None = None,
        max_digits: int = 4,
        answer: str | None = None,
    ) -> dict[str, Any]:
        """Generate a Python code challenge and score the model output."""
        start = time.time()
        allowed = tuple(allowed_ops) if allowed_ops else DEFAULT_ALLOWED_OPS
        seed_value = seed if seed is not None else (task_id if task_id is not None else random.getrandbits(32))
        api_key_value = api_key or self.api_key

        code_lines, ops_used = build_program(
            seed=seed_value,
            op_count=op_count,
            allowed_ops=allowed,
            max_digits=max_digits,
        )
        prompt = format_prompt(code_lines)

        try:
            expected_output = execute_program(code_lines)
        except Exception as exc:  # noqa: BLE001 - propagate as structured error
            return {
                "task_name": "python-exec",
                "success": False,
                "score": 0.0,
                "error": f"Generated program failed: {exc}",
                "prompt": prompt,
                "seed": seed_value,
            }

        if answer is None:
            if not api_key_value:
                return {
                    "task_name": "python-exec",
                    "success": False,
                    "score": 0.0,
                    "error": "Provide CHUTES_API_KEY or pass `answer` to skip LLM call.",
                    "prompt": prompt,
                    "seed": seed_value,
                }
            try:
                model_response = await chat(
                    prompt=prompt,
                    model=model,
                    base_url=base_url,
                    timeout=timeout,
                    temperature=temperature,
                    api_key=api_key_value,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "task_name": "python-exec",
                    "success": False,
                    "score": 0.0,
                    "error": str(exc),
                    "prompt": prompt,
                    "seed": seed_value,
                }
        else:
            model_response = answer

        provided_output = extract_answer(model_response)
        expected_canonical = expected_output.rstrip("\n")
        provided_canonical = provided_output.rstrip("\n") if provided_output is not None else None
        success = provided_canonical is not None and provided_canonical == expected_canonical

        return {
            "task_name": "python-exec",
            "success": success,
            "score": 1.0 if success else 0.0,
            "expected_output": expected_output,
            "provided_output": provided_output,
            "model_response": model_response,
            "prompt": prompt,
            "seed": seed_value,
            "ops_used": ops_used,
            "time_taken": time.time() - start,
        }
