"""Bug-finding environment with principled evaluation."""

from __future__ import annotations

import os
import random
import time
from typing import Any, Sequence

from _bug_mutation import evaluate_response, format_prompt, generate_challenge
from _llm import chat
from _program import DEFAULT_ALLOWED_OPS


class Actor:
    """Bug-finding environment actor."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

    async def evaluate(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 120,
        temperature: float = 0.0,
        api_key: str | None = None,
        seed: int | None = None,
        task_id: int | None = None,
        op_count: int = 8,
        allowed_ops: Sequence[str] | None = None,
        max_digits: int = 4,
        answer: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run evaluation.

        Returns
        -------
        dict
            score: 1.0 if correct, 0.0 otherwise
            success: bool
            prompt, response, ground truth details
        """
        start = time.time()
        seed_val = seed if seed is not None else (task_id if task_id is not None else random.getrandbits(32))
        allowed = tuple(allowed_ops) if allowed_ops else DEFAULT_ALLOWED_OPS

        challenge = generate_challenge(seed_val, allowed_ops=allowed, op_count=op_count, max_digits=max_digits)
        if challenge is None:
            return {
                "task_name": "bug-find",
                "success": False,
                "score": 0.0,
                "error": "Failed to generate challenge",
                "seed": seed_val,
            }

        prompt = format_prompt(challenge)

        if answer is None:
            key = api_key or self.api_key
            if not key:
                return {
                    "task_name": "bug-find",
                    "success": False,
                    "score": 0.0,
                    "error": "No API key",
                    "seed": seed_val,
                    "prompt": prompt,
                }
            try:
                response = await chat(
                    prompt=prompt,
                    model=model,
                    base_url=base_url,
                    timeout=timeout,
                    temperature=temperature,
                    api_key=key,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "task_name": "bug-find",
                    "success": False,
                    "score": 0.0,
                    "error": str(exc),
                    "seed": seed_val,
                    "prompt": prompt,
                }
        else:
            response = answer

        correct = evaluate_response(response, challenge)

        return {
            "task_name": "bug-find",
            "success": correct,
            "score": 1.0 if correct else 0.0,
            "seed": seed_val,
            "prompt": prompt,
            "response": response,
            "mutation_operator": challenge.mutation.operator,
            "mutation_line": challenge.mutation.line,
            "mutation_original": challenge.mutation.original,
            "mutation_mutated": challenge.mutation.mutated,
            "time": time.time() - start,
        }
