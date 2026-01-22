"""ARC-GEN Environment Actor."""

import gc
import os
import random
import sys
import time

import httpx
import openai

# Add /app to path to import local modules
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from arc_task import ArcGenTask


class Actor:
    """ARC-GEN evaluation actor."""

    def __init__(self, api_key: str = None) -> None:
        """Initialize Actor with API key."""
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.arc_task = ArcGenTask()

    async def _llm_chat(self, prompt, model, base_url, timeout, temperature, current_api_key, seed=None):
        """Call LLM API with specified API key and optional seed (streaming mode)."""
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=current_api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )

        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if seed is not None:
            params["seed"] = seed

        stream = await client.chat.completions.create(**params)

        content_parts = []
        reasoning_parts = []  # Collect reasoning content for o1-style models
        usage = None
        async for chunk in stream:
            # Collect content chunks and reasoning chunks
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta

                # Collect regular content
                if delta.content:
                    content_parts.append(delta.content)

                # Collect reasoning content (for o1-style reasoning models)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_parts.append(delta.reasoning_content)

            if chunk.usage:
                usage = chunk.usage.model_dump()

        if not content_parts:
            # Return None for empty content (e.g., token limit exhausted during reasoning)
            # This will result in 0 score rather than raising an error
            return None, usage

        content = "".join(content_parts)
        if not content:
            # Return None for empty content (e.g., token limit exhausted during reasoning)
            return None, usage

        return content.strip(), usage

    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=0.7,
        api_key: str = None,
        seed: int = None,
        task_id: int = None,
        num_train: int = 3,
    ):
        """Run evaluation on a single ARC-GEN task."""
        # Use task_id as seed for deterministic generation if seed not provided
        if seed is None:
            seed = random.randint(0, 99_999_999)

        current_api_key = api_key or self.api_key

        start = time.time()

        challenge = await self.arc_task.generate(task_id=task_id, num_train=num_train)

        usage = None
        try:
            resp, usage = await self._llm_chat(
                challenge.prompt,
                model,
                base_url,
                timeout,
                temperature,
                current_api_key,
                seed,
            )
            error = None
        except Exception as e:
            import traceback

            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        score = 0.0
        cell_accuracy = None
        if resp:
            score, cell_accuracy, _ = await self.arc_task.evaluate(resp, challenge)

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp},
        ]

        result = {
            "task_name": "affine:arc-gen",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "task_id": task_id,
                "cell_accuracy": cell_accuracy,
                "expected_output": challenge.extra.get("expected_output"),
                "usage": usage,
            },
        }

        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        gc.collect()

        return result
