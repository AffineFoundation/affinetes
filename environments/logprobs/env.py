"""Logprobs V1 Environment Actor.

Given a task_id, generates a prompt and returns the top-K logprobs
for the model's first N output tokens.
"""

import gc
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

import httpx
import openai

from task_generator import generate_prompt


class Actor:
    """Returns top-K logprobs for the first N generated tokens.

    Task ID encoding:
      task_type = TASK_NAMES[task_id // 100_000_000]
      seed      = task_id %  100_000_000

    Supported ranges:
      0–99,999,999:   common_sense_combo
      100M–199,999,999: lgc_v2
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

    async def get_logprobs(
        self,
        task_id: int,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: str = None,
        n_tokens: int = 20,
        top_k: int = 3,
        temperature: float = 0.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """Generate a prompt from task_id and return top-K logprobs for n_tokens.

        Args:
            task_id: Encodes task type and seed (see class docstring).
            model: Model name; must support the logprobs parameter.
            base_url: OpenAI-compatible API endpoint.
            api_key: Override API key; falls back to CHUTES_API_KEY env var.
            n_tokens: Number of tokens to sample (default: 20).
            top_k: Top logprobs per token position (default: 3, max: 20).
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.

        Returns:
            Dict matching the standard affinetes evaluate() schema:
            {
                "task_name": "logprobs:{task_type}",
                "score": 0.0,
                "success": True,
                "time_taken": float,
                "extra": {
                    "task_id": int,
                    "task_type": str,
                    "seed": int,
                    "prompt": str,
                    "model": str,
                    "n_tokens": int,
                    "top_k": int,
                    "tokens": [
                        {
                            "position": int,
                            "token": str,
                            "logprob": float,
                            "top_k": [{"token": str, "logprob": float, "prob": float}]
                        },
                        ...
                    ],
                    "task_metadata": dict,
                    "usage": dict | None,
                }
            }
        """
        current_api_key = api_key or self.api_key
        start = time.time()

        # Generate prompt deterministically from task_id
        prompt, task_type, seed, task_metadata = await generate_prompt(task_id)

        # Non-streaming API call with logprobs enabled
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=current_api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )

        error = None
        tokens: List[Dict[str, Any]] = []
        usage = None

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=n_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=top_k,
                stream=False,
            )

            usage = resp.usage.model_dump() if resp.usage else None

            logprobs_content = (
                resp.choices[0].logprobs.content
                if resp.choices and resp.choices[0].logprobs
                else []
            )
            for i, tok in enumerate(logprobs_content[:n_tokens]):
                top_entries = [
                    {
                        "token": entry.token,
                        "logprob": round(entry.logprob, 6),
                        "prob": round(math.exp(entry.logprob), 6),
                    }
                    for entry in (tok.top_logprobs or [])
                ]
                tokens.append(
                    {
                        "position": i,
                        "token": tok.token,
                        "logprob": round(tok.logprob, 6),
                        "top_k": top_entries,
                    }
                )

        except Exception as e:
            import traceback
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        finally:
            await client.close()

        result = {
            "task_name": f"logprobs:{task_type}",
            "score": 0.0,
            "success": error is None,
            "time_taken": time.time() - start,
            "extra": {
                "task_id": task_id,
                "task_type": task_type,
                "seed": seed,
                "prompt": prompt,
                "model": model,
                "n_tokens": n_tokens,
                "top_k": top_k,
                "tokens": tokens,
                "task_metadata": task_metadata,
                "usage": usage,
            },
        }

        if error:
            result["error"] = error
            result["error_type"] = "api_failure"

        gc.collect()
        return result
