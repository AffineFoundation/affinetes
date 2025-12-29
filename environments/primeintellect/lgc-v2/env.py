"""Logic V2 Environment Actor - Seed-based generation"""

import os
import time
import gc
import httpx
import openai
import sys
import random

# Add /app to path to import local modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from logic_task_v2 import LogicTaskV2


class Actor:
    """Logic V2 task evaluation actor with seed-based generation"""

    def __init__(self, api_key: str = None, task_configs: dict = None, max_cache_size: int = 10000):
        """
        Initialize Actor with API key and task configurations

        Args:
            api_key: API key for LLM service. If not provided, will use CHUTES_API_KEY env var
            task_configs: Optional dict of task-specific configurations
                         Format: {"dyck_language": {"n_types": 4, "total_length": 20}}
            max_cache_size: Maximum number of challenges to cache (default: 10000)
                           Set to 0 to disable caching
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

        # Initialize logic task V2 instance with caching
        self.logic_task = LogicTaskV2(task_configs=task_configs, max_cache_size=max_cache_size)

    async def _llm_chat(self, prompt, model, base_url, timeout, temperature, current_api_key, seed=None):
        """Call LLM API with specified API key and optional seed (streaming mode)"""
        # Unset SSL_CERT_FILE to avoid certificate path issues in container
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('REQUESTS_CA_BUNDLE', None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip('/'),
            api_key=current_api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0
        )

        # Prepare API call parameters with streaming enabled
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        # Add seed if provided
        if seed is not None:
            params["seed"] = seed

        stream = await client.chat.completions.create(**params)

        # Collect streamed content and usage
        content_parts = []
        usage = None

        async for chunk in stream:
            # Collect content chunks
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

            # Collect usage information from the final chunk
            if chunk.usage:
                usage = chunk.usage.model_dump()

        # Combine all content parts
        if not content_parts:
            raise ValueError("LLM API returned empty content stream")

        content = "".join(content_parts)
        if not content:
            raise ValueError("LLM API returned None content (possible content filtering or API error)")

        # Return both content and usage information
        return content.strip(), usage

    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=0.7,
        api_key: str = None,
        task_id: int = None
    ):
        """
        Run evaluation on a single logic task

        Args:
            model: Model name to use for evaluation
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API calls
            temperature: Temperature for LLM generation
            api_key: Override API key for this evaluation. If not provided, uses instance api_key
            task_id: Task ID that encodes both task type and seed.
                     Task type is determined by: task_id // 100,000,000
                     Seed is determined by: task_id % 100,000,000

                     Examples:
                       - task_id=500 -> dyck_language with seed=500
                       - task_id=100_000_500 -> future_task with seed=500

                     If not provided, a random task_id for dyck_language will be generated.

        Returns:
            Dict with evaluation results including score, conversation, and metadata
        """
        # Generate random task_id if not provided (default to dyck_language)
        if task_id is None:
            task_id = random.randint(0, 99_999_999)  # dyck_language range

        # Allow per-call api_key override
        current_api_key = api_key or self.api_key

        start = time.time()

        # Generate challenge using task_id (auto-detects task type)
        try:
            challenge = await self.logic_task.generate(task_id=task_id)
        except ValueError as e:
            # Only catch expected generation failures
            if "Failed to generate valid sequence" in str(e):
                import traceback
                return {
                    "task_name": "logic-v2:unknown",
                    "score": 0.0,
                    "success": False,
                    "time_taken": time.time() - start,
                    "extra": {
                        "error": "generation_failed",
                        "error_detail": str(e),
                        "task_id": task_id,
                        "traceback": traceback.format_exc()
                    }
                }
            else:
                # Other ValueErrors are unexpected, let them propagate
                raise

        # Call LLM using task_id as seed
        usage = None
        try:
            resp, usage = await self._llm_chat(challenge.prompt, model, base_url, timeout, temperature, current_api_key, task_id)
            error = None
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Evaluate
        score = 0.0
        if resp:
            try:
                score = await self.logic_task.evaluate(resp, challenge)
            except Exception as e:
                import traceback
                error = f"Evaluation error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        task_type = challenge.extra.get("task_type")

        result = {
            "task_name": f"logic-v2:{task_type}",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "task_id": task_id,
                "task_type": task_type,
                "seed": challenge.extra.get("seed"),
                "task_metadata": challenge.extra.get("metadata", {}),
                "usage": usage
            }
        }

        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Force garbage collection to free memory immediately
        gc.collect()

        return result
