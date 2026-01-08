"""Trace Environment Actor"""

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

from trace_task import TraceTask

# Import shared logging utilities
from request_logger import RequestLogger, log_event


class Actor:
    """Trace task evaluation actor"""
    
    def __init__(
        self,
        api_key: str = None,
    ):
        """
        Initialize Actor with API key
        
        Args:
            api_key: API key for LLM service. If not provided, will use CHUTES_API_KEY env var
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        
        # Initialize trace task instance
        self.trace_task = TraceTask()
    
    async def _llm_chat(self, prompt, model, base_url, timeout, temperature, current_api_key, seed=None):
        """Call LLM API with specified API key and optional seed (streaming mode)"""
        # Unset SSL_CERT_FILE to avoid certificate path issues in container
        # Let httpx/certifi use default certificate bundle
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

            # Collect usage information from the final chunk
            if chunk.usage:
                usage = chunk.usage.model_dump()

        # Combine all content parts
        if not content_parts:
            # Return None for empty content (e.g., token limit exhausted during reasoning)
            # This will result in 0 score rather than raising an error
            return None, usage

        content = "".join(content_parts)
        if not content:
            # Return None for empty content (e.g., token limit exhausted during reasoning)
            return None, usage

        # Return both content and usage information
        return content.strip(), usage
    
    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=0.7,
        api_key: str = None,
        seed: int = None,
        task_id: int = None
    ):
        """
        Run evaluation on a single trace task
        
        Args:
            model: Model name to use for evaluation
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API calls
            temperature: Temperature for LLM generation
            api_key: Override API key for this evaluation. If not provided, uses instance api_key
            seed: Random seed for LLM generation. Used to ensure reproducible results. If not provided, a random seed will be generated.
            task_id: Optional task ID for deterministic task selection.
                     If provided, used as index into dataset.
                     If not provided, random sample is selected.
        """
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Allow per-call api_key override
        current_api_key = api_key or self.api_key

        start = time.time()

        # Create request logger context
        with RequestLogger(
            task_id=task_id if task_id is not None else "random",
            task_type="trace",
            seed=seed,
            model=model,
            base_url=base_url
        ):
            # Generate challenge
            challenge = await self.trace_task.generate(task_id=task_id)
            log_event("challenge_generated", dataset_index=challenge.extra.get("dataset_index"))

            # Add model and base_url info to challenge.extra for logging
            challenge.extra["model"] = model
            challenge.extra["base_url"] = base_url

            # Call LLM
            log_event("llm_call_start")
            usage = None
            try:
                resp, usage = await self._llm_chat(challenge.prompt, model, base_url, timeout, temperature, current_api_key, seed)
                error = None
                log_event("llm_call_complete", response_length=len(resp) if resp else 0)
            except Exception as e:
                import traceback
                resp = None
                error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                log_event("llm_call_failed", level='error', error=str(e), error_type=type(e).__name__)

            # Evaluate
            log_event("evaluation_start")
            score = 0.0
            test_result = "0/1"
            if resp:
                score, test_result = await self.trace_task.evaluate(resp, challenge)
                log_event("evaluation_complete", score=score, test_result=test_result)

            conversation = [
                {"role": "user", "content": challenge.prompt},
                {"role": "assistant", "content": resp}
            ]

            result = {
                "task_name": "Trace",
                "score": score,
                "success": score > 0,
                "time_taken": time.time() - start,
                "extra": {
                    "conversation": conversation,
                    "seed": seed,
                    "test_result": test_result,
                    "dataset_index": challenge.extra.get("dataset_index"),
                    "usage": usage
                }
            }

            # Add error info if present
            if error:
                result["error"] = error
                result["error_type"] = "llm_failure"

            log_event("request_complete", score=score, success=score > 0, total_time_ms=int((time.time() - start) * 1000))

            # Force garbage collection to free memory immediately
            gc.collect()

            return result

