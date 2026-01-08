"""Logic V2 Environment Actor - Seed-based generation"""

import os
import time
import gc
import httpx
import openai
import sys
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add /app to path to import local modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from logic_task_v2 import LogicTaskV2


class Actor:
    """Logic V2 task evaluation actor with seed-based generation"""

    def __init__(self, api_key: str = None, task_configs: dict = None, max_cache_size: int = 1000, max_client_cache: int = 200):
        """
        Initialize Actor with API key and task configurations

        Args:
            api_key: API key for LLM service. If not provided, will use CHUTES_API_KEY env var
            task_configs: Optional dict of task-specific configurations
                         Format: {"dyck_language": {"n_types": 4, "total_length": 20}}
            max_cache_size: Maximum number of challenges to cache (default: 1000)
            max_client_cache: Maximum number of OpenAI clients to cache (default: 200)
                             Prevents unlimited growth when using multiple base_urls
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

        # Initialize logic task V2 instance with caching
        self.logic_task = LogicTaskV2(task_configs=task_configs, max_cache_size=max_cache_size)

        # Reusable OpenAI client to avoid connection leaks (LRU cache with size limit)
        from collections import OrderedDict
        self._client_cache = OrderedDict()
        self._max_client_cache = max_client_cache

    def _get_or_create_client(self, base_url, current_api_key, timeout):
        """Get cached client or create new one with LRU eviction to avoid connection leaks"""
        # Create cache key based on base_url and api_key
        cache_key = f"{base_url}:{current_api_key[:10]}"

        # If client exists, move to end (mark as recently used)
        if cache_key in self._client_cache:
            self._client_cache.move_to_end(cache_key)
            return self._client_cache[cache_key]

        # Create new client
        # Unset SSL_CERT_FILE to avoid certificate path issues in container
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('REQUESTS_CA_BUNDLE', None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip('/'),
            api_key=current_api_key,
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=20.0,     # Read timeout (per chunk)
                write=10.0,    # Write timeout
                pool=10.0      # Pool timeout
            ),
            max_retries=0
        )

        self._client_cache[cache_key] = client

        # Evict oldest client if cache is full (LRU)
        if len(self._client_cache) > self._max_client_cache:
            oldest_key, oldest_client = self._client_cache.popitem(last=False)
            # Note: We don't await close() here since it's sync method
            # The client will be garbage collected and connections will eventually close
            # For proper cleanup, consider using async cleanup in Actor.__del__ or cleanup method

        return client

    async def _llm_chat(self, prompt, model, base_url, timeout, temperature, current_api_key, seed=None):
        """Call LLM API with specified API key and optional seed (streaming mode)"""
        import asyncio

        # Get client (timeout is configured in httpx.Timeout when creating client)
        client = self._get_or_create_client(base_url, current_api_key, timeout)

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

        # Create stream (httpx client already has 20s read timeout configured)
        stream = await client.chat.completions.create(**params)

        # Collect streamed content and usage
        content_parts = []
        reasoning_parts = []  # Collect reasoning content for o1-style models
        usage = None
        chunk_count = 0
        max_chunks = 32000  # Limit output to ~32k tokens (assuming ~1 chunk per token)
        chunk_timeout = 30.0  # Max time between chunks

        try:
            # Create async iterator from stream
            chunk_iter = stream.__aiter__()

            while True:
                try:
                    # Wait for next chunk with timeout
                    chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=chunk_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Stream timeout: no chunk received for {chunk_timeout}s")

                chunk_count += 1

                # Collect content chunks and reasoning chunks
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta

                    # Collect regular content
                    if delta.content:
                        content_parts.append(delta.content)

                    # Collect reasoning content (for o1-style reasoning models)
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_parts.append(delta.reasoning_content)

                    # Apply chunk limit (approximate token limit)
                    if chunk_count >= max_chunks:
                        break

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

        finally:
            # Critical: Close stream to return connection to pool
            # Use timeout protection to avoid hanging on close
            try:
                await asyncio.wait_for(stream.response.aclose(), timeout=5.0)
            except asyncio.TimeoutError:
                # If close hangs, log and continue to avoid blocking error propagation
                # The connection will be cleaned up by garbage collector
                import logging
                logging.warning("Stream close timed out after 5s, connection may leak")
            except Exception as e:
                # Best effort cleanup - don't let close failure block error propagation
                import logging
                logging.debug(f"Stream close failed: {e}")

    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=0.7,
        api_key: str = None,
        task_id: int = None,
        seed: int = None,
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
                # Try to decode task_type for task_name
                try:
                    from logic_task_v2 import LogicTaskV2
                    task_type, seed = LogicTaskV2.decode_task_id(task_id)
                    task_name = f"logic-v2:{task_type}"
                except:
                    task_type = "unknown"
                    seed = task_id
                    task_name = "logic-v2:unknown"

                # Return 0 score with same format as success, failure info in conversation
                error_message = f"Task generation failed: {str(e)}"
                conversation = [
                    {"role": "system", "content": error_message},
                    {"role": "assistant", "content": None}
                ]

                return {
                    "task_name": task_name,
                    "score": 0.0,
                    "success": True,  # Hide failure from external view
                    "time_taken": time.time() - start,
                    "extra": {
                        "conversation": conversation,
                        "task_id": task_id,
                        "task_type": task_type,
                        "seed": seed,
                        "task_metadata": {
                            "generation_failed": True,
                            "generation_error": str(e),
                            "generation_traceback": traceback.format_exc()
                        },
                        "usage": None
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

    async def _llm_chat_local(self, prompt, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, seed=None):
        """Call local LLM model for inference"""
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=4096)

        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    async def local_evaluate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        timeout=600,
        temperature=0.7,
        api_key: str = None,
        task_id: int = None,
        seed: int = None,
    ):
        """
        Run evaluation on a single logic task

        Args:
            model: Model to use for evaluation
            tokenizer: Tokenizer to use for evaluation
            seed: Random seed for LLM generation. Used to ensure reproducible results. If not provided, a random seed will be generated.
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
                # Try to decode task_type for task_name
                try:
                    from logic_task_v2 import LogicTaskV2
                    task_type, seed = LogicTaskV2.decode_task_id(task_id)
                    task_name = f"logic-v2:{task_type}"
                except:
                    task_type = "unknown"
                    seed = task_id
                    task_name = "logic-v2:unknown"

                # Return 0 score with same format as success, failure info in conversation
                error_message = f"Task generation failed: {str(e)}"
                conversation = [
                    {"role": "system", "content": error_message},
                    {"role": "assistant", "content": None}
                ]

                return {
                    "task_name": task_name,
                    "score": 0.0,
                    "success": True,  # Hide failure from external view
                    "time_taken": time.time() - start,
                    "extra": {
                        "conversation": conversation,
                        "task_id": task_id,
                        "task_type": task_type,
                        "seed": seed,
                        "task_metadata": {
                            "generation_failed": True,
                            "generation_error": str(e),
                            "generation_traceback": traceback.format_exc()
                        },
                        "usage": None
                    }
                }
            else:
                # Other ValueErrors are unexpected, let them propagate
                raise

        # Call LLM using task_id as seed
        try:
            resp = await self._llm_chat_local(challenge.prompt, model, tokenizer, task_id)
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
                "task_metadata": challenge.extra.get("metadata", {})
            }
        }

        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Force garbage collection to free memory immediately
        gc.collect()

        return result