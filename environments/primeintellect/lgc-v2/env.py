"""Logic V2 Environment Actor - Seed-based generation

Supports two modes:
1. evaluate() - One-shot LLM evaluation with internal generation + scoring
2. reset/step/stop - OpenEnv training interface for external control
"""

import os
import time
import gc
import httpx
import openai
import sys
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Add /app to path to import local modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from logic_task_v2 import LogicTaskV2

# Import shared logging utilities
from request_logger import RequestLogger, log_event

from affinetes.core.openenv import OpenEnvResponse


@dataclass
class EpisodeState:
    """Training episode state for Logic V2 tasks"""
    episode_id: str
    task_id: int
    task_type: str
    seed: int
    challenge: Any  # Challenge object
    done: bool = False
    truncated: bool = False
    step_count: int = 0
    response: Optional[str] = None
    score: float = 0.0


class Actor:
    """Logic V2 task evaluation actor with seed-based generation

    Provides two modes:
    1. evaluate() - One-shot LLM evaluation with internal generation + scoring
    2. reset/step/stop - OpenEnv training interface for external control
    """

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

        # Training episode states - supports concurrent episodes
        self._episodes: Dict[str, EpisodeState] = {}
        self._last_observations: Dict[str, str] = {}

    # ========== Helper methods for training interface ==========

    def _info(self, ep: Optional[EpisodeState] = None, *, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build info dictionary for OpenEnv response"""
        info: Dict[str, Any] = {
            "task_id": ep.task_id if ep else None,
            "task_type": ep.task_type if ep else None,
            "seed": ep.seed if ep else None,
            "step_count": ep.step_count if ep else 0,
            "score": ep.score if ep else 0.0,
        }
        if ep and ep.challenge:
            info["task_metadata"] = ep.challenge.extra.get("metadata", {})
        if error:
            info["error"] = error
        return info

    def _resp(
        self,
        observation: str,
        *,
        episode_id: Optional[str] = None,
        reward: float = 0.0,
        done: bool = False,
        truncated: bool = False,
        info: Dict[str, Any],
    ) -> OpenEnvResponse:
        """Build OpenEnv response"""
        if episode_id:
            self._last_observations[episode_id] = observation
        return OpenEnvResponse(
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            episode_id=episode_id,
            info=info,
        )

    # ========== OpenEnv Training Interface ==========

    async def reset(
        self,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> OpenEnvResponse:
        """
        Reset environment and start a new logic task episode.

        Args:
            task_id: Task identifier that encodes both task type and seed.
                     Task type is determined by: task_id // 100,000,000
                     Seed is determined by: task_id % 100,000,000
            seed: Random seed (unused - seed is encoded in task_id)

        Returns:
            OpenEnvResponse with challenge prompt as observation
        """
        # Generate random task_id if not provided (default to dyck_language)
        resolved_task_id = task_id if task_id is not None else random.randint(0, 99_999_999)

        # Decode task_type for tracking
        try:
            task_type, actual_seed = LogicTaskV2.decode_task_id(resolved_task_id)
        except Exception:
            task_type = "unknown"
            actual_seed = resolved_task_id

        # Generate challenge
        try:
            challenge = await self.logic_task.generate(task_id=resolved_task_id)
        except ValueError as e:
            if "Failed to generate valid sequence" in str(e):
                return self._resp(
                    f"Task generation failed: {str(e)}",
                    episode_id=None,
                    done=True,
                    truncated=True,
                    info=self._info(None, error={"type": "generation_error", "message": str(e), "retryable": True}),
                )
            raise
        except Exception as e:
            return self._resp(
                f"Error generating challenge: {str(e)}",
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "generation_error", "message": str(e), "retryable": True}),
            )

        # Generate episode ID
        episode_id = uuid.uuid4().hex

        # Create episode state
        ep = EpisodeState(
            episode_id=episode_id,
            task_id=resolved_task_id,
            task_type=task_type,
            seed=actual_seed,
            challenge=challenge,
        )

        # Store in concurrent episodes dict
        self._episodes[episode_id] = ep

        # Return challenge prompt as observation
        return self._resp(challenge.prompt, episode_id=episode_id, info=self._info(ep))

    async def step(
        self,
        action: str,
        episode_id: Optional[str] = None,
    ) -> OpenEnvResponse:
        """
        Execute an action (submit response) for the logic task.

        Args:
            action: The answer/response to evaluate
            episode_id: Episode identifier

        Returns:
            OpenEnvResponse with evaluation result
        """
        # Validate episode_id is provided
        if not episode_id:
            return self._resp(
                "No episode_id provided. Call reset() first to get an episode_id.",
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "no_episode_id", "message": "episode_id is required for step().", "retryable": True}),
            )

        # Look up episode from concurrent episodes dict
        ep = self._episodes.get(episode_id)
        if not ep:
            return self._resp(
                f"Episode not found: {episode_id}. Call reset() to start a new episode.",
                episode_id=episode_id,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "episode_not_found", "message": f"Episode {episode_id} not found.", "retryable": True}),
            )

        # Check if episode already done
        if ep.done:
            return self._resp(
                f"Episode already completed with score {ep.score}. Call reset() to start a new episode.",
                episode_id=ep.episode_id,
                reward=ep.score,
                done=True,
                truncated=False,
                info=self._info(ep, error={"type": "episode_done", "message": "Episode already finished. Call reset().", "retryable": True}),
            )

        # Evaluate the response
        ep.step_count += 1
        ep.response = action

        try:
            score = await self.logic_task.evaluate(action, ep.challenge)
            ep.score = score
            ep.done = True

            # Build result observation
            result_obs = f"""# Evaluation Result
Task Type: {ep.task_type}
Score: {score}
Success: {"Yes" if score > 0 else "No"}

## Your Answer:
{action[:500]}{"..." if len(action) > 500 else ""}"""

            return self._resp(
                result_obs,
                episode_id=ep.episode_id,
                reward=score,
                done=True,
                info=self._info(ep),
            )

        except Exception as e:
            ep.done = True
            ep.truncated = True
            return self._resp(
                f"Evaluation error: {str(e)}",
                episode_id=ep.episode_id,
                reward=0.0,
                done=True,
                truncated=True,
                info=self._info(ep, error={"type": "evaluation_error", "message": str(e), "retryable": False}),
            )

    async def state(
        self,
        episode_id: Optional[str] = None,
    ) -> OpenEnvResponse:
        """
        Get current task state without advancing (no state transition).

        Args:
            episode_id: Episode identifier

        Returns:
            OpenEnvResponse with current observation
        """
        if not episode_id:
            return self._resp(
                "No episode_id provided. Call reset() first to get an episode_id.",
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "no_episode_id", "message": "episode_id is required.", "retryable": True}),
            )

        ep = self._episodes.get(episode_id)
        if not ep:
            obs = self._last_observations.get(episode_id, "Episode not found.")
            return self._resp(
                obs,
                episode_id=episode_id,
                done=True,
                truncated=True,
                info=self._info(None, error={"type": "episode_not_found", "message": f"Episode {episode_id} not found.", "retryable": True}),
            )

        if ep.done:
            obs = f"Episode completed with score {ep.score}."
        else:
            obs = ep.challenge.prompt

        return self._resp(obs, episode_id=ep.episode_id, done=ep.done, truncated=ep.truncated, info=self._info(ep))

    async def stop(
        self,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stop (terminate) the active episode and release resources.

        Args:
            episode_id: Episode identifier

        Returns:
            Status dictionary
        """
        if not episode_id:
            return {"status": "ok", "stopped": False, "message": "No episode_id provided"}

        # Remove from concurrent episodes dict
        ep = self._episodes.pop(episode_id, None)
        self._last_observations.pop(episode_id, None)

        if not ep:
            return {"status": "ok", "stopped": False, "message": f"Episode {episode_id} not found"}

        return {"status": "ok", "stopped": True, "episode_id": episode_id}

    # ========== Original Evaluation Interface ==========

    def _get_or_create_client(self, base_url, current_api_key, timeout):
        """Get cached client or create new one with LRU eviction to avoid connection leaks"""
        # Create cache key based on base_url and api_key
        cache_key = f"{base_url}:{current_api_key[:10]}"

        # If client exists, move to end (mark as recently used)
        if cache_key in self._client_cache:
            self._client_cache.move_to_end(cache_key)
            log_event("client_cache_hit", cache_size=len(self._client_cache), cache_key=cache_key[:30])
            return self._client_cache[cache_key]

        # Create new client
        log_event("client_cache_miss", cache_size=len(self._client_cache), cache_key=cache_key[:30])

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
        log_event("client_created", cache_size=len(self._client_cache))

        # Evict oldest client if cache is full (LRU)
        if len(self._client_cache) > self._max_client_cache:
            oldest_key, oldest_client = self._client_cache.popitem(last=False)
            log_event("client_evicted", evicted_key=oldest_key[:30], cache_size=len(self._client_cache))
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
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        # Add temperature if provided
        if temperature is not None:
            params["temperature"] = temperature

        # Add seed if provided
        if seed is not None:
            params["seed"] = seed

        # Create stream (httpx client already has 20s read timeout configured)
        log_event("stream_creating")
        stream_start = time.time()
        stream = await client.chat.completions.create(**params)
        log_event("stream_created")

        # Collect streamed content and usage
        content_parts = []
        reasoning_parts = []  # Collect reasoning content for o1-style models
        usage = None
        chunk_count = 0
        max_chunks = 32000  # Limit output to ~32k tokens (assuming ~1 chunk per token)
        chunk_timeout = 30.0  # Max time between chunks

        first_chunk_received = False

        try:
            # Create async iterator from stream
            chunk_iter = stream.__aiter__()

            while True:
                try:
                    # Wait for next chunk with timeout
                    chunk_receive_start = time.time()
                    chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=chunk_timeout)
                    chunk_receive_time = time.time()

                    # Track first chunk (TTFB)
                    if not first_chunk_received:
                        ttfb_ms = int((chunk_receive_time - stream_start) * 1000)
                        log_event("first_chunk_received", ttfb_ms=ttfb_ms)
                        first_chunk_received = True

                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    log_event("stream_chunk_timeout", level='error', timeout_seconds=chunk_timeout, chunks_received=chunk_count)
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
            content = "".join(content_parts) if content_parts else None
            reasoning = "".join(reasoning_parts) if reasoning_parts else None

            if not content:
                # Return None for empty content (e.g., token limit exhausted during reasoning)
                # This will result in 0 score rather than raising an error
                log_event("stream_empty_content", level='warning', chunks=chunk_count, has_reasoning=bool(reasoning))
                return None, reasoning, usage

            # Log stream completion stats
            stream_duration_ms = int((time.time() - stream_start) * 1000)
            log_event("stream_complete",
                     total_chunks=chunk_count,
                     content_length=len(content),
                     reasoning_length=len(reasoning) if reasoning else 0,
                     stream_duration_ms=stream_duration_ms)

            # Return content, reasoning, and usage information
            return content.strip(), reasoning, usage

        finally:
            # Critical: Close stream to return connection to pool
            # Use timeout protection to avoid hanging on close
            try:
                close_start = time.time()
                await asyncio.wait_for(stream.response.aclose(), timeout=5.0)
                close_duration_ms = int((time.time() - close_start) * 1000)
                log_event("stream_closed", close_duration_ms=close_duration_ms)
            except asyncio.TimeoutError:
                # If close hangs, log and continue to avoid blocking error propagation
                # The connection will be cleaned up by garbage collector
                log_event("stream_close_timeout", level='warning', timeout_seconds=5.0)
            except Exception as e:
                # Best effort cleanup - don't let close failure block error propagation
                log_event("stream_close_failed", level='warning', error=str(e))

    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=None,
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

        # Decode task_type for logging
        try:
            from logic_task_v2 import LogicTaskV2
            task_type, actual_seed = LogicTaskV2.decode_task_id(task_id)
        except:
            task_type = "unknown"
            actual_seed = task_id

        start = time.time()

        # Setup request logger
        logger = RequestLogger(
            task_id=task_id,
            task_type=task_type,
            seed=actual_seed,
            model=model,
            base_url=base_url
        )
        logger.__enter__()

        # Generate challenge using task_id (auto-detects task type)
        try:
            challenge = await self.logic_task.generate(task_id=task_id)
            log_event("challenge_generated")
        except ValueError as e:
            # Only catch expected generation failures
            if "Failed to generate valid sequence" in str(e):
                import traceback
                log_event("challenge_generation_failed", level='error', error=str(e))

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

                result = {
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
                logger.__exit__(None, None, None)
                return result
            else:
                # Other ValueErrors are unexpected, let them propagate
                raise

        # Call LLM using task_id as seed
        log_event("llm_call_start")
        usage = None
        reasoning = None
        try:
            resp, reasoning, usage = await self._llm_chat(challenge.prompt, model, base_url, timeout, temperature, current_api_key, task_id)
            error = None
            log_event("llm_call_complete", response_length=len(resp) if resp else 0, reasoning_length=len(reasoning) if reasoning else 0)
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            log_event("llm_call_failed", level='error', error=str(e), error_type=type(e).__name__)

        # Evaluate
        log_event("evaluation_start")
        score = 0.0
        if resp:
            try:
                score = await self.logic_task.evaluate(resp, challenge)
                log_event("evaluation_complete", score=score)
            except Exception as e:
                import traceback
                error = f"Evaluation error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                log_event("evaluation_failed", level='error', error=str(e))

        # Build assistant message with optional reasoning
        assistant_message = {"role": "assistant", "content": resp}
        if reasoning:
            assistant_message["reasoning_content"] = reasoning

        conversation = [
            {"role": "user", "content": challenge.prompt},
            assistant_message
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

        log_event("request_complete", score=score, success=score > 0, total_time_ms=int((time.time() - start) * 1000))

        # Force garbage collection to free memory immediately
        gc.collect()

        logger.__exit__(None, None, None)
        return result
