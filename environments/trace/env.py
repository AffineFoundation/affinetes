"""Trace Environment Actor

Supports two modes:
1. evaluate() - One-shot LLM evaluation with internal generation + scoring
2. reset/step/stop - OpenEnv training interface for external control
"""

import os
import time
import gc
import sys
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Add /app to path to import local modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from trace_task import TraceTask

# Import shared logging utilities
from request_logger import RequestLogger, log_event

from affinetes.core.openenv import OpenEnvResponse
from affinetes.core.llm_chat import llm_chat


@dataclass
class EpisodeState:
    """Training episode state for Trace tasks"""
    episode_id: str
    task_id: int
    seed: int
    challenge: Any  # Challenge object
    done: bool = False
    truncated: bool = False
    step_count: int = 0
    response: Optional[str] = None
    score: float = 0.0


class Actor:
    """Trace task evaluation actor with training support

    Provides two modes:
    1. evaluate() - One-shot LLM evaluation with internal generation + scoring
    2. reset/step/stop - OpenEnv training interface for external control
    """

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

        # Training episode states - supports concurrent episodes
        self._episodes: Dict[str, EpisodeState] = {}
        self._last_observations: Dict[str, str] = {}

    # ========== Helper methods for training interface ==========

    def _info(self, ep: Optional[EpisodeState] = None, *, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build info dictionary for OpenEnv response"""
        info: Dict[str, Any] = {
            "task_id": ep.task_id if ep else None,
            "seed": ep.seed if ep else None,
            "step_count": ep.step_count if ep else 0,
            "score": ep.score if ep else 0.0,
        }
        if ep and ep.challenge:
            info["dataset_index"] = ep.challenge.extra.get("dataset_index")
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
        Reset environment and start a new trace task episode.

        Args:
            task_id: Task identifier (index into dataset)
            seed: Random seed for reproducibility (used for print injection)

        Returns:
            OpenEnvResponse with challenge prompt as observation
        """
        resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        resolved_task_id = task_id if task_id is not None else random.randint(0, 10**6 - 1)

        # Generate challenge
        try:
            challenge = await self.trace_task.generate(task_id=resolved_task_id)
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
            seed=resolved_seed,
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
        Execute an action (submit response) for the trace task.

        Args:
            action: The predicted stdout output
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
            score, test_result = await self.trace_task.evaluate(action, ep.challenge)
            ep.score = score
            ep.done = True

            # Build result observation
            ground_truth = ep.challenge.extra.get("ground_truth", "")
            result_obs = f"""# Evaluation Result
Score: {score}
Test Result: {test_result}

## Your Prediction:
{action[:500]}{"..." if len(action) > 500 else ""}

## Expected Output:
{ground_truth[:500]}{"..." if len(ground_truth) > 500 else ""}"""

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

    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=None,
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
            temperature: Temperature for LLM generation (None = use model default)
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

        # Setup request logger
        logger = RequestLogger(
            task_id=task_id if task_id is not None else "random",
            task_type="trace",
            seed=seed,
            model=model,
            base_url=base_url
        )
        logger.__enter__()

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
            result = await llm_chat(
                messages=[{"role": "user", "content": challenge.prompt}],
                model=model,
                base_url=base_url,
                api_key=current_api_key,
                timeout=timeout,
                temperature=temperature,
                seed=seed,
                stream=True,
            )
            resp, usage = result.content, result.usage
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

        logger.__exit__(None, None, None)
        return result

