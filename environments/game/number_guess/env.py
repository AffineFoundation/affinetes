"""Number Guessing interactive environment

Clean implementation with simple function signatures.
HTTP layer handles request parsing, env layer focuses on business logic.
"""

import os
import time
import random
import re
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any

from affinetes.core.openenv import OpenEnvResponse
from affinetes.core.llm_chat import llm_chat


@dataclass
class EpisodeState:
    """Internal episode state"""
    episode_id: str
    task_id: int
    seed: int
    target: int
    attempts_used: int = 0
    done: bool = False


class Actor:
    """Interactive Number Guessing game environment

    Supports three calling patterns:
    1. SDK evaluate: await env.evaluate(model=..., task_id=...)
    2. SDK openenv:  sess = await env.openenv().reset(task_id=...)
                     resp = await sess.step("500")
    3. HTTP direct:  POST /reset {"task_id": 42, "seed": 123}
                     POST /step {"action": "500", "episode_id": "xxx"}
    """

    MIN_RANGE = 1
    MAX_RANGE = 1000
    MAX_ATTEMPTS = 10

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self._episodes: Dict[str, EpisodeState] = {}

    def _parse_guess(self, response: str) -> Optional[int]:
        """Parse integer from response string."""
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        return None

    def _prompt(self) -> str:
        """Initial game prompt."""
        return f"""You are playing a number guessing game.

Rules:
- I have chosen a secret number between {self.MIN_RANGE} and {self.MAX_RANGE} (inclusive)
- You have {self.MAX_ATTEMPTS} attempts to guess the number
- After each guess, I will tell you if the secret number is higher or lower
- Try to find the number in as few attempts as possible

To make a guess, respond with just the number.
Example: "500"

What is your first guess?"""

    def _info(self, ep: Optional[EpisodeState], error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build info dict."""
        info = {
            "task_id": ep.task_id if ep else None,
            "seed": ep.seed if ep else None,
            "attempts_left": max(self.MAX_ATTEMPTS - (ep.attempts_used if ep else 0), 0),
        }
        if error:
            info["error"] = error
        return info

    # ========== OpenEnv Interface ==========

    async def reset(
        self,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> OpenEnvResponse:
        """Reset environment and start a new episode.

        Args:
            task_id: Task identifier for reproducibility
            seed: Random seed

        Returns:
            OpenEnvResponse with initial observation
        """
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        task_id = task_id if task_id is not None else (seed & 0x7FFFFFFF)

        target = random.Random(task_id).randint(self.MIN_RANGE, self.MAX_RANGE)
        episode_id = uuid.uuid4().hex

        ep = EpisodeState(
            episode_id=episode_id,
            task_id=task_id,
            seed=seed,
            target=target,
        )
        self._episodes[episode_id] = ep

        return OpenEnvResponse(
            observation=self._prompt(),
            episode_id=episode_id,
            info=self._info(ep),
        )

    async def step(
        self,
        action: str,
        episode_id: Optional[str] = None,
    ) -> OpenEnvResponse:
        """Execute an action (guess).

        Args:
            action: The guess as a string
            episode_id: Episode identifier

        Returns:
            OpenEnvResponse with result
        """
        # Validate episode
        if not episode_id:
            return OpenEnvResponse(
                observation="No episode_id provided. Call reset() first.",
                done=True,
                truncated=True,
                info=self._info(None, {"type": "no_episode_id", "retryable": True}),
            )

        ep = self._episodes.get(episode_id)
        if not ep:
            return OpenEnvResponse(
                observation=f"Episode {episode_id} not found. Call reset() first.",
                done=True,
                truncated=True,
                info=self._info(None, {"type": "episode_not_found", "retryable": True}),
            )

        if ep.done:
            return OpenEnvResponse(
                observation="Episode already finished. Call reset() to start a new one.",
                episode_id=episode_id,
                done=True,
                info=self._info(ep, {"type": "episode_done", "retryable": True}),
            )

        # Parse guess
        guess = self._parse_guess(action)
        if guess is None:
            return OpenEnvResponse(
                observation="Cannot parse your guess. Please respond with just a number.\n\nWhat is your guess?",
                episode_id=episode_id,
                info=self._info(ep, {"type": "action_parse", "retryable": True}),
            )

        # Validate range
        if guess < self.MIN_RANGE or guess > self.MAX_RANGE:
            return OpenEnvResponse(
                observation=f"Your guess {guess} is out of range. Please guess between {self.MIN_RANGE} and {self.MAX_RANGE}.\n\nWhat is your guess?",
                episode_id=episode_id,
                info=self._info(ep, {"type": "out_of_range", "retryable": True}),
            )

        # Process guess
        ep.attempts_used += 1
        attempts_left = self.MAX_ATTEMPTS - ep.attempts_used

        if guess == ep.target:
            ep.done = True
            return OpenEnvResponse(
                observation=f"Correct! You found the secret number {guess} in {ep.attempts_used} attempts!",
                episode_id=episode_id,
                reward=1.0,
                done=True,
                info=self._info(ep),
            )

        if attempts_left <= 0:
            ep.done = True
            return OpenEnvResponse(
                observation=f"Game over! You've used all {ep.attempts_used} attempts.\nThe secret number was {ep.target}.",
                episode_id=episode_id,
                reward=0.0,
                done=True,
                info=self._info(ep),
            )

        hint = "higher" if guess < ep.target else "lower"
        return OpenEnvResponse(
            observation=f"Your guess: {guess}\nResult: The secret number is {hint} than {guess}.\n\nAttempts remaining: {attempts_left}\n\nWhat is your next guess?",
            episode_id=episode_id,
            info=self._info(ep),
        )

    async def state(self, episode_id: Optional[str] = None) -> OpenEnvResponse:
        """Get current state without advancing."""
        if not episode_id:
            return OpenEnvResponse(
                observation="No episode_id provided.",
                done=True,
                truncated=True,
                info=self._info(None, {"type": "no_episode_id"}),
            )

        ep = self._episodes.get(episode_id)
        if not ep:
            return OpenEnvResponse(
                observation=f"Episode {episode_id} not found.",
                done=True,
                truncated=True,
                info=self._info(None, {"type": "episode_not_found"}),
            )

        return OpenEnvResponse(
            observation=self._prompt() if ep.attempts_used == 0 else f"Attempts used: {ep.attempts_used}",
            episode_id=episode_id,
            done=ep.done,
            info=self._info(ep),
        )

    async def stop(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop and cleanup an episode."""
        if not episode_id:
            return {"status": "ok", "stopped": False, "message": "No episode_id provided"}

        stopped = self._episodes.pop(episode_id, None) is not None
        return {"status": "ok", "stopped": stopped, "episode_id": episode_id}

    # ========== Evaluate Interface ==========

    async def evaluate(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        timeout: int = 600,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a full evaluation with an LLM."""
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        api_key = api_key or self.api_key
        start = time.time()

        # Reset
        reset_resp = await self.reset(task_id=task_id, seed=seed)
        episode_id = reset_resp.episode_id
        resolved_task_id = reset_resp.info.get("task_id")

        conversation = [{"role": "user", "content": reset_resp.observation}]
        usage = None
        success = False

        for _ in range(self.MAX_ATTEMPTS + 5):
            content, usage = await llm_chat(
                messages=conversation,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                temperature=temperature,
                seed=seed,
                stream=False,
            )
            action_text = content or ""
            conversation.append({"role": "assistant", "content": action_text})

            step_resp = await self.step(action=action_text, episode_id=episode_id)
            conversation.append({"role": "user", "content": step_resp.observation})

            if step_resp.done:
                success = step_resp.reward > 0.0
                break

        # Cleanup
        await self.stop(episode_id=episode_id)

        return {
            "task_name": "game:number_guess",
            "score": 1.0 if success else 0.0,
            "success": success,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "task_id": resolved_task_id,
                "usage": usage,
            }
        }
