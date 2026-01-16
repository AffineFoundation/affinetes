"""Number Guessing interactive environment"""

import os
import time
import random
import re
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any

from affinetes.core.openenv import ResetRequest, StepRequest, OpenEnvResponse
from affinetes.core.llm_chat import llm_chat


@dataclass
class EpisodeState:
    episode_id: str
    task_id: int
    seed: int
    target: int
    attempts_used: int = 0
    done: bool = False


class Actor:
    """Interactive Number Guessing game environment"""
    
    MIN_RANGE = 1
    MAX_RANGE = 1000
    MAX_ATTEMPTS = 10
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self._episode: Optional[EpisodeState] = None
        self._last_observation: Optional[str] = None
    
    def _parse_guess(self, response: str) -> Optional[int]:
        numbers = re.findall(r'-?\d+', response)
        
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        
        return None

    def _info(self, *, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Minimal info (A): flat info dict (no ident/public nesting)."""
        ep = self._episode
        attempts_left = max(self.MAX_ATTEMPTS - (ep.attempts_used if ep else 0), 0)
        info: Dict[str, Any] = {
            "task_id": ep.task_id if ep else None,
            "seed": ep.seed if ep else None,
            "attempts_left": attempts_left,
        }
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
    ) -> Dict[str, Any]:
        self._last_observation = observation
        return OpenEnvResponse(
            episode_id=episode_id,
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        ).model_dump()

    def _initial_prompt(self) -> str:
        return f"""You are playing a number guessing game.

Rules:
- I have chosen a secret number between {self.MIN_RANGE} and {self.MAX_RANGE} (inclusive)
- You have {self.MAX_ATTEMPTS} attempts to guess the number
- After each guess, I will tell you if the secret number is higher or lower
- Try to find the number in as few attempts as possible

To make a guess, respond with just the number.
Example: "500"

What is your first guess?"""

    async def reset(
        self,
        request: Optional[Dict[str, Any]] = None,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if request is None:
            request = {"task_id": task_id, "seed": seed, "kwargs": kwargs}
        rr = ResetRequest.model_validate(request)

        resolved_seed = rr.seed if rr.seed is not None else random.randint(0, 2**32 - 1)
        resolved_task_id = int(rr.task_id) if rr.task_id is not None else (int(resolved_seed) & 0x7FFFFFFF)

        target = random.Random(resolved_task_id).randint(self.MIN_RANGE, self.MAX_RANGE)
        episode_id = uuid.uuid4().hex
        self._episode = EpisodeState(
            episode_id=episode_id,
            task_id=resolved_task_id,
            seed=int(resolved_seed),
            target=target,
        )

        obs = self._initial_prompt()
        return self._resp(obs, episode_id=episode_id, info=self._info())

    async def state(
        self,
        request: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return current observation for the active episode (no state transition)."""
        if request is None:
            request = {"episode_id": episode_id}
        req_episode_id = (request or {}).get("episode_id")

        if not self._episode:
            obs = self._last_observation or "No active episode. Call reset() first."
            return self._resp(
                obs,
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(error={"type": "no_active_episode", "message": "Call reset() before state().", "retryable": True}),
            )

        if req_episode_id is not None and req_episode_id != self._episode.episode_id:
            obs = self._last_observation or "Episode mismatch."
            return self._resp(
                obs,
                episode_id=self._episode.episode_id,
                done=True,
                truncated=True,
                info=self._info(error={"type": "episode_mismatch", "message": f"Expected episode_id={self._episode.episode_id}, got {req_episode_id}", "retryable": False}),
            )

        obs = self._last_observation or self._initial_prompt()
        return self._resp(
            obs,
            episode_id=self._episode.episode_id,
            done=self._episode.done,
            truncated=False,
            info=self._info(),
        )

    async def stop(
        self,
        request: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop (terminate) the active episode and release server-side state."""
        if request is None:
            request = {"episode_id": episode_id}
        req_episode_id = (request or {}).get("episode_id")

        if not self._episode:
            return {"status": "ok", "stopped": False}

        if req_episode_id is not None and req_episode_id != self._episode.episode_id:
            return {
                "status": "failed",
                "error": f"episode_mismatch: expected {self._episode.episode_id}, got {req_episode_id}",
            }

        self._episode = None
        self._last_observation = None
        return {"status": "ok", "stopped": True}

    async def step(
        self,
        request: Optional[Dict[str, Any]] = None,
        action: Optional[str] = None,
        episode_id: Optional[str] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if request is None:
            request = {"action": action, "episode_id": episode_id, "kwargs": kwargs}
        sr = StepRequest.model_validate(request)
        skw = sr.kwargs or {}
        invalid_action_penalty = float(skw.get("invalid_action_penalty", 0.0))
        out_of_range_penalty = float(skw.get("out_of_range_penalty", 0.0))

        # No active episode
        if not self._episode:
            obs = "No active episode. Call reset() first."
            return self._resp(
                obs,
                episode_id=None,
                done=True,
                truncated=True,
                info=self._info(error={"type": "no_active_episode", "message": "Call reset() before step().", "retryable": True}),
            )

        if sr.episode_id is not None and sr.episode_id != self._episode.episode_id:
            obs = "Episode mismatch. Call reset() to start a new episode."
            return self._resp(
                obs,
                episode_id=self._episode.episode_id,
                done=True,
                truncated=True,
                info=self._info(error={"type": "episode_mismatch", "message": f"Expected episode_id={self._episode.episode_id}, got {sr.episode_id}", "retryable": False}),
            )

        if self._episode.done:
            obs = "Episode already finished. Call reset() to start a new episode."
            return self._resp(
                obs,
                episode_id=self._episode.episode_id,
                done=True,
                truncated=False,
                info=self._info(error={"type": "episode_done", "message": "Episode is done; call reset().", "retryable": True}),
            )

        guess = self._parse_guess(sr.action)
        if guess is None:
            obs = "Cannot parse your guess. Please respond with just a number.\n\nWhat is your guess?"
            return self._resp(
                obs,
                reward=invalid_action_penalty,
                episode_id=self._episode.episode_id,
                info=self._info(error={"type": "action_parse", "message": "Could not parse an integer from action.", "retryable": True}),
            )

        if guess < self.MIN_RANGE or guess > self.MAX_RANGE:
            obs = (
                f"Your guess {guess} is out of range. "
                f"Please guess a number between {self.MIN_RANGE} and {self.MAX_RANGE}.\n\n"
                "What is your guess?"
            )
            return self._resp(
                obs,
                reward=out_of_range_penalty,
                episode_id=self._episode.episode_id,
                info=self._info(error={"type": "out_of_range", "message": "Guess out of valid range.", "retryable": True}),
            )

        self._episode.attempts_used += 1
        target = self._episode.target
        attempts_left = self.MAX_ATTEMPTS - self._episode.attempts_used

        if guess == target:
            self._episode.done = True
            obs = f"Correct! You found the secret number {guess} in {self._episode.attempts_used} attempts!"
            return self._resp(obs, episode_id=self._episode.episode_id, reward=1.0, done=True, info=self._info())

        if attempts_left <= 0:
            self._episode.done = True
            obs = f"Game over! You've used all {self._episode.attempts_used} attempts.\nThe secret number was {target}."
            return self._resp(obs, episode_id=self._episode.episode_id, reward=0.0, done=True, info=self._info())

        hint = "higher" if guess < target else "lower"
        obs = f"""Your guess: {guess}
Result: The secret number is {hint} than {guess}.

Attempts remaining: {attempts_left}

What is your next guess?"""
        return self._resp(obs, episode_id=self._episode.episode_id, info=self._info())

    
    async def evaluate(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        task_id: Optional[int] = None,
        timeout: int = 600,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Play number guessing game interactively"""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        current_api_key = api_key or self.api_key
        start = time.time()

        # Reuse the OpenEnv loop to avoid duplicated game logic
        reset_resp = await self.reset(task_id=task_id, seed=seed, kwargs=None)
        episode_id = reset_resp.get("episode_id")
        resolved_task_id = (reset_resp.get("info") or {}).get("task_id")

        conversation = [{"role": "user", "content": reset_resp["observation"]}]
        usage: Optional[Dict[str, Any]] = None
        success = False

        # Upper bound: allow some extra turns for invalid parses without consuming attempts
        for _ in range(self.MAX_ATTEMPTS + 5):
            content, usage = await llm_chat(
                messages=conversation,
                model=model,
                base_url=base_url,
                api_key=current_api_key,
                timeout=timeout,
                temperature=temperature,
                seed=seed,
                stream=False,
            )
            action_text = content or ""
            conversation.append({"role": "assistant", "content": action_text})

            step_resp = await self.step(action=action_text, episode_id=episode_id, kwargs=None)
            conversation.append({"role": "user", "content": step_resp["observation"]})

            if step_resp.get("done"):
                success = float(step_resp.get("reward", 0.0)) > 0.0
                break

        result = {
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
        
        return result