"""OpenEnv client helpers (SDK-side).

Implements 2A:
  oe = env.openenv()
  sess = await oe.reset(...)
  resp = await sess.step("...")

This hides episode_id plumbing from user code while keeping the underlying
env methods unchanged (still dynamic dispatch via EnvironmentWrapper).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _extract_episode_id(resp: Dict[str, Any]) -> str:
    """Best-effort extraction of episode_id from an OpenEnvResponse dict."""
    # Preferred (new): top-level episode_id on OpenEnvResponse
    if resp.get("episode_id"):
        return str(resp["episode_id"])
    info = resp.get("info") or {}
    # Legacy shapes (kept for backward compatibility)
    if isinstance(info, dict):
        if info.get("episode_id"):
            return str(info["episode_id"])
        ident = info.get("ident")
        if isinstance(ident, dict) and ident.get("episode_id"):
            return str(ident["episode_id"])
    raise ValueError(f"Cannot find episode_id in reset response info: keys={list(info.keys()) if isinstance(info, dict) else type(info)}")


@dataclass
class OpenEnvSession:
    """Episode-bound OpenEnv session (auto-injects episode_id into step calls)."""

    _env: Any
    episode_id: str
    last: Dict[str, Any]
    _stopped: bool = False

    @property
    def observation(self) -> str:
        return str(self.last.get("observation", ""))

    async def state(self) -> Dict[str, Any]:
        """Fetch current state/observation (no transition)."""
        resp = await self._env.state(request={"episode_id": self.episode_id})
        if not isinstance(resp, dict):
            raise TypeError(f"OpenEnv state expected dict response, got {type(resp)}")
        # state() should return an OpenEnvResponse-like dict; keep last if present
        if "observation" in resp:
            self.last = resp
        return resp

    async def stop(self) -> Any:
        """Terminate the episode on the environment (best-effort, idempotent)."""
        if self._stopped:
            return None
        self._stopped = True
        if hasattr(self._env, "stop"):
            return await self._env.stop(request={"episode_id": self.episode_id})
        # fallback (older envs)
        if hasattr(self._env, "close"):
            return await self._env.close()
        return None

    async def step(
        self,
        action: str,
        *,
        kwargs: Optional[Dict[str, Any]] = None,
        request: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call env.step while automatically providing episode_id."""
        if request is None:
            request = {"action": action, "episode_id": self.episode_id, "kwargs": kwargs}
            resp = await self._env.step(request=request)
        else:
            # Ensure episode_id is present unless user intentionally overrides it.
            request = dict(request)
            request.setdefault("episode_id", self.episode_id)
            resp = await self._env.step(request=request)

        if not isinstance(resp, dict):
            raise TypeError(f"OpenEnv step expected dict response, got {type(resp)}")
        self.last = resp
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()
        return False

    def __del__(self):
        # Best-effort auto-stop when session object is garbage collected.
        # Note: __del__ is not guaranteed to run; prefer `async with` or explicit stop().
        if getattr(self, "_stopped", True):
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.stop())
        except Exception:
            # During interpreter shutdown or if no loop exists, skip silently.
            pass


class OpenEnvClient:
    """OpenEnv client wrapper around an EnvironmentWrapper instance."""

    def __init__(self, env: Any):
        self._env = env

    async def reset(
        self,
        *,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        request: Optional[Dict[str, Any]] = None,
    ) -> OpenEnvSession:
        """Call env.reset and return an OpenEnvSession bound to episode_id."""
        if request is None:
            request = {"task_id": task_id, "seed": seed, "kwargs": kwargs}
        resp = await self._env.reset(request=request)
        if not isinstance(resp, dict):
            raise TypeError(f"OpenEnv reset expected dict response, got {type(resp)}")
        episode_id = _extract_episode_id(resp)
        return OpenEnvSession(_env=self._env, episode_id=episode_id, last=resp)


