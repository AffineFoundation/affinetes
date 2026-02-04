"""MiniSWE Fixer Agent - wraps minisweagent library"""

import os
import re
import sys
import asyncio
import tempfile
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from .base import BaseFixerAgent, FixerConfig, FixerResult
from utils import SANITIZE_GIT_SCRIPT


def _strip_thinking_tags(content: str) -> str:
    """Strip <think>...</think> tags from model output.

    Some models (e.g., DeepSeek R1) return thinking content wrapped in these tags
    when using extended thinking mode via OpenAI-compatible API. This can interfere
    with action parsing if the thinking content contains code blocks.
    """
    if "</think>" in content:
        # Take only the content after the last </think> tag
        content = content.split("</think>")[-1].strip()
    return content

# Suppress verbose logging from minisweagent
logging.getLogger("minisweagent").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class MiniSWEFixerAgent(BaseFixerAgent):
    """Fixer agent using the minisweagent library"""

    def __init__(self, config: FixerConfig):
        super().__init__(config)
        self._env = None
        self._agent = None
        self._container_name = None

    def _sanitize_git_history(self) -> bool:
        """Remove git history to prevent cheating by looking at past commits."""
        if not self._env:
            return False

        try:
            result = self._env.execute(SANITIZE_GIT_SCRIPT, timeout=60)
            result_str = result.get("stdout", "") if isinstance(result, dict) else str(result or "")
            print(f"[SWE-SYNTH] Git history sanitization: {result_str[:200] if result_str else 'done'}")
            return True
        except Exception as e:
            print(f"[SWE-SYNTH] Warning: Failed to sanitize git history: {e}")
            return False

    def _apply_patches(
        self,
        gold_patch: Optional[str],
        bug_patch: Optional[str],
    ) -> bool:
        """Apply gold_patch and bug_patch inside container using docker cp.

        Note: SWE-bench Docker images are already at base_commit, no need to reset.

        Returns:
            True if patches applied successfully, False otherwise.

        Raises:
            RuntimeError: If patch application fails critically.
        """
        if not self._env or not self._container_name:
            return False

        patch_names = ["gold_patch", "bug_patch"]

        # Apply patches using docker cp to avoid argument length limits
        for idx, patch in enumerate([gold_patch, bug_patch]):
            if patch:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as f:
                    f.write(patch)
                    temp_path = f.name
                try:
                    subprocess.run(
                        ["docker", "cp", temp_path, f"{self._container_name}:/tmp/patch_{idx}.diff"],
                        check=True, capture_output=True, timeout=30
                    )
                    # Apply patch and capture result (no || true - we want to see failures)
                    result = self._env.execute(
                        f"cd /app && git apply -v /tmp/patch_{idx}.diff 2>&1",
                        timeout=120
                    )
                    # Handle result (may be dict or string depending on environment)
                    if isinstance(result, dict):
                        result_str = result.get("stdout", "") or result.get("output", "") or str(result)
                    else:
                        result_str = str(result) if result else ""
                    # Check for apply errors
                    result_lower = result_str.lower()
                    if "error" in result_lower or "rejected" in result_lower or "patch failed" in result_lower:
                        print(f"[SWE-SYNTH] Warning: {patch_names[idx]} may have failed to apply: {result_str[:500]}")
                    else:
                        print(f"[SWE-SYNTH] {patch_names[idx]} applied successfully")
                finally:
                    os.unlink(temp_path)

        # Sanitize git history to prevent cheating via git log/show
        self._sanitize_git_history()

        # Verify code state after patches
        git_status_raw = self._env.execute("cd /app && git status --porcelain", timeout=30)
        git_diff_raw = self._env.execute("cd /app && git diff --stat", timeout=30)
        # Handle result (may be dict or string)
        git_status = git_status_raw.get("stdout", "") if isinstance(git_status_raw, dict) else str(git_status_raw or "")
        git_diff_stat = git_diff_raw.get("stdout", "") if isinstance(git_diff_raw, dict) else str(git_diff_raw or "")
        print(f"[SWE-SYNTH] Post-patch git status: {git_status[:200] if git_status else 'clean'}")
        print(f"[SWE-SYNTH] Post-patch git diff stat: {git_diff_stat[:200] if git_diff_stat else 'no changes'}")

        return True

    async def fix(
        self,
        problem_statement: str,
        docker_image: str,
        repo_path: Optional[str] = None,
        gold_patch: Optional[str] = None,
        bug_patch: Optional[str] = None,
        base_commit: Optional[str] = None,
    ) -> FixerResult:
        """Run MiniSWE agent to fix the bug"""
        try:
            from minisweagent.agents.default import DefaultAgent, FormatError
            from minisweagent.environments.docker import DockerEnvironment
            from minisweagent.models.litellm_model import LitellmModel

            # Custom agent that strips thinking tags before parsing actions
            class ThinkingAwareAgent(DefaultAgent):
                """DefaultAgent with support for models that output <think> tags."""

                def parse_action(self, response: dict) -> dict:
                    """Parse action, stripping thinking tags first."""
                    content = _strip_thinking_tags(response["content"])
                    actions = re.findall(self.config.action_regex, content, re.DOTALL)
                    if len(actions) == 1:
                        return {"action": actions[0].strip(), **response}
                    raise FormatError(
                        self.render_template(self.config.format_error_template, actions=actions)
                    )

            # Pull image first (must succeed, otherwise fail fast)
            print(f"Pulling image: {docker_image}")
            pull_result = subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True,
                timeout=300,
                text=True,
            )
            if pull_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to pull image {docker_image}: {pull_result.stderr}"
                )

            # Initialize model
            model_name = self.config.model
            if not model_name.startswith(("openai/", "anthropic/", "azure/", "bedrock/", "claude")):
                model_name = f"openai/{model_name}"

            model_kwargs = {"temperature": self.config.temperature}
            if self.config.seed is not None:
                model_kwargs["seed"] = self.config.seed

            is_anthropic = "claude" in model_name or "anthropic/" in model_name
            if is_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
            else:
                if self.config.api_base:
                    model_kwargs["api_base"] = self.config.api_base
                model_kwargs["api_key"] = self.config.api_key

            # Clear litellm's cached HTTP clients to prevent "client has been closed" errors
            # This happens when cached httpx clients are reused across async/sync boundaries
            import litellm
            if hasattr(litellm.in_memory_llm_clients_cache, 'flush_cache'):
                litellm.in_memory_llm_clients_cache.flush_cache()
            elif hasattr(litellm.in_memory_llm_clients_cache, 'cache_dict'):
                litellm.in_memory_llm_clients_cache.cache_dict.clear()

            model_obj = LitellmModel(
                model_name=model_name,
                model_kwargs=model_kwargs,
                cost_tracking="ignore_errors",
            )

            # Initialize Docker environment
            self._container_name = f"swe-synth-fixer-{int(time.time() * 1000)}"
            self._env = DockerEnvironment(
                image=docker_image,
                cwd=self.config.cwd,
                timeout=self.config.timeout,
                executable="docker",
                run_args=["--rm", "--entrypoint", "", "--name", self._container_name],
                container_timeout=str(self.config.timeout),
            )

            # Apply patches (image is already at base_commit)
            if gold_patch or bug_patch:
                self._apply_patches(gold_patch, bug_patch)

            # Load agent config
            config_path = Path(__file__).parent.parent / "config.yaml"
            agent_config = {}
            if config_path.exists():
                with open(config_path, "r") as f:
                    agent_config = yaml.safe_load(f).get("agent", {}).copy()

            agent_config["step_limit"] = self.config.max_iterations
            agent_config["cost_limit"] = self.config.cost_limit

            # Run agent (use ThinkingAwareAgent to handle <think> tags)
            self._agent = ThinkingAwareAgent(model_obj, self._env, **agent_config)
            patch = ""
            error = None

            try:
                loop = asyncio.get_event_loop()
                _, result = await loop.run_in_executor(None, self._agent.run, problem_statement)
                patch = result
            except Exception as e:
                import traceback
                error = traceback.format_exc()
            finally:
                self.cleanup()

            # Extract usage stats
            total_tokens = 0
            clean_conversation = []

            for msg in self._agent.messages:
                if isinstance(msg, dict):
                    extra = msg.get("extra", {})
                    if isinstance(extra, dict):
                        usage = extra.get("usage") or extra.get("response", {}).get("usage")
                        if usage:
                            total_tokens += usage.get("total_tokens", 0)
                    clean_conversation.append({k: v for k, v in msg.items() if k != "extra"})
                else:
                    clean_conversation.append(msg)

            return FixerResult(
                patch=patch or "",
                model_calls=self._agent.model.n_calls if self._agent else 0,
                model_cost=self._agent.model.cost if self._agent else 0.0,
                total_tokens=total_tokens,
                conversation=clean_conversation,
                success=bool(patch) and error is None,
                error=error,
            )

        except Exception as e:
            import traceback
            return FixerResult(patch="", success=False, error=traceback.format_exc())

    def cleanup(self):
        """Clean up Docker environment"""
        if self._env:
            try:
                self._env.cleanup()
            except Exception:
                pass
            self._env = None
        self._container_name = None
