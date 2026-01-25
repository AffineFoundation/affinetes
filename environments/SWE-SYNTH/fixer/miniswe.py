"""MiniSWE Fixer Agent - wraps minisweagent library"""

import os
import asyncio
import base64
import time
from pathlib import Path
from typing import Optional

import yaml

from .base import BaseFixerAgent, FixerConfig, FixerResult


class MiniSWEFixerAgent(BaseFixerAgent):
    """Fixer agent using the minisweagent library"""

    def __init__(self, config: FixerConfig):
        super().__init__(config)
        self._env = None
        self._agent = None

    def _apply_patches(
        self,
        base_commit: Optional[str],
        gold_patch: Optional[str],
        bug_patch: Optional[str],
    ) -> bool:
        """Apply gold_patch and bug_patch inside container"""
        if not self._env:
            return False

        commands = ["cd /app"]

        if base_commit:
            commands.append(f"git reset --hard {base_commit}")
            commands.append(f"git checkout {base_commit}")

        for patch in [gold_patch, bug_patch]:
            if patch:
                patch_b64 = base64.b64encode(patch.encode('utf-8')).decode('ascii')
                commands.append(f'echo "{patch_b64}" | base64 -d > /tmp/patch.diff')
                commands.append("git apply -v /tmp/patch.diff || true")

        self._env.execute(" && ".join(commands), timeout=60)
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
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.env import DockerEnvironment
            from minisweagent.models import LitellmModel

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

            model_obj = LitellmModel(
                model_name=model_name,
                model_kwargs=model_kwargs,
                cost_tracking="ignore_errors",
            )

            # Initialize Docker environment
            container_name = f"swe-synth-fixer-{int(time.time() * 1000)}"
            self._env = DockerEnvironment(
                image=docker_image,
                cwd=self.config.cwd,
                timeout=self.config.timeout,
                executable="docker",
                run_args=["--rm", "--entrypoint", "", "--name", container_name],
                container_timeout=str(self.config.timeout),
            )

            # Apply patches
            if gold_patch or bug_patch:
                self._apply_patches(base_commit, gold_patch, bug_patch)

            # Load agent config
            config_path = Path(__file__).parent.parent / "config.yaml"
            agent_config = {}
            if config_path.exists():
                with open(config_path, "r") as f:
                    agent_config = yaml.safe_load(f).get("agent", {}).copy()

            agent_config["step_limit"] = self.config.max_iterations
            agent_config["cost_limit"] = self.config.cost_limit

            # Run agent
            self._agent = DefaultAgent(model_obj, self._env, **agent_config)
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
