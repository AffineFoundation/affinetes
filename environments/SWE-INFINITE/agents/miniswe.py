"""MiniSWE Agent — uses minisweagent library for multi-turn coding inside Docker."""

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any

import litellm
import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import (
    LitellmModel,
    logger as _litellm_model_logger,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Allow importing from parent directory (SWE-INFINITE/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    SANITIZE_GIT_SCRIPT,
    NORMALIZE_TIMESTAMPS_SCRIPT,
    NETWORK_BLOCKLIST_SCRIPT,
    is_image_prepared,
    ContainerLostError,
    is_blacklisted_command,
    is_container_running,
)
from agents.outcome import AgentOutcome

try:
    from minisweagent.exceptions import FormatError
except ImportError:  # mini-swe-agent 1.x exports it from agents.default
    from minisweagent.agents.default import FormatError


_MODEL_FAILURE_EXCEPTIONS = (
    FormatError,
    litellm.exceptions.ContextWindowExceededError,
)


def outcome_from_exception(exc: BaseException) -> AgentOutcome:
    """Classify typed MiniSWE exceptions without inspecting their messages."""
    if isinstance(exc, _MODEL_FAILURE_EXCEPTIONS):
        return AgentOutcome.MODEL_FAILURE
    return AgentOutcome.INFRA_FAILURE


class FailFastLitellmModel(LitellmModel):
    """LitellmModel that aborts immediately on any 4xx BadRequest.

    LitellmModel's default tenacity decorator only blacklists
    ContextWindowExceededError. sglang / vllm return 400 with messages
    like "Input length (N tokens) exceeds the maximum allowed length"
    that don't match LiteLLM's context-window keyword list, so they
    stay typed as BadRequestError and the default decorator burns
    ~5-6 minutes on 10 doomed retries — slot stays occupied that whole
    time. Adding BadRequestError to the blacklist surfaces 4xx on the
    first try; only 5xx / network errors / KeyboardInterrupt still
    retry. Only LiteLLM's explicit ``ContextWindowExceededError`` is a
    model failure; an untyped BadRequest remains invalid for scoring.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(_litellm_model_logger, logging.WARNING),
        retry=retry_if_not_exception_type((
            litellm.exceptions.UnsupportedParamsError,
            litellm.exceptions.NotFoundError,
            litellm.exceptions.PermissionDeniedError,
            litellm.exceptions.ContextWindowExceededError,
            litellm.exceptions.BadRequestError,
            litellm.exceptions.APIError,
            litellm.exceptions.AuthenticationError,
            KeyboardInterrupt,
        )),
    )
    def _query(self, messages, **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e


def _strip_thinking_tags(content: str) -> str:
    """Strip <think>...</think> tags from model output.

    Some models (e.g., DeepSeek R1) wrap reasoning in these tags,
    which can interfere with action parsing if they contain code blocks.
    """
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


class _BlacklistDockerEnv:
    """Wraps DockerEnvironment to block fingerprinting commands."""

    def __init__(self, env):
        self._env = env

    def execute(self, command, **kwargs):
        if is_blacklisted_command(str(command)):
            print(f"[SWE-INFINITE] Blocked: {str(command)[:200]}")
            return {
                "stdout": "Command not permitted in this environment.",
                "output": "Command not permitted in this environment.",
                "returncode": 1,
            }
        result = self._env.execute(command, **kwargs)
        # Detect container loss: docker daemon errors mean the agent cannot
        # make any further progress, so abort the run and let the caller retry.
        if result.get("returncode", 0) != 0 and not is_container_running(
            self._env.container_id
        ):
            raise ContainerLostError(
                f"docker container unavailable: {self._env.container_id}"
            )
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)


@dataclass
class MiniSWEConfig:
    model: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    max_iterations: int = 100
    cost_limit: float = 3.0
    timeout: int = 300
    seed: Optional[int] = None
    cwd: str = "/app"


@dataclass
class MiniSWEResult:
    patch: str
    model_calls: int = 0
    model_cost: float = 0.0
    total_tokens: int = 0
    conversation: List[Any] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    outcome: AgentOutcome = AgentOutcome.COMPLETED


class MiniSWEAgent:
    """Runs minisweagent inside a task Docker container."""

    def __init__(self, config: MiniSWEConfig):
        self.config = config
        self._env = None
        self._agent = None
        self._container_name: Optional[str] = None

    def _apply_patch(self, patch: str, label: str = "augmented test") -> None:
        """Apply a patch inside the container via base64 pipe."""
        import base64
        patch_b64 = base64.b64encode(patch.encode('utf-8')).decode('ascii')
        result = subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-c",
             f"cd /app && echo '{patch_b64}' | base64 -d | git apply -v --allow-empty"],
            capture_output=True, text=True, timeout=60,
        )
        print(f"[MINISWE] Applied {label} patch: {result.stdout[:200]}")

    def _prepare_container(self) -> None:
        """Apply network blocklist, sanitize git history, normalize timestamps.

        Fast path: if the image baked sanitize+normalize at build time
        (sentinel present), only apply the network blocklist — docker
        always rebuilds /etc/hosts on container start, so that step
        cannot be moved to image build.
        """
        # Network blocklist must always run at container start.
        subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-c", NETWORK_BLOCKLIST_SCRIPT],
            capture_output=True, text=True, timeout=10,
        )
        print("[MINISWE] Network blocklist applied")

        if is_image_prepared(self._container_name):
            print("[MINISWE] Image pre-prepared, skipping sanitize+normalize")
            return

        result = subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-c", SANITIZE_GIT_SCRIPT],
            capture_output=True, text=True, timeout=60,
        )
        print(f"[MINISWE] Git sanitized: {result.stdout[:200]}")

        # Warm up login shell (conda activation, .pyc compilation)
        subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-lc", "true"],
            capture_output=True, text=True, timeout=60,
        )

        subprocess.run(
            ["docker", "exec", self._container_name, "bash", "-lc", NORMALIZE_TIMESTAMPS_SCRIPT],
            capture_output=True, text=True, timeout=120,
        )
        print("[MINISWE] Timestamps normalized")

    async def solve(
        self,
        problem_statement: str,
        docker_image: str,
        repo: str = "",
        language: str = "",
        test_command: str = "",
        fail_to_pass: list = None,
    ) -> MiniSWEResult:
        """Run MiniSWE agent to implement the change."""
        try:
            from minisweagent.environments.docker import DockerEnvironment

            # Subclass to handle <think> tags in model output
            class ThinkingAwareAgent(DefaultAgent):
                def parse_action(self, response: dict) -> dict:
                    content = _strip_thinking_tags(response["content"])
                    actions = re.findall(self.config.action_regex, content, re.DOTALL)
                    if len(actions) == 1:
                        return {"action": actions[0].strip(), **response}
                    raise FormatError(
                        self.render_template(self.config.format_error_template, actions=actions)
                    )

            # 1. Pull image
            print(f"[MINISWE] Pulling image: {docker_image}")
            pull_result = subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True, text=True, timeout=300,
            )
            if pull_result.returncode != 0:
                inspect = subprocess.run(
                    ["docker", "image", "inspect", docker_image],
                    capture_output=True, timeout=10,
                )
                if inspect.returncode != 0:
                    return MiniSWEResult(
                        patch="", success=False,
                        error=f"Failed to pull image: {pull_result.stderr}",
                        outcome=AgentOutcome.INFRA_FAILURE,
                    )
                print(f"[MINISWE] Using local image: {docker_image}")

            # 2. Initialize model
            model_name = self.config.model
            if not model_name.startswith(("openai/", "anthropic/", "azure/", "bedrock/", "claude")):
                model_name = f"openai/{model_name}"

            model_kwargs = {
                "temperature": self.config.temperature,
            }
            if self.config.seed is not None:
                model_kwargs["seed"] = self.config.seed

            is_anthropic = "claude" in model_name or "anthropic/" in model_name
            if is_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
            else:
                if self.config.api_base:
                    model_kwargs["api_base"] = self.config.api_base
                model_kwargs["api_key"] = self.config.api_key

            # Clear litellm cached HTTP clients to prevent stale connection errors
            if hasattr(litellm.in_memory_llm_clients_cache, 'flush_cache'):
                litellm.in_memory_llm_clients_cache.flush_cache()
            elif hasattr(litellm.in_memory_llm_clients_cache, 'cache_dict'):
                litellm.in_memory_llm_clients_cache.cache_dict.clear()

            # Suppress verbose logging
            import logging
            logging.getLogger("minisweagent").setLevel(logging.WARNING)
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)

            model_obj = FailFastLitellmModel(
                model_name=model_name,
                model_kwargs=model_kwargs,
                cost_tracking="ignore_errors",
            )

            # 3. Initialize Docker environment
            self._container_name = f"swe-infinite-miniswe-{int(time.time() * 1000)}"
            self._env = DockerEnvironment(
                image=docker_image,
                cwd=self.config.cwd,
                timeout=self.config.timeout,
                executable="docker",
                run_args=["--rm", "--entrypoint", "", "--memory", "4g", "--name", self._container_name],
                container_timeout=str(self.config.timeout),
            )

            # 4. Sanitize git and normalize timestamps
            self._prepare_container()

            # 5. Load agent config from config.yaml
            config_path = Path(__file__).parent / "config.yaml"
            agent_config = {}
            if config_path.exists():
                with open(config_path, "r") as f:
                    agent_config = yaml.safe_load(f).get("agent", {}).copy()

            agent_config["step_limit"] = self.config.max_iterations
            agent_config["cost_limit"] = self.config.cost_limit

            # 6. Build prompt with task context
            prompt = self._build_prompt(
                problem_statement, repo, language, test_command, fail_to_pass,
            )

            # 7. Run agent
            self._agent = ThinkingAwareAgent(
                model_obj, _BlacklistDockerEnv(self._env), **agent_config,
            )
            patch = ""
            error = None
            outcome = AgentOutcome.COMPLETED

            try:
                loop = asyncio.get_event_loop()
                _, result = await loop.run_in_executor(None, self._agent.run, prompt)
                patch = result
            except Exception as exc:
                import traceback
                error = traceback.format_exc()
                outcome = outcome_from_exception(exc)
            finally:
                self.cleanup()

            # 8. Extract usage stats
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

            return MiniSWEResult(
                patch=patch or "",
                model_calls=self._agent.model.n_calls if self._agent else 0,
                model_cost=self._agent.model.cost if self._agent else 0.0,
                total_tokens=total_tokens,
                conversation=clean_conversation,
                success=bool(patch) and error is None and outcome.valid_for_scoring,
                error=error,
                outcome=outcome,
            )

        except Exception:
            import traceback
            return MiniSWEResult(
                patch="",
                success=False,
                error=traceback.format_exc(),
                outcome=AgentOutcome.INFRA_FAILURE,
            )

    def _build_prompt(
        self,
        problem_statement: str,
        repo: str = "",
        language: str = "",
        test_command: str = "",
        fail_to_pass: list = None,
    ) -> str:
        """Wrap PR description into a structured task prompt."""
        lines = []
        if repo:
            lines.append(f"Repository: {repo}")
        if language:
            lines.append(f"Language: {language}")
        if lines:
            lines.append("")

        lines.append(problem_statement.strip())

        return "\n".join(lines)

    def cleanup(self):
        """Clean up Docker environment."""
        if self._env:
            try:
                self._env.cleanup()
            except Exception:
                pass
            self._env = None
        self._container_name = None
