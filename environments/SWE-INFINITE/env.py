"""
SWE-INFINITE Environment

Evaluates coding agents on real GitHub PRs (expansion tasks from the mining pipeline).

Supports two modes:
1. evaluate() - One-shot evaluation with Codex agent
2. reset/step/stop - OpenEnv training interface for external control

Flow:
    instance_id → load from R2 → agent implements change → verify via tests
"""

import base64
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
from jinja2 import StrictUndefined, Template

from cache import TwoLevelCache
from agents import (
    CodexAgent, CodexConfig,
    MiniSWEAgent, MiniSWEConfig,
    SUPPORTED_AGENTS, select_agent,
)
from affinetes.core.openenv import OpenEnvResponse
from utils import (
    SANITIZE_GIT_SCRIPT,
    NORMALIZE_TIMESTAMPS_SCRIPT,
    DIFF_EXTENSIONS,
    parse_test_output,
)

# Timeout constants (seconds)
DOCKER_PULL_TIMEOUT = 300
VERIFY_TIMEOUT = 1800

# OpenEnv constants
DEFAULT_STEP_LIMIT = 100
DEFAULT_COMMAND_TIMEOUT = 300
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
ACTION_REGEX = r"```bash\s*\n(.*?)\n```"

# Output markers for verification
STDOUT_BEGIN = "===SWE_INFINITE_STDOUT_BEGIN==="
STDOUT_END = "===SWE_INFINITE_STDOUT_END==="
STDERR_BEGIN = "===SWE_INFINITE_STDERR_BEGIN==="
STDERR_END = "===SWE_INFINITE_STDERR_END==="
# Legacy combined marker (kept for backward compat)
OUTPUT_BEGIN = "===SWE_INFINITE_OUTPUT_BEGIN==="
OUTPUT_END = "===SWE_INFINITE_OUTPUT_END==="



@dataclass
class EpisodeState:
    """Training episode state."""
    episode_id: str
    task_id: str
    seed: int
    task: Dict[str, Any]

    # Docker environment
    container_id: str
    docker_image: str

    # Agent state
    messages: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    truncated: bool = False
    submitted_patch: str = ""

    # Configuration
    step_limit: int = DEFAULT_STEP_LIMIT
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    start_time: float = field(default_factory=time.time)


class InfiniteActor:
    """SWE-INFINITE evaluation actor.

    Evaluates coding agents on real GitHub PRs loaded from R2.
    Tasks are produced by the affine-swe-infinite mining pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "/tmp/swe-infinite-cache",
        # R2 public bucket
        r2_base_url: Optional[str] = None,
        r2_prefix: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

        # Authenticate with Docker Hub to avoid pull rate limits
        self._setup_docker_auth()

        # Initialize two-level cache (L1 local + L2 R2 public HTTP)
        self.cache = TwoLevelCache(
            local_cache_dir=cache_dir,
            r2_base_url=r2_base_url,
            r2_prefix=r2_prefix,
        )

        # Cleanup stale resources from previous runs
        self._cleanup_stale_containers()
        self._start_periodic_cleanup()

        # OpenEnv: episode states
        self._episodes: Dict[str, EpisodeState] = {}

        # OpenEnv: agent config from config.yaml
        self._agent_config = self._load_agent_config()

    # ===== Infrastructure =====

    def _setup_docker_auth(self) -> None:
        """Log in to Docker Hub using env vars."""
        username = os.getenv("DOCKER_HUB_USERNAME")
        token = os.getenv("DOCKER_HUB_TOKEN")
        if not username or not token:
            print("[SWE-INFINITE] DOCKER_HUB_USERNAME/DOCKER_HUB_TOKEN not set, skipping docker login")
            return
        result = subprocess.run(
            ["docker", "login", "-u", username, "--password-stdin"],
            input=token, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"[SWE-INFINITE] Docker Hub login succeeded for {username}")
        else:
            print(f"[SWE-INFINITE] Warning: Docker Hub login failed: {result.stderr.strip()}")

    def _cleanup_stale_containers(self, min_age_minutes: int = 2):
        """Clean up stale containers from previous runs."""
        try:
            from datetime import datetime, timezone

            prefixes = ["swe-infinite-codex-", "swe-infinite-openenv-", "swe-infinite-verify-"]
            now = datetime.now(timezone.utc)
            cleaned = 0

            for prefix in prefixes:
                result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name={prefix}",
                     "--format", "{{.ID}} {{.CreatedAt}}"],
                    capture_output=True, text=True, timeout=30,
                )
                if not result.stdout.strip():
                    continue

                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        continue
                    cid, created_str = parts[0], parts[1]
                    try:
                        created_str = created_str.rsplit(' ', 1)[0] if 'UTC' in created_str else created_str
                        created = datetime.strptime(created_str.strip(), "%Y-%m-%d %H:%M:%S %z")
                        if (now - created).total_seconds() / 60 >= min_age_minutes:
                            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=30)
                            cleaned += 1
                    except (ValueError, TypeError):
                        continue

            if cleaned > 0:
                print(f"[SWE-INFINITE] Cleaned up {cleaned} stale containers")
        except Exception as e:
            print(f"[SWE-INFINITE] Warning: Failed to cleanup stale containers: {e}")

    def _start_periodic_cleanup(self, interval_hours: float = 6):
        """Start background thread for periodic resource cleanup."""
        import threading

        def _cleanup_loop():
            while True:
                time.sleep(interval_hours * 3600)
                try:
                    self._cleanup_stale_containers()
                except Exception as e:
                    print(f"[SWE-INFINITE] Periodic cleanup error: {e}")

        t = threading.Thread(target=_cleanup_loop, daemon=True, name="swe-infinite-cleanup")
        t.start()

    # ===== Task Loading =====

    def _load_task(self, task_id) -> Dict[str, Any]:
        """Load task from R2 by task_id (int or string).

        Args:
            task_id: Numeric task ID (e.g. 1) or instance_id string

        Returns:
            SWESynthTask dict

        Raises:
            ValueError: If task not found
        """
        key = str(task_id)
        task = self.cache.load(key)
        if task is None:
            raise ValueError(
                f"Task '{key}' not found in cache. "
                f"Ensure the task was generated by the mining pipeline."
            )
        # Inject numeric task_id for agent selection (alternating by parity)
        try:
            task["_task_id"] = int(key)
        except (ValueError, TypeError):
            pass
        return task

    # ===== Verification =====

    def _verify(
        self,
        task: Dict[str, Any],
        fix_patch: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Verify if the agent's patch passes the required tests.

        The Docker image already has test_patch baked in (base_commit + test_patch).
        We apply augmented_test_patch (if present), then fix_patch, then run tests.

        Returns:
            (score, test_stats) where score is 1.0 if all required tests pass.
        """
        if not fix_patch or not fix_patch.strip():
            return 0.0, {"error": "no patch"}

        docker_image = task["dockerhub_tag"]
        test_command = task.get("test_command", "pytest -v --tb=no")
        fail_to_pass = task.get("fail_to_pass", [])
        pass_to_pass = task.get("pass_to_pass", [])
        test_patch = task.get("test_patch", "")
        augmented_test_patch = task.get("augmented_test_patch", "")

        # Ensure lists (may be stored as JSON strings)
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except (json.JSONDecodeError, TypeError):
                fail_to_pass = []
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except (json.JSONDecodeError, TypeError):
                pass_to_pass = []

        f2p = set(fail_to_pass)
        p2p = set(pass_to_pass)
        all_required = f2p | p2p
        if not all_required:
            return 0.0, {"error": "No required tests defined"}

        try:
            # Build verification entry script
            # Apply order: test_patch -> augmented_test_patch -> fix_patch
            # test_patch contains the augmented/enhanced tests not baked into the image.
            apply_steps = []
            if test_patch and test_patch.strip():
                apply_steps.append(
                    'git apply --recount --whitespace=fix /workspace/test_patch.diff 2>&1 || echo "TEST_PATCH_APPLY_FAILED"'
                )
            if augmented_test_patch and augmented_test_patch.strip():
                apply_steps.append(
                    'git apply --recount --whitespace=fix /workspace/augmented_test.diff 2>&1 || echo "AUGMENTED_PATCH_APPLY_FAILED"'
                )
            apply_cmds = "\n".join(apply_steps)

            entryscript = f"""
cd /app
{apply_cmds}
git apply --recount --whitespace=fix /workspace/fix_patch.diff 2>&1 || {{ echo "PATCH_APPLY_FAILED"; }}
{test_command} > /workspace/stdout.log 2> /workspace/stderr.log || true
echo "{STDOUT_BEGIN}"
cat /workspace/stdout.log
echo "{STDOUT_END}"
echo "{STDERR_BEGIN}"
cat /workspace/stderr.log
echo "{STDERR_END}"
"""

            fix_patch_b64 = base64.b64encode(fix_patch.encode('utf-8')).decode('ascii')
            entryscript_b64 = base64.b64encode(entryscript.encode('utf-8')).decode('ascii')

            test_patch_lines = ""
            if test_patch and test_patch.strip():
                tp_b64 = base64.b64encode(test_patch.encode('utf-8')).decode('ascii')
                test_patch_lines = f'echo "{tp_b64}" | base64 -d > /workspace/test_patch.diff'
            augmented_lines = ""
            if augmented_test_patch and augmented_test_patch.strip():
                aug_b64 = base64.b64encode(augmented_test_patch.encode('utf-8')).decode('ascii')
                augmented_lines = f'echo "{aug_b64}" | base64 -d > /workspace/augmented_test.diff'

            full_script = f"""#!/bin/bash
mkdir -p /workspace
{test_patch_lines}
{augmented_lines}
echo "{fix_patch_b64}" | base64 -d > /workspace/fix_patch.diff
echo "{entryscript_b64}" | base64 -d > /workspace/entryscript.sh
chmod +x /workspace/entryscript.sh
bash /workspace/entryscript.sh
"""

            # Pull image and run verification container
            print(f"[SWE-INFINITE] Pulling image: {docker_image}")
            subprocess.run(
                ["docker", "pull", docker_image],
                check=False, capture_output=True, timeout=DOCKER_PULL_TIMEOUT,
            )

            verify_ctr = f"swe-infinite-verify-{uuid.uuid4().hex[:12]}"
            print(f"[SWE-INFINITE] Running verification (timeout={VERIFY_TIMEOUT}s)...")
            try:
                result = subprocess.run(
                    ["docker", "run", "--rm", "--init", "-i",
                     "--name", verify_ctr,
                     "--stop-timeout", "10",
                     "--memory", "4g",
                     "--network=host",
                     "--entrypoint", "/bin/bash",
                     docker_image],
                    input=full_script,
                    capture_output=True,
                    timeout=VERIFY_TIMEOUT,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", verify_ctr], capture_output=True, timeout=15)
                return 0.0, {"error": "timeout"}

            print("[SWE-INFINITE] Verification container completed.")
            container_stdout = result.stdout

            if "PATCH_APPLY_FAILED" in container_stdout:
                # Check specific patch failures first (test_patch / augmented_test_patch)
                if "TEST_PATCH_APPLY_FAILED" in container_stdout:
                    return 0.0, {"error": "test_patch apply failed"}
                if "AUGMENTED_PATCH_APPLY_FAILED" in container_stdout:
                    return 0.0, {"error": "augmented_test_patch apply failed"}
                return 0.0, {"error": "patch apply failed"}

            if STDOUT_BEGIN not in container_stdout or STDERR_BEGIN not in container_stdout:
                return 0.0, {"error": "No output markers", "stderr": result.stderr[:500]}

            test_stdout = container_stdout[
                container_stdout.index(STDOUT_BEGIN) + len(STDOUT_BEGIN):
                container_stdout.index(STDOUT_END)
            ].strip()
            test_stderr = container_stdout[
                container_stdout.index(STDERR_BEGIN) + len(STDERR_BEGIN):
                container_stdout.index(STDERR_END)
            ].strip()

            language = task.get("repo_language", "")
            passed_tests, failed_tests = parse_test_output(test_stdout, test_stderr, language, test_command)

            # Fallback: if no individual tests were parsed, check summary line.
            # Handles non-verbose test runners (e.g. Minitest without -v).
            if not passed_tests and not failed_tests:
                summary_m = re.search(
                    r"(\d+) runs?.*?(\d+) failures?.*?(\d+) errors?", test_stdout + test_stderr
                )
                if summary_m:
                    total = int(summary_m.group(1))
                    failures = int(summary_m.group(2))
                    errors = int(summary_m.group(3))
                    if total > 0 and failures == 0 and errors == 0:
                        passed_tests = all_required.copy()

            f2p_passed = len(f2p & passed_tests)
            all_passed_count = len(all_required & passed_tests)
            all_pass = all_required <= passed_tests

            test_stats = {
                "f2p_result": f"{f2p_passed}/{len(f2p)}",
                "all_result": f"{all_passed_count}/{len(all_required)}",
                "all_passed": all_pass,
            }

            if all_pass:
                return 1.0, test_stats

            test_stats["missing_tests"] = sorted(all_required - passed_tests)
            return 0.0, test_stats

        except subprocess.TimeoutExpired:
            return 0.0, {"error": "timeout"}
        except Exception:
            import traceback
            return 0.0, {"error": traceback.format_exc()}

    # ===== OpenEnv Helper Methods =====

    def _load_agent_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("agent", {})
        return {}

    def _render_template(self, template: str, **kwargs) -> str:
        return Template(template, undefined=StrictUndefined).render(**kwargs)

    def _info(
        self,
        ep: Optional[EpisodeState] = None,
        *,
        error: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "task_id": ep.task_id if ep else None,
            "seed": ep.seed if ep else None,
            "step_count": ep.step_count if ep else 0,
            "instance_id": ep.task_id if ep else None,
        }
        if error:
            info["error"] = error
        return info

    def _start_container(self, docker_image: str, container_name: str) -> str:
        """Start a Docker container and return the container ID."""
        print(f"[SWE-INFINITE] Pulling image: {docker_image}")
        subprocess.run(
            ["docker", "pull", docker_image],
            capture_output=True, timeout=DOCKER_PULL_TIMEOUT,
        )

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-w", "/app",
            "--rm", "--init",
            "--stop-timeout", "10",
            "--memory", "4g",
            "--memory-swap", "6g",
            "--network=host",
            "--entrypoint", "",
            docker_image,
            "sleep", "7200",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        container_id = result.stdout.strip()
        print(f"[SWE-INFINITE] Started container {container_name} ({container_id[:12]})")
        return container_id

    def _execute_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int = DEFAULT_COMMAND_TIMEOUT,
    ) -> Dict[str, Any]:
        """Execute a command in a Docker container."""
        cmd = ["docker", "exec", "-w", "/app", container_id, "bash", "-lc", command]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            return {"output": output, "returncode": -1, "timeout": True}

    def _apply_patch_in_container(self, container_id: str, patch: str, label: str = "augmented test") -> None:
        """Apply a patch inside a running container via base64 pipe."""
        patch_b64 = base64.b64encode(patch.encode('utf-8')).decode('ascii')
        cmd = f"echo '{patch_b64}' | base64 -d | git apply -v --allow-empty"
        result = self._execute_in_container(container_id, cmd, timeout=60)
        print(f"[SWE-INFINITE] Applied {label} patch: {result.get('output', '')[:200]}")

    def _sanitize_git_in_container(self, container_id: str) -> None:
        """Sanitize git history to prevent cheating."""
        result = self._execute_in_container(container_id, SANITIZE_GIT_SCRIPT, timeout=60)
        print(f"[SWE-INFINITE] Git sanitized: {result.get('output', '')[:200]}")

        # Warm up login shell before normalizing timestamps
        self._execute_in_container(container_id, "true", timeout=60)

        self._execute_in_container(container_id, NORMALIZE_TIMESTAMPS_SCRIPT, timeout=120)
        print("[SWE-INFINITE] Timestamps normalized")

    def _parse_action(self, action: str) -> Optional[str]:
        """Parse bash command from action string."""
        if "</think>" in action:
            action = action.split("</think>")[-1].strip()
        actions = re.findall(ACTION_REGEX, action, re.DOTALL)
        if len(actions) == 1:
            return actions[0].strip()
        return None

    def _extract_diff_from_container(self, container_id: str) -> str:
        """Extract code diff from container."""
        result = self._execute_in_container(
            container_id,
            f"cd /app && git add -A && git diff --cached -- {DIFF_EXTENSIONS}",
            timeout=60,
        )
        diff = result.get("output", "").lstrip()
        return diff.rstrip('\n') + '\n' if diff else ""

    def _stop_container(self, container_id: str):
        """Stop and remove a Docker container asynchronously."""
        try:
            subprocess.Popen(
                f"(timeout 60 docker stop {container_id} || docker rm -f {container_id}) >/dev/null 2>&1 &",
                shell=True,
            )
        except Exception as e:
            print(f"[SWE-INFINITE] Warning: Failed to stop container: {e}")

    # ===== OpenEnv Training Interface =====

    async def reset(
        self,
        task_id,
        seed: Optional[int] = None,
        step_limit: int = DEFAULT_STEP_LIMIT,
        command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    ) -> OpenEnvResponse:
        """Reset environment and start a new episode.

        Args:
            task_id: Instance ID string (e.g. "boto__boto3-4503")
            seed: Random seed
            step_limit: Maximum steps before truncation
            command_timeout: Timeout per command execution
        """
        import random
        resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        try:
            task = self._load_task(task_id)
            docker_image = task["dockerhub_tag"]
            problem_statement = task["problem_statement"]

            print(f"[SWE-INFINITE] Loaded task: {task_id}")

            episode_id = uuid.uuid4().hex
            container_name = f"swe-infinite-openenv-{episode_id[:8]}"

            # Start container (image has base_commit only)
            container_id = self._start_container(docker_image, container_name)

            # Sanitize git history
            self._sanitize_git_in_container(container_id)

            # Create episode state
            ep = EpisodeState(
                episode_id=episode_id,
                task_id=task_id,
                seed=resolved_seed,
                task=task,
                container_id=container_id,
                docker_image=docker_image,
                step_limit=step_limit,
                command_timeout=command_timeout,
            )

            # Render initial prompts
            system_template = self._agent_config.get("system_template", "")
            instance_template = self._agent_config.get("instance_template", "")
            system_msg = self._render_template(system_template)
            instance_msg = self._render_template(instance_template, task=problem_statement)

            ep.messages.append({"role": "system", "content": system_msg, "timestamp": time.time()})
            ep.messages.append({"role": "user", "content": instance_msg, "timestamp": time.time()})

            self._episodes[episode_id] = ep
            observation = f"{system_msg}\n\n{instance_msg}"

            return OpenEnvResponse(
                observation=observation,
                episode_id=episode_id,
                info=self._info(ep),
            )

        except ValueError as e:
            return OpenEnvResponse(
                observation=f"Error: {str(e)}",
                done=True, truncated=True,
                info=self._info(None, error={"type": "task_not_found", "message": str(e), "retryable": False}),
            )
        except Exception as e:
            import traceback
            return OpenEnvResponse(
                observation=f"Error initializing episode: {str(e)}",
                done=True, truncated=True,
                info=self._info(None, error={"type": "init_error", "message": traceback.format_exc(), "retryable": True}),
            )

    async def step(
        self,
        action: str,
        episode_id: Optional[str] = None,
    ) -> OpenEnvResponse:
        """Execute an action in the current episode."""
        if not episode_id:
            return OpenEnvResponse(
                observation="No episode_id provided. Call reset() first.",
                done=True, truncated=True,
                info=self._info(None, error={"type": "no_episode_id", "retryable": True}),
            )

        ep = self._episodes.get(episode_id)
        if not ep:
            return OpenEnvResponse(
                observation=f"Episode {episode_id} not found. Call reset() first.",
                episode_id=episode_id, done=True, truncated=True,
                info=self._info(None, error={"type": "episode_not_found", "retryable": True}),
            )

        if ep.done:
            return OpenEnvResponse(
                observation="Episode already finished. Call reset() to start a new one.",
                episode_id=episode_id, done=True,
                info=self._info(ep, error={"type": "episode_done", "retryable": True}),
            )

        # Check step limit
        if ep.step_limit > 0 and ep.step_count >= ep.step_limit:
            ep.done = True
            ep.truncated = True
            return OpenEnvResponse(
                observation=f"Step limit ({ep.step_limit}) exceeded.",
                episode_id=episode_id, done=True, truncated=True,
                info=self._info(ep, error={"type": "step_limit_exceeded", "retryable": False}),
            )

        ep.messages.append({"role": "assistant", "content": action, "timestamp": time.time()})
        ep.step_count += 1

        # Parse action to extract bash command
        bash_cmd = self._parse_action(action)
        if bash_cmd is None:
            format_error_template = self._agent_config.get(
                "format_error_template",
                "Please always provide EXACTLY ONE action in triple backticks.",
            )
            actions = re.findall(ACTION_REGEX, action, re.DOTALL)
            error_msg = self._render_template(format_error_template, actions=actions)
            ep.messages.append({"role": "user", "content": error_msg, "timestamp": time.time()})
            return OpenEnvResponse(
                observation=error_msg,
                episode_id=episode_id,
                info=self._info(ep, error={"type": "format_error", "retryable": True}),
            )

        # Execute command
        try:
            output = self._execute_in_container(ep.container_id, bash_cmd, timeout=ep.command_timeout)
        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            ep.messages.append({"role": "user", "content": error_msg, "timestamp": time.time()})
            return OpenEnvResponse(
                observation=error_msg,
                episode_id=episode_id,
                info=self._info(ep, error={"type": "execution_error", "retryable": True}),
            )

        # Handle timeout
        if output.get("timeout"):
            timeout_msg = "The last command timed out. Please try another command."
            ep.messages.append({"role": "user", "content": timeout_msg, "timestamp": time.time()})
            return OpenEnvResponse(
                observation=timeout_msg,
                episode_id=episode_id,
                info=self._info(ep, error={"type": "command_timeout", "retryable": True}),
            )

        # Check for submission
        output_text = output.get("output", "")
        lines = output_text.lstrip().splitlines(keepends=True)

        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", SUBMIT_MARKER]:
            if output.get("returncode", 0) != 0:
                pass  # Let agent see error and retry
            else:
                ep.done = True
                ep.submitted_patch = "".join(lines[1:])

                fix_patch = self._extract_diff_from_container(ep.container_id)
                score, test_stats = self._verify(ep.task, fix_patch)

                final_observation = (
                    f"Submitted.\n\nFix patch:\n"
                    f"{fix_patch[:2000]}{'...(truncated)' if len(fix_patch) > 2000 else ''}"
                )

                info = self._info(ep)
                info["test_stats"] = test_stats
                info["fix_patch"] = fix_patch
                info["conversation"] = ep.messages

                return OpenEnvResponse(
                    observation=final_observation,
                    episode_id=episode_id,
                    reward=score,
                    done=True,
                    info=info,
                )

        # Normal step - format observation
        action_observation_template = self._agent_config.get(
            "action_observation_template",
            "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>",
        )
        observation = self._render_template(action_observation_template, output=output)
        ep.messages.append({"role": "user", "content": observation, "timestamp": time.time()})

        return OpenEnvResponse(
            observation=observation,
            episode_id=episode_id,
            info=self._info(ep),
        )

    async def stop(
        self,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop and cleanup an episode."""
        if not episode_id:
            return {"status": "ok", "stopped": False, "message": "No episode_id provided"}

        ep = self._episodes.pop(episode_id, None)
        if not ep:
            return {"status": "ok", "stopped": False, "message": f"Episode {episode_id} not found"}

        self._stop_container(ep.container_id)
        return {
            "status": "ok",
            "stopped": True,
            "episode_id": episode_id,
            "step_count": ep.step_count,
            "done": ep.done,
        }

    # ===== One-shot Evaluation Interface =====

    async def evaluate(
        self,
        task_id,
        model: str = "affine/Kimi-K2.5",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        timeout: int = 1800,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        agent: str = "",
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate an agent on a real PR task.

        Args:
            task_id: Numeric task ID (e.g. 1) or instance_id string
            model: Model name
            base_url: API base URL
            api_key: API key (uses env var CHUTES_API_KEY if not provided)
            timeout: Timeout for agent execution
            temperature: Model temperature
            seed: Random seed for LLM inference
            agent: Agent type — "miniswe" or "codex". Empty = auto-select from task metadata.
            max_iterations: Max agent iterations (miniswe only)
        """
        start = time.time()

        eval_api_key = api_key or self.api_key
        if not eval_api_key:
            raise ValueError("api_key required (pass to evaluate() or set CHUTES_API_KEY env var)")

        # Load task
        task = self._load_task(task_id)
        instance_id = task["instance_id"]
        docker_image = task["dockerhub_tag"]
        problem_statement = task["problem_statement"]

        # Select agent (explicit override > task metadata > default)
        agent = select_agent(task, override=agent)
        print(f"[SWE-INFINITE] Loaded task: {instance_id} (agent={agent})")

        # Create and run agent (no test patches — agent should not see test cases)
        solve_kwargs = dict(
            problem_statement=problem_statement,
            docker_image=docker_image,
            repo=task.get("repo", ""),
            language=task.get("repo_language", ""),
            test_command=task.get("test_command", ""),
            fail_to_pass=task.get("fail_to_pass"),
        )

        if agent == "codex":
            config = CodexConfig(
                model=model, api_base=base_url, api_key=eval_api_key, timeout=timeout,
            )
            agent_obj = CodexAgent(config)
        else:
            config = MiniSWEConfig(
                model=model, api_base=base_url, api_key=eval_api_key,
                temperature=temperature, timeout=timeout, seed=seed,
                max_iterations=max_iterations,
            )
            agent_obj = MiniSWEAgent(config)

        try:
            agent_result = await agent_obj.solve(**solve_kwargs)
            fix_patch = agent_result.patch
            conversation = agent_result.conversation or []
            model_cost = getattr(agent_result, "model_cost", 0.0)
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": agent_result.total_tokens,
            }
        finally:
            agent_obj.cleanup()

        # Verify fix
        print("[SWE-INFINITE] Verifying fix...")
        if not fix_patch or not fix_patch.strip():
            if agent_result.error:
                print(f"[SWE-INFINITE] Agent error:\n{agent_result.error}")
                error_msg = agent_result.error.lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    error_type = "agent_timeout"
                elif error_msg.startswith("api_error") or any(kw in error_msg for kw in (
                    "api", "authentication", "connection", "network",
                    "404", "401", "403", "no matching chute", "reconnecting",
                )):
                    error_type = "api_error"
                elif "docker" in error_msg or "container" in error_msg:
                    error_type = "docker_error"
                else:
                    error_type = "agent_error"
                if error_type in ("api_error", "docker_error", "agent_timeout"):
                    raise RuntimeError(f"{error_type}: {agent_result.error}")
                test_stats = {"error": agent_result.error, "error_type": error_type}
            else:
                print("[SWE-INFINITE] No patch generated")
                test_stats = {"failure_reason": "no_patch_generated"}
            score = 0.0
        else:
            score, test_stats = self._verify(task, fix_patch)

        return {
            "task_name": "swe-infinite",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "task_id": task_id,
                "task_type": "swe-infinite",
                "agent_type": agent,
                "instance_id": instance_id,
                "repo": task.get("repo", ""),
                "repo_language": task.get("repo_language", ""),
                "problem_statement": problem_statement,
                "fix_patch": fix_patch or "",
                "conversation": conversation,
                "model_calls": agent_result.model_calls,
                "model_cost": model_cost,
                "total_tokens": agent_result.total_tokens,
                "test_stats": test_stats,
                "usage": usage,
            },
        }

    async def verify(self, task_id, fix_patch: str) -> Dict[str, Any]:
        """Verify a patch against a task's test suite (for testing/debugging).

        Args:
            task_id: Numeric task ID or instance_id string
            fix_patch: The patch to verify (unified diff)
        """
        task = self._load_task(task_id)
        score, test_stats = self._verify(task, fix_patch)
        return {
            "score": score,
            "success": score > 0.0,
            "task_id": task_id,
            "repo_language": task.get("repo_language", ""),
            "test_stats": test_stats,
        }


# Framework requires class named 'Actor'
Actor = InfiniteActor
