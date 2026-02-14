"""Codex CLI Fixer Agent - runs OpenAI Codex CLI inside SWE-bench Docker container"""

import json
import os
import subprocess
import tempfile
from typing import Optional, List, Dict, Any, Tuple

from .base import BaseFixerAgent, FixerConfig, FixerResult
from utils import SANITIZE_GIT_SCRIPT, DIFF_EXTENSIONS

DOCKER_PULL_TIMEOUT = 300

# Pre-built static codex binary path inside the SWE-SYNTH container (pinned to 0.94.0).
# Copied into SWE-bench containers via docker cp, avoiding npm install at runtime.
CODEX_STATIC_BINARY = "/usr/local/bin/codex-static"


class CodexFixerAgent(BaseFixerAgent):
    """Fixer agent that runs OpenAI Codex CLI inside SWE-bench Docker container.

    Similar to MiniSWEFixerAgent: starts the SWE-bench container, applies patches,
    then runs codex exec inside it so codex has full access to the project environment
    (dependencies, test suite, etc.).
    """

    def __init__(self, fixer_config: FixerConfig):
        super().__init__(fixer_config)
        self._container_name = None

    def _exec_in_container(
        self,
        cmd: str,
        timeout: int = 60,
        env: Optional[Dict[str, str]] = None,
        stdin_data: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Execute a command inside the Docker container."""
        docker_cmd = ["docker", "exec"]
        if stdin_data is not None:
            docker_cmd.append("-i")
        if env:
            for k, v in env.items():
                docker_cmd.extend(["-e", f"{k}={v}"])
        docker_cmd.extend([self._container_name, "bash", "-c", cmd])
        return subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin_data,
        )

    def _apply_patches(
        self,
        gold_patch: Optional[str],
        bug_patch: Optional[str],
    ) -> bool:
        """Apply gold_patch and bug_patch inside the container using docker cp."""
        for idx, (name, patch) in enumerate([("gold", gold_patch), ("bug", bug_patch)]):
            if patch:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".diff", delete=False
                ) as f:
                    f.write(patch)
                    temp_path = f.name
                try:
                    subprocess.run(
                        ["docker", "cp", temp_path,
                         f"{self._container_name}:/tmp/patch_{idx}.diff"],
                        check=True, capture_output=True, timeout=30,
                    )
                    result = self._exec_in_container(
                        f"cd /app && git apply -v /tmp/patch_{idx}.diff 2>&1",
                        timeout=120,
                    )
                    if result.returncode != 0:
                        print(f"[CODEX] Warning: {name}_patch may have failed: "
                              f"{result.stdout[:500]}")
                    else:
                        print(f"[CODEX] {name}_patch applied successfully")
                finally:
                    os.unlink(temp_path)
        return True

    def _install_codex(self) -> bool:
        """Copy pre-built codex binary into the SWE-bench container."""
        print("[CODEX] Copying codex binary into container...")
        # docker cp from SWE-SYNTH host into the SWE-bench container
        cp_result = subprocess.run(
            ["docker", "cp", CODEX_STATIC_BINARY,
             f"{self._container_name}:/usr/local/bin/codex"],
            capture_output=True, text=True, timeout=30,
        )
        if cp_result.returncode != 0:
            print(f"[CODEX] Failed to copy codex binary: {cp_result.stderr[:500]}")
            return False
        result = self._exec_in_container("codex --version", timeout=10)
        if result.returncode != 0:
            print(f"[CODEX] Codex binary not working: {result.stderr[:500]}")
            return False
        print(f"[CODEX] Codex ready: {result.stdout.strip()}")
        return True

    def _write_codex_config(self) -> None:
        """Write codex config.toml inside the container.

        Configures a custom provider with wire_api="chat" so codex uses
        /v1/chat/completions instead of /v1/responses (which most
        OpenAI-compatible endpoints don't support).
        """
        config_toml = (
            f'model = {json.dumps(self.config.model)}\n'
            f'model_provider = "chutes"\n'
            f'\n'
            f'[model_providers.chutes]\n'
            f'name = "Chutes"\n'
            f'env_key = "CODEX_API_KEY"\n'
        )
        if self.config.api_base:
            config_toml += f'base_url = {json.dumps(self.config.api_base)}\n'
        config_toml += 'wire_api = "chat"\n'

        self._exec_in_container(
            f"mkdir -p /root/.codex && "
            f"cat > /root/.codex/config.toml << 'TOMLEOF'\n{config_toml}TOMLEOF",
            timeout=10,
        )

    def _parse_codex_json_output(
        self, stdout: str
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Parse JSONL output from codex --experimental-json.

        Captures ALL item.completed events as the full trajectory:
        - agent_message: codex's text messages/reasoning
        - command_execution: commands codex executed and their output
        - file changes, web searches, etc.

        Returns:
            (total_tokens, model_calls, conversation)
        """
        total_input = 0
        total_output = 0
        model_calls = 0
        conversation: List[Dict[str, Any]] = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "turn.completed":
                model_calls += 1
                usage = event.get("usage", {})
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)

            elif event_type == "item.completed":
                # Capture all item types for full trajectory
                item = event.get("item", {})
                conversation.append(item)

        total_tokens = total_input + total_output
        return total_tokens, model_calls, conversation

    async def fix(
        self,
        problem_statement: str,
        docker_image: str,
        repo_path: Optional[str] = None,
        gold_patch: Optional[str] = None,
        bug_patch: Optional[str] = None,
        base_commit: Optional[str] = None,
    ) -> FixerResult:
        """Run Codex CLI inside SWE-bench Docker container to fix the bug"""
        try:
            # 1. Pull Docker image
            print(f"[CODEX] Pulling image: {docker_image}")
            pull_result = subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True, text=True,
                timeout=DOCKER_PULL_TIMEOUT,
            )
            if pull_result.returncode != 0:
                # Check if image exists locally
                inspect = subprocess.run(
                    ["docker", "image", "inspect", docker_image],
                    capture_output=True, timeout=10,
                )
                if inspect.returncode != 0:
                    return FixerResult(
                        patch="", success=False,
                        error=f"Failed to pull image: {pull_result.stderr}",
                    )
                print(f"[CODEX] Using local image: {docker_image}")

            # 2. Start container
            self._container_name = f"codex-fixer-{os.urandom(4).hex()}"
            print(f"[CODEX] Starting container {self._container_name}")
            run_result = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", self._container_name,
                    "--memory", "4g",
                    "--entrypoint", "",
                    docker_image,
                    "sleep", str(self.config.timeout + 300),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if run_result.returncode != 0:
                return FixerResult(
                    patch="", success=False,
                    error=f"Failed to start container: {run_result.stderr}",
                )

            # 3. Apply patches
            if gold_patch or bug_patch:
                self._apply_patches(gold_patch, bug_patch)

            # 4. Sanitize git history
            self._exec_in_container(SANITIZE_GIT_SCRIPT, timeout=60)
            print("[CODEX] Git history sanitized")

            # 5. Install codex CLI in container
            if not self._install_codex():
                return FixerResult(
                    patch="", success=False,
                    error="Failed to install Codex CLI in container",
                )

            # 6. Write codex config (sets wire_api, model, provider)
            self._write_codex_config()

            # 7. Run codex exec (pass prompt via stdin to avoid shell escaping)
            codex_env = {"CODEX_API_KEY": self.config.api_key}

            codex_cmd = (
                "cd /app && codex exec "
                "--dangerously-bypass-approvals-and-sandbox "
                "--experimental-json "
                "-"  # Read prompt from stdin
            )

            print(f"[CODEX] Running codex exec (timeout={self.config.timeout}s)...")
            try:
                result = self._exec_in_container(
                    codex_cmd,
                    timeout=self.config.timeout,
                    env=codex_env,
                    stdin_data=problem_statement,
                )
            except subprocess.TimeoutExpired:
                return FixerResult(
                    patch="", success=False,
                    error=f"Codex timed out after {self.config.timeout}s",
                )

            # 8. Parse JSON output
            total_tokens, model_calls, conversation = \
                self._parse_codex_json_output(result.stdout)

            print(
                f"[CODEX] Exit code: {result.returncode}, "
                f"turns: {model_calls}, tokens: {total_tokens}"
            )
            if result.returncode != 0:
                if result.stderr:
                    print(f"[CODEX] stderr: {result.stderr[:1000]}")
                if result.stdout:
                    print(f"[CODEX] stdout: {result.stdout[:1000]}")

            # 9. Extract diff from container
            diff_result = self._exec_in_container(
                f"cd /app && git add -A && git diff --cached -- {DIFF_EXTENSIONS}",
                timeout=60,
            )
            patch = diff_result.stdout.lstrip()
            if patch:
                patch = patch.rstrip("\n") + "\n"

            return FixerResult(
                patch=patch,
                model_calls=model_calls,
                total_tokens=total_tokens,
                conversation=conversation,
                success=bool(patch),
                error=None if patch else "No changes produced by codex",
            )

        except subprocess.TimeoutExpired:
            return FixerResult(
                patch="", success=False,
                error="Operation timed out",
            )
        except Exception:
            import traceback
            return FixerResult(patch="", success=False, error=traceback.format_exc())

        finally:
            self.cleanup()

    def cleanup(self):
        """Stop and remove the Docker container"""
        if self._container_name:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self._container_name],
                    capture_output=True, timeout=30,
                )
                print(f"[CODEX] Container {self._container_name} removed")
            except Exception:
                pass
            self._container_name = None
