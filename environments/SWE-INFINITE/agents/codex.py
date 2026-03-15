"""Codex CLI Agent — runs OpenAI Codex CLI inside a Docker container to solve tasks."""

import json
import os
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# Allow importing from parent directory (SWE-INFINITE/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import SANITIZE_GIT_SCRIPT, NORMALIZE_TIMESTAMPS_SCRIPT, DIFF_EXTENSIONS

DOCKER_PULL_TIMEOUT = 300

# Pre-built static codex binary — search common locations.
def _find_codex_binary() -> str:
    for path in ["/usr/local/bin/codex-static", os.path.expanduser("~/codex-static")]:
        if os.path.isfile(path):
            return path
    return "codex-static"  # fallback to PATH lookup

CODEX_STATIC_BINARY = _find_codex_binary()


@dataclass
class CodexConfig:
    model: str
    api_base: str
    api_key: str
    timeout: int = 1800


@dataclass
class CodexResult:
    patch: str
    model_calls: int = 0
    total_tokens: int = 0
    conversation: List[Any] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class CodexAgent:
    """Runs Codex CLI inside a task Docker container."""

    def __init__(self, config: CodexConfig):
        self.config = config
        self._container_name: Optional[str] = None

    def _exec(
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

    def _install_codex(self) -> bool:
        """Copy pre-built codex binary into the task container."""
        print("[CODEX] Copying codex binary into container...")
        cp_result = subprocess.run(
            ["docker", "cp", CODEX_STATIC_BINARY,
             f"{self._container_name}:/usr/local/bin/codex"],
            capture_output=True, text=True, timeout=30,
        )
        if cp_result.returncode != 0:
            print(f"[CODEX] Failed to copy codex binary: {cp_result.stderr[:500]}")
            return False
        result = self._exec("codex --version", timeout=10)
        if result.returncode != 0:
            print(f"[CODEX] Codex binary not working: {result.stderr[:500]}")
            return False
        print(f"[CODEX] Codex ready: {result.stdout.strip()}")
        return True

    def _write_codex_config(self) -> None:
        """Write codex config.toml inside the container (wire_api=chat for OpenAI-compatible endpoints)."""
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

        self._exec(
            f"mkdir -p /root/.codex && "
            f"cat > /root/.codex/config.toml << 'TOMLEOF'\n{config_toml}TOMLEOF",
            timeout=10,
        )

    def _parse_json_output(self, stdout: str) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Parse JSONL output from codex --experimental-json.

        Returns (total_tokens, model_calls, conversation).
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
                conversation.append(event.get("item", {}))

        return total_input + total_output, model_calls, conversation

    def _apply_patch(self, patch: str, label: str = "augmented test") -> None:
        """Apply a patch inside the container via base64 pipe."""
        import base64
        patch_b64 = base64.b64encode(patch.encode('utf-8')).decode('ascii')
        result = self._exec(
            f"cd /app && echo '{patch_b64}' | base64 -d | git apply -v --allow-empty",
            timeout=60,
        )
        print(f"[CODEX] Applied {label} patch: {result.stdout[:200]}")

    def _prepare_container(self) -> None:
        """Sanitize git history and normalize timestamps."""
        result = self._exec(SANITIZE_GIT_SCRIPT, timeout=60)
        print(f"[CODEX] Git sanitized: {result.stdout[:200]}")

        # Warm up login shell so conda activation happens before normalization
        self._exec("bash -lc true", timeout=60)

        self._exec(NORMALIZE_TIMESTAMPS_SCRIPT, timeout=120)
        print("[CODEX] Timestamps normalized")

    def _build_prompt(
        self,
        problem_statement: str,
        repo: str = "",
        language: str = "",
        test_command: str = "",
        fail_to_pass: list = None,
    ) -> str:
        """Wrap raw PR description into a structured SWE task prompt."""
        lines = [
            "You are solving a software engineering task. A GitHub repository has an open issue or pull request.",
            "Your goal is to implement the necessary code changes to resolve it.",
            "",
        ]
        if repo:
            lines.append(f"Repository: {repo}")
        if language:
            lines.append(f"Language: {language}")
        lines.append("")

        lines.append("## Issue / PR Description")
        lines.append("")
        lines.append(problem_statement.strip())
        lines.append("")

        lines.append("## Instructions")
        lines.append("")
        lines.append("- Modify ONLY source code files under /app. Do NOT modify tests or config files.")
        lines.append("- Read relevant source files to understand the codebase before making changes.")
        lines.append("- Make minimal, focused changes that directly address the issue.")

        return "\n".join(lines)

    async def solve(
        self,
        problem_statement: str,
        docker_image: str,
        repo: str = "",
        language: str = "",
        test_command: str = "",
        fail_to_pass: list = None,
    ) -> CodexResult:
        """Run Codex CLI inside Docker container to implement the change."""
        prompt = self._build_prompt(
            problem_statement, repo, language, test_command, fail_to_pass,
        )
        try:
            # 1. Pull Docker image
            print(f"[CODEX] Pulling image: {docker_image}")
            pull_result = subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True, text=True, timeout=DOCKER_PULL_TIMEOUT,
            )
            if pull_result.returncode != 0:
                inspect = subprocess.run(
                    ["docker", "image", "inspect", docker_image],
                    capture_output=True, timeout=10,
                )
                if inspect.returncode != 0:
                    return CodexResult(
                        patch="", success=False,
                        error=f"Failed to pull image: {pull_result.stderr}",
                    )
                print(f"[CODEX] Using local image: {docker_image}")

            # 2. Start container
            self._container_name = f"swe-infinite-codex-{os.urandom(4).hex()}"
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
                return CodexResult(
                    patch="", success=False,
                    error=f"Failed to start container: {run_result.stderr}",
                )

            # 3. Sanitize git history and normalize timestamps
            self._prepare_container()

            # 4. Install codex CLI
            if not self._install_codex():
                return CodexResult(
                    patch="", success=False,
                    error="Failed to install Codex CLI in container",
                )

            # 5. Write codex config
            self._write_codex_config()

            # 6. Run codex exec (pass prompt via stdin)
            codex_cmd = (
                "cd /app && codex exec "
                "--dangerously-bypass-approvals-and-sandbox "
                "--experimental-json "
                "-"
            )
            print(f"[CODEX] Running codex exec (timeout={self.config.timeout}s)...")
            try:
                result = self._exec(
                    codex_cmd,
                    timeout=self.config.timeout,
                    env={"CODEX_API_KEY": self.config.api_key},
                    stdin_data=prompt,
                )
            except subprocess.TimeoutExpired:
                return CodexResult(
                    patch="", success=False,
                    error=f"Codex timed out after {self.config.timeout}s",
                )

            # 7. Parse output
            total_tokens, model_calls, conversation = self._parse_json_output(result.stdout)
            print(f"[CODEX] Exit code: {result.returncode}, turns: {model_calls}, tokens: {total_tokens}")

            if result.returncode != 0:
                if result.stderr:
                    print(f"[CODEX] stderr: {result.stderr[:1000]}")
                if result.stdout:
                    print(f"[CODEX] stdout: {result.stdout[:1000]}")

            if result.returncode != 0 and model_calls == 0:
                error_detail = (result.stderr or result.stdout or "")[:500]
                return CodexResult(
                    patch="", success=False,
                    model_calls=0, total_tokens=0,
                    conversation=conversation,
                    error=f"Codex failed to start (exit {result.returncode}): {error_detail}",
                )

            # 8. Extract diff from container
            diff_result = self._exec(
                f"cd /app && git add -A && git diff --cached -- {DIFF_EXTENSIONS}",
                timeout=60,
            )
            patch = diff_result.stdout.lstrip()
            if patch:
                patch = patch.rstrip("\n") + "\n"

            return CodexResult(
                patch=patch,
                model_calls=model_calls,
                total_tokens=total_tokens,
                conversation=conversation,
                success=bool(patch),
            )

        except subprocess.TimeoutExpired:
            return CodexResult(patch="", success=False, error="Operation timed out")
        except Exception:
            import traceback
            return CodexResult(patch="", success=False, error=traceback.format_exc())
        finally:
            self.cleanup()

    def cleanup(self):
        """Stop and remove the Docker container."""
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
