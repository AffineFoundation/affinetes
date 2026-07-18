"""Codex CLI Agent — runs OpenAI Codex CLI inside a Docker container to solve tasks."""

import json
import os
import queue
import sys
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# Allow importing from parent directory (SWE-INFINITE/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    SANITIZE_GIT_SCRIPT,
    NORMALIZE_TIMESTAMPS_SCRIPT,
    NETWORK_BLOCKLIST_SCRIPT,
    DIFF_EXTENSIONS,
    ContainerLostError,
    is_container_running,
    is_image_prepared,
)
from agents.outcome import (
    AgentOutcome,
    failure_kind_from_provider_error,
    outcome_from_process_exit_code,
)

DOCKER_PULL_TIMEOUT = 300
SHELL_EDIT_INSTRUCTIONS = Path(__file__).with_name("codex_system_prompt_shell_edits.txt")

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
    outcome: AgentOutcome = AgentOutcome.COMPLETED
    process_exit_code: Optional[int] = None
    failure_kind: Optional[str] = None


def _codex_error_kind(payload: object) -> Optional[str]:
    """Extract a native or provider-structured Codex failure kind."""
    if not isinstance(payload, dict):
        return None
    value = payload.get("codex_error_info") or payload.get("codexErrorInfo")
    native_kind: Optional[str] = None
    if isinstance(value, str) and value.strip():
        native_kind = value.strip()
    elif isinstance(value, dict):
        error_type = value.get("type")
        if isinstance(error_type, str) and error_type.strip():
            native_kind = error_type.strip()
        elif len(value) == 1:
            key = next(iter(value))
            if isinstance(key, str) and key.strip():
                native_kind = key.strip()

    if native_kind in (None, "other", "badRequest"):
        provider_kind = failure_kind_from_provider_error(payload.get("message"))
        if provider_kind is not None:
            return provider_kind
    return native_kind


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
        """Execute a command inside the Docker container.

        Raises ContainerLostError when docker daemon reports the container is
        gone (No such container, is not running, etc.) so the run aborts and
        the caller can mark the eval as a retryable docker_error instead of a
        zero-score sample.
        """
        docker_cmd = ["docker", "exec"]
        if stdin_data is not None:
            docker_cmd.append("-i")
        if env:
            for k, v in env.items():
                docker_cmd.extend(["-e", f"{k}={v}"])
        docker_cmd.extend([self._container_name, "bash", "-c", cmd])
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin_data,
        )
        if result.returncode != 0 and not is_container_running(self._container_name):
            raise ContainerLostError(
                f"docker container unavailable: {self._container_name}"
            )
        return result

    def _run_codex_app_server(
        self,
        prompt: str,
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """Run one Codex turn through its typed app-server protocol."""
        command = [
            "docker",
            "exec",
            "-i",
            "-e",
            f"CODEX_API_KEY={self.config.api_key}",
            self._container_name,
            "bash",
            "-c",
            "cd /app && codex app-server",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("failed to open Codex app-server stdio")

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        stdout_queue: queue.Queue[Optional[str]] = queue.Queue()
        captured_methods = {
            "error",
            "item/completed",
            "thread/tokenUsage/updated",
            "turn/completed",
        }
        deadline = time.monotonic() + timeout
        returncode = 1
        timeout_error: Optional[subprocess.TimeoutExpired] = None

        def _drain_stdout() -> None:
            for line in process.stdout:
                stdout_queue.put(line)
            stdout_queue.put(None)

        def _drain_stderr() -> None:
            for line in process.stderr:
                stderr_lines.append(line)

        stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        def _send(message: dict) -> None:
            process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            process.stdin.flush()

        def _read_until(predicate) -> dict:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    line = stdout_queue.get(timeout=remaining)
                except queue.Empty:
                    raise subprocess.TimeoutExpired(command, timeout)
                if line is None:
                    code = process.poll()
                    raise RuntimeError(
                        f"Codex app-server exited before turn completion: {code}"
                    )
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("method") in captured_methods:
                    stdout_lines.append(line)
                if message.get("id") is not None and "error" in message:
                    raise RuntimeError(
                        "Codex app-server request failed: "
                        + json.dumps(message["error"], ensure_ascii=True)
                    )
                if predicate(message):
                    return message

        try:
            _send({
                "method": "initialize",
                "id": 0,
                "params": {
                    "clientInfo": {
                        "name": "swe_infinite",
                        "title": "SWE-INFINITE",
                        "version": "1",
                    },
                },
            })
            _read_until(lambda message: message.get("id") == 0)
            _send({"method": "initialized", "params": {}})
            _send({
                "method": "thread/start",
                "id": 1,
                "params": {
                    "model": self.config.model,
                    "cwd": "/app",
                    "approvalPolicy": "never",
                    "sandbox": "danger-full-access",
                    "ephemeral": True,
                },
            })
            thread_response = _read_until(
                lambda message: message.get("id") == 1
            )
            thread_id = (
                thread_response.get("result", {})
                .get("thread", {})
                .get("id")
            )
            if not isinstance(thread_id, str) or not thread_id:
                raise RuntimeError("Codex app-server returned no thread id")

            _send({
                "method": "turn/start",
                "id": 2,
                "params": {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                    "cwd": "/app",
                    "approvalPolicy": "never",
                    "sandboxPolicy": {"type": "dangerFullAccess"},
                },
            })
            terminal = _read_until(
                lambda message: message.get("method") == "turn/completed"
            )
            status = (
                terminal.get("params", {})
                .get("turn", {})
                .get("status")
            )
            returncode = 0 if status == "completed" else 1
        except subprocess.TimeoutExpired as exc:
            timeout_error = exc
        except Exception as exc:
            stderr_lines.append(f"Codex app-server protocol error: {exc}\n")
            process_code = process.poll()
            returncode = process_code if process_code not in (None, 0) else 1
        finally:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        if timeout_error is not None:
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
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

        # Expose codex's argv[0]-dispatched apply_patch in the default PATH
        # (/usr/local/bin) so `bash -lc apply_patch ...` works. Login shells
        # reset PATH from /etc/profile and drop codex's auto-injected
        # ~/.codex/tmp/path/ entry, breaking the model's frequent
        # `bash -lc apply_patch '...'` calls otherwise.
        self._exec(
            "ln -sf /usr/local/bin/codex /usr/local/bin/apply_patch && "
            "ln -sf /usr/local/bin/codex /usr/local/bin/applypatch",
            timeout=5,
        )
        return True

    def _write_codex_config(self) -> None:
        """Write codex config.toml inside the container (wire_api=chat for OpenAI-compatible endpoints)."""
        base_url = self.config.api_base
        instructions = SHELL_EDIT_INSTRUCTIONS.read_text()
        self._exec(
            "mkdir -p /root/.codex && cat > /root/.codex/model_instructions.md",
            timeout=10,
            stdin_data=instructions,
        )
        config_toml = (
            f'model = {json.dumps(self.config.model)}\n'
            f'model_provider = "chutes"\n'
            f'model_instructions_file = "/root/.codex/model_instructions.md"\n'
            f'\n'
            f'[model_providers.chutes]\n'
            f'name = "Chutes"\n'
            f'env_key = "CODEX_API_KEY"\n'
        )
        if base_url:
            config_toml += f'base_url = {json.dumps(base_url)}\n'
        config_toml += 'wire_api = "chat"\n'

        self._exec(
            f"mkdir -p /root/.codex && "
            f"cat > /root/.codex/config.toml << 'TOMLEOF'\n{config_toml}TOMLEOF",
            timeout=10,
        )

    def _parse_json_output(
        self, stdout: str
    ) -> Tuple[
        int,
        int,
        List[Dict[str, Any]],
        Optional[str],
        Optional[str],
    ]:
        """Parse Codex app-server JSONL, plus legacy exec JSONL if present.

        Returns tokens, calls, conversation, error detail, and native error kind.
        The app-server terminal event retains `codexErrorInfo`, which the
        legacy `codex exec --json` projection omits.
        """
        total_input = 0
        total_output = 0
        latest_total_tokens: Optional[int] = None
        model_calls = 0
        conversation: List[Dict[str, Any]] = []
        last_error: Optional[str] = None
        last_failure_kind: Optional[str] = None

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            event_type = event.get("type", "")
            method = event.get("method", "")
            params = event.get("params") or {}
            if event_type == "turn.completed":
                model_calls += 1
                usage = event.get("usage", {})
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
            elif event_type == "item.completed":
                conversation.append(event.get("item", {}))
            elif event_type == "error":
                error_payload = event.get("error")
                if not isinstance(error_payload, dict):
                    error_payload = event
                msg = error_payload.get("message")
                if isinstance(msg, str) and msg.strip():
                    last_error = msg.strip()
                last_failure_kind = _codex_error_kind(error_payload)
            elif event_type == "turn.failed":
                err = event.get("error") or {}
                msg = err.get("message") if isinstance(err, dict) else None
                if isinstance(msg, str) and msg.strip():
                    last_error = msg.strip()
                last_failure_kind = _codex_error_kind(err)
            elif method == "thread/tokenUsage/updated" and isinstance(params, dict):
                total = (params.get("tokenUsage") or {}).get("total") or {}
                value = total.get("totalTokens")
                if isinstance(value, int):
                    latest_total_tokens = value
            elif method == "item/completed" and isinstance(params, dict):
                item = params.get("item")
                if isinstance(item, dict) and item.get("type") != "userMessage":
                    conversation.append(item)
            elif method == "error" and isinstance(params, dict):
                error_payload = params.get("error") or {}
                if isinstance(error_payload, dict):
                    msg = error_payload.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_error = msg.strip()
                    last_failure_kind = _codex_error_kind(error_payload)
            elif method == "turn/completed" and isinstance(params, dict):
                turn = params.get("turn") or {}
                if not isinstance(turn, dict):
                    continue
                if turn.get("status") == "completed":
                    model_calls += 1
                error_payload = turn.get("error") or {}
                if isinstance(error_payload, dict) and error_payload:
                    msg = error_payload.get("message")
                    if isinstance(msg, str) and msg.strip():
                        last_error = msg.strip()
                    last_failure_kind = _codex_error_kind(error_payload)

        return (
            latest_total_tokens
            if latest_total_tokens is not None
            else total_input + total_output,
            model_calls,
            conversation,
            last_error,
            last_failure_kind,
        )

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
        """Apply network blocklist, sanitize git history, normalize timestamps.

        Fast path: skip sanitize+normalize if the image already baked them
        (sentinel /etc/swe-infinite-prepared present).
        """
        # Network blocklist must always run at container start (docker
        # rebuilds /etc/hosts from scratch on every run).
        self._exec(NETWORK_BLOCKLIST_SCRIPT, timeout=10)
        print("[CODEX] Network blocklist applied")

        if is_image_prepared(self._container_name):
            print("[CODEX] Image pre-prepared, skipping sanitize+normalize")
            return

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
        lines.append("")
        lines.append("## Tool Contract")
        lines.append("")
        lines.append("- Do not emit XML, JSON, or pseudo tool calls in assistant text.")
        lines.append("- To edit files, call the available `shell` tool and run commands that actually modify files under /app.")
        lines.append("- Recommended edit methods include `sed -i`, `perl -0pi`, small Python rewrite scripts, `cat > file`, or `git apply` from a unified diff file.")
        lines.append("- The evaluator scores the real working tree diff; assistant text describing an edit is not applied.")

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
                        outcome=AgentOutcome.INFRA_FAILURE,
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
                    outcome=AgentOutcome.INFRA_FAILURE,
                )

            # 3. Sanitize git history and normalize timestamps
            self._prepare_container()

            # 4. Install codex CLI
            if not self._install_codex():
                return CodexResult(
                    patch="", success=False,
                    error="Failed to install Codex CLI in container",
                    outcome=AgentOutcome.INFRA_FAILURE,
                )

            # 5. Write codex config
            self._write_codex_config()

            # 6. Run one turn through app-server. Unlike `codex exec --json`,
            # this protocol preserves the native `codexErrorInfo` enum.
            print(
                f"[CODEX] Running codex app-server "
                f"(timeout={self.config.timeout}s)..."
            )
            try:
                result = self._run_codex_app_server(
                    prompt,
                    timeout=self.config.timeout,
                )
            except subprocess.TimeoutExpired:
                return CodexResult(
                    patch="", success=False,
                    error=f"Codex timed out after {self.config.timeout}s",
                    outcome=AgentOutcome.INFRA_FAILURE,
                )

            # 7. Parse output
            total_tokens, model_calls, conversation, last_error, failure_kind = (
                self._parse_json_output(result.stdout)
            )
            # Prepend initial prompt as first conversation entry
            conversation.insert(0, {"role": "user", "content": prompt})
            print(
                f"[CODEX] Exit code: {result.returncode}, "
                f"turns: {model_calls}, tokens: {total_tokens}"
            )

            if result.returncode != 0:
                if last_error:
                    print(f"[CODEX] event error: {last_error[:1000]}")
                if result.stderr:
                    print(f"[CODEX] stderr: {result.stderr[:1000]}")
                if result.stdout:
                    print(f"[CODEX] stdout: {result.stdout[:1000]}")

            # Capture failure detail for the caller, but always harvest the
            # patch: a failed turn can leave useful partial edits on disk.
            outcome = outcome_from_process_exit_code(
                result.returncode,
                failure_kind=failure_kind,
            )
            error_detail: Optional[str] = None
            if result.returncode != 0:
                detail = (
                    last_error
                    or (result.stderr or result.stdout or "")[:500]
                )
                error_detail = f"codex exited with {result.returncode}: {detail}"

            # 8. Extract diff from container — run unconditionally so that
            # partial edits left by a failed codex run are not silently lost.
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
                success=bool(patch) and outcome is AgentOutcome.COMPLETED,
                error=error_detail,
                outcome=outcome,
                process_exit_code=result.returncode,
                failure_kind=failure_kind,
            )

        except subprocess.TimeoutExpired:
            return CodexResult(
                patch="",
                success=False,
                error="Operation timed out",
                outcome=AgentOutcome.INFRA_FAILURE,
            )
        except ContainerLostError as exc:
            return CodexResult(
                patch="",
                success=False,
                error=str(exc),
                outcome=AgentOutcome.INFRA_FAILURE,
            )
        except Exception:
            import traceback
            return CodexResult(
                patch="",
                success=False,
                error=traceback.format_exc(),
                outcome=AgentOutcome.INFRA_FAILURE,
            )
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
