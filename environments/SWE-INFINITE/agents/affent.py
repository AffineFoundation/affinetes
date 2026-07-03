"""Affent Agent — runs the affent agent loop CLI inside the task container.

Same delivery model as the codex agent: docker-cp the static binary into
the per-task container, exec it with a workspace pointed at /app, parse
its SSE trace JSONL for tokens/calls/conversation, and pull the patch
out of the container with `git diff` afterwards.
"""

import base64
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any

# Allow importing from parent directory (SWE-INFINITE/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    SANITIZE_GIT_SCRIPT,
    NORMALIZE_TIMESTAMPS_SCRIPT,
    NETWORK_BLOCKLIST_SCRIPT,
    DIFF_EXTENSIONS,
    ContainerLostError,
    is_container_lost,
    is_image_prepared,
)

DOCKER_PULL_TIMEOUT = 300


def _find_affent_binary() -> str:
    for path in [
        "/usr/local/bin/affent-static",
        os.path.expanduser("~/affent-static"),
    ]:
        if os.path.isfile(path):
            return path
    return "affent-static"  # fallback to PATH


AFFENT_STATIC_BINARY = _find_affent_binary()


# affent's built-in system prompt is "dev-box" flavored (mentions schedule_*
# tools wired by the gateway). For SWE tasks we replace it with a focused
# instruction set: minimal source-only edits, stop when done — no submit
# protocol, the host extracts the diff post-run.
SWE_SYSTEM_PROMPT = """\
You are a software engineering agent solving a real GitHub PR task.

Your workspace is /app — the repository root. Use the available tools
(shell, read_file, write_file, edit_file, list_files) to:

  1. Read the relevant source files to understand the codebase.
  2. Make minimal, focused changes that directly address the task.
  3. Modify ONLY source code files. Do NOT touch tests, fixtures, or
     configuration files.
  4. Keep changes contained: prefer the smallest diff that resolves
     the issue. Do not refactor unrelated code.

# Turn protocol (IMPORTANT — read carefully)

EVERY assistant turn must do exactly one of these two things:

  (a) Call one or more tools to make progress (preferred — keep working).
  (b) Stop the run by replying with the literal token `TASK_COMPLETE` on
      its own line. Only do this when you are confident the diff in
      /app already resolves the task. Do NOT print the patch — the
      framework extracts it automatically.

Never reply with planning prose alone ("let me look at...", "next I'll
check...") without an accompanying tool call. If you need to think,
just call the next tool. A turn with neither a tool call nor the
TASK_COMPLETE token is treated as a protocol error and the framework
will automatically re-prompt you to continue.
"""


@dataclass
class AffentConfig:
    model: str
    api_base: str
    api_key: str
    timeout: int = 1800
    max_turns: int = 100
    max_call_timeout: str = "5m"


@dataclass
class AffentResult:
    patch: str
    model_calls: int = 0
    total_tokens: int = 0
    conversation: List[Any] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class AffentAgent:
    """Runs affent CLI inside a task Docker container."""

    def __init__(self, config: AffentConfig):
        self.config = config
        self._container_name: Optional[str] = None

    def _exec(
        self,
        cmd: str,
        timeout: int = 60,
        env: Optional[Dict[str, str]] = None,
        stdin_data: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Execute a command inside the task container.

        Raises ContainerLostError on docker daemon container-loss errors so
        the caller can mark the eval as a retryable docker_error.
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
        if result.returncode != 0 and is_container_lost(result.stderr or ""):
            raise ContainerLostError(
                f"docker container lost: {(result.stderr or '').strip()[:300]}"
            )
        return result

    def _install_affent(self) -> bool:
        """Copy the prebuilt affent binary into the task container."""
        print("[AFFENT] Copying affent binary into container...")
        cp_result = subprocess.run(
            ["docker", "cp", AFFENT_STATIC_BINARY,
             f"{self._container_name}:/usr/local/bin/affentctl"],
            capture_output=True, text=True, timeout=30,
        )
        if cp_result.returncode != 0:
            print(f"[AFFENT] Failed to copy affent binary: {cp_result.stderr[:500]}")
            return False
        # Smoke test: -h is the cheapest invocation that proves the binary loads.
        result = self._exec("affentctl help", timeout=10)
        # `help` exits with code 2 in some flag.FlagSet conventions; we just
        # check the binary produces usage text on stderr/stdout.
        combined = (result.stdout or "") + (result.stderr or "")
        if "affentctl" not in combined.lower():
            print(f"[AFFENT] affent binary not working: {combined[:500]}")
            return False
        print("[AFFENT] affent ready")
        return True

    def _write_text_file(self, container_path: str, content: str) -> None:
        """Write a multi-line text file inside the container via base64 pipe.

        Avoids shell-escaping landmines for prompts containing quotes,
        backticks, dollar signs, etc.
        """
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        result = self._exec(
            f"echo '{b64}' | base64 -d > {container_path}",
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to write {container_path}: {result.stderr[:200]}"
            )

    def _prepare_container(self) -> None:
        """Apply network blocklist, sanitize git history, normalize timestamps.

        Fast path: skip sanitize+normalize if the image already baked them
        (sentinel /etc/swe-infinite-prepared present).
        """
        # Network blocklist must always run (docker recreates /etc/hosts).
        self._exec(NETWORK_BLOCKLIST_SCRIPT, timeout=10)
        print("[AFFENT] Network blocklist applied")

        if is_image_prepared(self._container_name):
            print("[AFFENT] Image pre-prepared, skipping sanitize+normalize")
            return

        result = self._exec(SANITIZE_GIT_SCRIPT, timeout=60)
        print(f"[AFFENT] Git sanitized: {result.stdout[:200]}")

        self._exec("bash -lc true", timeout=60)

        self._exec(NORMALIZE_TIMESTAMPS_SCRIPT, timeout=120)
        print("[AFFENT] Timestamps normalized")

    def _build_user_prompt(
        self,
        problem_statement: str,
        repo: str = "",
        language: str = "",
    ) -> str:
        lines: list[str] = []
        if repo:
            lines.append(f"Repository: {repo}")
        if language:
            lines.append(f"Language: {language}")
        if lines:
            lines.append("")
        lines.append("## Issue / PR Description")
        lines.append("")
        lines.append(problem_statement.strip())
        return "\n".join(lines)

    def _parse_jsonl_trace(self, jsonl_text: str) -> tuple[int, int, list]:
        """Walk affent SSE trace JSONL → (total_tokens, model_calls, conversation).

        Each line: {"id": int, "type": "...", "data": <obj>}. We fold deltas
        into OpenAI-style messages so downstream analysis matches the codex
        and miniswe agents.

        model_calls counts `message.done` events (one per LLM round trip),
        matching codex's `turn.completed` semantics — affent's `turn.end`
        is per-user-message and would always be 1 here.
        """
        total_tokens = 0
        model_calls = 0
        conversation: list[dict] = []
        cur_assistant_parts: list[str] = []
        cur_tool_calls: list[dict] = []

        def _flush_assistant() -> None:
            text = "".join(cur_assistant_parts).strip()
            if text or cur_tool_calls:
                msg: dict = {"role": "assistant", "content": text}
                if cur_tool_calls:
                    msg["tool_calls"] = list(cur_tool_calls)
                conversation.append(msg)
            cur_assistant_parts.clear()
            cur_tool_calls.clear()

        for line in jsonl_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type", "")
            d = ev.get("data") or {}
            if t == "user.message":
                _flush_assistant()
                conversation.append({"role": "user", "content": d.get("text", "")})
            elif t == "message.delta":
                cur_assistant_parts.append(d.get("delta", ""))
            elif t == "message.done":
                # Canonical full text overrides concatenated deltas.
                full = d.get("text")
                if full is not None:
                    cur_assistant_parts[:] = [full]
                model_calls += 1
            elif t == "tool.request":
                cur_tool_calls.append({
                    "call_id": d.get("call_id", ""),
                    "tool": d.get("tool", ""),
                    "args": d.get("args", {}),
                })
            elif t == "tool.result":
                _flush_assistant()
                conversation.append({
                    "role": "tool",
                    "call_id": d.get("call_id", ""),
                    "exit_code": d.get("exit_code", 0),
                    "content": d.get("result_summary", ""),
                })
            elif t == "usage":
                total_tokens += int(d.get("input_tokens", 0)) + int(d.get("output_tokens", 0))
            elif t == "turn.end":
                _flush_assistant()
        _flush_assistant()
        return total_tokens, model_calls, conversation

    async def solve(
        self,
        problem_statement: str,
        docker_image: str,
        repo: str = "",
        language: str = "",
        test_command: str = "",
        fail_to_pass: list = None,
    ) -> AffentResult:
        """Run affentctl inside Docker container to implement the change."""
        try:
            # 1. pull image
            print(f"[AFFENT] Pulling image: {docker_image}")
            pull = subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True, text=True, timeout=DOCKER_PULL_TIMEOUT,
            )
            if pull.returncode != 0:
                inspect = subprocess.run(
                    ["docker", "image", "inspect", docker_image],
                    capture_output=True, timeout=10,
                )
                if inspect.returncode != 0:
                    return AffentResult(
                        patch="", success=False,
                        error=f"Failed to pull image: {pull.stderr}",
                    )
                print(f"[AFFENT] Using local image: {docker_image}")

            # 2. start container
            self._container_name = f"swe-infinite-affent-{os.urandom(4).hex()}"
            print(f"[AFFENT] Starting container {self._container_name}")
            run = subprocess.run(
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
            if run.returncode != 0:
                return AffentResult(
                    patch="", success=False,
                    error=f"Failed to start container: {run.stderr}",
                )

            # 3. sanitize git, network blocklist, normalize timestamps
            self._prepare_container()

            # 4. install affent CLI
            if not self._install_affent():
                return AffentResult(
                    patch="", success=False,
                    error="Failed to install affent in container",
                )

            # 5. stage prompts
            user_prompt = self._build_user_prompt(problem_statement, repo, language)
            self._write_text_file("/tmp/affent_prompt.txt", user_prompt)
            self._write_text_file("/tmp/affent_system.txt", SWE_SYSTEM_PROMPT)

            # 6. run affentctl. Trace goes to a file (clean stdout — final
            # assistant text doesn't get mixed with JSONL events).
            cmd = (
                "cd /app && affentctl run "
                "--workspace /app "
                f"--base-url {self.config.api_base} "
                f"--model {self.config.model} "
                "--prompt @/tmp/affent_prompt.txt "
                "--system-prompt @/tmp/affent_system.txt "
                "--trace /tmp/affent_trace.jsonl "
                "--trace-skip-deltas "
                f"--max-turns {self.config.max_turns} "
                f"--max-call-timeout {self.config.max_call_timeout} "
                "--quiet"
            )
            print(f"[AFFENT] Running affentctl (timeout={self.config.timeout}s)...")
            try:
                result = self._exec(
                    cmd,
                    timeout=self.config.timeout,
                    env={"AFFENTCTL_API_KEY": self.config.api_key},
                )
            except subprocess.TimeoutExpired:
                return AffentResult(
                    patch="", success=False,
                    error=f"affent timed out after {self.config.timeout}s",
                )

            # 7. persist trace to host BEFORE cleanup deletes the container,
            # then read + parse. Useful for debugging — the JSONL stream
            # captures every SSE event affent emitted (thinking deltas,
            # tool requests, tool results, usage, …).
            host_trace_dir = Path("/tmp/affent-traces")
            host_trace_dir.mkdir(parents=True, exist_ok=True)
            host_trace_path = host_trace_dir / f"{self._container_name}.jsonl"
            cp_trace = subprocess.run(
                ["docker", "cp",
                 f"{self._container_name}:/tmp/affent_trace.jsonl",
                 str(host_trace_path)],
                capture_output=True, text=True, timeout=30,
            )
            if cp_trace.returncode == 0:
                print(f"[AFFENT] Trace persisted to {host_trace_path}")
                trace_text = host_trace_path.read_text(errors="replace")
            else:
                # Fall back to docker exec cat — keeps the agent running
                # even if the trace file vanished mid-run.
                trace_read = self._exec(
                    "cat /tmp/affent_trace.jsonl 2>/dev/null || true",
                    timeout=30,
                )
                trace_text = trace_read.stdout or ""

            total_tokens, model_calls, conversation = self._parse_jsonl_trace(trace_text)
            if not any(m.get("role") == "user" for m in conversation):
                conversation.insert(0, {"role": "user", "content": user_prompt})

            print(
                f"[AFFENT] Exit: {result.returncode}, turns: {model_calls}, "
                f"tokens: {total_tokens}"
            )
            if result.returncode != 0:
                if result.stderr:
                    print(f"[AFFENT] stderr: {result.stderr[:1000]}")

            # Pull out the most informative LLM error from the trace, if any.
            # affent's stderr is mostly logging — the actual failure (HTTP 429,
            # timeout, auth) lives in the SSE error events. Surfacing it lets
            # env.py's _classify_agent_error route the failure to the correct
            # bucket (api_error → retryable) instead of agent_error.
            trace_errors = []
            for line in trace_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "error":
                    trace_errors.append((ev.get("data") or {}).get("message", ""))

            # Always attempt to extract the diff. Even when affentctl exits
            # with an error mid-loop, prior tool calls may have written real
            # changes to /app — discarding them would hide partial work and
            # make a transient LLM blip look like the model produced nothing.
            diff = self._exec(
                f"cd /app && git add -A && git diff --cached -- {DIFF_EXTENSIONS}",
                timeout=60,
            )
            patch = diff.stdout.lstrip()
            if patch:
                patch = patch.rstrip("\n") + "\n"

            error_str: Optional[str] = None
            if result.returncode != 0:
                # Prefer the trace's last error message (richer than affentctl's
                # stderr, which is just "exit 3"). Build a string env.py's
                # classifier can grep — keep "429" / "rate limit" / "timeout"
                # verbatim so it routes to api_error.
                msg = trace_errors[-1] if trace_errors else (
                    (result.stderr or result.stdout or "").strip()
                )
                lower = msg.lower()
                if any(k in lower for k in (
                    "429", "rate limit", "ratelimit", "too many requests",
                    "infrastructure is at maximum capacity",
                    "timeout", "timed out", "deadline exceeded",
                    "connection", "network", "reconnecting",
                    "401", "403", "authentication",
                    "404", "no matching chute", "not found",
                )):
                    prefix = "api_error"
                else:
                    prefix = "affent_error"
                error_str = f"{prefix}: exit {result.returncode}: {msg[:500]}"

            return AffentResult(
                patch=patch,
                model_calls=model_calls,
                total_tokens=total_tokens,
                conversation=conversation,
                success=bool(patch) and error_str is None,
                error=error_str,
            )

        except subprocess.TimeoutExpired:
            return AffentResult(patch="", success=False, error="Operation timed out")
        except ContainerLostError as e:
            return AffentResult(patch="", success=False, error=f"docker_error: {e}")
        except Exception:
            import traceback
            return AffentResult(patch="", success=False, error=traceback.format_exc())
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
                print(f"[AFFENT] Container {self._container_name} removed")
            except Exception:
                pass
            self._container_name = None
