"""Affent agent — runs the AffineFoundation/affent `affentctl` loop inside the task image.

Workflow:
  1. Create a host workspace dir; pre-stage stub solution.py + oracle.sh.
  2. Start a solving container from the release image as root, bind-mounting
     host workspace at /workspace, with /task/verify hidden (chmod 000) so
     the agent can't read oracle vectors.
  3. Copy the affent-static binary + the MCP verify forwarder into the
     container, write an mcp config that points the forwarder at our
     host-side HTTP verify server.
  4. Run a host-side verify HTTP server in a daemon thread. The forwarder
     in the container POSTs to it; the server runs the IFS two-stage
     verifier in fresh containers and returns aggregate scores (no
     per-case expecteds — that would defeat grading).
  5. Run `affentctl run --mcp-config ...` with system prompt + one-line
     kickoff. The agent has tools: shell / read_file / write_file /
     edit_file / list_files plus `ifs__verify` from the MCP server.
  6. Stream the JSONL trace, summarising each event for the operator log.
  7. After affent exits, env.py runs the authoritative final verify on the
     same host workspace.

affentctl was chosen over codex / goose because:
  - tiny static Go binary (~5MB vs goose's 270MB) → fast docker cp
  - stream-json trace format is simpler and we control the parser
  - per-call timeout + max_turns are first-class flags
  - tool surface (shell + read/write/edit/list_files) is exactly what
    we need; no model-side toolshim quirks
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from wsgiref.simple_server import WSGIServer, make_server

DOCKER_PULL_TIMEOUT = 300

# Stub-kick attempts: when the agent exits cleanly but never replaces
# the seeded stub files, we re-prompt it with an explicit kick. affent
# itself already retries transient LLM errors (5xx/429/EOF/timeout) via
# --retry-transient, so we don't need a wrapper retry for those.
MAX_AFFENT_ATTEMPTS = 3

# affent's per-LLM-call ceiling. GLM-class reasoning models can take
# minutes to first-token on long system prompts; we go well past the
# default 3min so a single slow inference doesn't kill the run before
# affent's own watchdog/retry has anything to work with.
AFFENT_PER_CALL_TIMEOUT = "15m"
AFFENT_RETRY_TRANSIENT = "3"
AFFENT_RETRY_BACKOFF = "4s"

# Verify-via-MCP: how many times the agent may invoke `ifs__verify` per
# session. Each call spawns two fresh task containers and grades the
# whole vector set, so it's expensive — but this is what lets the agent
# actually iterate towards the correct answer.
VERIFY_BUDGET_PER_SESSION = 20
VERIFY_TIMEOUT_SEC = 1800


def _find_affent_binary() -> str:
    for path in ["/usr/local/bin/affent-static", os.path.expanduser("~/affent-static")]:
        if os.path.isfile(path):
            return path
    return "affent-static"


# Path to the in-container MCP forwarder script. Lives next to this file.
MCP_FORWARDER_SCRIPT = str(Path(__file__).parent / "mcp_verify_forwarder.py")


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _strip_score_for_agent(score_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Project a UnifiedScore down to what the agent is allowed to see.

    The agent gets aggregate scores per milestone — enough to know which
    parts of its implementation are still wrong — but never per-case
    expected values or anything that would let it overfit. The breakdown
    fields (OK / FAILED counts) ARE included because they're aggregate
    and they massively help the model decide where to look next.
    """
    milestones = {}
    for k, v in (score_doc.get("milestones") or {}).items():
        if not isinstance(v, dict):
            continue
        milestones[k] = {
            "score": v.get("score", 0.0),
            "weight": v.get("weight", 0.0),
            "pass": v.get("pass", False),
            # Counts only — no per-case detail.
            "breakdown": {
                bk: v.get("breakdown", {}).get(bk)
                for bk in ("OK", "FAILED", "MISSING", "MATCHED_FIELDS")
                if bk in (v.get("breakdown") or {})
            },
        }
    return {
        "overall_score": score_doc.get("overall_score", 0.0),
        "overall_pass": score_doc.get("overall_pass", False),
        "milestones": milestones,
    }


@dataclass
class _VerifyServerState:
    image: str
    host_workspace: Path
    token: str
    budget: int
    calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _run_verify_two_stage(
    image: str, host_workspace: Path,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Run the two-stage IFS verifier in fresh containers.

    Returns (score_doc, error_message). On success score_doc is the parsed
    /out/score.json, error_message is "". On failure score_doc is None and
    error_message describes the failure.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="swe-frontier-mcp-out-"))
    try:
        for stage, mount_workspace in (("solver", True), ("score", False)):
            cmd = [
                "docker", "run", "--rm", "--init",
                "--user", "0:0",
                "--memory", "4g",
                "--network", "none",
                "-e", f"IFS_STAGE={stage}",
                "-v", f"{out_dir}:/out",
            ]
            if mount_workspace:
                cmd += ["-v", f"{host_workspace}:/workspace:ro"]
            cmd.append(image)
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT_SEC,
            )
            if r.returncode != 0:
                return None, (
                    f"stage_{stage}_failed exit={r.returncode}: "
                    + (r.stdout + r.stderr)[-1000:]
                )
        score_path = out_dir / "score.json"
        if not score_path.is_file():
            return None, "no_score_json"
        try:
            return json.loads(score_path.read_text()), ""
        except Exception as e:
            return None, f"bad_score_json: {e}"
    finally:
        # out_dir will be root-owned because the verify containers ran as
        # root; rmtree from a non-root caller fails. Best effort.
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass


def _build_verify_app(state: _VerifyServerState):
    def app(environ, start_response):
        def respond(status: str, payload: Dict[str, Any]):
            body = json.dumps(payload).encode("utf-8")
            start_response(status, [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        if environ.get("REQUEST_METHOD") != "POST":
            return respond("405 Method Not Allowed", {"error": "method_not_allowed"})
        if environ.get("PATH_INFO") != "/verify":
            return respond("404 Not Found", {"error": "not_found"})
        token = environ.get("HTTP_X_VERIFY_TOKEN", "")
        if token != state.token:
            return respond("403 Forbidden", {"error": "bad_token"})

        with state.lock:
            if state.calls >= state.budget:
                return respond("429 Too Many Requests", {
                    "error": "budget_exhausted",
                    "used": state.calls,
                    "limit": state.budget,
                })
            state.calls += 1
            call_idx = state.calls

        score_doc, err = _run_verify_two_stage(state.image, state.host_workspace)
        if err:
            return respond("200 OK", {
                "error": err,
                "budget": {"used": call_idx, "limit": state.budget},
            })
        payload = _strip_score_for_agent(score_doc)
        payload["budget"] = {"used": call_idx, "limit": state.budget}
        return respond("200 OK", payload)

    return app


@contextlib.contextmanager
def _run_verify_server(state: _VerifyServerState, *, host: str = "127.0.0.1"):
    """Yield (host, port). Server runs on a daemon thread."""
    server: WSGIServer = make_server(host, 0, _build_verify_app(state))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield (host, server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


AFFENT_STATIC_BINARY = _find_affent_binary()

SYSTEM_INSTRUCTIONS_HEAD = """\
You are an autonomous coding agent. ONLY files at /workspace are graded;
chat is ignored. The grader picks an entrypoint by trying, in order:
the .py file named in the BRIEF (e.g. `rfc3550.py`), then `oracle.sh`,
then `solution.py`. Whichever the grader runs is invoked as
`python3 -u <file>` (or `bash` for oracle.sh) with the test vectors
piped on stdin and JSON answers expected one-per-line on stdout.

THIS MEANS: any .py entrypoint you create MUST end with a
`if __name__ == "__main__": main()` guard that drives the stdin->stdout
loop. Importing the file from a thin wrapper does NOT count — the
grader runs the file directly. Do not assume `from <module> import
main; main()` will be reached.

GRADING: the grader compares each output object field-by-field against
a hidden reference. ANY single field mismatch (wrong value, wrong
type, wrong shape) fails the whole case. Subtle distinctions that
matter:
  - JSON null vs "" vs 0 vs false vs missing key — these are all
    different to the grader, and the right one is task-specific.
  - int vs float, list vs dict, hex string vs int.
The BRIEF will not enumerate every type rule; the only reliable way
to know what the grader expects is to call `ifs__verify` (see below)
and read the OK / FAILED counts per milestone.

/workspace starts empty. Read the BRIEF below to learn what file the
grader expects to run (some BRIEFs explicitly name an entrypoint like
`rfc3550.py`; others let you pick — solution.py / oracle.sh / main.py
all work). After writing your code, sanity-check with shell that
`python3 -u /workspace/<entrypoint>` actually reads stdin and prints
JSON one object per line.

GROUND-TRUTH FEEDBACK: you have an MCP tool `ifs__verify` that runs the
official grader against your current /workspace and returns aggregate
scores per milestone (no per-case expected values). Use it. The
grader scoring is the only authoritative signal — your shell tests
only check that JSON parses, not that fields match expectations. A
typical workflow:
  1. Write a first cut of every operation.
  2. Call ifs__verify. Note which milestones still have OK == 0.
  3. Examine your output for that milestone, reason about likely
     mismatches (null vs ""? int vs float? wrong field names?), apply
     a fix with edit_file.
  4. Call ifs__verify again. If a milestone's OK count went up, your
     fix helped — repeat for the next milestone. If it went down,
     revert. Verify is rate-limited (~12 calls per session); spend
     calls on substantive changes, not after every line edit.

The agent loop's session log is at /workspace/.affentctl/ — that's
internal bookkeeping, not part of the task. Don't read or write it.

The full task brief follows. Begin work on your first turn rather than
reading the spec exhaustively first.

──────────────────────── BRIEF (full text) ────────────────────────
"""

SYSTEM_INSTRUCTIONS_TAIL = """\
──────────────────────── end BRIEF ────────────────────────
"""

# One-line kickoff — the system prompt carries the contract.
SOLVER_PROMPT_TEMPLATE = (
    "Begin. First action: replace the stub at /workspace/solution.py with "
    "a real scaffold via the `write_file` tool, then iterate with "
    "`edit_file` patches, testing with `shell` between edits."
)

# NOTE: we used to seed /workspace with stub `solution.py` + `oracle.sh`
# files marked with a STUB sentinel, then detect "agent wrote real code"
# by checking whether the sentinel was gone. That bumps into BRIEFs
# that name a different entrypoint (e.g. `rfc3550.py`): the grader
# would prefer the agent's new file but our wrapper would still see
# stale stubs and falsely conclude no work happened. We now leave
# /workspace empty and detect "real work" by counting any file in the
# workspace, ignoring affent's own .affentctl/ session log.


@dataclass
class AffentConfig:
    model: str
    api_base: str
    api_key: str
    timeout: int = 1800
    # affent's own per-turn round limit. We raise it well above the
    # default 10 because larger benchmark tasks need many tool calls,
    # but cap it so a confused model doesn't burn the wall budget on a
    # cycle.
    max_turns: int = 80


@dataclass
class AffentResult:
    success: bool
    workspace_dir: Optional[Path] = None
    model_calls: int = 0
    total_tokens: int = 0
    conversation: List[Any] = field(default_factory=list)
    error: Optional[str] = None


class AffentAgent:
    """Runs affentctl inside a SWE-FRONTIER task release image."""

    def __init__(self, config: AffentConfig):
        self.config = config
        self._container_name: Optional[str] = None
        self._host_workspace: Optional[Path] = None
        self._verify_server_ctx = None  # contextmanager handle
        self._verify_state: Optional[_VerifyServerState] = None

    # ---- container helpers ---------------------------------------------------

    def _exec(
        self,
        cmd: str,
        timeout: int = 60,
        env: Optional[Dict[str, str]] = None,
        stdin_data: Optional[str] = None,
        user: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        docker_cmd = ["docker", "exec"]
        if stdin_data is not None:
            docker_cmd.append("-i")
        if user:
            docker_cmd += ["--user", user]
        if env:
            for k, v in env.items():
                docker_cmd += ["-e", f"{k}={v}"]
        docker_cmd += [self._container_name, "bash", "-c", cmd]
        return subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin_data,
        )

    def _install_affent(self) -> bool:
        cp_result = subprocess.run(
            ["docker", "cp", AFFENT_STATIC_BINARY,
             f"{self._container_name}:/usr/local/bin/affentctl"],
            capture_output=True, text=True, timeout=60,
        )
        if cp_result.returncode != 0:
            print(f"[AFFENT] Failed to copy affentctl: {cp_result.stderr[:500]}")
            return False
        result = self._exec("affentctl help", timeout=10)
        if result.returncode != 0:
            print(f"[AFFENT] affentctl smoke check failed: {result.stderr[:500]}")
            return False
        print("[AFFENT] affentctl ready")
        return True

    # ---- prompt construction -------------------------------------------------

    def _read_task_files(self) -> Tuple[str, str, str]:
        brief = self._exec("cat /task/BRIEF.md 2>/dev/null || true", timeout=10).stdout
        # Plain grep to avoid depending on PyYAML in the release image.
        lang_proc = self._exec(
            r"grep -E '^language:[[:space:]]+' /task/task.yaml | "
            r"head -n1 | awk -F':' '{print $2}' | tr -d '[:space:]'",
            timeout=10,
        )
        language = (lang_proc.stdout or "").strip() or "python"
        forb_proc = self._exec(
            "cat /task/forbidden.yaml 2>/dev/null || true", timeout=10,
        )
        forbidden_text = forb_proc.stdout.strip()
        forbidden_block = (
            "FORBIDDEN LIBRARIES (banned in your solution; verifier rejects on hit):\n"
            f"{forbidden_text}\n"
            if forbidden_text else ""
        )
        return brief, language, forbidden_block

    def _build_prompts(self) -> Tuple[str, str]:
        brief, language, forbidden_block = self._read_task_files()
        sys_text = SYSTEM_INSTRUCTIONS_HEAD + brief + "\n" + SYSTEM_INSTRUCTIONS_TAIL
        if forbidden_block:
            sys_text += "\n" + forbidden_block
        sys_text += f"\nLANGUAGE: {language}\n"
        return sys_text, SOLVER_PROMPT_TEMPLATE

    # ---- output parsing ------------------------------------------------------

    @staticmethod
    def _format_event(event: Dict[str, Any]) -> Optional[str]:
        """Render an affent SSE event as a single short progress line.

        Returns None for events that should not be printed (token-level
        deltas + per-line tool output, which would flood the log).
        """
        etype = event.get("type", "")
        data = event.get("data", {}) or {}
        if etype == "tool.request":
            tool = data.get("tool", "?")
            args = data.get("args", {})
            args_text = json.dumps(args, default=str)[:200]
            return f"tool.request {tool} {args_text}"
        if etype == "tool.result":
            return (
                f"tool.result exit={data.get('exit_code')} "
                f"{(data.get('result_summary') or '')[:200]}"
            )
        if etype == "usage":
            return (
                f"usage in={data.get('input_tokens', 0)} "
                f"out={data.get('output_tokens', 0)}"
            )
        if etype == "turn.end":
            return f"turn.end reason={data.get('reason', '?')}"
        if etype == "turn.start":
            return "turn.start"
        if etype == "error":
            return f"ERROR {json.dumps(data)[:400]}"
        return None  # message.delta, thinking.delta, tool.output, etc.

    @staticmethod
    def _parse_trace(stdout: str) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Aggregate (total_tokens, model_calls, conversation) from a trace."""
        events: List[Dict[str, Any]] = []
        total_input = 0
        total_output = 0
        model_calls = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(ev)
            etype = ev.get("type")
            data = ev.get("data") or {}
            if etype == "usage":
                # affent emits one usage event per LLM turn — use that as
                # both the model-call count and the token aggregator.
                model_calls += 1
                total_input += int(data.get("input_tokens") or 0)
                total_output += int(data.get("output_tokens") or 0)
        # Build a lightweight conversation view: user.message,
        # message.end (assistant final), tool.request/result.
        conversation: List[Dict[str, Any]] = []
        for ev in events:
            etype = ev.get("type")
            data = ev.get("data") or {}
            if etype == "user.message":
                conversation.append({"role": "user", "content": data.get("text", "")})
            elif etype == "message.end":
                conversation.append({"role": "assistant", "content": data.get("text", "")})
            elif etype == "tool.request":
                conversation.append({
                    "role": "tool_request",
                    "tool": data.get("tool"),
                    "args": data.get("args"),
                })
            elif etype == "tool.result":
                conversation.append({
                    "role": "tool_result",
                    "exit_code": data.get("exit_code"),
                    "summary": data.get("result_summary"),
                })
        return total_input + total_output, model_calls, conversation

    # ---- streaming runner ----------------------------------------------------

    def _run_affent_streaming(
        self,
        *,
        prompt: str,
        timeout: int,
    ) -> Tuple[int, str, str]:
        """Run `affentctl run` via Popen, streaming the JSONL trace as it arrives.

        Returns (returncode, full_stdout, full_stderr). Raises
        subprocess.TimeoutExpired on wall-time exhaustion.
        """
        # affentctl prints final assistant text to stdout when --trace points
        # to a file. We redirect both to stdout via `--trace -`, so stdout
        # carries the JSONL trace + final text. We separate them by parsing
        # each line — JSONL events are JSON, the final text is plain.
        affent_cmd = (
            "cd /workspace && affentctl run "
            "--prompt - "
            f"--workspace /workspace "
            f"--base-url {self.config.api_base} "
            f"--api-key {self.config.api_key} "
            f"--model {self.config.model} "
            f"--max-turns {self.config.max_turns} "
            f"--max-call-timeout {AFFENT_PER_CALL_TIMEOUT} "
            f"--retry-transient {AFFENT_RETRY_TRANSIENT} "
            f"--retry-backoff {AFFENT_RETRY_BACKOFF} "
            "--system-prompt /tmp/affent_system.txt "
            "--mcp-config /tmp/affent_mcp.json "
            "--quiet --trace -"
        )
        docker_cmd = [
            "docker", "exec", "-i",
            self._container_name,
            "bash", "-c", affent_cmd,
        ]
        proc = subprocess.Popen(
            docker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        def _pump_stdout() -> None:
            assert proc.stdout is not None
            last_label = ""
            for line in proc.stdout:
                stdout_chunks.append(line)
                stripped = line.rstrip()
                if not stripped:
                    continue
                # Trace lines are JSON; the final assistant text is plain.
                if stripped.startswith("{"):
                    try:
                        ev = json.loads(stripped)
                    except json.JSONDecodeError:
                        print(f"[affent/raw] {stripped[:300]}")
                        continue
                    rendered = self._format_event(ev)
                    if rendered is not None and rendered != last_label:
                        print(f"[affent] {rendered}")
                        last_label = rendered
                # else: final assistant text (affent prints after turn.end)
                sys.stdout.flush()

        def _pump_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)
                stripped = line.rstrip()
                if stripped:
                    print(f"[affent/stderr] {stripped[:400]}", file=sys.stderr)
                    sys.stderr.flush()

        t_out = threading.Thread(target=_pump_stdout, daemon=True)
        t_err = threading.Thread(target=_pump_stderr, daemon=True)
        t_out.start()
        t_err.start()

        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
        except BrokenPipeError:
            pass

        deadline = time.monotonic() + timeout
        while True:
            try:
                rc = proc.wait(timeout=min(5.0, max(0.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    proc.kill()
                    proc.wait(timeout=10)
                    t_out.join(timeout=5)
                    t_err.join(timeout=5)
                    raise

        t_out.join(timeout=10)
        t_err.join(timeout=10)
        full_stdout = "".join(stdout_chunks)
        full_stderr = "".join(stderr_chunks)
        # Dump the raw trace for offline debugging.
        try:
            dump_path = Path("/tmp/affent_dumps")
            dump_path.mkdir(exist_ok=True)
            stamp = int(time.time() * 1000)
            (dump_path / f"affent_{stamp}.stdout.ndjson").write_text(full_stdout)
            if full_stderr.strip():
                (dump_path / f"affent_{stamp}.stderr.log").write_text(full_stderr)
        except Exception:
            pass
        return rc, full_stdout, full_stderr

    # ---- workspace extraction ------------------------------------------------

    def _extract_workspace(self, dest_dir: Path) -> bool:
        dest_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            f"docker cp {self._container_name}:/workspace/. - | "
            f"tar -x -C {dest_dir}",
            shell=True, capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            print(f"[AFFENT] Workspace extraction failed: {proc.stderr[:500]}")
            return False
        return True

    # ---- main entrypoint -----------------------------------------------------

    def solve(self, docker_image: str) -> AffentResult:
        try:
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
                        success=False,
                        error=f"Failed to pull image: {pull.stderr[:500]}",
                    )
                print(f"[AFFENT] Using local image: {docker_image}")

            # Host workspace shared with the agent container (bind mount) and
            # with the verify containers spawned later by the MCP server.
            # Starts empty — the agent reads the BRIEF and decides what
            # entrypoint to write.
            self._host_workspace = Path(tempfile.mkdtemp(prefix="swe-frontier-ws-"))
            os.chmod(self._host_workspace, 0o777)

            # Stage the MCP forwarder under a host path that gets mounted
            # into the container at /opt/ifs/mcp_verify_forwarder.py.
            forwarder_host = self._host_workspace.parent / (
                self._host_workspace.name + ".mcp_forwarder.py"
            )
            shutil.copy(MCP_FORWARDER_SCRIPT, forwarder_host)

            self._container_name = f"swe-frontier-affent-{uuid.uuid4().hex[:12]}"
            print(f"[AFFENT] Starting container {self._container_name}")
            run = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", self._container_name,
                    "--memory", "4g",
                    "--user", "0:0",
                    "--network", "host",
                    "--entrypoint", "",
                    "-v", f"{self._host_workspace}:/workspace:rw",
                    "-v", f"{forwarder_host}:/opt/ifs/mcp_verify_forwarder.py:ro",
                    docker_image,
                    "sleep", str(self.config.timeout + 300),
                ],
                capture_output=True, text=True, timeout=60,
            )
            if run.returncode != 0:
                return AffentResult(
                    success=False,
                    error=f"Failed to start container: {run.stderr[:500]}",
                )

            # Hide oracle vectors + scoring from the agent. The bind-mounted
            # /workspace is rw so the agent can edit; /task lives in the
            # release image's RO layer and we lock down /task/verify.
            self._exec("chmod -R 000 /task/verify", timeout=10)

            if not self._install_affent():
                return AffentResult(
                    success=False,
                    error="Failed to install affentctl in container",
                )

            # Start the host-side verify HTTP server. Token is short-lived
            # per session; the MCP forwarder gets it via env var so a stray
            # process on the same loopback port can't hit /verify.
            verify_token = uuid.uuid4().hex
            self._verify_state = _VerifyServerState(
                image=docker_image,
                host_workspace=self._host_workspace,
                token=verify_token,
                budget=VERIFY_BUDGET_PER_SESSION,
            )
            self._verify_server_ctx = _run_verify_server(self._verify_state)
            verify_host, verify_port = self._verify_server_ctx.__enter__()
            verify_url = f"http://{verify_host}:{verify_port}/verify"
            print(
                f"[AFFENT] Verify MCP server bound at {verify_url} "
                f"(budget={self._verify_state.budget})"
            )

            # MCP config that points the in-container forwarder at our
            # host server. The agent will see the tool as `ifs__verify`.
            mcp_cfg = {
                "servers": [{
                    "name": "ifs",
                    "command": "python3",
                    "args": ["/opt/ifs/mcp_verify_forwarder.py"],
                    "env": [
                        f"VERIFY_URL={verify_url}",
                        f"VERIFY_TOKEN={verify_token}",
                    ],
                }],
            }
            mcp_b64 = base64.b64encode(
                json.dumps(mcp_cfg).encode("utf-8")
            ).decode()
            self._exec(
                f"echo {mcp_b64} | base64 -d > /tmp/affent_mcp.json",
                timeout=10,
            )

            sys_text, prompt = self._build_prompts()
            sys_b64 = base64.b64encode(sys_text.encode("utf-8")).decode()
            self._exec(
                f"echo {sys_b64} | base64 -d > /tmp/affent_system.txt",
                timeout=10,
            )

            max_attempts = MAX_AFFENT_ATTEMPTS
            wall_deadline = time.monotonic() + self.config.timeout
            attempts = 0
            rc = 1
            full_stdout = ""
            full_stderr = ""
            agg_stdout_parts: List[str] = []
            agg_conversation: List[Dict[str, Any]] = []
            agg_tokens = 0
            agg_calls = 0

            while attempts < max_attempts:
                attempts += 1
                remaining = int(wall_deadline - time.monotonic())
                if remaining <= 30:
                    print("[AFFENT] wall budget exhausted, stopping retries")
                    break

                if attempts == 1:
                    attempt_prompt = prompt
                else:
                    # "Real work" = any non-hidden file in /workspace
                    # (ignoring affent's own .affentctl/ session log dir).
                    prev_marker = self._exec(
                        "find /workspace -mindepth 1 -maxdepth 1 "
                        "-not -name '.affentctl' 2>/dev/null | wc -l",
                        timeout=10,
                    )
                    file_count = int((prev_marker.stdout or "0").strip() or 0)
                    if file_count == 0:
                        attempt_prompt = (
                            "STOP. Your previous turn ended without writing any "
                            "implementation file to /workspace. That scores zero "
                            "— chat is not graded. RIGHT NOW call write_file "
                            "with the entrypoint named in the BRIEF (or pick a "
                            "name like solution.py if the BRIEF doesn't name "
                            "one) and a real implementation of the operations. "
                            "Then iterate with edit_file."
                        )
                    else:
                        attempt_prompt = (
                            "Continuation: your in-progress files are still in "
                            "/workspace from the previous attempt. Inspect them "
                            "with shell (`ls -la /workspace`, `cat …`) and "
                            "continue refining via edit_file / write_file calls. "
                            "Use ifs__verify to see if your changes improved "
                            "any milestone. Do NOT start over."
                        )

                print(
                    f"[AFFENT] Running affentctl attempt={attempts}/{max_attempts} "
                    f"(remaining={remaining}s)..."
                )
                sys.stdout.flush()
                try:
                    rc, full_stdout, full_stderr = self._run_affent_streaming(
                        prompt=attempt_prompt,
                        timeout=remaining,
                    )
                except subprocess.TimeoutExpired:
                    # Wall budget hit. Don't drop the workspace — the agent
                    # may have written real code mid-attempt. Fall through
                    # to the post-loop verification path so env.py can grade
                    # whatever's there.
                    print(
                        f"[AFFENT] attempt {attempts} hit wall-time limit; "
                        "preserving workspace for grading"
                    )
                    rc = 124
                    full_stdout = ""
                    full_stderr = "wall_time_exhausted"
                    break

                agg_stdout_parts.append(full_stdout)
                tokens, calls, conv = self._parse_trace(full_stdout)
                agg_tokens += tokens
                agg_calls += calls
                agg_conversation.extend(conv)
                print(
                    f"[AFFENT] attempt {attempts} exit={rc} turns={calls} "
                    f"tokens={tokens}"
                )

                marker = self._exec(
                    "find /workspace -mindepth 1 -maxdepth 1 "
                    "-not -name '.affentctl' 2>/dev/null | wc -l",
                    timeout=10,
                )
                file_count = int((marker.stdout or "0").strip() or 0)
                workspace_empty = file_count == 0
                stubs_unchanged = workspace_empty  # name kept for the rest of the flow

                if not stubs_unchanged:
                    # Real work was produced (stubs were replaced). Whether
                    # affentctl exited cleanly or not, the workspace is
                    # gradable — affent already retried transient upstream
                    # errors internally before giving up.
                    break
                if rc == 0:
                    print(
                        "[AFFENT] attempt finished cleanly but workspace empty; "
                        "retrying with kick-prompt"
                    )
                else:
                    # affent exhausted its own retry budget without producing
                    # any code. One more wrapper-level attempt with the same
                    # kick still has a chance — the next attempt seeds a
                    # fresh `affentctl run` invocation with a new HTTP client.
                    print(
                        f"[AFFENT] attempt {attempts} failed (exit={rc}) "
                        f"with workspace empty; retrying with kick-prompt"
                    )

                if attempts < max_attempts:
                    time.sleep(min(5.0, max(1.0, 2.0 * attempts)))

            full_stdout = "".join(agg_stdout_parts)
            total_tokens = agg_tokens
            model_calls = agg_calls
            conversation = agg_conversation
            print(
                f"[AFFENT] Done. attempts={attempts} final_exit={rc} "
                f"turns={model_calls} tokens={total_tokens}"
            )

            # Workspace is bind-mounted, so the host already has the
            # agent's final state — no docker-cp extraction step needed.
            agent_warning: Optional[str] = None
            if rc != 0:
                detail = (full_stderr or full_stdout or "")[:500]
                agent_warning = f"affentctl exited {rc}: {detail}"
                print(f"[AFFENT] WARNING: {agent_warning[:400]}")

            verify_calls = (
                self._verify_state.calls if self._verify_state else 0
            )
            print(
                f"[AFFENT] Agent invoked verify {verify_calls}/"
                f"{VERIFY_BUDGET_PER_SESSION} times"
            )

            return AffentResult(
                success=True,
                workspace_dir=self._host_workspace,
                model_calls=model_calls,
                total_tokens=total_tokens,
                conversation=conversation,
                error=agent_warning,
            )
        except subprocess.TimeoutExpired:
            return AffentResult(success=False, error="Operation timed out")
        except Exception:
            import traceback
            return AffentResult(success=False, error=traceback.format_exc())
        finally:
            self.cleanup()

    def cleanup(self):
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
        if self._verify_server_ctx is not None:
            try:
                self._verify_server_ctx.__exit__(None, None, None)
            except Exception as e:
                print(f"[AFFENT] Verify server shutdown failed: {e!r}")
            self._verify_server_ctx = None
        # NOTE: we don't shutil.rmtree(self._host_workspace) here because
        # env.py needs to run the authoritative final verify on it. The
        # caller is responsible for cleaning up workspace_dir afterwards.
