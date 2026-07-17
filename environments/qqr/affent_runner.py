"""
Drive QQR evaluations with affent (Go agent loop) instead of an in-env LLM loop.

affent (github.com/AffineFoundation/affent) gives us streaming + reasoning_content,
HTTP 408/429/5xx retry with Retry-After, stream watchdog, context compaction,
and turn-step budgeting — none of which the old _call_llm had. The agent process
is spawned as `affentctl run` per evaluation; we read back its session JSONL and
trace JSONL to reconstruct the conversation + tool_trace that scorer.py expects.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from affent_protocol import (
    AffentDisposition,
    classify_affent_run,
    failure_kind_from_provider_error,
    verify_affent_binary,
)
from config import (
    AMAP_MAPS_API_KEY,
    MAX_TOOL_STEPS,
    PYTHONPATH,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

AFFENTCTL_BIN = os.getenv("AFFENTCTL_BIN", "/usr/local/bin/affentctl")

# MCP server names — affent prefixes tool names with these (e.g. "AMap_poi_search").
# Keep short so the model's tool-name token cost stays small.
_AMAP_SERVER = "AMap"
_TRANSPORT_SERVER = "Transport"
_MCP_SERVER_PREFIXES = (f"{_AMAP_SERVER}_", f"{_TRANSPORT_SERVER}_")


def _build_mcp_config(workspace: Path, epoch_salt: str) -> Path:
    """Write MCP config JSON so affent spawns qqr's stdio MCP servers.

    affent's ServerSpec.Env is `[]string` — list of "KEY=VALUE" strings layered
    on top of the parent process env — not a {key: value} object.
    """
    cfg = {
        "servers": [
            {
                "name": _AMAP_SERVER,
                "command": "python",
                "args": ["-m", "amap_server"],
                "env": [
                    f"AMAP_MAPS_API_KEY={AMAP_MAPS_API_KEY or ''}",
                    f"PYTHONPATH={PYTHONPATH or ''}",
                ],
            },
            {
                "name": _TRANSPORT_SERVER,
                "command": "python",
                "args": ["-m", "mock_transport.server"],
                "env": [
                    f"PYTHONPATH={PYTHONPATH or ''}",
                    f"TRANSPORT_SALT={epoch_salt}",
                ],
            },
        ]
    }
    path = workspace / "mcp_config.json"
    path.write_text(json.dumps(cfg))
    return path


def _strip_mcp_prefix(name: str) -> Tuple[str, bool]:
    """Return (unprefixed_name, is_mcp_tool). Non-MCP tools (affent builtins)
    return is_mcp_tool=False so callers can filter them out of tool_trace."""
    for prefix in _MCP_SERVER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):], True
    return name, False


def _parse_trace_events(trace_path: Path) -> Dict[str, Any]:
    """Walk affent's trace JSONL and pull out tool_trace + usage.

    Tool requests/results are matched by call_id. Builtin tool calls (shell/read_file/
    write_file/edit_file/list_files) are filtered out — qqr scoring only looks at
    travel-planning tool calls.

    final_answer extraction is NOT done here. A `message.done` event fires for any
    assistant content, including the preamble paragraph the model emits alongside
    its first tool_calls; using it as the final answer would let mid-turn chatter
    score as if the agent had produced a plan. Derive final_answer from the session
    conversation instead — see _extract_final_answer.
    """
    pending_requests: Dict[str, Dict[str, Any]] = {}
    tool_trace: List[Dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    turn_end_reason: Optional[str] = None
    error_message: Optional[str] = None
    failure_kind: Optional[str] = None

    if not trace_path.exists():
        return {
            "tool_trace": tool_trace,
            "usage": usage,
            "turn_end_reason": None,
            "error_message": None,
            "failure_kind": None,
        }

    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue

        etype = ev.get("type")
        payload = ev.get("data") or {}
        if not isinstance(payload, dict):
            payload = {}

        if etype == "tool.request":
            pending_requests[payload.get("call_id", "")] = {
                "name": payload.get("tool", ""),
                "args": payload.get("args", {}) or {},
            }
        elif etype == "tool.result":
            cid = payload.get("call_id", "")
            req = pending_requests.pop(cid, {"name": "", "args": {}})
            unprefixed, is_mcp = _strip_mcp_prefix(req["name"])
            if not is_mcp:
                # Skip builtins (shell, read_file, etc.) — not relevant to travel scoring
                continue
            result_text = payload.get("result") or payload.get("result_summary") or ""
            tool_trace.append({
                "name": unprefixed,
                "arguments": req["args"],
                "result": {"text": result_text},
            })
        elif etype == "usage":
            usage["prompt_tokens"] += payload.get("input_tokens", 0) or 0
            usage["completion_tokens"] += payload.get("output_tokens", 0) or 0
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        elif etype == "error":
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                error_message = message.strip()
            if payload.get("recoverable") is not False:
                failure_kind = None
            else:
                kind = payload.get("failure_kind")
                if isinstance(kind, str) and kind.strip():
                    failure_kind = kind.strip()
                else:
                    failure_kind = failure_kind_from_provider_error(message)
        elif etype == "turn.end":
            reason = payload.get("reason")
            turn_end_reason = (
                reason.strip()
                if isinstance(reason, str) and reason.strip()
                else None
            )

    return {
        "tool_trace": tool_trace,
        "usage": usage,
        "turn_end_reason": turn_end_reason,
        "error_message": error_message,
        "failure_kind": failure_kind,
    }


def _extract_final_answer(conversation: List[Dict[str, Any]]) -> str:
    """Return the agent's genuine final plan, or "" if it never produced one.

    A "final answer" is the last assistant message with non-empty content AND no
    tool_calls — i.e. the model deliberately stopped calling tools and synthesized
    a plan. Assistant messages that still carry tool_calls are mid-turn preamble
    text that travels alongside a tool dispatch; scoring those as final answers
    rewards models for talking while they call tools rather than for actually
    producing a plan.
    """
    for msg in reversed(conversation):
        if msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue
        content = (msg.get("content") or "").strip()
        if content:
            return content
    return ""


def _parse_session_log(session_path: Path) -> List[Dict[str, Any]]:
    """affent's per-session JSONL contains OpenAI-compatible ChatMessage entries
    (one per line). The scorer doesn't require any reshape — we just return them."""
    if not session_path.exists():
        return []
    msgs: List[Dict[str, Any]] = []
    for line in session_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            msgs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return msgs


async def run_affent_agent(
    user_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    epoch_salt: str,
    timeout: int = 600,
    max_turns: int = MAX_TOOL_STEPS + 2,
) -> Dict[str, Any]:
    """Spawn `affentctl run`, wait for it to finish, return reconstructed state.

    Returns:
        {
            "conversation": [...ChatMessage dicts...],
            "tool_trace":  [{name, arguments, result: {text}}, ...],
            "final_answer": str,
            "usage":       {prompt_tokens, completion_tokens, total_tokens},
            "exit_code":   int,
            "turn_end_reason": str,
            "failure_kind": Optional[str],
            "disposition": str,
            "affent_ref": str,
            "affent_sha256": str,
            "stderr_tail": str,  # last few KB of affentctl stderr for debugging
        }

    Raises:
        RuntimeError if the binary does not match its build manifest, the
        subprocess times out, or affent reports an invalid protocol outcome.
    """
    identity = verify_affent_binary(AFFENTCTL_BIN)

    with tempfile.TemporaryDirectory(prefix="affent-qqr-") as ws:
        workspace = Path(ws)
        session_id = uuid.uuid4().hex

        mcp_cfg = _build_mcp_config(workspace, epoch_salt)
        sys_prompt_file = workspace / "system_prompt.txt"
        sys_prompt_file.write_text(SYSTEM_PROMPT)
        trace_file = workspace / "trace.jsonl"

        cmd = [
            AFFENTCTL_BIN, "run",
            "--workspace", str(workspace),
            "--session-id", session_id,
            "--base-url", base_url,
            "--api-key", api_key,
            "--model", model,
            "--mcp-config", str(mcp_cfg),
            "--system-prompt", str(sys_prompt_file),
            "--prompt", "-",
            "--max-turns", str(max_turns),
            "--trace", str(trace_file),
            "--trace-skip-deltas",
            "--project-context=false",
        ]

        logger.debug("Spawning affentctl: model=%s max_turns=%d", model, max_turns)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(input=user_prompt.encode("utf-8")),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            raise RuntimeError(f"affentctl timed out after {timeout}s")

        stderr_tail = stderr.decode(errors="replace")[-2048:]
        trace = _parse_trace_events(trace_file)
        disposition = classify_affent_run(
            proc.returncode,
            turn_end_reason=trace["turn_end_reason"],
            failure_kind=trace["failure_kind"],
        )
        if disposition is AffentDisposition.INFRA_FAILURE:
            detail = trace["error_message"] or stderr_tail
            if not detail:
                detail = "missing or inconsistent turn.end"
            raise RuntimeError(
                f"affentctl protocol failure: exit={proc.returncode}, "
                f"turn_end={trace['turn_end_reason']!r}: {detail}"
            )

        session_log = workspace / ".affentctl" / f"{session_id}.jsonl"
        conversation = _parse_session_log(session_log)

        return {
            "conversation": conversation,
            "tool_trace": trace["tool_trace"],
            "final_answer": _extract_final_answer(conversation),
            "usage": trace["usage"],
            "exit_code": proc.returncode,
            "turn_end_reason": trace["turn_end_reason"],
            "failure_kind": trace["failure_kind"],
            "failure_detail": trace["error_message"],
            "disposition": disposition.value,
            "affent_ref": identity.revision,
            "affent_sha256": identity.sha256,
            "binary_path": AFFENTCTL_BIN,
            "stderr_tail": stderr_tail,
        }
