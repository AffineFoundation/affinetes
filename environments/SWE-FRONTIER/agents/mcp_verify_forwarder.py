"""Stdio MCP server that forwards `verify` calls to a host HTTP server.

Why a forwarder: afent only supports stdio MCP servers, but our verify
needs host-side docker access (to spawn fresh /task verify containers).
This script runs inside the agent's solving container, speaks JSON-RPC
2.0 over stdin/stdout to afent, and bridges every `verify` tool call to
an HTTP POST against the verify server on the docker host.

Tools exposed:
  verify  — run the workspace through the IFS two-stage verifier and
            return a UnifiedScore-like aggregate (overall_score,
            overall_pass, milestone scores). Per-case expected values
            are NEVER returned; only the aggregate the agent needs to
            know which milestones still fail.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

VERIFY_URL = os.environ.get("VERIFY_URL", "")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "")

PROTOCOL_VERSION = "2025-06-18"

TOOL_DEFS = [
    {
        "name": "verify",
        "description": (
            "Run the IFS two-stage verifier against the current /workspace "
            "and return aggregate scores per milestone. Use this between "
            "edits to find out which milestones still fail. Calls are "
            "rate-limited; each call takes 30-90 seconds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }
]


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _ok(req_id, result):
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id, code, message):
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _do_verify() -> dict:
    if not VERIFY_URL:
        return {"error": "VERIFY_URL not set in MCP server env"}
    req = urllib.request.Request(
        VERIFY_URL,
        method="POST",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "X-Verify-Token": VERIFY_TOKEN,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"verify http {e.code}: {body[:500]}"}
    except Exception as e:
        return {"error": f"verify request failed: {e!r}"}
    try:
        return json.loads(body)
    except Exception as e:
        return {"error": f"verify response not JSON: {e!r}", "raw": body[:500]}


def _handle(req: dict) -> None:
    rid = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}

    if method == "initialize":
        _ok(rid, {
            "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ifs-verify-forwarder", "version": "0.1"},
        })
        return
    if method == "notifications/initialized":
        # No response for notifications.
        return
    if method == "tools/list":
        _ok(rid, {"tools": TOOL_DEFS})
        return
    if method == "tools/call":
        name = params.get("name", "")
        if name != "verify":
            _err(rid, -32601, f"unknown tool: {name}")
            return
        result = _do_verify()
        text = json.dumps(result, indent=2)
        is_error = "error" in result and not result.get("milestones")
        _ok(rid, {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        })
        return
    if method == "ping":
        _ok(rid, {})
        return
    _err(rid, -32601, f"method not implemented: {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            _handle(req)
        except Exception as e:
            rid = req.get("id") if isinstance(req, dict) else None
            _err(rid, -32603, f"internal error: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
