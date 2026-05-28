"""Thin structlog wrapper.

Eval-only slice: only ``get_logger`` / ``trace_context`` /
``current_trace_id`` are needed. Configuration (JSON formatter, DB
event sink, level routing) lives upstream in cortex's generator
service — the eval environment runs short-lived processes where
structlog's stock defaults are fine.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog
from structlog.contextvars import bind_contextvars

_trace_id_ctx: ContextVar[str | None] = ContextVar("distill_v2_eval_trace_id", default=None)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name or "distill_v2_eval")


def new_trace_id() -> str:
    return uuid4().hex


def current_trace_id() -> str | None:
    return _trace_id_ctx.get()


@contextmanager
def trace_context(trace_id: str | None = None, **extra: Any):
    """Bind a trace_id (and optional extra fields) for the duration of a block."""
    tid = trace_id or new_trace_id()
    token = _trace_id_ctx.set(tid)
    bind_contextvars(trace_id=tid, **extra)
    start = time.monotonic()
    try:
        yield tid
    finally:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        bind_contextvars(elapsed_ms=elapsed_ms)
        _trace_id_ctx.reset(token)
