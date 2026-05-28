"""Per-family renderer registry."""

from __future__ import annotations

from typing import Any

from distill_v2_eval.adapters.forward_engines.render import (
    LlamaRenderer,
    QwenRenderer,
    RenderConfig,
    TrajectoryRenderer,
)
from distill_v2_eval.domain.errors import EvalError


def build_renderer(
    family: str, tokenizer: Any, cfg: RenderConfig | None = None,
) -> TrajectoryRenderer:
    """Resolve a renderer by student family.

    Falls back to Qwen-flavoured ChatML for unknown families because most
    open-weight chat models inherit that template. The fallback is
    intentional — it produces a usable score on day 1 for a new arch and
    can be replaced with a family-specific subclass when the bias matters.
    """
    if family in ("qwen", "qwen2", "qwen3"):
        return QwenRenderer(tokenizer, family=family, cfg=cfg)
    if family in ("llama", "llama3"):
        return LlamaRenderer(tokenizer, family=family, cfg=cfg)
    if family in ("generic", "chatml"):
        return TrajectoryRenderer(tokenizer, family=family, cfg=cfg)
    raise EvalError(
        f"no renderer registered for family={family!r}. "
        "Add an adapter under adapters/forward_engines/render.py or pass "
        "family='generic' to use ChatML."
    )


def known_families() -> list[str]:
    return ["qwen", "llama", "generic"]
