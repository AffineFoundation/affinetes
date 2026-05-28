"""Render NormalizedTrajectory -> RenderedSample for student forward.

The forward engine consumes ``RenderedSample`` (input_ids + loss_mask +
per-assistant-message spans). Rendering is the **only** place the student's
chat template touches the data — every other layer works with the canonical
NormalizedTrajectory. Splitting it out keeps the eval pipeline trivially
testable without a model.

Loss masking
------------

The mask is 1 only over tokens that represent the *assistant's commitment*
to the next move:

  - assistant ``content`` (the natural-language portion)
  - assistant ``tool_calls`` serialised by the student's template

``reasoning`` is masked by default (``mask_reasoning=True``). The flip is a
single boolean on the renderer so eval runs can a/b reasoning's contribution
to the score without changing code (see ``docs/modules/eval.md``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from distill_v2_eval.domain.errors import EvalError
from distill_v2_eval.domain.ids import RolloutId
from distill_v2_eval.domain.models import (
    NormalizedMessage,
    NormalizedTrajectory,
    RenderedSample,
    Rollout,
    StudentSubmission,
)


@dataclass(frozen=True)
class RenderConfig:
    mask_reasoning: bool = True
    max_seq_len: int = 32_768
    truncate_strategy: str = "keep_head_tail"   # keep_head_tail | drop_oldest


class TrajectoryRenderer:
    """Convert NormalizedTrajectory to RenderedSample for a specific student family.

    Strategy is family-specific because:
      - the chat template differs (Qwen Hermes, Llama 3, ...)
      - tool-call serialisation differs
      - the tokenizer's special tokens differ

    The implementation here uses HuggingFace's ``apply_chat_template`` for
    correctness, plus a *manual span pass* to compute the loss mask
    because ``apply_chat_template`` strips message boundaries.
    """

    def __init__(self, tokenizer, family: str, cfg: RenderConfig | None = None) -> None:  # type: ignore[no-untyped-def]
        self._tok = tokenizer
        self.family = family
        self.cfg = cfg or RenderConfig()

    # ----- public API ---- #

    def render(
        self, trajectory: NormalizedTrajectory, rollout_id: RolloutId,
    ) -> RenderedSample:
        token_ids: list[int] = []
        loss_mask: list[int] = []
        spans: list[tuple[int, int]] = []
        for msg in trajectory.messages:
            chunks = self._chunks_for_message(msg)
            for span_kind, text in chunks:
                ids = self._tok.encode(text, add_special_tokens=False)
                mask_val = 1 if span_kind == "assistant_loss" else 0
                start = len(token_ids)
                token_ids.extend(ids)
                loss_mask.extend([mask_val] * len(ids))
                if mask_val == 1:
                    spans.append((start, start + len(ids)))
        token_ids, loss_mask, spans = self._truncate(token_ids, loss_mask, spans)
        return RenderedSample(
            rollout_id=rollout_id,
            input_ids=token_ids,
            loss_mask=loss_mask,
            message_spans=spans,
            meta={"family": self.family, "mask_reasoning": self.cfg.mask_reasoning},
        )

    # ----- per-message chunking --------------------------------------------- #

    def _chunks_for_message(self, msg: NormalizedMessage) -> list[tuple[str, str]]:
        """Return list of ``(kind, text)`` chunks where ``kind`` is either
        ``context`` (mask=0) or ``assistant_loss`` (mask=1)."""
        if msg.role == "system":
            return [("context", self._format_system(msg.content or ""))]
        if msg.role == "user":
            return [("context", self._format_user(msg.content or ""))]
        if msg.role == "tool":
            return [("context", self._format_tool_result(msg))]
        if msg.role == "assistant":
            return self._chunks_for_assistant(msg)
        raise EvalError(f"unknown message role: {msg.role}")

    def _chunks_for_assistant(self, msg: NormalizedMessage) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        head = self._format_assistant_open(msg)
        if head:
            out.append(("context", head))
        if msg.reasoning:
            kind = "context" if self.cfg.mask_reasoning else "assistant_loss"
            out.append((kind, self._format_reasoning(msg.reasoning)))
        if msg.content:
            out.append(("assistant_loss", msg.content))
        for tc in msg.tool_calls:
            out.append(("assistant_loss", self._format_tool_call(tc)))
        tail = self._format_assistant_close()
        if tail:
            out.append(("context", tail))
        return out

    # ----- truncation ------------------------------------------------------- #

    def _truncate(
        self,
        token_ids: list[int],
        loss_mask: list[int],
        spans: list[tuple[int, int]],
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        if len(token_ids) <= self.cfg.max_seq_len:
            return token_ids, loss_mask, spans
        keep = self.cfg.max_seq_len
        if self.cfg.truncate_strategy == "keep_head_tail":
            head = keep // 4
            tail = keep - head
            new_ids = token_ids[:head] + token_ids[-tail:]
            new_mask = loss_mask[:head] + loss_mask[-tail:]
        elif self.cfg.truncate_strategy == "drop_oldest":
            new_ids = token_ids[-keep:]
            new_mask = loss_mask[-keep:]
        else:
            raise EvalError(f"unknown truncate strategy: {self.cfg.truncate_strategy}")
        # Re-derive spans from the surviving mask (cheaper than tracking
        # head/tail offsets for every span).
        new_spans = _spans_from_mask(new_mask)
        return new_ids, new_mask, new_spans

    # ----- family-overridable formatters ------------------------------------ #
    # Default is Qwen-flavoured ChatML which most modern open-weight chat
    # models inherit. Concrete families override in subclasses.

    def _format_system(self, text: str) -> str:
        return f"<|im_start|>system\n{text}<|im_end|>\n"

    def _format_user(self, text: str) -> str:
        return f"<|im_start|>user\n{text}<|im_end|>\n"

    def _format_assistant_open(self, _msg: NormalizedMessage) -> str:
        return "<|im_start|>assistant\n"

    def _format_assistant_close(self) -> str:
        return "<|im_end|>\n"

    def _format_reasoning(self, text: str) -> str:
        return f"<think>\n{text}\n</think>\n"

    def _format_tool_call(self, tc) -> str:  # type: ignore[no-untyped-def]
        # Hermes-style: <tool_call>{...}</tool_call>
        body = json.dumps({"name": tc.name, "arguments": tc.arguments},
                          separators=(",", ":"), ensure_ascii=False)
        return f"<tool_call>\n{body}\n</tool_call>\n"

    def _format_tool_result(self, msg: NormalizedMessage) -> str:
        return (f"<|im_start|>tool\n<tool_response>\n{msg.content or ''}\n"
                f"</tool_response><|im_end|>\n")


class QwenRenderer(TrajectoryRenderer):
    """Default ChatML / Hermes — same as base."""


class LlamaRenderer(TrajectoryRenderer):
    """Llama 3 chat template."""

    def _format_system(self, text: str) -> str:
        return f"<|start_header_id|>system<|end_header_id|>\n\n{text}<|eot_id|>"

    def _format_user(self, text: str) -> str:
        return f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"

    def _format_assistant_open(self, _msg: NormalizedMessage) -> str:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def _format_assistant_close(self) -> str:
        return "<|eot_id|>"

    def _format_reasoning(self, text: str) -> str:
        return f"<thinking>\n{text}\n</thinking>\n\n"

    def _format_tool_call(self, tc) -> str:  # type: ignore[no-untyped-def]
        body = json.dumps({"name": tc.name, "parameters": tc.arguments},
                          separators=(",", ":"), ensure_ascii=False)
        return f"{body}\n"

    def _format_tool_result(self, msg: NormalizedMessage) -> str:
        return (f"<|start_header_id|>ipython<|end_header_id|>\n\n"
                f"{msg.content or ''}<|eot_id|>")


# --------------------------------------------------------------------------- #


def _spans_from_mask(mask: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, m in enumerate(mask):
        if m == 1 and start is None:
            start = i
        elif m == 0 and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def render_pair(
    win: Rollout, lose: Rollout, student: StudentSubmission, *,
    renderer: TrajectoryRenderer,
) -> tuple[RenderedSample, RenderedSample]:
    """Convenience: render both halves of a pair under the same renderer.

    Legacy helper — used by ``application.eval_student`` (pair-based flow).
    New evaluation code should call :func:`render_rollout` instead.
    """
    return (
        render_rollout(win, student, renderer=renderer),
        render_rollout(lose, student, renderer=renderer),
    )


def render_rollout(
    rollout: Rollout, student: StudentSubmission, *,
    renderer: TrajectoryRenderer,
) -> RenderedSample:
    """Decompress one rollout's trajectory and render it under ``renderer``.

    Used by the rollout-level eval flow
    (``application.eval_student_rollout``).
    """
    from distill_v2_eval.infrastructure.compression import decompress_json
    trajectory = NormalizedTrajectory.model_validate(
        decompress_json(rollout.extra_compressed)
    )
    return renderer.render(trajectory, rollout.rollout_id)
