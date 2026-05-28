"""Forward engine port.

Encapsulates GPU model lifecycle + teacher-forcing forward pass. Implementations
(``hf_flash``, ``vllm_logprob``) are swapped via config without touching the
application layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from distill_v2_eval.domain.models import RenderedSample, StudentSubmission


@runtime_checkable
class ForwardEngine(Protocol):
    """Owns one loaded student model on one GPU group."""

    name: str

    def load(self, student: StudentSubmission, *, dtype: str, max_seq_len: int) -> None: ...

    def session(self) -> AbstractContextManager[None]:
        """Reserve the GPU group for a batch of calls; releases on exit."""

    def compute_ce(self, samples: Sequence[RenderedSample]) -> list["CEMeasurement"]:
        """Sum cross-entropy over masked tokens for each sample. Same order."""

    def unload(self) -> None: ...


class CEMeasurement:
    """Lightweight result struct. Avoids importing torch in domain code."""

    __slots__ = ("ce_sum", "n_tokens", "per_message_ce")

    def __init__(self, ce_sum: float, n_tokens: int, per_message_ce: list[float] | None = None) -> None:
        self.ce_sum = ce_sum
        self.n_tokens = n_tokens
        self.per_message_ce = per_message_ce
