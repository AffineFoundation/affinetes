"""vLLM-backed forward engine — teacher-forcing CE via the OpenAI completion
API's prompt-logprob mode.

Why this exists
---------------

``hf_flash`` only handles models whose architecture matches
``AutoModelForCausalLM``. Newer Qwen 3.5 / 3.6 releases ship under
``Qwen3_5*ForConditionalGeneration`` (multimodal, often MoE), which
``AutoModelForCausalLM`` cannot load. vLLM, on the other hand, supports
these architectures natively and is typically already up as a serving
process. We piggy-back on it: send the renderer's ``input_ids`` to
``/v1/completions`` with ``max_tokens=0 echo=true logprobs=0`` and sum
the returned per-position logprobs over the ``loss_mask`` positions.

Wire shape (vLLM's OpenAI-compatible response, with echo=true):

    response.choices[0].logprobs.token_logprobs
        # length == len(prompt). First entry is null
        # (no preceding token to predict it). Each subsequent entry is
        # log p(prompt_token_i | prompt[:i]) under the model.

We deliberately do *not* call ``/v1/chat/completions`` — the renderer
already applied the student's chat template; we want the raw token-id
forward, not vLLM's re-templating.

Tokenizer alignment
-------------------

The renderer must encode trajectories under the **vLLM-served model's
own tokenizer** for the input_ids to line up with the served vocab. We
expose that tokenizer via ``.tokenizer``, identical contract to
``hf_flash`` — eval consumer / CLI uses it transparently.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Iterator

import orjson

from distill_v2_eval.domain.errors import EvalError
from distill_v2_eval.domain.models import RenderedSample, StudentSubmission
from distill_v2_eval.domain.ports.forward_engine import CEMeasurement
from distill_v2_eval.infrastructure.logging import get_logger

log = get_logger("forward.vllm_logprob")


class VLLMLogprobEngine:
    """Forward engine that scores a teacher-forced prompt via vLLM."""

    name = "vllm_logprob"

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        # Per-forward read timeout against the miner's vLLM endpoint. A
        # single cell forwards N (8~13) rollouts serially, each an
        # ``echo=true`` full-prompt prefill of up to ~tens of thousands of
        # tokens. Slow / loaded miner endpoints routinely take minutes on
        # one such request; at the old 300s default every such cell hit
        # ReadTimeout and the whole task was scored INVALID, so distill-v2
        # produced almost no samples. 600s lets a slow endpoint finish a
        # single forward while staying under cortex's outer proxy timeout
        # (1260s) for the cell as a whole.
        request_timeout_s: float = 600.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = request_timeout_s
        self._tokenizer = None
        self._client = None
        self._dtype = "bfloat16"

    # ---- lifecycle ---- #

    def load(self, student: StudentSubmission, *, dtype: str, max_seq_len: int) -> None:  # noqa: ARG002
        """Build the local tokenizer and verify vLLM is serving the model.

        ``dtype``/``max_seq_len`` are informational — the vLLM server decided
        those at its own startup. We accept them so the call signature stays
        interchangeable with :class:`HFForwardEngine`.
        """
        import httpx

        from distill_v2_eval.adapters.forward_engines.tokenizer_cache import (
            get_tokenizer,
        )

        if self._client is not None:
            raise EvalError("engine already loaded; call unload() first")

        source = student.hf_repo or (
            student.meta.get("local_path") if isinstance(student.meta, dict) else None
        )
        if not source:
            raise EvalError(
                f"student {student.student_name}@{student.revision} has no hf_repo / local_path"
            )

        # Shared, content-deduplicated instance — do NOT mutate it here.
        self._tokenizer = get_tokenizer(source)

        self._client = httpx.Client(timeout=self._timeout)
        # Sanity check the served model id matches what we plan to request.
        resp = self._client.get(f"{self._base_url}/models")
        resp.raise_for_status()
        served = [m["id"] for m in resp.json().get("data", [])]
        if self._model_name not in served:
            raise EvalError(
                f"vLLM at {self._base_url} does not serve {self._model_name!r}; got {served}"
            )
        log.info("forward.load",
                 source=source, base_url=self._base_url,
                 model_name=self._model_name, dtype=dtype)

    def unload(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
        self._tokenizer = None

    @contextmanager
    def session(self) -> Iterator[None]:
        """No batching state to reserve — vLLM owns its own KV cache."""
        yield

    @property
    def tokenizer(self):  # type: ignore[no-untyped-def]
        if self._tokenizer is None:
            raise EvalError("engine not loaded")
        return self._tokenizer

    # ---- forward + CE ---- #

    def compute_ce(self, samples: Sequence[RenderedSample]) -> list[CEMeasurement]:
        if self._client is None:
            raise EvalError("engine not loaded")
        return [self._score_one(s) for s in samples]

    def _score_one(self, sample: RenderedSample) -> CEMeasurement:
        # vLLM accepts a list of token IDs as the prompt; sending IDs avoids
        # re-tokenization drift (the renderer already used the right vocab).
        #
        # ``max_tokens`` and ``logprobs`` are both set to 1 (not 0) because
        # sglang rejects ``max_tokens=0`` (400 "must be positive") and
        # crashes on ``logprobs=0`` with echo=true ("input_top_logprobs"
        # KeyError, 500). vLLM accepts these values too. ``_ce_from_logprobs``
        # truncates to ``len(loss_mask)`` so the extra trailing generated
        # token doesn't enter the CE sum.
        payload = {
            "model": self._model_name,
            "prompt": list(sample.input_ids),
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
            "temperature": 0,
        }
        resp = self._client.post(f"{self._base_url}/completions", json=payload)  # type: ignore[union-attr]
        resp.raise_for_status()
        # The echo response also carries tokens / top_logprobs / text_offset —
        # tens of MB of Python objects at 32k positions. Extract the one field
        # we use and drop the rest before CE computation, so concurrent
        # forwards don't stack full parsed bodies.
        body = orjson.loads(resp.content)
        token_lp = body["choices"][0]["logprobs"]["token_logprobs"]
        del body, resp
        return _ce_from_logprobs(
            token_logprobs=token_lp,
            loss_mask=list(sample.loss_mask),
            message_spans=list(sample.message_spans),
        )


# --------------------------------------------------------------------------- #


def _ce_from_logprobs(
    *,
    token_logprobs: list[float | None],
    loss_mask: list[int],
    message_spans: list[tuple[int, int]],
) -> CEMeasurement:
    """Convert vLLM's prompt logprobs into a teacher-forcing CE measurement.

    vLLM's ``token_logprobs[i]`` is ``log p(token[i] | token[:i])`` — exactly
    the teacher-forcing signal we want. The first entry is None (the model
    can't score the very first token without context); the renderer's
    loss_mask should be 0 there anyway, but we guard against off-by-one with
    an explicit ``is None`` check.

    Mirrors :func:`hf_flash._compute_ce`:

      * total CE is the sum of -log p over masked positions;
      * per-message CE sums the same quantity within each
        ``message_spans`` range, used for diagnostics.
    """
    n = min(len(token_logprobs), len(loss_mask))
    ce_sum = 0.0
    n_tokens = 0
    for i in range(n):
        if loss_mask[i] != 1:
            continue
        lp = token_logprobs[i]
        if lp is None:
            # Should not happen if the renderer plays nice (mask=0 on the
            # leading boundary token), but if it does we skip to avoid
            # silently zeroing out CE.
            continue
        ce_sum += -float(lp)
        n_tokens += 1

    per_message: list[float] = []
    for start, end in message_spans:
        m_sum = 0.0
        for i in range(max(start, 0), min(end, n)):
            lp = token_logprobs[i]
            if lp is None or loss_mask[i] != 1:
                continue
            m_sum += -float(lp)
        per_message.append(m_sum)

    return CEMeasurement(ce_sum=ce_sum, n_tokens=n_tokens, per_message_ce=per_message)
