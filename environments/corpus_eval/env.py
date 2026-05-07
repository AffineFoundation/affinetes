"""corpus_eval Actor.

Given a ``task_id``, deterministically samples a text slice from
``karpathy/climbmix-400b-shuffle`` as a prompt, then calls a teacher
model to produce a fixed-length continuation with logprobs. The
resulting rollout is returned in a shape compatible with the
downstream ``distill`` environment (see
``environments/distill/env.py``), and in a shape compatible with
``knowledge_eval`` (task_name/score/success/time_taken/extra).

Loop / degeneration handling:
    Both the sliced prompt and the teacher continuation are checked
    for heavy repetition. If either fails, the task_id is advanced
    deterministically (shard-local first, preserving cache locality)
    and retried, up to ``max_skip`` times. All skips are recorded in
    ``extra.skipped`` / ``extra.skip_reasons`` so the caller can audit.

Shard layout:
    998 public parquet shards on HuggingFace, ~86k rows each. Shards
    are fetched on demand into an LRU disk cache (``CORPUS_CACHE_DIR``,
    default ``/tmp/corpus_cache``). See :class:`ShardCache`.

task_id encoding (locality-preserving):
    N_SHARDS   = 998
    ROW_STRIDE = 64

    shard_idx = (task_id // ROW_STRIDE) % N_SHARDS
    slot      = task_id % ROW_STRIDE
    group     = task_id // (ROW_STRIDE * N_SHARDS)

    row_flat  = slot + group * ROW_STRIDE
    row_idx   = row_flat % rows_in_shard   # on filtered rows
    variant   = row_flat // rows_in_shard  # char-offset seed
"""

import asyncio
import math
import os
import sys
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai

if "/app" not in sys.path:
    sys.path.insert(0, "/app")


N_SHARDS = 998
ROW_STRIDE = 64
SHARD_URL = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/"
    "resolve/main/shard_{idx:05d}.parquet"
)
DEFAULT_CACHE_DIR = os.getenv("CORPUS_CACHE_DIR", "/tmp/corpus_cache")
MAX_CACHED_SHARDS = int(os.getenv("CORPUS_MAX_CACHED_SHARDS", "16"))
DOWNLOAD_TIMEOUT = float(os.getenv("CORPUS_DOWNLOAD_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

# zlib compression ratio below this flags "heavy repetition" (plain
# English prose sits around 0.45-0.55). Chosen so repeated paragraphs
# or single-token loops are caught, but varied prose is not.
_LOOP_ZLIB_RATIO = 0.35
# Any 80-char window appearing twice non-overlapping => exact repeat.
# 80 chars ≈ one typical line; short enough to catch paragraph repeats
# like the Armstrong Creek degeneration seen during local tests, long
# enough that natural language bigrams don't false-positive.
_LOOP_WINDOW = 80


def _has_loop(text: str) -> Optional[str]:
    """Return a reason string if ``text`` looks degenerate, else None.

    Two cheap linear-time signals:
        * zlib compression ratio (catches dense local repetition)
        * long exact substring repeat (catches paragraph-level loops)
    """
    if not text or len(text) < 200:
        return None

    # 1. Compression ratio on bytes.
    data = text.encode("utf-8", errors="ignore")
    if len(data) >= 200:
        compressed = zlib.compress(data, 6)
        ratio = len(compressed) / len(data)
        if ratio < _LOOP_ZLIB_RATIO:
            return f"zlib_ratio={ratio:.3f}"

    # 2. Long exact-repeat detection with a rolling window.
    if len(text) >= _LOOP_WINDOW * 2:
        seen: Dict[str, int] = {}
        for i in range(len(text) - _LOOP_WINDOW + 1):
            w = text[i : i + _LOOP_WINDOW]
            prev = seen.get(w)
            if prev is not None and (i - prev) >= _LOOP_WINDOW:
                return f"exact_repeat@{prev}->{i}"
            if prev is None:
                seen[w] = i

    return None


# ---------------------------------------------------------------------------
# Shard cache
# ---------------------------------------------------------------------------


class ShardCache:
    """LRU cache of parsed climbmix shards (parsed rows + parquet file)."""

    def __init__(self, cache_dir: str, max_entries: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._parsed: "OrderedDict[int, List[str]]" = OrderedDict()
        self._locks: Dict[int, asyncio.Lock] = {}

    def _shard_path(self, shard_idx: int) -> Path:
        return self.cache_dir / f"shard_{shard_idx:05d}.parquet"

    def _evict_if_needed(self) -> None:
        while len(self._parsed) > self.max_entries:
            old_idx, _ = self._parsed.popitem(last=False)
            try:
                self._shard_path(old_idx).unlink(missing_ok=True)
            except OSError:
                pass

    async def get(self, shard_idx: int, min_chars: int) -> List[str]:
        if shard_idx in self._parsed:
            self._parsed.move_to_end(shard_idx)
            return self._parsed[shard_idx]

        lock = self._locks.setdefault(shard_idx, asyncio.Lock())
        async with lock:
            if shard_idx in self._parsed:
                self._parsed.move_to_end(shard_idx)
                return self._parsed[shard_idx]

            path = self._shard_path(shard_idx)
            if not path.exists():
                await self._download(shard_idx, path)

            texts = self._load_and_filter(path, min_chars)
            self._parsed[shard_idx] = texts
            self._evict_if_needed()
            return texts

    async def _download(self, shard_idx: int, path: Path) -> None:
        url = SHARD_URL.format(idx=shard_idx)
        tmp = path.with_suffix(path.suffix + ".tmp")
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(DOWNLOAD_TIMEOUT, connect=30.0),
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread())[:300]
                    raise RuntimeError(
                        f"Failed to download shard {shard_idx}: "
                        f"HTTP {resp.status_code} {body!r}"
                    )
                with open(tmp, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
        tmp.rename(path)

    @staticmethod
    def _load_and_filter(path: Path, min_chars: int) -> List[str]:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path), columns=["text"], memory_map=True)
        texts: List[str] = []
        for t in table.column("text").to_pylist():
            if t and len(t) >= min_chars:
                texts.append(t)
        return texts


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


def _decode_task_id(vid: int) -> Tuple[int, int]:
    """Return ``(shard_idx, row_flat)`` for a non-negative ``task_id``."""
    shard_idx = (vid // ROW_STRIDE) % N_SHARDS
    slot = vid % ROW_STRIDE
    group = vid // (ROW_STRIDE * N_SHARDS)
    row_flat = slot + group * ROW_STRIDE
    return shard_idx, row_flat


class Actor:
    """Climbmix prompt + teacher rollout generator."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self._cache = ShardCache(DEFAULT_CACHE_DIR, MAX_CACHED_SHARDS)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    async def list_tasks(self) -> Dict[str, Any]:
        return {
            "dataset": "karpathy/climbmix-400b-shuffle",
            "n_shards": N_SHARDS,
            "row_stride": ROW_STRIDE,
            "note": (
                "task_id is unbounded; corpus_eval produces a deterministic "
                "(prompt, teacher_continuation) rollout per task_id."
            ),
        }

    # ------------------------------------------------------------------
    # Prompt sampling
    # ------------------------------------------------------------------
    async def _resolve_prompt(
        self, vid: int, prompt_chars: int
    ) -> Dict[str, Any]:
        """Decode a task_id into a concrete prompt slice."""
        shard_idx, row_flat = _decode_task_id(vid)
        texts = await self._cache.get(shard_idx, min_chars=prompt_chars)
        if not texts:
            raise RuntimeError(
                f"shard {shard_idx} has no rows with len>={prompt_chars}"
            )
        rows_in_shard = len(texts)
        row_idx = row_flat % rows_in_shard
        variant = row_flat // rows_in_shard
        text = texts[row_idx]
        max_start = len(text) - prompt_chars
        if max_start <= 0:
            start_offset = 0
        else:
            start_offset = (variant * prompt_chars) % (max_start + 1)
        prompt = text[start_offset : start_offset + prompt_chars]
        return {
            "prompt": prompt,
            "shard_idx": shard_idx,
            "row_idx": row_idx,
            "variant": variant,
            "start_offset": start_offset,
        }

    # ------------------------------------------------------------------
    # Teacher call
    # ------------------------------------------------------------------
    async def _teacher_generate(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> Tuple[str, Dict[str, Any]]:
        """Plain generation call: return (completion_text, usage)."""
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )
        try:
            resp = await client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
        finally:
            await client.close()

        if not resp.choices:
            raise ValueError("teacher returned no choices")
        completion_text = resp.choices[0].text or ""
        if not completion_text:
            raise ValueError("teacher returned empty completion")
        usage = resp.usage.model_dump() if resp.usage else {}
        return completion_text, usage

    async def _teacher_echo_logprobs(
        self,
        prompt: str,
        completion: str,
        model: str,
        base_url: str,
        api_key: str,
        top_logprobs: int,
        timeout: float,
    ) -> Dict[str, Any]:
        """Echo forward pass on ``prompt + completion`` for full logprobs.

        Returns a distill-compatible dict:
            {
                "full": prompt + completion,
                "token_ids": [int, ...],      # vLLM text_offset per token
                "logprobs": [dict|None, ...]  # top-k dict for completion
                                              # tokens, None for prompt
                                              # tokens (distill skips)
            }

        The mask is built by ``text_offset`` comparison against
        ``len(prompt)``: any token whose offset starts before the
        prompt/completion boundary is masked. This makes the result
        independent of how the teacher tokenizer segments the prompt
        side, and aligns exactly with what a student echo forward
        pass will produce on the same ``full`` string.
        """
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        full_text = prompt + completion
        prompt_len = len(prompt)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )
        try:
            resp = await client.completions.create(
                model=model,
                prompt=full_text,
                max_tokens=1,
                temperature=0.0,
                logprobs=top_logprobs,
                echo=True,
                stream=False,
            )
        finally:
            await client.close()

        if not resp.choices:
            raise ValueError("teacher echo returned no choices")
        lp = resp.choices[0].logprobs
        if not lp or not lp.tokens:
            raise ValueError(
                "teacher echo returned no logprobs (endpoint may not "
                "support echo=True + logprobs)"
            )

        tokens = lp.tokens
        top_list = lp.top_logprobs or []
        text_offsets = lp.text_offset or []

        token_ids_out: List[int] = []
        logprobs_out: List[Optional[Dict[str, float]]] = []
        for i in range(len(tokens)):
            offset = int(text_offsets[i]) if i < len(text_offsets) else 0
            token_ids_out.append(offset)
            top = top_list[i] if i < len(top_list) else None
            if offset < prompt_len:
                # Prompt segment: mask so distill skips these positions.
                logprobs_out.append(None)
            elif isinstance(top, dict) and top:
                logprobs_out.append(
                    {str(k): float(v) for k, v in top.items()}
                )
            else:
                # Completion token without a top-k dict (e.g. the very
                # first echoed token can be None on some backends).
                logprobs_out.append(None)

        return {
            "full": full_text,
            "token_ids": token_ids_out,
            "logprobs": logprobs_out,
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    async def evaluate(
        self,
        task_id: Optional[int] = None,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        prompt_chars: int = 2048,
        max_completion_tokens: int = 512,
        temperature: float = 0.0,
        top_logprobs: int = 20,
        max_skip: int = 32,
        timeout: float = 300.0,
        seed: Optional[int] = None,  # accepted for framework compat
        **_: Any,
    ) -> Dict[str, Any]:
        """Produce a deterministic (prompt, teacher continuation) rollout.

        Args:
            task_id: Non-negative integer. Deterministically maps to a
                climbmix text slice; may be advanced forward if the
                slice or teacher output is degenerate.
            model, base_url, api_key: Teacher LLM endpoint. ``api_key``
                falls back to ``CHUTES_API_KEY`` env.
            prompt_chars: Length of the prompt slice in characters.
            max_completion_tokens: Teacher generation length.
            temperature: Teacher sampling temperature (default 0.0 for
                determinism; degenerate outputs are filtered via
                :func:`_has_loop` and skipped).
            top_logprobs: Top-k logprobs per completion token.
            max_skip: Upper bound on forward-scan attempts when the
                prompt or teacher output is flagged as looping.
            timeout: Per-request timeout for the teacher call.
            seed: Accepted and forwarded for framework compatibility
                but not used (teacher call is deterministic via
                temperature=0 and, if the backend honours it, seed).

        Returns: knowledge_eval-compatible result dict. On success the
        score is 1.0 and ``extra`` carries the prompt, the teacher
        continuation, a distill-compatible ``full_logprobs`` block, a
        ``conversation`` mirror, and bookkeeping for skip history.
        """
        start = time.time()
        current_api_key = api_key or self.api_key

        # --- input validation --------------------------------------------
        if task_id is None:
            return self._error(None, "validation", "task_id is required", start)
        try:
            vid_in = int(task_id)
        except (TypeError, ValueError):
            return self._error(
                task_id, "validation",
                f"task_id must be an integer, got {type(task_id).__name__}",
                start,
            )
        if vid_in < 0:
            return self._error(
                vid_in, "validation",
                f"task_id must be non-negative, got {vid_in}",
                start,
            )
        if not isinstance(prompt_chars, int) or not (0 < prompt_chars <= 1_000_000):
            return self._error(
                vid_in, "validation",
                f"prompt_chars out of range: {prompt_chars!r}",
                start,
            )
        if not isinstance(max_completion_tokens, int) or not (
            0 < max_completion_tokens <= 8192
        ):
            return self._error(
                vid_in, "validation",
                f"max_completion_tokens out of range: {max_completion_tokens!r}",
                start,
            )
        if not current_api_key:
            return self._error(
                vid_in, "validation",
                "api_key is required (pass api_key= or set CHUTES_API_KEY)",
                start,
            )

        # --- forward scan loop -------------------------------------------
        skip_reasons: List[Dict[str, Any]] = []
        last_error: Optional[Tuple[str, str]] = None  # (type, msg)
        vid = vid_in

        for attempt in range(max_skip + 1):
            # Resolve prompt for current vid.
            try:
                slice_info = await self._resolve_prompt(vid, prompt_chars)
            except Exception as e:  # noqa: BLE001
                last_error = (
                    "shard_load_failed",
                    f"{type(e).__name__}: {e}",
                )
                # Shard failures are infra — don't silently skip forever.
                break

            prompt = slice_info["prompt"]
            loop_reason = _has_loop(prompt)
            if loop_reason is not None:
                skip_reasons.append(
                    {"task_id": vid, "stage": "prompt", "reason": loop_reason}
                )
                vid += 1
                continue

            # Step 1: generation call. Cheap-ish; loop-check before
            # paying for the echo forward pass.
            try:
                completion, usage = await self._teacher_generate(
                    prompt=prompt,
                    model=model,
                    base_url=base_url,
                    api_key=current_api_key,
                    max_tokens=max_completion_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
            except Exception as e:  # noqa: BLE001
                last_error = (
                    "teacher_call_failed",
                    f"{type(e).__name__}: {e}",
                )
                break

            loop_reason = _has_loop(completion)
            if loop_reason is not None:
                skip_reasons.append(
                    {
                        "task_id": vid,
                        "stage": "completion",
                        "reason": loop_reason,
                    }
                )
                vid += 1
                continue

            # Step 2: echo forward pass for per-token top-k logprobs
            # over prompt + completion. The prompt segment is masked
            # to None so distill's index-based student alignment is
            # correct regardless of teacher/student tokenizer layout
            # on the prompt side.
            try:
                full_logprobs = await self._teacher_echo_logprobs(
                    prompt=prompt,
                    completion=completion,
                    model=model,
                    base_url=base_url,
                    api_key=current_api_key,
                    top_logprobs=top_logprobs,
                    timeout=timeout,
                )
            except Exception as e:  # noqa: BLE001
                last_error = (
                    "teacher_echo_failed",
                    f"{type(e).__name__}: {e}",
                )
                break

            # --- success path --------------------------------------------
            extra = {
                "task_id": vid_in,
                "resolved_task_id": vid,
                "shard_idx": slice_info["shard_idx"],
                "row_idx": slice_info["row_idx"],
                "variant": slice_info["variant"],
                "start_offset": slice_info["start_offset"],
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "completion": completion,
                "full_logprobs": full_logprobs,
                "model": model,
                "temperature": temperature,
                "usage": usage,
                "skipped": len(skip_reasons),
                "skip_reasons": skip_reasons,
                "dataset": "karpathy/climbmix-400b-shuffle",
                "conversation": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            }
            return {
                "task_name": "corpus_eval",
                "score": 1.0,
                "success": True,
                "time_taken": time.time() - start,
                "extra": extra,
            }

        # --- exhausted or errored ----------------------------------------
        if last_error is not None:
            err_type, err_msg = last_error
        else:
            err_type = "max_skip_exhausted"
            err_msg = (
                f"no clean rollout within {max_skip} attempts starting "
                f"at task_id={vid_in}"
            )
        return self._error(
            vid_in,
            err_type,
            err_msg,
            start,
            extra={
                "skipped": len(skip_reasons),
                "skip_reasons": skip_reasons,
                "last_attempt_task_id": vid,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _error(
        task_id: Optional[int],
        error_type: str,
        message: str,
        start_time: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base_extra: Dict[str, Any] = {
            "task_id": task_id,
            "error_type": error_type,
        }
        if extra:
            base_extra.update(extra)
        return {
            "task_name": "corpus_eval",
            "score": 0.0,
            "success": False,
            "time_taken": time.time() - start_time,
            "extra": base_extra,
            "error": message,
            "error_type": error_type,
        }
