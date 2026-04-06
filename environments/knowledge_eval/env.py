"""knowledge_eval Actor.

A single environment that exposes ``evaluate(task_id, ...)`` over four
public knowledge / instruction-following benchmarks: GPQA-diamond,
MMLU-Pro, HLE (multiple choice text-only) and IFEval (simple subset).

Routing rules:
    * If ``task_id`` is provided alone, it is treated as a *global* id
      and the manifest decides which dataset to use.
    * If ``task_type`` is also provided, ``task_id`` is treated as a
      local index inside that dataset.
    * If neither is provided, a random global id is drawn.
"""

import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
import openai

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from batching import build_batch_prompt, split_batch_response  # noqa: E402
from distractor_pool import DistractorPool  # noqa: E402
from gpqa import GPQATask  # noqa: E402
from hle import HLETask  # noqa: E402
from ifeval import IFEvalTask  # noqa: E402
from manifest import Manifest  # noqa: E402
from mmlu_pro import MMLUProTask  # noqa: E402

# Loaded once at process start; ~14k entries totalling a few MB.
_MANIFEST = Manifest()

_TASKS = {
    "gpqa": GPQATask(),
    "mmlu_pro": MMLUProTask(),
    "hle": HLETask(),
    "ifeval": IFEvalTask(),
}

# Build cross-question distractor pools for every MC task. HLE's pool
# only contributes to the cross_pool *extend* mode (its options are
# too question-specific to fully replace via distractor_swap, so that
# mode still falls back to shuffle for HLE).
_POOLS = {
    "gpqa": DistractorPool(_MANIFEST.rows("gpqa"), "gpqa"),
    "mmlu_pro": DistractorPool(_MANIFEST.rows("mmlu_pro"), "mmlu_pro"),
    "hle": DistractorPool(_MANIFEST.rows("hle"), "hle"),
}

_VALID_MODES = {"off", "shuffle", "distractor_swap", "cross_pool"}


class Actor:
    """Multi-benchmark knowledge evaluator."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

    # ------------------------------------------------------------------
    # LLM call (mirrors affine env so behaviour stays consistent)
    # ------------------------------------------------------------------
    async def _llm_chat(
        self,
        prompt: str,
        model: str,
        base_url: str,
        timeout: float,
        temperature: float,
        api_key: str,
        seed: Optional[int] = None,
    ):
        # SSL_CERT_FILE / REQUESTS_CA_BUNDLE in some container images point
        # at a path that doesn't exist; let httpx fall back to certifi.
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )

        params: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            params["seed"] = seed

        stream = await client.chat.completions.create(**params)

        parts = []
        usage = None
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                usage = chunk.usage.model_dump()

        if not parts:
            raise ValueError("LLM API returned empty content stream")
        content = "".join(parts).strip()
        if not content:
            raise ValueError("LLM API returned empty content")
        return content, usage

    # ------------------------------------------------------------------
    # Logprobs side-channel
    # ------------------------------------------------------------------
    async def _attach_full_logprobs(
        self,
        result: Dict[str, Any],
        *,
        conversation: List[Dict[str, Any]],
        model: str,
        base_url: str,
        api_key: str,
    ) -> None:
        """Run a forward pass over ``conversation`` with ``echo=True`` and
        attach the resulting per-token logprobs to ``result['extra']``.

        Failures are non-fatal: ``full_logprobs`` becomes ``None`` and
        the error message goes into ``logprobs_error`` so the caller
        can audit. Imports are lazy so the module still loads in
        environments where ``affinetes`` isn't on the path (unit tests).
        """
        try:
            from affinetes.core.logprobs_utils import collect_full_logprobs
            full_logprobs = await collect_full_logprobs(
                conversation=conversation,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            result["extra"]["full_logprobs"] = full_logprobs
            if full_logprobs is None:
                # Distinguish "framework asked for logprobs but the
                # upstream returned nothing usable" from a successful
                # echo. The most common cause is an inference endpoint
                # (e.g. chutes) that doesn't honour echo+logprobs on
                # /v1/completions; users should point base_url at a
                # vLLM/SGLang instance that does.
                result["extra"]["logprobs_error"] = (
                    "endpoint returned no logprobs (echo+logprobs not supported)"
                )
        except Exception as e:  # noqa: BLE001
            result["extra"]["full_logprobs"] = None
            result["extra"]["logprobs_error"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _global_id_for(task_type: str, local_id: int) -> int:
        for entry in _MANIFEST.ranges:
            if entry["task_type"] == task_type:
                return entry["start"] + local_id
        raise ValueError(f"Unknown task_type {task_type!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def list_tasks(self) -> Dict[str, Any]:
        """Inspect available task types and their global id ranges."""
        return _MANIFEST.stats()

    async def evaluate(
        self,
        task_id: Optional[int] = None,
        task_type: Optional[str] = None,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: float = 600,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        anti_contam: str = "shuffle",
        perturb_seed: Optional[int] = None,
        extra_distractors: int = 4,
        collect_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Run a single evaluation.

        Args:
            task_id: Global task id (no ``task_type``) or local id within a
                dataset (with ``task_type``). If omitted, a random global
                id is drawn.
            task_type: Optional dataset selector
                (``gpqa`` / ``mmlu_pro`` / ``hle`` / ``ifeval``).
            model, base_url, timeout, temperature, api_key, seed:
                Standard LLM call knobs. ``api_key`` falls back to the
                instance value (CHUTES_API_KEY env at construction time).
            anti_contam: Anti-contamination mode for MC tasks.
                ``off`` keeps the original option order; ``shuffle``
                deterministically reorders options; ``distractor_swap``
                replaces incorrect options with same-category
                distractors from other questions and then shuffles.
                IFEval ignores this. HLE falls back to ``shuffle`` for
                ``distractor_swap``.
            perturb_seed: Seed for the deterministic shuffle / swap.
                Defaults to ``task_id`` so a bare ``task_id`` is fully
                reproducible. Pass a different value to materialise an
                independent variant of the same question.
            extra_distractors: Only used when ``anti_contam='cross_pool'``.
                Number of cross-question options to *append* to the
                original choices. The original options are kept intact
                so the question's intrinsic difficulty is preserved;
                the extras come from a pool that mixes other questions'
                distractors AND correct answers, so memorising the
                answer text alone doesn't help. Defaults to 4.
            collect_logprobs: When True, after the LLM response is
                obtained, do a forward pass with ``echo=True`` to
                recover full per-token logprobs for the conversation.
                The result is placed in
                ``result['extra']['full_logprobs']`` (None on failure,
                error string in ``result['extra']['logprobs_error']``).
        """
        try:
            if anti_contam not in _VALID_MODES:
                raise ValueError(
                    f"anti_contam must be one of {sorted(_VALID_MODES)}, "
                    f"got {anti_contam!r}"
                )
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            current_api_key = api_key or self.api_key

            # Resolve which sample to use. task_id is *virtual* — values
            # past the canonical range wrap and the integer quotient
            # becomes the auto perturb_seed, so a single int fully
            # determines a deterministic variant of a question.
            if task_type is None:
                if task_id is None:
                    task_id = random.randint(0, _MANIFEST.total - 1)
                if int(task_id) < 0:
                    raise ValueError(
                        f"task_id must be non-negative, got {task_id}"
                    )
                base = _MANIFEST.total
                canonical = int(task_id) % base
                auto_seed_from_id = int(task_id) // base
                resolved_type, local_id, sample = _MANIFEST.resolve(canonical)
                global_id = int(task_id)
            else:
                if task_type not in _TASKS:
                    raise ValueError(
                        f"Unknown task_type {task_type!r}. Available: {list(_TASKS)}"
                    )
                if task_id is None:
                    task_id = random.randint(0, _MANIFEST.count(task_type) - 1)
                base = _MANIFEST.count(task_type)
                if int(task_id) < 0:
                    raise ValueError(
                        f"local task_id must be non-negative, got {task_id}"
                    )
                local_id = int(task_id) % base
                auto_seed_from_id = int(task_id) // base
                sample = _MANIFEST.get_local(task_type, local_id)
                resolved_type = task_type
                global_id = self._global_id_for(task_type, local_id)

            task = _TASKS[resolved_type]
            # If perturb_seed is explicit, it wins. Otherwise the high
            # bits of the virtual task_id supply it (so task_id alone is
            # enough to uniquely pick a variant).
            effective_perturb_seed = (
                perturb_seed if perturb_seed is not None else auto_seed_from_id
            )
            pool = _POOLS.get(resolved_type)

            start = time.time()
            challenge = await task.generate(
                sample=sample,
                task_id=local_id,
                mode=anti_contam,
                perturb_seed=effective_perturb_seed,
                pool=pool,
                extra_distractors=extra_distractors,
            )
        except (ValueError, KeyError, TypeError, IndexError) as ve:
            # Validation / build-time errors get a clean structured
            # response instead of a 500 from the framework. We catch the
            # full family of programming-style exceptions that can occur
            # during sample resolution and challenge generation, but
            # *not* runtime LLM errors — those have their own handler
            # below.
            return {
                "task_name": "knowledge_eval",
                "score": 0.0,
                "success": False,
                "time_taken": 0.0,
                "extra": {"anti_contam": anti_contam, "task_id": task_id},
                "error": f"{type(ve).__name__}: {ve}",
                "error_type": "validation",
            }

        usage = None
        error = None
        resp = None
        try:
            resp, usage = await self._llm_chat(
                challenge.prompt,
                model,
                base_url,
                timeout,
                temperature,
                current_api_key,
                seed,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        score = 0.0
        if resp:
            score = await task.evaluate(resp, challenge)

        # Result extra inherits everything from challenge.extra (so
        # task-specific fields like correct_letter, valid_letters,
        # rendered_options, instruction_id_list, kwargs, base_prompt_idx
        # are automatically surfaced) and is then overlaid with the
        # framework's bookkeeping fields.
        ch_extra = dict(challenge.extra)
        result_extra: Dict[str, Any] = {
            **ch_extra,
            "task_type": resolved_type,
            "global_task_id": global_id,
            "local_task_id": local_id,
            "seed": seed,
            "anti_contam": anti_contam,
            "perturb_seed": effective_perturb_seed,
            "n_options": (
                len(ch_extra["rendered_options"])
                if ch_extra.get("rendered_options") is not None
                else None
            ),
            "usage": usage,
            "conversation": [
                {"role": "user", "content": challenge.prompt},
                {"role": "assistant", "content": resp},
            ],
        }
        result: Dict[str, Any] = {
            "task_name": f"knowledge_eval:{resolved_type}",
            "score": float(score),
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": result_extra,
        }

        # IFEval: include per-instruction breakdown for debugging.
        if resolved_type == "ifeval" and resp is not None:
            _, breakdown = _TASKS["ifeval"].detail(resp, challenge)
            result["extra"]["ifeval_breakdown"] = breakdown

        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Optional post-rollout forward pass for token-level logprobs.
        if collect_logprobs and resp:
            await self._attach_full_logprobs(
                result,
                conversation=[
                    {"role": "user", "content": challenge.prompt},
                    {"role": "assistant", "content": resp},
                ],
                model=model,
                base_url=base_url,
                api_key=current_api_key,
            )

        return result

    async def evaluate_batch(
        self,
        task_ids: List[int],
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: float = 600,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        anti_contam: str = "shuffle",
        perturb_seed: Optional[int] = None,
        extra_distractors: int = 4,
        collect_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate multiple tasks (possibly across benchmarks) in **one** LLM call.

        The model receives a single prompt that bundles all the
        questions, separated by ``--- Question K ---`` headers, and is
        instructed to emit ``=== Answer K ===`` blocks. Each block is
        then routed back to its task's evaluator.

        Why this exists:
            * Each task is evaluated in the *context* of unrelated
              tasks, which defeats per-prompt cache lookups and any
              "memorise the exact prompt" strategies.
            * Cross-benchmark batches mix MC and IFEval prompts in one
              shot — the model can't pre-commit a single output format.

        Args:
            task_ids: List of *global* task ids. Use :meth:`list_tasks`
                to look up the ranges. Order matters — answer K
                corresponds to ``task_ids[K-1]``.
            anti_contam, perturb_seed: Same semantics as
                :meth:`evaluate`. Applied to every MC item in the
                batch. ``perturb_seed`` defaults to each item's
                ``local_task_id``.
            Other args: same as :meth:`evaluate`.

        Returns a dict with a top-level mean ``score``, an ``items``
        breakdown, and the joint conversation for inspection.
        """
        try:
            if not isinstance(task_ids, (list, tuple)):
                raise ValueError(
                    f"task_ids must be a list or tuple of ints, got {type(task_ids).__name__}"
                )
            if not task_ids:
                raise ValueError("evaluate_batch requires a non-empty task_ids list")
            if anti_contam not in _VALID_MODES:
                raise ValueError(
                    f"anti_contam must be one of {sorted(_VALID_MODES)}, "
                    f"got {anti_contam!r}"
                )
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            current_api_key = api_key or self.api_key

            start = time.time()

            # 1. Resolve every id and generate its individual challenge.
            items_meta: List[Dict[str, Any]] = []
            per_prompts: List[str] = []
            challenges = []
            for raw_id in task_ids:
                vid = int(raw_id)
                if vid < 0:
                    raise ValueError(f"task_id must be non-negative, got {raw_id}")
                base = _MANIFEST.total
                canonical = vid % base
                auto_seed_from_id = vid // base
                resolved_type, local_id, sample = _MANIFEST.resolve(canonical)
                task = _TASKS[resolved_type]
                item_perturb = (
                    perturb_seed if perturb_seed is not None else auto_seed_from_id
                )
                challenge = await task.generate(
                    sample=sample,
                    task_id=local_id,
                    mode=anti_contam,
                    perturb_seed=item_perturb,
                    pool=_POOLS.get(resolved_type),
                    extra_distractors=extra_distractors,
                )
                challenges.append(challenge)
                per_prompts.append(challenge.prompt)
                items_meta.append(
                    {
                        "task_type": resolved_type,
                        "global_task_id": vid,
                        "local_task_id": local_id,
                        "perturb_seed": item_perturb,
                    }
                )
        except (ValueError, KeyError, TypeError, IndexError) as ve:
            safe_ids = list(task_ids) if isinstance(task_ids, (list, tuple)) else None
            return {
                "task_name": "knowledge_eval:batch",
                "score": 0.0,
                "success": False,
                "time_taken": 0.0,
                "extra": {"anti_contam": anti_contam, "task_ids": safe_ids},
                "error": f"{type(ve).__name__}: {ve}",
                "error_type": "validation",
            }

        # 2. Build joint prompt and call the model exactly once.
        joint_prompt = build_batch_prompt(per_prompts)

        usage = None
        error = None
        resp = None
        try:
            resp, usage = await self._llm_chat(
                joint_prompt,
                model,
                base_url,
                timeout,
                temperature,
                current_api_key,
                seed,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        # 3. Slice response and route each chunk to its task evaluator.
        chunks = split_batch_response(resp or "", len(task_ids))
        per_item_results: List[Dict[str, Any]] = []
        scores: List[float] = []
        for meta, challenge, chunk in zip(items_meta, challenges, chunks):
            task = _TASKS[meta["task_type"]]
            item_score = 0.0
            if chunk:
                item_score = float(await task.evaluate(chunk, challenge))
            scores.append(item_score)
            # Inherit challenge.extra so per-item entries carry
            # correct_letter / valid_letters / rendered_options / kwargs
            # / instruction_id_list etc. without needing per-task patching.
            entry = {
                **dict(challenge.extra),
                **meta,
                "score": item_score,
                "success": item_score > 0,
                "response": chunk,
            }
            if meta["task_type"] == "ifeval" and chunk:
                _, breakdown = _TASKS["ifeval"].detail(chunk, challenge)
                entry["ifeval_breakdown"] = breakdown
            per_item_results.append(entry)

        mean_score = sum(scores) / len(scores) if scores else 0.0

        result: Dict[str, Any] = {
            "task_name": "knowledge_eval:batch",
            "score": mean_score,
            "success": all(s > 0 for s in scores),
            "time_taken": time.time() - start,
            "extra": {
                "batch_size": len(task_ids),
                "anti_contam": anti_contam,
                "seed": seed,
                "usage": usage,
                "items": per_item_results,
                "conversation": [
                    {"role": "user", "content": joint_prompt},
                    {"role": "assistant", "content": resp},
                ],
            },
        }
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Optional forward pass for the joint conversation. The joint
        # batch prompt is treated as a single user/assistant turn —
        # logprobs are computed over the entire combined response.
        if collect_logprobs and resp:
            await self._attach_full_logprobs(
                result,
                conversation=[
                    {"role": "user", "content": joint_prompt},
                    {"role": "assistant", "content": resp},
                ],
                model=model,
                base_url=base_url,
                api_key=current_api_key,
            )
        return result
