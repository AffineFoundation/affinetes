"""Distill-V2 Environment Actor.

Per-cell rollout-level CE-diff evaluator.

A "cell" is one ``(list_name, env_name, task_id, teacher_name)`` group
of N rollouts, published as an immutable shard by cortex's
``af servers generator-publisher``. The Actor receives ``task_id``
(which we treat as the cortex-side ``task_idx`` into the manifest),
resolves it to a shard URL, scores the miner against those N rollouts
under the caller-supplied vLLM endpoint, and returns one score.

Same (task_id, miner, base_url) → same score, always: the shard is
immutable, the renderer is deterministic, and the vLLM endpoint
performs only teacher-forcing (max_tokens=0, echo=true, no sampling).

Distinct from cortex's pre-existing ``distill`` env: that one is a
direct token-level KL between student and stored teacher logprobs.
This one (``distill-v2``) is REINFORCE/GRPO-style — per-rollout
advantage-weighted CE on tokens the agent actually committed to, with
advantage computed strictly within the cell (one teacher) to avoid
style leakage between teachers.

All evaluator code is vendored under ``distill_v2_eval/`` in this
image; the environment is self-contained.

Contract (affinetes' standard Actor protocol):

    await actor.evaluate(
        task_id=<task_idx into manifest>,    # required
        model="<served-model>",              # vLLM --served-model-name
        base_url="http://...",               # vLLM OpenAI-compatible
        student_name="<hotkey>",             # required
        revision="<commit>",                 # required
        hf_repo="<owner/repo>",              # tokenizer source; defaults to model
        arch="qwen",                         # renderer family
        dataset_base_url="https://...",      # R2 base URL (or DISTILL_V2_DATASET_BASE_URL env var)
        scoring_name="reward_weighted_ce",
        max_seq_len=32768,
    )
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any

if "/app" not in sys.path:
    sys.path.insert(0, "/app")


# Public R2 bucket holding the published rollout shards. Hardcoded so
# callers don't need to pass ``dataset_base_url`` or set the env var on
# every host; override at runtime via ``DISTILL_V2_DATASET_BASE_URL`` if
# the dataset ever moves.
_DEFAULT_DATASET_BASE_URL = "https://pub-f6e8aaa82d31450daad5b16e3ca040b1.r2.dev"


class Actor:
    """Distill-V2 evaluator actor.

    The heavy import (vendored distill_v2_eval package) is deferred to
    first use so ``afs build`` can introspect this module before deps
    install.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # ``api_key`` accepted for parity with other affinetes actors; the
        # evaluator never calls an LLM API directly — the caller-supplied
        # vLLM endpoint owns inference.
        self._api_key = api_key or os.getenv("CHUTES_API_KEY")
        self._evaluator = None  # lazy

    def _ensure(self):  # type: ignore[no-untyped-def]
        if self._evaluator is None:
            from distill_v2_eval.driver import DistillV2Evaluator
            self._evaluator = DistillV2Evaluator()
        return self._evaluator

    async def evaluate(
        self,
        *,
        task_id: int | None = None,
        model: str = "",
        base_url: str = "",
        # protocol parity (unused by evaluator)
        seed: int | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        # miner identification
        student_name: str = "",
        revision: str = "",
        hf_repo: str | None = None,
        arch: str = "qwen",
        # rollout source — base URL of the cortex publisher's R2 prefix.
        # Falls back to DISTILL_V2_DATASET_BASE_URL env var when the
        # caller doesn't pass it, so the validator can configure the
        # dataset location once at container start.
        dataset_base_url: str = "",
        # scoring knobs
        scoring_name: str = "reward_weighted_ce",
        dtype: str = "bfloat16",
        max_seq_len: int = 32768,
        mask_reasoning: bool = True,
        **_unused: Any,
    ) -> dict[str, Any]:
        del seed, api_key, temperature, timeout, _unused

        start = time.time()
        try:
            dataset_base_url = (
                dataset_base_url
                or os.getenv("DISTILL_V2_DATASET_BASE_URL", "")
                or _DEFAULT_DATASET_BASE_URL
            )
            self._require("model", model)
            self._require("base_url", base_url)
            self._require("student_name", student_name)
            self._require("revision", revision)
            self._require("dataset_base_url", dataset_base_url)
            if task_id is None:
                raise ValueError("distill-v2.evaluate requires `task_id`")

            evaluator = self._ensure()
            snapshot = await evaluator.evaluate(
                student_name=student_name,
                revision=revision,
                hf_repo=hf_repo or model,
                arch=arch,
                base_url=base_url,
                served_model_name=model,
                dataset_base_url=dataset_base_url,
                task_idx=int(task_id),
                scoring_name=scoring_name,
                dtype=dtype,
                max_seq_len=max_seq_len,
                mask_reasoning=mask_reasoning,
            )
        except Exception as exc:
            return {
                "task_name": "distill-v2",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": "distill_v2_eval_failure",
                "extra": {
                    "task_id": task_id,
                    "traceback": traceback.format_exc()[:4000],
                },
            }

        return {
            "task_name": "distill-v2",
            "score": float(snapshot["mean_score"]),
            "success": snapshot["n_rollouts"] > 0,
            "time_taken": time.time() - start,
            "extra": {
                "task_id": task_id,
                "n_rollouts": snapshot["n_rollouts"],
                "win_rate": snapshot["win_rate"],
                "total_forward_tokens": snapshot.get("total_forward_tokens"),
                "cell": snapshot.get("cell"),
                "per_rollout": snapshot.get("per_rollout"),
                "dataset_base_url": dataset_base_url,
            },
        }

    @staticmethod
    def _require(name: str, value: str) -> None:
        if not value:
            raise ValueError(f"distill-v2.evaluate requires non-empty `{name}`")
