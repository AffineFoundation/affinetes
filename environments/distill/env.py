"""KL Divergence Environment.

Computes KL divergence between a student model and pre-generated teacher rollouts.

Flow:
    1. Load teacher rollout from R2 by task_id (same pattern as SWE-INFINITE)
       Rollout contains full_logprobs: {full, token_ids, logprobs}
    2. Student forward pass: /v1/completions with echo=True + logprobs=1
    3. Align by token position, compute KL on assistant tokens (non-None logprobs)

Task ID:
    Plain integer, maps to a teacher rollout file in R2:
      {R2_BASE_URL}/{R2_PREFIX}/task_{task_id:011d}.json

    The rollout JSON must contain a "full_logprobs" key with:
      full: str           - formatted conversation text
      token_ids: [int]    - token positions (or offsets)
      logprobs: [float|None] - None = masked, float = assistant token logprob
"""

import math
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import httpx


class LocalCache:
    """Local filesystem cache for teacher rollouts."""

    def __init__(self, cache_dir: str = "/tmp/kl-cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: int) -> Path:
        return self.cache_dir / f"task_{task_id:011d}.json"

    def load(self, task_id: int) -> Optional[Dict]:
        p = self._path(task_id)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def save(self, task_id: int, data: Dict) -> None:
        with open(self._path(task_id), "w") as f:
            json.dump(data, f, separators=(",", ":"))


class R2Cache:
    """Read-only R2 public HTTP cache for teacher rollouts."""

    def __init__(self, base_url: str, prefix: str):
        self.base_url = base_url.rstrip("/")
        self.prefix = prefix.rstrip("/")

    def _url(self, task_id: int) -> str:
        filename = f"task_{task_id:011d}.json"
        return f"{self.base_url}/{self.prefix}/{filename}"

    def load(self, task_id: int) -> Optional[Dict]:
        url = self._url(task_id)
        try:
            req = Request(url, headers={"User-Agent": "kl-env/1.0"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code != 404:
                print(f"[KL] R2 error for {task_id}: {e.code}")
            return None
        except (URLError, TimeoutError) as e:
            print(f"[KL] R2 fetch error for {task_id}: {e}")
            return None


class Actor:
    """Computes KL divergence between student model and teacher rollouts."""

    def __init__(self):
        r2_base = os.getenv("KL_R2_BASE_URL", "https://pub-7882418a56434a479bf9a7febd660b36.r2.dev")
        r2_prefix = os.getenv("KL_R2_PREFIX", "teacher_rollouts")
        cache_dir = os.getenv("KL_CACHE_DIR", "/tmp/kl-cache")

        self._local = LocalCache(cache_dir)
        self._r2 = R2Cache(r2_base, r2_prefix)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
            verify=False,
        )

    def _load_rollout(self, task_id: int) -> Optional[Dict]:
        """Load teacher rollout: L1 local -> L2 R2."""
        data = self._local.load(task_id)
        if data is not None:
            return data
        data = self._r2.load(task_id)
        if data is not None:
            self._local.save(task_id, data)
            return data
        return None

    async def _student_forward_pass(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float,
    ) -> tuple:
        """Student forward pass with echo=True to get logprobs for all tokens.

        Returns (logprobs_dict, error_info) where:
          - On success: (logprobs_dict, None)
          - On failure: (None, {"error_type": str, "message": str, "status": int|None})

        Error types:
          - rate_limit      : HTTP 429 (Chutes infra capacity / upstream rate limit)
          - no_instance     : HTTP 503 (miner chute cold / no instances)
          - upstream_error  : HTTP 502 / 504 (gateway/upstream)
          - bad_request     : HTTP 400 / 404 (malformed prompt, unknown model)
          - auth_error      : HTTP 401 / 403
          - http_error      : any other non-200
          - timeout         : httpx.TimeoutException
          - connect_error   : httpx connection failure
          - empty_response  : 200 but no choices
          - no_logprobs     : 200 but choices[0].logprobs is None
          - unexpected      : anything else
        """
        try:
            resp = await self._http.post(
                f"{base_url.rstrip('/')}/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "logprobs": 1,
                    "echo": True,
                    "stream": False,
                },
                timeout=timeout,
            )
        except httpx.TimeoutException:
            return None, {
                "error_type": "timeout",
                "status": None,
                "message": f"Student forward pass timed out after {timeout}s",
            }
        except httpx.ConnectError as e:
            return None, {
                "error_type": "connect_error",
                "status": None,
                "message": f"Student connect error: {e}",
            }
        except Exception as e:
            return None, {
                "error_type": "unexpected",
                "status": None,
                "message": f"Student HTTP error: {type(e).__name__}: {e}",
            }

        if resp.status_code != 200:
            body = resp.text[:300]
            status = resp.status_code
            if status == 429:
                err_type = "rate_limit"
            elif status == 503:
                err_type = "no_instance"
            elif status in (502, 504):
                err_type = "upstream_error"
            elif status in (400, 404):
                err_type = "bad_request"
            elif status in (401, 403):
                err_type = "auth_error"
            else:
                err_type = "http_error"
            msg = f"Student forward pass failed: HTTP {status} ({err_type}): {body}"
            print(f"[DISTILL] {msg}")
            return None, {"error_type": err_type, "status": status, "message": msg}

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None, {
                "error_type": "empty_response",
                "status": 200,
                "message": "Student response has no choices",
            }

        lp = choices[0].get("logprobs")
        if lp is None:
            return None, {
                "error_type": "no_logprobs",
                "status": 200,
                "message": "Student response has no logprobs (model may not support echo+logprobs)",
            }
        return lp, None

    def _compute_kl(
        self,
        teacher_logprobs: List,
        student_lp_dict: Dict,
    ) -> Dict[str, Any]:
        """Compute KL divergence between teacher and student logprobs.

        Both teacher and student have logprobs for the same token sequence.
        Teacher logprobs[i] is None for non-assistant tokens.
        Student logprobs come from echo=True (covers all tokens).

        We align by position: same tokenizer -> same token sequence.
        """
        student_token_logprobs = student_lp_dict.get("token_logprobs", [])

        total_kl = 0.0
        matched = 0
        total_teacher = sum(1 for lp in teacher_logprobs if lp is not None)

        for i, t_lp in enumerate(teacher_logprobs):
            if t_lp is None:
                continue
            if i >= len(student_token_logprobs):
                break
            s_lp = student_token_logprobs[i]
            if s_lp is None:
                continue
            # Per-token KL contribution: teacher_lp - student_lp
            total_kl += t_lp - s_lp
            matched += 1

        avg_kl = total_kl / matched if matched > 0 else 0.0

        return {
            "kl": avg_kl,
            "total_kl": total_kl,
            "matched_tokens": matched,
            "total_teacher_tokens": total_teacher,
            "match_rate": matched / total_teacher if total_teacher > 0 else 0.0,
        }

    async def evaluate(
        self,
        task_id: int,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: str = None,
        seed: int = None,
        timeout: float = 300.0,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluate KL divergence between student and teacher.

        Args:
            task_id: Maps to teacher rollout in R2.
            model: Student model name.
            base_url: Student model API endpoint.
            api_key: API key.
            seed: Unused (framework compat).
            timeout: Request timeout.
            temperature: Unused (forward pass only).
        """
        api_key = api_key or os.getenv("CHUTES_API_KEY")
        start = time.time()

        # Load teacher rollout
        rollout = self._load_rollout(task_id)
        if rollout is None:
            return {
                "task_name": "distill",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "error": f"Teacher rollout not found: {task_id}",
                "error_type": "rollout_not_found",
                "extra": {"task_id": task_id, "error_type": "rollout_not_found"},
            }

        teacher_lp = rollout.get("full_logprobs")
        if not teacher_lp or not teacher_lp.get("full"):
            return {
                "task_name": "distill",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "error": "Teacher rollout has no full_logprobs",
                "error_type": "rollout_malformed",
                "extra": {"task_id": task_id, "error_type": "rollout_malformed"},
            }

        full_text = teacher_lp["full"]
        teacher_logprobs = teacher_lp["logprobs"]

        # Student forward pass
        error = None
        error_type = None
        error_status = None
        kl_result = None
        student_lp = None
        try:
            student_lp, err_info = await self._student_forward_pass(
                prompt=full_text,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )

            if student_lp is None:
                error = err_info["message"]
                error_type = err_info["error_type"]
                error_status = err_info.get("status")
            else:
                kl_result = self._compute_kl(teacher_logprobs, student_lp)
        except Exception as e:
            import traceback
            error_type = "unexpected"
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Score: exp(-|kl|), perfect match = 1.0
        score = 0.0
        if kl_result and kl_result["matched_tokens"] > 0:
            score = math.exp(-abs(kl_result["kl"]))

        # Student per-token logprobs (None if forward pass failed)
        student_token_logprobs = (
            student_lp.get("token_logprobs") if student_lp else None
        )

        result = {
            "task_name": "distill",
            "score": score,
            "success": error is None and kl_result is not None,
            "time_taken": time.time() - start,
            "extra": {
                "task_id": task_id,
                "kl": kl_result,
                "teacher_tokens_count": sum(1 for lp in teacher_logprobs if lp is not None),
                "model": model,
                "error_type": error_type,
                "error_status": error_status,
                "full": full_text,
                "teacher_logprobs": teacher_logprobs,
                "student_logprobs": student_token_logprobs,
            },
        }

        if error:
            result["error"] = error
            result["error_type"] = error_type

        return result
