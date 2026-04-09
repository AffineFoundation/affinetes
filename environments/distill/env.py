"""KL Divergence Environment.

Computes KL divergence between a student model and pre-generated teacher rollouts.

Flow:
    1. Load teacher rollout from R2 by task_id (same pattern as SWE-INFINITE)
       Rollout contains full_logprobs: {full, token_ids, logprobs}
    2. Student forward pass: /v1/completions with echo=True + logprobs=20
       to recover the student's top-20 distribution at every position.
    3. At each assistant position we have:
         teacher top-20 dict P = {token_str: logprob}
         student top-20 dict Q = {token_str: logprob}
       Compute KL restricted to P's top-20 support:
         KL_i = sum_{t in P} exp(P[t]) * (P[t] - log Q(t))
       where log Q(t) is taken from the student top-20 if present, and
       otherwise bounded by the smallest logprob in the student top-20
       (a conservative fallback for tokens outside student top-k).

Task ID:
    Plain integer, maps to a teacher rollout file in R2:
      {R2_BASE_URL}/{R2_PREFIX}/task_{task_id:011d}.json

    The rollout JSON must contain a "full_logprobs" key with:
      full: str                 - formatted conversation text
      token_ids: [int]          - token positions (or offsets)
      logprobs: [dict|None]     - None = masked; dict = teacher top-20
                                  {token_str: logprob} at that position
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
        if self.prefix:
            return f"{self.base_url}/{self.prefix}/{filename}"
        return f"{self.base_url}/{filename}"

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
        # Default to the public distill bucket. Files live at the bucket
        # root (no prefix) — see teacher_mover for the promotion pipeline.
        r2_base = os.getenv("KL_R2_BASE_URL", "https://pub-4546777cb27840ec91b892f19eb5742b.r2.dev")
        r2_prefix = os.getenv("KL_R2_PREFIX", "")
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
                    "logprobs": 20,
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
        """Compute a [0, 1] closeness score per position, then average.

        Top-k path (teacher dict per position): renormalize both
        teacher and student onto the teacher's top-k support set, then
        compute a proper (non-negative) KL on that restricted simplex:

            KL_i = sum_t p_i(t) * (log p_i(t) - log q_i(t))

        and map to [0, 1] via score_i = exp(-KL_i).

        Top-1 / legacy path (teacher float per position): the rollout
        only stored the chosen token's logprob, sampled from teacher.
        Use Schulman's k3 estimator as an unbiased, non-negative,
        low-variance single-sample estimate of KL(P || Q):

            r      = Q(t) / P(t) = exp(s_lp - t_lp)
            kl_i  ~= (r - 1) - log r        (>= 0)
            score_i = exp(-kl_i)            in (0, 1]
        """
        student_top_logprobs = student_lp_dict.get("top_logprobs") or []
        student_token_logprobs = student_lp_dict.get("token_logprobs") or []

        total_score = 0.0
        # Geometric-mean score: accumulate clipped kl, map exp(-mean) at end.
        # Clip guards legacy k3 outliers (r>>1 tokens blow up single-sample kl).
        KL_CLIP = 10.0
        total_kl_clipped = 0.0
        total_kl = 0.0
        kl_positions = 0
        matched = 0
        total_teacher = sum(1 for lp in teacher_logprobs if lp is not None)

        def _log_softmax(lps: List[float]) -> List[float]:
            m = max(lps)
            exps = [math.exp(x - m) for x in lps]
            lse = m + math.log(sum(exps))
            return [x - lse for x in lps]

        for i, t_entry in enumerate(teacher_logprobs):
            if t_entry is None:
                continue

            if isinstance(t_entry, dict):
                # Top-k path.
                if i >= len(student_top_logprobs):
                    break
                s_top = student_top_logprobs[i]
                if not s_top or not isinstance(s_top, dict):
                    continue
                if not t_entry:
                    continue

                # For tokens in teacher top-k but outside student top-k,
                # under-estimate student logprob with min of student top-k.
                s_fallback = min(s_top.values())

                toks = list(t_entry.keys())
                t_lps = [t_entry[t] for t in toks]
                s_lps = [s_top.get(t, s_fallback) for t in toks]

                # Renormalize both sides on the teacher's top-k support.
                t_log = _log_softmax(t_lps)
                s_log = _log_softmax(s_lps)

                kl_i = 0.0
                for tl, sl in zip(t_log, s_log):
                    kl_i += math.exp(tl) * (tl - sl)
                # Guard against tiny float noise only.
                if kl_i < 0.0:
                    kl_i = 0.0

                total_kl += kl_i
                total_kl_clipped += min(kl_i, KL_CLIP)
                kl_positions += 1
                matched += 1
            else:
                # Top-1 / legacy path: Schulman k3 estimator of KL(P || Q).
                if i >= len(student_token_logprobs):
                    break
                s_lp = student_token_logprobs[i]
                if s_lp is None:
                    continue
                log_r = s_lp - float(t_entry)
                r = math.exp(log_r)
                kl_i = (r - 1.0) - log_r   # >= 0 by convexity
                if kl_i < 0.0:
                    kl_i = 0.0             # guard float noise only
                total_kl += kl_i
                total_kl_clipped += min(kl_i, KL_CLIP)
                kl_positions += 1
                matched += 1

        avg_kl_clipped = total_kl_clipped / matched if matched > 0 else 0.0
        avg_score = math.exp(-avg_kl_clipped) if matched > 0 else 0.0
        avg_kl = total_kl / kl_positions if kl_positions > 0 else 0.0

        return {
            "score": avg_score,
            "kl": avg_kl,
            "kl_positions": kl_positions,
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

        # Per-position closeness in [0, 1], averaged. Perfect match = 1.0.
        score = 0.0
        if kl_result and kl_result["matched_tokens"] > 0:
            score = kl_result["score"]

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
            },
        }

        if error:
            result["error"] = error
            result["error_type"] = error_type

        return result
