"""Tree environment actor – compact, principled entrypoint."""

from __future__ import annotations

import gc
import os
import random
import sys
import time
from typing import List, Optional, Tuple

# Add /app to path for container imports
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from ._task import TreeReconstructionTask, TreeChallenge
from ._session import get_session_manager
from ._deduction import TreeDeductionTask


class LLMConfig:
    """LLM call configuration."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: int,
        temperature: float,
        api_key: str,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.api_key = api_key
        self.seed = seed


async def chat(messages: List[dict], config: LLMConfig) -> str:
    """Minimal OpenAI-compatible chat helper."""
    import httpx
    import openai

    os.environ.pop("SSL_CERT_FILE", None)
    os.environ.pop("REQUESTS_CA_BUNDLE", None)

    client = openai.AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=httpx.Timeout(config.timeout),
        max_retries=0,
    )

    params = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "stream": False,
    }
    if config.seed is not None:
        params["seed"] = config.seed

    response = await client.chat.completions.create(**params)
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("LLM API returned no content")
    return response.choices[0].message.content.strip()


# --- reconstruction helpers -------------------------------------------------

def _parse_queries(response: str) -> List[Tuple[str, List[int]]]:
    out: List[Tuple[str, List[int]]] = []
    for match in TreeReconstructionTask.QUERY_PATTERN.finditer(response):
        args = [int(match.group(2))]
        if match.group(3):
            args.append(int(match.group(3)))
        out.append((match.group(1).upper(), args))
    return out


def _format_query(query_type: str, args: List[int], result: dict) -> str:
    value = result["result"]
    if query_type == "ANCESTOR":
        answer = "YES" if value else "NO"
    elif query_type in ("CHILDREN", "PATH"):
        answer = "[" + ", ".join(map(str, value)) + "]"
    else:
        answer = str(value)
    return f"{query_type} {' '.join(map(str, args))}: {answer}"


async def _run_multi_turn(task: TreeReconstructionTask, challenge: TreeChallenge, convo: List[dict], config: LLMConfig):
    sm = task.session_manager
    for _ in range(TreeReconstructionTask.MAX_TURNS):
        response = await chat(convo, config)
        convo.append({"role": "assistant", "content": response})

        feedback: List[str] = []
        for qtype, args in _parse_queries(response):
            try:
                feedback.append(_format_query(qtype, args, sm.query(challenge.session_id, qtype, args)))
            except ValueError as exc:
                feedback.append(f"Error: {exc}")

        done = False
        result = None
        submit_msg = None
        if task.SUBMIT_PATTERN.search(response):
            submit_msg, done, result = await task.process_response(challenge.session_id, response)

        if feedback or submit_msg:
            combined = feedback + ([submit_msg] if submit_msg else [])
            convo.append({"role": "user", "content": "\n".join(combined)})

        if done:
            score = result.get("score", 0.0) if result else 0.0
            return score, result

    return 0.0, {"error": "max_turns_reached"}


async def _run_single_turn(task: TreeReconstructionTask, challenge: TreeChallenge, convo: List[dict], config: LLMConfig):
    sm = task.session_manager
    response = await chat(convo, config)
    convo.append({"role": "assistant", "content": response})

    for qtype, args in _parse_queries(response):
        try:
            sm.query(challenge.session_id, qtype, args)
        except ValueError:
            continue

    score = await task.evaluate(response, challenge)
    try:
        state = sm.get_session_state(challenge.session_id)
        final = {
            "score": score,
            "query_count": state.get("query_count", 0),
            "total_bits": state.get("total_bits", 0),
            "status": state.get("status", "unknown"),
        }
    except ValueError:
        final = {"score": score}

    return score, final


# --- Actor ------------------------------------------------------------------

class Actor:
    """Thin façade delegating to reconstruction or deduction tasks."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.session_manager = get_session_manager()

    def _llm_config(
        self,
        model: str,
        base_url: str,
        timeout: int,
        temperature: float,
        api_key: Optional[str],
        seed: Optional[int],
    ) -> LLMConfig:
        return LLMConfig(
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            api_key=api_key or self.api_key,
            seed=seed,
        )

    # Reconstruction -----------------------------------------------------
    async def evaluate(
        self,
        n: int = 20,
        method: str = "prufer",
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 600,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        task_id: Optional[int] = None,
        max_queries: Optional[int] = None,
        allowed_query_types: Optional[List[str]] = None,
        multi_turn: bool = True,
    ):
        llm_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        tree_seed = task_id if task_id is not None else (seed if seed is not None else random.randint(0, 2**63 - 1))
        llm_config = self._llm_config(model, base_url, timeout, temperature, api_key, llm_seed)

        task = TreeReconstructionTask(
            default_n=n,
            default_method=method,
            max_queries=max_queries,
            allowed_query_types=allowed_query_types,
            session_manager=self.session_manager,
        )

        start = time.time()
        challenge = await task.generate(n=n, seed=tree_seed, method=method, task_id=task_id)
        convo = [{"role": "user", "content": challenge.prompt}]

        try:
            if multi_turn:
                score, final = await _run_multi_turn(task, challenge, convo, llm_config)
            else:
                score, final = await _run_single_turn(task, challenge, convo, llm_config)
            error = None
        except Exception as exc:  # pragma: no cover
            import traceback

            score = 0.0
            final = None
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        finally:
            self.session_manager.close_session(challenge.session_id)
            gc.collect()

        result = {
            "task_name": "tree:reconstruction",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": convo,
                "seed": llm_seed,
                "tree_seed": tree_seed,
                "task_id": task_id,
                "n": n,
                "method": method,
                "evaluation": final,
            },
        }
        if error:
            result["error"] = error
            result["error_type"] = "evaluation_failure"
        return result

    async def create_session(self, n: int = 20, seed: Optional[int] = None, method: str = "prufer"):
        actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
        task = TreeReconstructionTask(default_n=n, default_method=method, session_manager=self.session_manager)
        challenge = await task.generate(n=n, seed=actual_seed, method=method)
        return {
            "session_id": challenge.session_id,
            "n": n,
            "seed": actual_seed,
            "prompt": challenge.prompt,
            "info_lower_bound": challenge.extra["info_lower_bound"],
        }

    async def query(self, session_id: str, query_type: str, args: list):
        return self.session_manager.query(session_id, query_type, args)

    async def submit(self, session_id: str, parent_array: list):
        state = self.session_manager.get_session_state(session_id)
        n = state["n"]
        normalized = [-1] + list(parent_array) if len(parent_array) == n - 1 else list(parent_array)
        if len(normalized) != n:
            raise ValueError(f"Expected {n-1} or {n} parent values, got {len(parent_array)}")
        return self.session_manager.submit(session_id, normalized)

    async def get_ground_truth(self, session_id: str):
        self.session_manager.get_session_state(session_id)
        return self.session_manager.sessions[session_id].tree.get_ground_truth()

    # Deduction -----------------------------------------------------------
    async def evaluate_deduction(
        self,
        n: int = 8,
        reveal_fraction: float = 0.5,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 300,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        task_id: Optional[int] = None,
    ):
        actual_seed = task_id if task_id is not None else (seed if seed is not None else random.randint(0, 2**32 - 1))
        llm_seed = seed if seed is not None else actual_seed
        config = self._llm_config(model, base_url, timeout, temperature, api_key, llm_seed)

        task = TreeDeductionTask(n=n, reveal_fraction=reveal_fraction)
        problem = await task.generate(n=n, reveal_fraction=reveal_fraction, seed=actual_seed, task_id=task_id)
        prompt = problem.to_prompt()

        start = time.time()
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await chat(messages, config)
            detailed = task.evaluate_detailed(response, problem)
            score = detailed["score"]
            error = None
        except Exception as exc:  # pragma: no cover
            import traceback

            response = None
            score = 0.0
            detailed = {"error": str(exc)}
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        finally:
            gc.collect()

        result = {
            "task_name": "tree:deduction",
            "score": score,
            "success": score == 1.0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "seed": actual_seed,
                "llm_seed": llm_seed,
                "n": n,
                "reveal_fraction": reveal_fraction,
                "ambiguity_bits": problem.ambiguity_bits,
                "evaluation": detailed,
                "ground_truth": problem.ground_truth,
            },
        }
        if error:
            result["error"] = error
            result["error_type"] = "evaluation_failure"
        return result


__all__ = ["Actor"]
