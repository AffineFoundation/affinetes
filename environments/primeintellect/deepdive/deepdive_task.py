"""DeepDive task generator and evaluator"""

from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter
from typing import Any
import random

import aiohttp
import httpx
from datasets import load_dataset
from openai import AsyncOpenAI

import sys
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from models import Challenge
from config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    METADATA_KEYS,
    PROMPT_SUFFIX,
    SERPER_API_URL,
)
from formatting import format_serper_results, truncate_text
from open_one import open_one
from rate_limit import with_rate_limit_retry

import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.utils.data_utils import extract_boxed_answer


class DeepDiveTask:
    """DeepDive task generator and evaluator"""
    
    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        dataset_split: str = DEFAULT_DATASET_SPLIT,
        dataset_test_size: float = 0.1,
        dataset_seed: int = 2025,
        serper_api_key: str = None,
        max_turns: int = 32,
        max_search_results: int = 10,
        max_response_chars: int = 20_000,
        serper_timeout: float = 15.0,
        finish_with_tool: bool = False,
        debug: bool = False,
    ):
        """
        Initialize DeepDiveTask
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_split: Dataset split to use
            dataset_test_size: Test split size
            dataset_seed: Random seed for split
            serper_api_key: Serper API key
            max_turns: Maximum conversation turns
            max_search_results: Maximum search results per query
            max_response_chars: Maximum response characters
            serper_timeout: Serper API timeout
            finish_with_tool: Whether to use finish tool
            debug: Enable debug logging
        """
        # Load dataset
        raw_split = load_dataset(dataset_name, split=dataset_split)
        
        # Process records
        def to_record(d):
            q = (d["question"] or "").rstrip()
            out = {
                "task": "deepdive",
                "info": {"raw_question": q},
                "prompt": q + ("" if finish_with_tool else PROMPT_SUFFIX),
            }
            for k in METADATA_KEYS:
                if k in d:
                    out[k] = d[k]
            return out
        
        raw_split = raw_split.map(to_record)
        split = raw_split.train_test_split(test_size=dataset_test_size, seed=dataset_seed)
        self.dataset = split["train"]
        self.eval_dataset = split["test"]
        
        # Store configuration
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.max_turns = max_turns
        self.max_search_results = max_search_results
        self.max_response_chars = max_response_chars
        self.serper_timeout = serper_timeout
        self.finish_with_tool = finish_with_tool
        self.debug = debug
        
        if not self.serper_api_key:
            raise ValueError("Missing Serper API key. Set SERPER_API_KEY environment variable.")
        
        # Rate limiting primitives
        self.concurrency_semaphore = asyncio.Semaphore(128)
        self.rate_limit_semaphore = asyncio.Semaphore(1)
        self.rate_limit_event = asyncio.Event()
    
    async def generate(self, task_id: int = None) -> Challenge:
        """
        Generate a DeepDive challenge
        
        Args:
            task_id: Optional task ID for deterministic selection
        """
        if task_id is not None:
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        return Challenge(
            env="deepdive",
            prompt=sample["prompt"],
            extra={
                "raw_question": sample["info"]["raw_question"],
                "task_id": task_id,
                "dataset_index": idx,
                "source": sample.get("source", ""),
                "category": sample.get("category", ""),
                "difficulty": sample.get("difficulty", ""),
            }
        )
    
    @with_rate_limit_retry
    async def _judge_with_rubric(self, question: str, completion: list, response: str, state: dict, judge_rubric, **kwargs):
        """Use JudgeRubric to evaluate answer (matches original implementation)"""
        judge_response = await judge_rubric.judge(question, completion, response, state, **kwargs)
        return judge_response
    
    async def evaluate(
        self,
        response: str,
        challenge: Challenge,
        judge_model: str = None,
        judge_base_url: str = None,
        judge_api_key: str = None
    ) -> tuple[float, dict]:
        """
        Evaluate response using judge model (matches original judge_reward_func logic)
        
        Args:
            response: Model response (can be string or conversation)
            challenge: Original challenge
            judge_model: Override judge model for this evaluation
            judge_base_url: Override judge base URL for this evaluation
            judge_api_key: Override judge API key for this evaluation
        
        Returns:
            Tuple of (score, extra_info)
        """
        # Create judge client with custom configuration
        api_key = judge_api_key or os.getenv("CHUTES_API_KEY", "EMPTY")
        httpx_timeout = httpx.Timeout(1200)
        httpx_limits = httpx.Limits(max_connections=8192, max_keepalive_connections=8192)
        httpx_client = httpx.AsyncClient(limits=httpx_limits, timeout=httpx_timeout)
        judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, http_client=httpx_client)
        
        # Setup judge rubric
        maybe_think_parser = vf.MaybeThinkParser(extract_fn=extract_boxed_answer)
        judge_rubric = JudgeRubric(
            judge_client=judge_client,
            judge_model=judge_model,
            parser=maybe_think_parser,
        )
        
        # Setup state matching original implementation
        state = {"info": challenge.extra}
        
        # Parse response into completion format
        try:
            if isinstance(response, str) and response.strip().startswith('['):
                completion = json.loads(response)
            else:
                # Convert string response to message format
                completion = [{"role": "assistant", "content": response}]
        except:
            completion = [{"role": "assistant", "content": str(response)}]
        
        # Extract final response (matching original: state.get("[[deepdive/FINAL_ANSWER]]", completion[-1]["content"]))
        final_response = state.get("[[deepdive/FINAL_ANSWER]]", completion[-1].get("content", response) if completion else response)
        
        # Use judge rubric with rate limiting (matches original implementation)
        async def judge_with_retry():
            async with self.concurrency_semaphore:
                if self.rate_limit_event.is_set():
                    await self.rate_limit_event.wait()
                    await asyncio.sleep(random.uniform(0, 2))
                
                return await self._judge_with_rubric(
                    challenge.extra["raw_question"],
                    completion,
                    final_response,
                    state,
                    judge_rubric
                )
        
        try:
            judge_response = await judge_with_retry()
        except Exception as e:
            if self.debug:
                print(f"Judge error: {e}")
            judge_response = "no"
        
        # Score calculation matches original: if "yes" in judge_response.lower(): return 1.0 else: return 0.0
        score = 1.0 if "yes" in judge_response.lower() else 0.0
        
        return score, {
            "judge_response": judge_response,
            "extracted_answer": extract_boxed_answer(final_response)
        }