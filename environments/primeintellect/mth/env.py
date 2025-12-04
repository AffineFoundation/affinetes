"""Math Environment Actor"""

import os
import time
import gc
import httpx
import openai
import sys
import random

# Add /app to path to import local modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from math_task import MathTask


class Actor:
    """Math task evaluation actor"""
    
    def __init__(
        self,
        api_key: str = None,
        judge_model: str = None,
        judge_base_url: str = None,
        judge_api_key: str = None
    ):
        """
        Initialize Actor with API key and judge configuration
        
        Args:
            api_key: API key for LLM service. If not provided, will use CHUTES_API_KEY env var
            judge_model: Judge model for LLM-based evaluation
            judge_base_url: Base URL for judge API
            judge_api_key: API key for judge model
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.judge_api_key = judge_api_key
        
        # Initialize math task instance
        self.math_task = MathTask(
            judge_model=self.judge_model,
            judge_base_url=self.judge_base_url,
            judge_api_key=self.judge_api_key
        )
    
    async def _llm_chat(self, prompt, model, base_url, timeout, temperature, current_api_key, seed=None):
        """Call LLM API with specified API key and optional seed"""
        # Unset SSL_CERT_FILE to avoid certificate path issues in container
        # Let httpx/certifi use default certificate bundle
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('REQUESTS_CA_BUNDLE', None)
        
        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip('/'),
            api_key=current_api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0
        )

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False
        }
        
        # Add seed if provided
        if seed is not None:
            params["seed"] = seed

        response = await client.chat.completions.create(**params)
        
        # Handle case where API returns None content
        if not response.choices:
            raise ValueError("LLM API returned empty choices list")
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM API returned None content (possible content filtering or API error)")
        
        return content.strip()
    
    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=0.7,
        api_key: str = None,
        seed: int = None,
        task_id: int = None
    ):
        """
        Run evaluation on a single math task
        
        Args:
            model: Model name to use for evaluation
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API calls
            temperature: Temperature for LLM generation
            api_key: Override API key for this evaluation. If not provided, uses instance api_key
            seed: Random seed for LLM generation. Used to ensure reproducible results. If not provided, a random seed will be generated.
            task_id: Optional task ID for deterministic task selection.
                     If provided, used as index into dataset.
                     If not provided, random sample is selected.
        """
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Allow per-call api_key override
        current_api_key = api_key or self.api_key
        
        start = time.time()
        
        # Generate challenge
        challenge = await self.math_task.generate(task_id=task_id)
        
        # Call LLM
        try:
            resp = await self._llm_chat(challenge.prompt, model, base_url, timeout, temperature, current_api_key, seed)
            error = None
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Evaluate
        score = 0.0
        if resp:
            score = await self.math_task.evaluate(resp, challenge)

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        result = {
            "task_name": "math:intellect-3-rl",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "answer": challenge.extra.get("answer", ""),
                "dataset_index": challenge.extra.get("dataset_index")
            }
        }
        
        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Force garbage collection to free memory immediately
        gc.collect()

        return result