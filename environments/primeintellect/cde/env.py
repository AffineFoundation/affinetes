"""Code Environment Actor"""

import os
import time
import gc
import httpx
import openai
import sys
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add /app to path to import local modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from code_task import CodeTask


class Actor:
    """Code task evaluation actor"""
    
    def __init__(
        self,
        api_key: str = None,
    ):
        """
        Initialize Actor with API key
        
        Args:
            api_key: API key for LLM service. If not provided, will use CHUTES_API_KEY env var
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        
        # Initialize code task instance
        self.code_task = CodeTask()
    
    async def _llm_chat(self, prompt, model, base_url, timeout, temperature, current_api_key, seed=None):
        """Call LLM API with specified API key and optional seed (streaming mode)"""
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

        # Prepare API call parameters with streaming enabled
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        # Add seed if provided
        if seed is not None:
            params["seed"] = seed

        stream = await client.chat.completions.create(**params)
        
        # Collect streamed content and usage
        content_parts = []
        usage = None
        
        async for chunk in stream:
            # Collect content chunks
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
            
            # Collect usage information from the final chunk
            if chunk.usage:
                usage = chunk.usage.model_dump()
        
        # Combine all content parts
        if not content_parts:
            raise ValueError("LLM API returned empty content stream")
        
        content = "".join(content_parts)
        if not content:
            raise ValueError("LLM API returned None content (possible content filtering or API error)")
        
        # Return both content and usage information
        return content.strip(), usage
    
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
        Run evaluation on a single code task
        
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
        challenge = await self.code_task.generate(task_id=task_id)
        
        # Add model and base_url info to challenge.extra for logging
        challenge.extra["model"] = model
        challenge.extra["base_url"] = base_url
        
        # Call LLM
        usage = None
        try:
            resp, usage = await self._llm_chat(challenge.prompt, model, base_url, timeout, temperature, current_api_key, seed)
            error = None
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Evaluate
        score = 0.0
        test_result = "0/0"
        if resp:
            score, test_result = await self.code_task.evaluate(resp, challenge)

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        result = {
            "task_name": "CDE",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "test_result": test_result,  # Format: "passed/total" (e.g., "7/15")
                "test_cases": challenge.extra.get("tests", ""),
                "dataset_index": challenge.extra.get("dataset_index"),
                "usage": usage
            }
        }
        
        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Force garbage collection to free memory immediately
        gc.collect()

        return result

    async def _llm_chat_local(self, prompt, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, seed=None):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=4096)

        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist() 
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    async def local_evaluate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task_type="cde",
        task_id: int = None,
        seed: int = None
    ):
        """
        Run evaluation on a single code task
        
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

        start = time.time()
        
        # Generate challenge
        challenge = await self.code_task.generate(task_id=task_id)
                
        # Call LLM
        try:
            resp = await self._llm_chat_local(challenge.prompt, model, tokenizer, seed)
            error = None
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Evaluate
        score = 0.0
        test_result = "0/0"
        if resp:
            score, test_result = await self.code_task.evaluate(resp, challenge)

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        result = {
            "task_name": "CDE",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "test_result": test_result,  # Format: "passed/total" (e.g., "7/15")
                "test_cases": challenge.extra.get("tests", ""),
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