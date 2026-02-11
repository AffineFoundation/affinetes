"""Codex Django Bug-Fix Environment Actor

This environment challenges models to fix bugs in Django that were
introduced by OpenAI Codex CLI. The model receives the buggy code
and must provide a fix that makes the tests pass.
"""

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

from codex_task import CodexTask


class Actor:
    """Codex Django bug-fix evaluation actor"""
    
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
        
        # Initialize Codex task instance
        self.codex_task = CodexTask()
    
    async def _llm_chat(
        self,
        prompt: str,
        model: str,
        base_url: str,
        timeout: int,
        temperature: float,
        current_api_key: str,
        seed: int = None
    ):
        """Call LLM API with specified API key and optional seed (streaming mode)"""
        # Unset SSL_CERT_FILE to avoid certificate path issues in container
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
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 600,
        temperature: float = 0.7,
        api_key: str = None,
        seed: int = None,
        task_id: int = None,
        bug_injection_model: str = None,
    ):
        """
        Run evaluation on a single Codex Django bug-fix task
        
        The evaluation flow:
        1. Use Codex to introduce a bug into Django (using bug_injection_model or same model)
        2. Generate a challenge for the model to fix the bug
        3. Call the model being evaluated to fix the bug
        4. Run Django tests to verify the fix
        5. Return score (1 if tests pass, 0 otherwise)
        
        Args:
            model: Model name to use for evaluation (the model being tested)
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API calls
            temperature: Temperature for LLM generation
            api_key: Override API key for this evaluation. If not provided, uses instance api_key
            seed: Random seed for reproducibility. If not provided, a random seed will be generated.
            task_id: Optional task ID for deterministic task selection.
            bug_injection_model: Model to use for bug injection. Defaults to same as evaluation model.
        """
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Allow per-call api_key override
        current_api_key = api_key or self.api_key
        
        if not current_api_key:
            return {
                "task_name": "codex_django_bugfix",
                "score": 0.0,
                "success": False,
                "time_taken": 0,
                "error": "No API key provided",
                "error_type": "config_error",
                "extra": {}
            }
        
        start = time.time()
        
        # Use bug_injection_model or default to evaluation model
        injection_model = bug_injection_model or model
        
        # Step 1: Inject bug using Codex
        try:
            bug_spec = await self.codex_task.inject_bug(
                api_key=current_api_key,
                base_url=base_url,
                model=injection_model,
                seed=seed if task_id is None else task_id,
            )
        except Exception as e:
            import traceback
            return {
                "task_name": "codex_django_bugfix",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "error": f"Bug injection failed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "error_type": "bug_injection_failure",
                "extra": {"seed": seed, "task_id": task_id}
            }
        
        if bug_spec is None:
            return {
                "task_name": "codex_django_bugfix",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "error": "Bug injection did not produce a valid bug",
                "error_type": "bug_injection_failure",
                "extra": {"seed": seed, "task_id": task_id}
            }
        
        # Step 2: Generate challenge
        try:
            challenge = await self.codex_task.generate(
                bug_spec=bug_spec,
                task_id=task_id,
            )
        except Exception as e:
            import traceback
            return {
                "task_name": "codex_django_bugfix",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "error": f"Challenge generation failed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "error_type": "challenge_generation_failure",
                "extra": {"seed": seed, "task_id": task_id}
            }
        
        # Add model info to challenge extra
        challenge.extra["model"] = model
        challenge.extra["base_url"] = base_url
        challenge.extra["bug_injection_model"] = injection_model
        
        # Step 3: Call the model being evaluated to fix the bug
        usage = None
        resp = None
        error = None
        
        try:
            resp, usage = await self._llm_chat(
                challenge.prompt,
                model,
                base_url,
                timeout,
                temperature,
                current_api_key,
                seed
            )
        except Exception as e:
            import traceback
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Step 4: Evaluate the fix
        score = 0.0
        test_result = "0/1"
        
        if resp:
            try:
                score, test_result = await self.codex_task.evaluate(resp, challenge)
            except Exception as e:
                import traceback
                error = f"Evaluation failed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        result = {
            "task_name": "codex_django_bugfix",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "task_id": task_id,
                "test_result": test_result,
                "file_path": bug_spec.file_path,
                "bug_description": bug_spec.bug_description,
                "usage": usage,
            }
        }
        
        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "evaluation_failure"

        # Force garbage collection to free memory
        gc.collect()

        return result


# For running as a module with the standard HTTP interface
if __name__ == "__main__":
    import asyncio
    import json
    
    async def main():
        api_key = os.environ.get("CHUTES_API_KEY")
        if not api_key:
            print("Error: CHUTES_API_KEY environment variable required")
            return
        
        actor = Actor(api_key=api_key)
        
        # Run a single evaluation
        result = await actor.evaluate(
            model="deepseek-ai/DeepSeek-V3",
            seed=42,
        )
        
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(main())

