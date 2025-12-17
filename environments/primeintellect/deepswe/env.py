"""DeepSWE Environment Actor"""

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

from datasets import load_dataset
from models import Challenge


class Actor:
    """DeepSWE task evaluation actor"""
    
    def __init__(
        self,
        api_key: str = None
    ):
        """
        Initialize Actor with API key
        
        Args:
            api_key: API key for LLM service. If not provided, will use CHUTES_API_KEY env var
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        
        # Initialize DeepSWE task instance once to avoid reloading dataset
        dataset_name = "R2E-Gym/R2E-Gym-Subset"
        dataset_split = "train"
        dataset_test_size = 0.1
        dataset_seed = 2025
        self.max_turns = 50
        
        split = "test" if dataset_name == "R2E-Gym/SWE-Bench-Verified" else dataset_split
        raw_dataset = load_dataset(dataset_name, split=split)
        
        # Process dataset
        def to_record(d):
            problem_statement = d.get("problem_statement", "")
            return {
                "prompt": self._format_prompt(problem_statement, self.max_turns),
                "info": {**d, "docker_image": d.get("docker_image", d.get("image_name"))},
                "answer": "",
            }
        
        raw_dataset = raw_dataset.map(to_record)
        split_data = raw_dataset.train_test_split(test_size=dataset_test_size, seed=dataset_seed)
        
        self.dataset = split_data["train"]
        self.eval_dataset = split_data["test"]
    
    def _format_prompt(self, problem_statement: str, max_turns: int) -> str:
        """Format the prompt for DeepSWE tasks"""
        return f"""Consider the following github issue:
<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

IMPORTANT TIP:
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
  2.1 reproduce_issue.py script finishes quickly after checking the error, fix etc. There should be no long running background servers for django for instance etc. It should be a quick script which checks the error and fix to provide a visible response.
  2.2 SUPER IMPORTANT: to ensure this reproduce_script.py must have a timeout logic of 20 seconds. If the script runs for more than 30 seconds, it should output a timeout message and you can interpret accordingly.
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well

VERY IMPORTANT: each response must include both reasoning and function call to solve the task.

You can take multiple turns to solve the task. So please only finish / submit when you are confident in your response. Don't rush. Be comprehensive.
You have to submit your final solution before reaching {max_turns} steps.
  
Your thinking should be thorough and so it's fine if it's very long.
VERY IMPORTANT: file_editor old_str and new_str must be w/o the line numbers. Line numbers are only shown in the view for clarity.

Also if a file_editor edit fails, it's a good idea to view the file near the edit location before trying to edit again. Don't keep trying the same edit over and over again. It will keep leading to the same failure.
Again, do not get stuck trying to do the same thing over and over again. Please be efficient.
"""
    
    async def generate(self, task_id: int = None) -> Challenge:
        """Generate a DeepSWE challenge
        
        Args:
            task_id: Optional task ID for deterministic selection
        
        Returns:
            Challenge object with prompt and metadata
        """
        if task_id is not None:
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        return Challenge(
            env="deepswe",
            prompt=sample["prompt"],
            extra={
                "task_id": task_id,
                "dataset_index": idx,
                "instance_id": sample["info"].get("instance_id", ""),
                "repo_name": sample["info"].get("repo_name", ""),
                "docker_image": sample["info"].get("docker_image", ""),
                "problem_statement": sample["info"].get("problem_statement", ""),
            }
        )
    
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
        Run evaluation on a single DeepSWE task
        
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
        challenge = await self.generate(task_id=task_id)
        
        # Call LLM
        usage = None
        try:
            resp, usage = await self._llm_chat(challenge.prompt, model, base_url, timeout, temperature, current_api_key, seed)
            error = None
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Note: Full evaluation requires sandbox execution
        # For now, we return a placeholder score
        score = 0.0

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        result = {
            "task_name": "deepswe",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "dataset_index": challenge.extra.get("dataset_index"),
                "instance_id": challenge.extra.get("instance_id", ""),
                "repo_name": challenge.extra.get("repo_name", ""),
                "usage": usage,
                "note": "DeepSWE evaluation requires full sandbox execution"
            }
        }
        
        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Force garbage collection to free memory immediately
        gc.collect()

        return result