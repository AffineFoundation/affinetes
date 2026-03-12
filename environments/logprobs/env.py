"""Logprobs Environment Actor.

Given a task_id, generates a prompt and returns the top-K logprobs
for the model's first N output tokens.

Uses /v1/completions (not /v1/chat/completions) to bypass reasoning parsers
that aggregate thinking tokens and suppress per-token logprobs.
"""

import gc
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

import httpx
import openai

from task_generator import generate_prompt

# Chat template formats for /v1/completions
CHAT_TEMPLATES = {
    "qwen": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "deepseek": "<|User|>{prompt}<|Assistant|>",
}


class Actor:
    """Returns top-K logprobs for the first N generated tokens.

    Task ID encoding:
      task_type = TASK_NAMES[task_id // 100_000_000]
      seed      = task_id %  100_000_000

    Supported ranges:
      0–99,999,999:     common_sense_combo
      100M–199,999,999: qqr_travel
      200M–299,999,999: long_context
      300M–399,999,999: openspiel_game
      400M–499,999,999: swe_agent_fix
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

    async def evaluate(
        self,
        task_id: int = None,
        seed: int = None,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: str = None,
        n_tokens: int = 20,
        top_k: int = 3,
        temperature: float = 0.0,
        timeout: float = 600.0,
        chat_template: str = "qwen",
    ) -> Dict[str, Any]:
        """Generate a prompt from task_id and return top-K logprobs for n_tokens.

        Uses /v1/completions to bypass reasoning parsers on thinking models.

        Args:
            task_id: Encodes task type and seed (see class docstring).
            seed: Ignored (framework compat); task_id is used.
            model: Model name; must support the logprobs parameter.
            base_url: OpenAI-compatible API endpoint.
            api_key: Override API key; falls back to CHUTES_API_KEY env var.
            n_tokens: Number of tokens to sample (default: 20).
            top_k: Top logprobs per token position (default: 3, max: 20).
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
            chat_template: Template name ("qwen", "llama", "deepseek") or
                           a custom format string with {prompt} placeholder.

        Returns:
            Dict matching the standard affinetes evaluate() schema.
        """
        current_api_key = api_key or self.api_key
        start = time.time()

        prompt, task_type, seed, task_metadata = await generate_prompt(task_id)

        # Format prompt for /v1/completions
        template = CHAT_TEMPLATES.get(chat_template, chat_template)
        formatted_prompt = template.format(prompt=prompt)

        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=current_api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )

        error = None
        tokens: List[Dict[str, Any]] = []
        usage = None
        hidden_states = None
        hidden_states_dim = None

        try:
            resp = await client.completions.create(
                model=model,
                prompt=formatted_prompt,
                max_tokens=n_tokens,
                temperature=temperature,
                logprobs=top_k,
                stream=False,
            )

            usage = resp.usage.model_dump() if resp.usage else None

            # Fetch hidden states via separate call (max_tokens=2 required)
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), verify=False) as http:
                    hs_resp = await http.post(
                        f"{base_url.rstrip('/')}/completions",
                        headers={
                            "Authorization": f"Bearer {current_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "prompt": formatted_prompt,
                            "max_tokens": 2,
                            "temperature": temperature,
                            "stream": False,
                            "return_hidden_states": True,
                        },
                    )
                    if hs_resp.status_code == 200:
                        hs_data = hs_resp.json()
                        choices = hs_data.get("choices", [])
                        if choices and "hidden_states" in choices[0]:
                            hs = choices[0]["hidden_states"]
                            if isinstance(hs, list):
                                hidden_states_dim = len(hs)
                                if len(hs) > 256:
                                    n_bins = 256
                                    bin_size = len(hs) / n_bins
                                    hidden_states = [
                                        sum(hs[int(i * bin_size):int((i + 1) * bin_size)]) / (int((i + 1) * bin_size) - int(i * bin_size))
                                        for i in range(n_bins)
                                    ]
                                else:
                                    hidden_states = hs
            except Exception:
                pass  # hidden states are best-effort

            lp = resp.choices[0].logprobs if resp.choices else None
            if lp and lp.tokens:
                raw_tokens = lp.tokens or []
                raw_logprobs = lp.token_logprobs or []
                raw_top = lp.top_logprobs or []
                for i in range(min(n_tokens, len(raw_tokens))):
                    lp_val = raw_logprobs[i] if i < len(raw_logprobs) else 0.0
                    top_dict = raw_top[i] if i < len(raw_top) else {}
                    top_entries = sorted(
                        [
                            {
                                "token": t,
                                "logprob": round(v, 6),
                                "prob": round(math.exp(v), 6),
                            }
                            for t, v in (top_dict or {}).items()
                        ],
                        key=lambda x: x["logprob"],
                        reverse=True,
                    )
                    tokens.append(
                        {
                            "position": i,
                            "token": raw_tokens[i],
                            "logprob": round(lp_val, 6),
                            "top_k": top_entries,
                        }
                    )

        except Exception as e:
            import traceback
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        finally:
            await client.close()

        result = {
            "task_name": f"logprobs:{task_type}",
            "score": 1.0,
            "success": error is None,
            "time_taken": time.time() - start,
            "extra": {
                "task_id": task_id,
                "task_type": task_type,
                "seed": seed,
                "prompt": prompt,
                "model": model,
                "n_tokens": n_tokens,
                "top_k": top_k,
                "chat_template": chat_template,
                "tokens": tokens,
                "task_metadata": task_metadata,
                "usage": usage,
                "hidden_states": hidden_states,
                "hidden_states_dim": hidden_states_dim,
            },
        }

        if error:
            result["error"] = error
            result["error_type"] = "api_failure"

        gc.collect()
        return result
