"""Shared LLM chat helper for environments.

Goal: remove boilerplate from env implementations.

Notes:
- Keep this dependency-light and container-safe.
- This is a *basic* helper. Environments with advanced needs (streaming, caching, retries)
  can still implement their own logic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai


async def llm_chat(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    timeout: float | int = 600,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    stream: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Minimal OpenAI-compatible chat completion helper.

    Returns:
        (content, usage_dict)
        - content: str | None
        - usage_dict: OpenAI usage dict | None
    """
    # Avoid SSL path issues commonly seen in containers
    os.environ.pop("SSL_CERT_FILE", None)
    os.environ.pop("REQUESTS_CA_BUNDLE", None)

    client = openai.AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=httpx.Timeout(timeout),
        max_retries=0,
    )

    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
    }
    if temperature is not None:
        params["temperature"] = temperature
    if seed is not None:
        params["seed"] = seed

    try:
        if not stream:
            resp = await client.chat.completions.create(**params)
            if not resp.choices:
                return None, getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else None
            content = resp.choices[0].message.content
            usage = resp.usage.model_dump() if resp.usage else None
            return content.strip() if content else None, usage

        # Stream mode (best-effort)
        params["stream_options"] = {"include_usage": True}
        stream_resp = await client.chat.completions.create(**params)
        parts: List[str] = []
        usage = None
        async for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                usage = chunk.usage.model_dump()
        content = "".join(parts).strip()
        return content if content else None, usage
    finally:
        await client.close()


