"""Shared LLM client helpers."""

from __future__ import annotations

import httpx
import openai


async def chat(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
    api_key: str,
) -> str:
    """Send a single chat completion request."""
    client = openai.AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=httpx.Timeout(timeout),
        max_retries=0,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if not response.choices:
        raise RuntimeError("LLM returned no choices")
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty content")
    return content
