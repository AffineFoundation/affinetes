"""Shared LLM chat helper for environments.

Provides a unified streaming/non-streaming chat completion interface with:
- Per-chunk timeout protection
- Exponential-backoff retries
- o1-style reasoning content collection
- Think-tag stripping
- Reusable client factory
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx
import openai


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChatResult:
    """Result of an LLM chat completion.

    Supports tuple unpacking for backward compatibility::

        content, usage = await llm_chat(...)          # 2-tuple
        content, reasoning, usage = result             # 3-fields via iter
    """

    content: Optional[str] = None
    reasoning: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    # Allow ``content, usage = await llm_chat(...)`` to keep working for
    # existing callers that expect a 2-tuple (content, usage).
    def __iter__(self) -> Iterator:
        return iter((self.content, self.usage))


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def create_client(
    base_url: str,
    api_key: str,
    *,
    connect_timeout: float = 10.0,
    read_timeout: float = 20.0,
    write_timeout: float = 10.0,
    pool_timeout: float = 10.0,
) -> openai.AsyncOpenAI:
    """Create an :class:`openai.AsyncOpenAI` client with fine-grained timeouts.

    The returned client is meant to be long-lived and reused across calls.
    The caller is responsible for closing it when done.
    """
    return openai.AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        ),
        max_retries=0,  # we handle retries ourselves
    )


# ---------------------------------------------------------------------------
# Think-tag removal
# ---------------------------------------------------------------------------

def remove_think_tags(text: str) -> str:
    """Remove ``<think>``/``<thinking>`` blocks and their content.

    Handles:
    - Complete ``<think>…</think>`` and ``<thinking>…</thinking>`` blocks
    - Malformed responses with only a closing tag (content after it is kept)
    - Truncated blocks (opening tag without a matching close — removed to end)
    - Collapses resulting excess blank lines
    """
    cleaned = text

    # 1. Complete blocks (non-greedy, case-insensitive)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 2. Residual closing tags → keep content after last occurrence
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    if "</thinking>" in cleaned:
        cleaned = cleaned.split("</thinking>")[-1]

    # 3. Unclosed opening tags → truncate from tag position
    for tag in ("<think>", "<thinking>"):
        match = re.search(tag, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[: match.start()]

    # 4. Collapse multiple blank lines → double newline, then strip
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------

# Errors that are safe to retry (transient / server-side).
_RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


async def llm_chat(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    timeout: float | int = 600,
    chunk_timeout: float = 30.0,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    stream: bool = True,
    max_chunks: int = 32_000,
    max_retries: int = 0,
    strip_think_tags: bool = False,
    client: Optional[openai.AsyncOpenAI] = None,
) -> ChatResult:
    """Unified OpenAI-compatible chat completion helper.

    Parameters
    ----------
    messages:
        Conversation messages in OpenAI format.
    model, base_url, api_key:
        Model endpoint configuration.
    timeout:
        Overall request timeout (used when *client* is ``None``).
    chunk_timeout:
        Maximum seconds to wait for the next streaming chunk.
    temperature, seed:
        Optional sampling parameters.
    stream:
        Use streaming mode (default ``True``).
    max_chunks:
        Upper bound on streamed chunks before stopping.
    max_retries:
        Number of retries on transient errors (0 = no retries).
    strip_think_tags:
        If ``True``, apply :func:`remove_think_tags` to the content.
    client:
        An existing :class:`openai.AsyncOpenAI` to reuse.  When provided the
        caller keeps ownership (the function will **not** close it).

    Returns
    -------
    ChatResult
        With ``content``, ``reasoning``, and ``usage`` fields.
    """
    # Avoid SSL path issues commonly seen in containers
    os.environ.pop("SSL_CERT_FILE", None)
    os.environ.pop("REQUESTS_CA_BUNDLE", None)

    own_client = client is None
    if own_client:
        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )

    try:
        return await _chat_with_retries(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            stream=stream,
            chunk_timeout=chunk_timeout,
            max_chunks=max_chunks,
            max_retries=max_retries,
            strip_think_tags=strip_think_tags,
        )
    finally:
        if own_client:
            await client.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _chat_with_retries(
    *,
    client: openai.AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float],
    seed: Optional[int],
    stream: bool,
    chunk_timeout: float,
    max_chunks: int,
    max_retries: int,
    strip_think_tags: bool,
) -> ChatResult:
    """Execute the chat call with optional retries."""
    attempts = max_retries + 1  # total attempts = retries + 1

    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if seed is not None:
        params["seed"] = seed

    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            if stream:
                result = await _stream_chat(client, params, chunk_timeout, max_chunks)
            else:
                result = await _non_stream_chat(client, params)

            content = result.content
            if strip_think_tags and content:
                content = remove_think_tags(content)
                if not content:
                    content = None

            return ChatResult(
                content=content,
                reasoning=result.reasoning,
                usage=result.usage,
            )

        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < attempts - 1:
                await asyncio.sleep(min(2 ** attempt, 32))
            continue

        except openai.BadRequestError as exc:
            msg = str(exc)
            if "is longer than the model" in msg or "context_length_exceeded" in msg:
                raise ValueError(f"Context length exceeded: {msg}") from exc
            raise ValueError(f"API error: {msg}") from exc

        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(min(2 ** attempt, 32))
                continue
            raise ValueError(f"API error {exc.status_code}: {exc.message}") from exc

    # All retries exhausted
    raise ValueError(f"API call failed after {attempts} attempts: {last_exc}")


async def _stream_chat(
    client: openai.AsyncOpenAI,
    params: Dict[str, Any],
    chunk_timeout: float,
    max_chunks: int,
) -> ChatResult:
    """Execute a streaming chat completion."""
    stream_params = {**params, "stream_options": {"include_usage": True}}
    stream = await client.chat.completions.create(**stream_params)

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    chunk_count = 0

    try:
        chunk_iter = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=chunk_timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Stream timeout: no chunk received for {chunk_timeout}s"
                )

            chunk_count += 1

            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                if getattr(delta, "reasoning_content", None):
                    reasoning_parts.append(delta.reasoning_content)
                if chunk_count >= max_chunks:
                    break

            if chunk.usage:
                usage = chunk.usage.model_dump()
    finally:
        # Best-effort stream close with timeout
        try:
            await asyncio.wait_for(stream.response.aclose(), timeout=5.0)
        except Exception:
            pass

    content = "".join(content_parts).strip() or None
    reasoning = "".join(reasoning_parts) or None
    return ChatResult(content=content, reasoning=reasoning, usage=usage)


async def _non_stream_chat(
    client: openai.AsyncOpenAI,
    params: Dict[str, Any],
) -> ChatResult:
    """Execute a non-streaming chat completion."""
    resp = await client.chat.completions.create(**params)
    usage = resp.usage.model_dump() if resp.usage else None

    if not resp.choices:
        return ChatResult(content=None, reasoning=None, usage=usage)

    msg = resp.choices[0].message
    content = msg.content.strip() if msg.content else None
    reasoning = getattr(msg, "reasoning_content", None)
    return ChatResult(content=content, reasoning=reasoning, usage=usage)
