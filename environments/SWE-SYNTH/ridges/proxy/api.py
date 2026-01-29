from fastapi import FastAPI, HTTPException
from model import InferenceRequest, InferenceResponse
import os
from openai import OpenAI
from collections import defaultdict
from typing import Dict, List, Any
import threading

app = FastAPI()

# Read environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
OPENAI_SEED = os.getenv("OPENAI_SEED")  # Optional, can be None

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

if not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Track conversation history and usage per evaluation_run_id
_conversation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_usage_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
    "total_requests": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
})
_lock = threading.Lock()

@app.post("/api/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    try:
        run_id = request.evaluation_run_id or "default"

        # Convert InferenceMessage to OpenAI format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        # Call OpenAI API (use env vars, ignore request values)
        kwargs = {
            "model": OPENAI_MODEL,
            "temperature": OPENAI_TEMPERATURE,
            "messages": messages
        }

        # Add seed if specified in environment
        if OPENAI_SEED is not None:
            kwargs["seed"] = int(OPENAI_SEED)

        response = client.chat.completions.create(**kwargs)

        # Extract content from response
        content = response.choices[0].message.content or ""

        # Track conversation and usage
        with _lock:
            # Record the last user message and assistant response
            if messages:
                last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
                if last_user:
                    _conversation_history[run_id].append(last_user)
            _conversation_history[run_id].append({"role": "assistant", "content": content})

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                _usage_stats[run_id]["total_requests"] += 1
                _usage_stats[run_id]["prompt_tokens"] += response.usage.prompt_tokens or 0
                _usage_stats[run_id]["completion_tokens"] += response.usage.completion_tokens or 0
                _usage_stats[run_id]["total_tokens"] += response.usage.total_tokens or 0

        # Return InferenceResponse with empty tool_calls
        return InferenceResponse(
            content=content,
            tool_calls=[]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


@app.get("/api/usage")
async def get_usage(evaluation_run_id: str = None):
    """
    Return usage information and conversation history.
    """
    run_id = evaluation_run_id or "default"
    with _lock:
        usage = _usage_stats.get(run_id, {
            "total_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })
        conversation = _conversation_history.get(run_id, [])

    return {
        "evaluation_run_id": run_id,
        "conversation": conversation,
        "usage": usage,
        **usage,  # Flatten usage for backward compatibility
    }


@app.delete("/api/usage")
async def clear_usage(evaluation_run_id: str = None):
    """Clear usage and conversation history for a run."""
    run_id = evaluation_run_id or "default"
    with _lock:
        _conversation_history.pop(run_id, None)
        _usage_stats.pop(run_id, None)
    return {"status": "cleared", "evaluation_run_id": run_id}

