from fastapi import FastAPI, HTTPException
from model import InferenceRequest, InferenceResponse
import os
from openai import OpenAI

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

@app.post("/api/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    try:
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
    Return usage information. 
    Since we're proxying to another service, we return minimal info.
    """
    return {
        "evaluation_run_id": evaluation_run_id,
        "total_requests": 0,
        "total_tokens": 0,
        "total_cost": 0.0
    }

