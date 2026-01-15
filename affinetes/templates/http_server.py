"""
Auto-injected HTTP server template for function-based environments.
This file is copied to container during image build via two-stage build.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import importlib.util
import asyncio
import inspect
import sys
import traceback
from typing import Any, Optional, Dict, List

app = FastAPI(title="affinetes HTTP Server")

# User module will be loaded at runtime
user_module = None
user_actor = None


class MethodCall(BaseModel):
    """Method call request"""
    method: str
    args: list = []
    kwargs: dict = {}


class MethodResponse(BaseModel):
    """Method call response"""
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


# =============================================================================
# OpenEnv Protocol Models
# =============================================================================

class ResetRequest(BaseModel):
    """OpenEnv reset request"""
    task_id: Optional[int] = None
    seed: Optional[int] = None
    # Additional environment-specific parameters can be passed
    kwargs: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    """OpenEnv step request"""
    action: str  # Text action (will be parsed by environment)


class OpenEnvResponse(BaseModel):
    """OpenEnv response for reset/step/state"""
    observation: str  # Text prompt for LLM
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: Dict[str, Any] = {}


def _load_user_env():
    """Load user's env.py module"""
    global user_module, user_actor
    
    spec = importlib.util.spec_from_file_location("user_env", "/app/env.py")
    user_module = importlib.util.module_from_spec(spec)
    sys.modules["user_env"] = user_module
    spec.loader.exec_module(user_module)
    
    # Initialize Actor if exists (lazy initialization - will be created when needed)
    # Don't create Actor in startup to avoid requiring env vars at startup
    if hasattr(user_module, "Actor"):
        user_actor = None  # Will be lazily initialized on first call


@app.on_event("startup")
async def startup():
    """Load user environment on startup"""
    _load_user_env()


@app.post("/call", response_model=MethodResponse)
async def call_method(call: MethodCall):
    """Generic method dispatcher for function-based environments"""
    global user_actor
    
    # Lazy initialize Actor on first call (allows env vars to be set at runtime)
    if hasattr(user_module, "Actor") and user_actor is None:
        try:
            user_actor = user_module.Actor()
        except Exception as e:
            raise HTTPException(500, f"Failed to initialize Actor: {str(e)}")
    
    # Find method
    func = None
    if user_actor and hasattr(user_actor, call.method):
        func = getattr(user_actor, call.method)
    elif user_module and hasattr(user_module, call.method):
        func = getattr(user_module, call.method)
    else:
        raise HTTPException(404, f"Method not found: {call.method}")
    
    # Execute method directly - timeout is handled by caller via _timeout parameter
    # The 'timeout' kwarg is passed through to the method for its internal use (e.g., LLM API timeout)
    try:
        if inspect.iscoroutinefunction(func):
            result = await func(*call.args, **call.kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(*call.args, **call.kwargs))

        return MethodResponse(status="success", result=result)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, f"{str(e)}\n{tb}")


@app.get("/methods")
async def list_methods():
    """List available methods with signatures"""
    methods = []
    
    # Get Actor methods (from class definition, not instance)
    if user_module and hasattr(user_module, "Actor"):
        actor_class = getattr(user_module, "Actor")
        for name in dir(actor_class):
            if name.startswith('_'):
                continue
            attr = getattr(actor_class, name)
            if callable(attr):
                try:
                    sig = inspect.signature(attr)
                    methods.append({
                        "name": name,
                        "signature": str(sig),
                        "source": "Actor"
                    })
                except Exception:
                    methods.append({
                        "name": name,
                        "signature": "(...)",
                        "source": "Actor"
                    })
    
    # Get module-level functions
    if user_module:
        for name in dir(user_module):
            if name.startswith('_'):
                continue
            attr = getattr(user_module, name)
            # Only include functions, not classes
            if callable(attr) and not inspect.isclass(attr):
                try:
                    sig = inspect.signature(attr)
                    methods.append({
                        "name": name,
                        "signature": str(sig),
                        "source": "module"
                    })
                except Exception:
                    methods.append({
                        "name": name,
                        "signature": "(...)",
                        "source": "module"
                    })
    
    return {"methods": methods}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


# =============================================================================
# OpenEnv Protocol Endpoints
# =============================================================================

def _get_actor():
    """Get or initialize actor instance"""
    global user_actor
    
    if hasattr(user_module, "Actor") and user_actor is None:
        try:
            user_actor = user_module.Actor()
        except Exception as e:
            raise HTTPException(500, f"Failed to initialize Actor: {str(e)}")
    
    return user_actor


@app.post("/reset", response_model=OpenEnvResponse)
async def reset(request: ResetRequest):
    """
    OpenEnv reset endpoint.
    
    Initializes a new episode and returns the initial observation.
    Requires Actor.reset(task_id, seed, **kwargs) method.
    """
    actor = _get_actor()
    
    if not actor or not hasattr(actor, "reset"):
        raise HTTPException(
            501, 
            "OpenEnv not supported: Actor.reset() method not implemented"
        )
    
    try:
        func = getattr(actor, "reset")
        kwargs = request.kwargs or {}
        
        if inspect.iscoroutinefunction(func):
            result = await func(
                task_id=request.task_id,
                seed=request.seed,
                **kwargs
            )
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: func(task_id=request.task_id, seed=request.seed, **kwargs)
            )
        
        # Handle both dict response and OpenEnvResponse-like objects
        if isinstance(result, dict):
            return OpenEnvResponse(
                observation=result.get("observation", ""),
                reward=result.get("reward", 0.0),
                done=result.get("done", False),
                truncated=result.get("truncated", False),
                info=result.get("info", {})
            )
        else:
            return result
            
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, f"reset() failed: {str(e)}\n{tb}")


@app.post("/step", response_model=OpenEnvResponse)
async def step(request: StepRequest):
    """
    OpenEnv step endpoint.
    
    Takes an action and returns the next observation, reward, and done flag.
    Requires Actor.step(action) method.
    """
    actor = _get_actor()
    
    if not actor or not hasattr(actor, "step"):
        raise HTTPException(
            501,
            "OpenEnv not supported: Actor.step() method not implemented"
        )
    
    try:
        func = getattr(actor, "step")
        
        if inspect.iscoroutinefunction(func):
            result = await func(action=request.action)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: func(action=request.action)
            )
        
        # Handle both dict response and OpenEnvResponse-like objects
        if isinstance(result, dict):
            return OpenEnvResponse(
                observation=result.get("observation", ""),
                reward=result.get("reward", 0.0),
                done=result.get("done", False),
                truncated=result.get("truncated", False),
                info=result.get("info", {})
            )
        else:
            return result
            
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, f"step() failed: {str(e)}\n{tb}")


@app.get("/state", response_model=OpenEnvResponse)
async def state():
    """
    OpenEnv state endpoint.
    
    Returns the current observation without taking any action.
    Requires Actor.state() method.
    """
    actor = _get_actor()
    
    if not actor or not hasattr(actor, "state"):
        raise HTTPException(
            501,
            "OpenEnv not supported: Actor.state() method not implemented"
        )
    
    try:
        func = getattr(actor, "state")
        
        if inspect.iscoroutinefunction(func):
            result = await func()
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func)
        
        # Handle both dict response and OpenEnvResponse-like objects
        if isinstance(result, dict):
            return OpenEnvResponse(
                observation=result.get("observation", ""),
                reward=result.get("reward", 0.0),
                done=result.get("done", False),
                truncated=result.get("truncated", False),
                info=result.get("info", {})
            )
        else:
            return result
            
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, f"state() failed: {str(e)}\n{tb}")