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
from typing import Any, Optional

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
    
    # Extract timeout parameter if present (only used for wait_for, not passed to method)
    timeout = call.kwargs.get("timeout", None)
    
    # Execute with optional timeout enforcement
    try:
        # Create the execution coroutine
        if inspect.iscoroutinefunction(func):
            exec_coro = func(*call.args, **call.kwargs)
        else:
            loop = asyncio.get_event_loop()
            exec_coro = loop.run_in_executor(None, lambda: func(*call.args, **call.kwargs))
        
        # Apply timeout only if specified
        if timeout is not None:
            try:
                result = await asyncio.wait_for(exec_coro, timeout=timeout)
            except asyncio.TimeoutError:
                # Task was cancelled, return error result
                return MethodResponse(
                    status="failed",
                    result={
                        "error": f"Task execution exceeded timeout of {timeout}s",
                    }
                )
        else:
            # No timeout specified, execute normally
            result = await exec_coro
        
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