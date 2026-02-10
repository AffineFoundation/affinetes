"""
Auto-injected HTTP server template for function-based environments.
This file is copied to container during image build via two-stage build.

Provides two ways to call methods:
1. POST /call {"method": "xxx", "args": [...], "kwargs": {...}}  (SDK style)
2. POST /{method_name} {"arg1": "val1", ...}  (RESTful style)
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import importlib.util
import asyncio
import inspect
import sys
import traceback
import dataclasses
from typing import Any, Optional, List

app = FastAPI(title="affinetes HTTP Server")


def _serialize_result(result: Any) -> Any:
    """Serialize result to JSON-compatible format.

    Handles:
    - dataclass instances (e.g., OpenEnvResponse)
    - objects with to_dict() method
    - plain dicts and primitives
    """
    if result is None:
        return None
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        # Convert dataclass to dict
        return dataclasses.asdict(result)
    if hasattr(result, "to_dict") and callable(result.to_dict):
        return result.to_dict()
    if hasattr(result, "model_dump") and callable(result.model_dump):
        # Pydantic v2
        return result.model_dump()
    if hasattr(result, "dict") and callable(result.dict):
        # Pydantic v1
        return result.dict()
    return result

# User module will be loaded at runtime
user_module = None
user_actor = None

# Track registered dynamic routes to avoid duplicates
_registered_routes: set = set()


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


def _collect_user_methods() -> List[str]:
    """Collect all callable method names from user module"""
    methods = []
    seen = set()

    # Collect Actor class methods
    if user_module and hasattr(user_module, "Actor"):
        actor_class = user_module.Actor
        for name in dir(actor_class):
            if name.startswith('_'):
                continue
            attr = getattr(actor_class, name, None)
            if callable(attr) and name not in seen:
                methods.append(name)
                seen.add(name)

    # Collect module-level functions
    if user_module:
        for name in dir(user_module):
            if name.startswith('_'):
                continue
            attr = getattr(user_module, name, None)
            if callable(attr) and not inspect.isclass(attr) and name not in seen:
                methods.append(name)
                seen.add(name)

    return methods


def _register_dynamic_routes():
    """Register dynamic HTTP routes for each user method"""
    methods = _collect_user_methods()

    for method_name in methods:
        if method_name in _registered_routes:
            continue
        _create_method_route(method_name)
        _registered_routes.add(method_name)


def _create_method_route(method_name: str):
    """Create a dedicated HTTP route for a single method"""

    async def method_handler(request: Request):
        """
        Handle direct method call via /{method_name}

        Supports two request formats:
        1. {"args": [...], "kwargs": {...}}  - explicit args/kwargs
        2. {"param1": "val1", ...}           - all params as kwargs
        """
        global user_actor

        # Lazy initialize Actor on first call
        if hasattr(user_module, "Actor") and user_actor is None:
            try:
                user_actor = user_module.Actor()
            except Exception as e:
                raise HTTPException(500, f"Failed to initialize Actor: {str(e)}")

        # Find method
        func = None
        if user_actor and hasattr(user_actor, method_name):
            func = getattr(user_actor, method_name)
        elif user_module and hasattr(user_module, method_name):
            func = getattr(user_module, method_name)
        else:
            raise HTTPException(404, f"Method not found: {method_name}")

        # Parse request body
        try:
            body = await request.body()
            if body:
                import json
                data = json.loads(body)
            else:
                data = {}
        except Exception:
            data = {}

        # Determine args and kwargs from request
        if isinstance(data, dict):
            if "args" in data or "kwargs" in data:
                # Explicit format: {"args": [...], "kwargs": {...}}
                args = data.get("args", [])
                kwargs = data.get("kwargs", {})
            else:
                # Direct format: all fields as kwargs
                args = []
                kwargs = data
        else:
            args = []
            kwargs = {}

        # Execute method
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))

            return MethodResponse(status="success", result=_serialize_result(result))
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(500, f"{str(e)}\n{tb}")

    # Register the route with FastAPI
    # Use unique function name to avoid conflicts
    method_handler.__name__ = f"handle_{method_name}"
    app.add_api_route(
        f"/{method_name}",
        method_handler,
        methods=["POST"],
        response_model=MethodResponse,
        name=f"call_{method_name}",
        summary=f"Call {method_name} method directly",
    )


@app.on_event("startup")
async def startup():
    """Load user environment and register dynamic routes on startup"""
    _load_user_env()
    _register_dynamic_routes()


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
    
    # Read timeout (seconds). Note: `timeout` is also forwarded to the user method via
    # **call.kwargs (unchanged).
    #
    # Why enforce a server-side timeout?
    # - If a call hangs (especially sync code running in the thread pool), the request can
    #   block forever. Over time, stuck calls can pile up and exhaust executor workers,
    #   eventually making the server unable to process new requests.
    # - `asyncio.wait_for` is a best-effort guardrail: it stops *waiting* and returns an
    #   error to the client, but it cannot forcibly kill a blocked thread (it may continue
    #   running in the background and occupy a worker).
    timeout = call.kwargs.get("timeout")
    timeout_s = None
    if timeout is not None:
        try:
            timeout_s = float(timeout)
        except (TypeError, ValueError):
            return MethodResponse(
                status="failed",
                result={"error": "Invalid timeout; expected a number of seconds."},
            )
    # Add a small grace period for server-side enforcement to avoid cutting off user code
    # that uses the same timeout internally.
    enforcement_timeout_s = None if timeout_s is None else timeout_s + 10.0

    # Execute with optional timeout enforcement
    try:
        # Build an awaitable unit of work (sync -> thread pool, async -> await directly)
        exec_coro = (
            func(*call.args, **call.kwargs)
            if inspect.iscoroutinefunction(func)
            else asyncio.to_thread(func, *call.args, **call.kwargs)
        )

        try:
            result = (
                await asyncio.wait_for(exec_coro, timeout=enforcement_timeout_s)
                if timeout_s is not None
                else await exec_coro
            )
        except asyncio.TimeoutError:
            return MethodResponse(
                status="failed",
                result={
                    "error": f"Task execution exceeded timeout of {timeout_s}s (+10s grace)"
                },
            )

        return MethodResponse(status="success", result=_serialize_result(result))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, f"{str(e)}\n{tb}")


@app.get("/methods")
async def list_methods():
    """List available methods with signatures and endpoint info"""
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
                        "source": "Actor",
                        "endpoint": f"POST /{name}",
                    })
                except Exception:
                    methods.append({
                        "name": name,
                        "signature": "(...)",
                        "source": "Actor",
                        "endpoint": f"POST /{name}",
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
                        "source": "module",
                        "endpoint": f"POST /{name}",
                    })
                except Exception:
                    methods.append({
                        "name": name,
                        "signature": "(...)",
                        "source": "module",
                        "endpoint": f"POST /{name}",
                    })

    return {
        "methods": methods,
        "usage": {
            "sdk_style": "POST /call {\"method\": \"xxx\", \"args\": [...], \"kwargs\": {...}}",
            "restful_style": "POST /{method_name} {\"param1\": \"val1\", ...}",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}