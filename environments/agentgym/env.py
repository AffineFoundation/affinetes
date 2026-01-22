#!/usr/bin/env python3

import os
import sys
import importlib
from loguru import logger
import time
import asyncio
import httpx
import traceback
import random
import secrets
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import partial

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Memory monitoring configuration
MEMORY_CHECK_INTERVAL = 60  # seconds
MEMORY_THRESHOLD = 0.90  # 90%
MEMORY_ENABLED = os.getenv("MEMORY_MONITOR_ENABLED", "true").lower() == "true"


async def monitor_memory():
    """Background task to monitor container memory usage and exit if threshold exceeded."""
    if not MEMORY_ENABLED:
        logger.info("Memory monitoring disabled")
        return
    
    logger.info(f"Memory monitoring started (threshold: {MEMORY_THRESHOLD*100}%, interval: {MEMORY_CHECK_INTERVAL}s)")
    
    while True:
        try:
            await asyncio.sleep(MEMORY_CHECK_INTERVAL)
            
            # Read cgroup v2 memory metrics
            mem_current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            mem_max_str = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            
            if mem_max_str == "max":
                # No memory limit set, skip check
                continue
            
            mem_max = int(mem_max_str)
            usage_ratio = mem_current / mem_max
            
            logger.debug(f"Memory usage: {usage_ratio*100:.1f}% ({mem_current}/{mem_max} bytes)")
            
            if usage_ratio >= MEMORY_THRESHOLD:
                logger.critical(
                    f"Memory threshold exceeded: {usage_ratio*100:.1f}% >= {MEMORY_THRESHOLD*100}%. "
                    f"Forcing process exit to trigger container restart."
                )
                os._exit(1)  # Force immediate exit to trigger Docker restart
                
        except FileNotFoundError:
            logger.warning("Memory cgroup files not found, disabling memory monitoring")
            break
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")
            # Continue monitoring despite errors
            continue


ENV_NAME = os.environ.get("ENV_NAME")
TOOL_NAME = os.environ.get("TOOL_NAME", "")

class EvaluatorRequest(BaseModel):
    model: str
    base_url: str = "https://llm.chutes.ai/v1"
    max_tokens: int = None
    temperature: float = 0.7
    top_p: float = 1.0
    task_id: int  # Required task ID for agentgym tasks
    max_round: int = 10
    env_server_base: Optional[str] = "http://localhost:8000"
    data_len: int = 200
    timeout: int = 2400
    api_key: Optional[str] = None  # Allow API key override via request parameter
    seed: Optional[int] = None  # Random seed for reproducible generation


class EvaluatorResponse(BaseModel):
    task_name: str
    score: float
    success: bool
    time_taken: float
    extra: Dict[str, Any]
    error: Optional[str] = None


def inject_health_endpoint(app: FastAPI):
    """Inject a health check endpoint into the existing FastAPI app."""

    for route in app.routes:
        if hasattr(route, 'path') and route.path == '/health':
            logger.info("Health endpoint already exists, skipping injection")
            return

    @app.get("/health")
    async def health_check():
        return "ok"

    logger.info("Health endpoint injected successfully")


async def validate_api_key(api_key: str, base_url: str) -> bool:
    """Validate the API key by making a test request to the API."""
    if not api_key:
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
            return response.status_code >= 200 and response.status_code < 300
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False


def inject_evaluator_endpoint(app: FastAPI):
    """Inject an evaluator endpoint into the existing FastAPI app."""
    
    for route in app.routes:
        if hasattr(route, 'path') and route.path == '/evaluator':
            logger.info("Evaluator endpoint already exists, skipping injection")
            return

    @app.post("/evaluate", response_model=EvaluatorResponse)
    async def evaluate_model(request: EvaluatorRequest):
        """
        Evaluate a model on AgentGym tasks.
        
        This endpoint allows evaluating language models on various AgentGym tasks
        by providing model configuration and task parameters.
        """
        try:
            from agentenv.controller import APIAgent, Evaluator

            # Import task classes dynamically
            task_modules = {
                "webshop": "WebshopTask",
                "alfworld": "AlfWorldTask",
                "babyai": "BabyAITask",
                "sciworld": "SciworldTask",
                "textcraft": "TextCraftTask",
                "webarena": "WebarenaTask",
                "sqlgym": "SqlGymTask",
                "maze": "MazeTask",
                "wordle": "WordleTask",
                "weather": "WeatherTask",
                "todo": "TodoTask",
                "movie": "MovieTask",
                "sheet": "SheetTask",
                "academia": "AcademiaTask",
                "searchqa": "SearchQATask",
            }
            if ENV_NAME == "tool" or ENV_NAME == "lmrlgym":
                class_name = task_modules[TOOL_NAME]
            else:
                class_name = task_modules[ENV_NAME]
            module = importlib.import_module("agentenv.envs")
            task_class = getattr(module, class_name)

            env_server_base = request.env_server_base
            if not env_server_base:
                env_server_base = f"http://localhost:8000"
            if ENV_NAME == "lmrlgym":
                env_server_base += f"/{TOOL_NAME}"

            # Generate random seed if not provided
            seed = request.seed
            if seed is None:
                seed = secrets.randbelow(2**32)

            env_args = {
                "env_server_base": env_server_base,
                "data_len": request.data_len,
                "timeout": request.timeout,
                "seed": seed,
            }

            # Only validate API key for chutes.ai base URLs
            api_key = None
            if "chutes" in request.base_url.lower():
                # Priority: request parameter > environment variable
                api_key = request.api_key or os.environ.get("CHUTES_API_KEY")
                if not api_key:
                    raise HTTPException(
                        status_code=401,
                        detail="API key not provided in request and CHUTES_API_KEY environment variable is not set"
                    )

                is_valid = await validate_api_key(api_key, request.base_url)
                if not is_valid:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Invalid API key for {request.base_url}. Please check your CHUTES_API_KEY environment variable."
                    )

                logger.info(f"API key validated successfully for {request.base_url}")
            else:
                # For non-chutes URLs, use provided API key without validation
                api_key = request.api_key or os.environ.get("CHUTES_API_KEY")
                logger.info(f"Using provided API key for {request.base_url} (validation skipped)")

            agent = APIAgent(
                api_key=api_key,
                base_url=request.base_url,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=seed,
            )
            
            loop = asyncio.get_event_loop()
            def create_evaluator():
                """Create evaluator in thread to avoid blocking"""
                task_instance = task_class(client_args=env_args, n_clients=1)
                return Evaluator(agent, [task_instance])
            evaluator = await loop.run_in_executor(None, create_evaluator)
            logger.info("Evaluator created successfully")

            start_time = time.time()
            logger.info(f"Evaluating task_id: {request.task_id}")
            
            try:
                exps = await loop.run_in_executor(
                    None,
                    partial(
                        evaluator.eval,
                        max_rounds=request.max_round,
                        idxs=[request.task_id]
                    )
                )
                
                score = exps.score
                success = exps.success
                experiences = vars(exps.experiences[0])
                error = None
                
            except Exception as e:
                logger.error(f"Error evaluating task_id {request.task_id}: {e}")
                traceback.print_exc()
                score = 0.0
                success = False
                experiences = {}
                error = str(e)
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            # Return response
            env_name = os.environ.get("ENV_NAME")
            # Add seed to extra (experiences)
            if experiences and isinstance(experiences, dict):
                experiences['seed'] = seed
            return EvaluatorResponse(
                task_name=env_name,
                score=score,
                success=success,
                time_taken=time_taken,
                extra=experiences,
                error=error
            )
            
        except HTTPException:
            # Re-raise HTTPException to preserve status codes (e.g., 401)
            raise
        except ImportError as e:
            logger.error(f"Import error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to import required modules: {e}")
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(tb_str)
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}, {tb_str}")
    
    logger.info("Evaluator endpoint injected successfully")


def create_app():
    logger.info(f"Loading {ENV_NAME} environment server")
    if ENV_NAME == "tool":
        module_name = f"agentenv_{TOOL_NAME}.{TOOL_NAME}_server"
    else:
        module_name = f"agentenv_{ENV_NAME}.server"
    try:
        logger.info(f"Importing module: {module_name}")
        os.chdir(f"/app/AgentGym/agentenv-{ENV_NAME}")
        if ENV_NAME == "lmrlgym":
            sys.path.insert(0, "/app/AgentGym/agentenv-lmrlgym/lmrlgym")
        if ENV_NAME == "sqlgym":
            sys.path.insert(0, "/app/AgentGym/agentenv-sqlgym")
            os.environ["AGENTENV_SQLGYM_BIRD_PATH"] = "/app/AgentGym/agentenv-sqlgym/bird/"
        if ENV_NAME == "tool":
            sys.path.insert(0, "/app/AgentGym/agentenv-tool/Toolusage/toolusage")
            sys.path.insert(0, "/app/AgentGym/agentenv-tool")
            os.environ["PROJECT_PATH"] = "/app/AgentGym/agentenv-tool/Toolusage"

        server_module = importlib.import_module(module_name)
        app = server_module.app
        logger.info(f"Successfully loaded {ENV_NAME} environment app")
        
        inject_health_endpoint(app)
        inject_evaluator_endpoint(app)
        
        return app
    except Exception as e:
        logger.error(f"Unexpected error loading environment: {e}")
        import traceback
        traceback.print_exc()
        

app = create_app()

@app.on_event("startup")
async def startup_event():
    env_name = os.environ.get("ENV_NAME", "unknown")
    logger.info(f"Environment server ready for: {env_name}")

    # Start background memory monitoring
    asyncio.create_task(monitor_memory())