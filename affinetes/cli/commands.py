"""CLI command implementations"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from ..api import load_env, build_image_from_env, get_environment
from ..utils.logger import logger
from .templates import (
    ACTOR_ENV_PY,
    BASIC_ENV_PY,
    FUNCTION_DOCKERFILE,
    FASTAPI_ENV_PY,
    FASTAPI_DOCKERFILE
)


async def run_environment(
    image: Optional[str],
    env_dir: Optional[str],
    tag: Optional[str],
    name: Optional[str],
    env_vars: Dict[str, str],
    pull: bool,
    mem_limit: Optional[str],
    no_cache: bool,
    host_network: bool = False,
    host_port: int = 8000
) -> None:
    """Start an environment container"""
    
    try:
        # Build image from directory if env_dir is provided
        if env_dir:
            if not tag:
                # Auto-generate tag from directory name
                dir_name = env_dir.rstrip('/').split('/')[-1]
                tag = f"{dir_name}:latest"
            
            logger.info(f"Building image '{tag}' from '{env_dir}'")
            
            # Build image
            image = build_image_from_env(
                env_path=env_dir,
                image_tag=tag,
                nocache=no_cache,
                quiet=False
            )
            
            logger.info(f"Image '{image}' built successfully")
        
        # Validate image parameter
        if not image:
            logger.error("Either image or env_dir must be specified")
            return
        
        if "CHUTES_API_KEY" not in env_vars and os.environ.get("CHUTES_API_KEY"):
            env_vars["CHUTES_API_KEY"] = os.environ.get("CHUTES_API_KEY")

        # Load environment using SDK
        env = load_env(
            image=image,
            container_name=name,
            env_vars=env_vars,
            cleanup=False,
            force_recreate=True,
            pull=pull,
            mem_limit=mem_limit,
            host_network=host_network,
            host_port=host_port
        )
        
        logger.info(f"✓ Environment started: {env.name}")
        
        # Show available methods immediately
        await env.list_methods(print_info=True)
        
        print(f"\nUsage:")
        print(f"  afs call {env.name} <method> --arg key=value")
    
    except Exception as e:
        logger.error(f"Failed to start environment: {e}")
        raise


async def call_method(
    name: str,
    method: str,
    args: Dict[str, Any],
    timeout: Optional[int] = 300
) -> None:
    """Call a method on running environment"""
    
    try:
        logger.info(f"Calling {method}({args}) on {name}...")
        
        # Try to get from registry first
        env = get_environment(name)
        
        if not env or not env.is_ready():
            # Not in registry, try to connect to existing container
            logger.debug(f"Environment '{name}' not in registry, connecting to container...")
            try:
                env = load_env(
                    container_name=name,
                    cleanup=False,
                    connect_only=True
                )
                logger.debug(f"Successfully connected to container '{name}'")
            except Exception as e:
                logger.error(
                    f"Failed to connect to container '{name}': {e}\n"
                    f"Please ensure the container is running with: docker ps"
                )
                return
        
        # Call method using SDK's dynamic dispatch
        method_func = getattr(env, method)
        result = await method_func(_timeout=timeout, **args)
        
        logger.info("✓ Method completed successfully")
        
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)
    
    except asyncio.TimeoutError:
        logger.error(f"Method call timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"Failed to call method: {e}")
        raise


async def build_and_push(
    env_dir: str,
    tag: str,
    push: bool,
    registry: Optional[str],
    no_cache: bool,
    quiet: bool,
    build_args: Optional[Dict[str, str]] = None
) -> None:
    """Build environment image and optionally push to registry"""
    
    try:
        env_path = Path(env_dir).resolve()
        
        # Validate environment directory
        if not env_path.exists():
            logger.error(f"Environment directory not found: {env_dir}")
            return
        
        if not (env_path / "env.py").exists():
            logger.error(f"Missing env.py in {env_dir}")
            return
        
        if not (env_path / "Dockerfile").exists():
            logger.error(f"Missing Dockerfile in {env_dir}")
            return
        
        logger.info(f"Building image '{tag}' from '{env_dir}'")
        
        # Build image
        final_tag = build_image_from_env(
            env_path=str(env_path),
            image_tag=tag,
            nocache=no_cache,
            quiet=quiet,
            buildargs=build_args,
            push=push,
            registry=registry
        )
        
        if push:
            logger.info(f"✓ Image built and pushed successfully: {final_tag}")
            logger.info(f"\nTo use this image:")
            logger.info(f"  afs run {final_tag}")
        else:
            logger.info(f"✓ Image built successfully: {final_tag}")
            logger.info(f"\nTo push to registry:")
            logger.info(f"  afs build {env_dir} --tag {tag} --push --registry <registry-url>")
            logger.info(f"\nTo run locally:")
            logger.info(f"  afs run {tag}")
    
    except Exception as e:
        logger.error(f"Failed to build image: {e}")
        raise


def init_environment(
    name: str,
    env_type: str,
    template: str
) -> None:
    """Initialize a new environment directory with template files"""
    
    try:
        env_path = Path(name)
        
        # Check if directory already exists
        if env_path.exists():
            logger.error(f"Directory '{name}' already exists")
            return
        
        # Create directory
        env_path.mkdir(parents=True)
        logger.info(f"Created directory: {name}/")
        
        # Generate files based on template
        if template == 'basic' or (template == 'actor' and env_type == 'function'):
            _create_function_based_env(env_path, use_actor=(template == 'actor'))
        elif template == 'fastapi' or env_type == 'http':
            _create_http_based_env(env_path)
        else:
            _create_function_based_env(env_path, use_actor=False)
        
        logger.info(f"✓ Environment '{name}' initialized successfully")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Build image:")
        logger.info(f"     afs build {name} --tag {name}:v1")
        logger.info(f"")
        logger.info(f"  2. Run environment:")
        logger.info(f"     afs run {name}:v1 --name {name}")
        logger.info(f"")
        logger.info(f"  3. Call methods:")
        logger.info(f"     afs call {name} add --arg a=10.5 --arg b=20.3")
        logger.info(f"     afs call {name} multiply --arg a=3.5 --arg b=4.2")
    
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise


def _create_function_based_env(env_path: Path, use_actor: bool = False) -> None:
    """Create function-based environment files"""
    
    # env.py
    env_py_content = ACTOR_ENV_PY if use_actor else BASIC_ENV_PY
    (env_path / "env.py").write_text(env_py_content)
    logger.info(f"  Created: env.py (function-based)")
    
    # Dockerfile
    (env_path / "Dockerfile").write_text(FUNCTION_DOCKERFILE)
    logger.info(f"  Created: Dockerfile")


def _create_http_based_env(env_path: Path) -> None:
    """Create HTTP-based environment files with FastAPI"""
    
    # env.py
    (env_path / "env.py").write_text(FASTAPI_ENV_PY)
    logger.info(f"  Created: env.py (HTTP-based)")
    
    # Dockerfile
    (env_path / "Dockerfile").write_text(FASTAPI_DOCKERFILE)
    logger.info(f"  Created: Dockerfile")