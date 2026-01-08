"""Public API for affinetes"""

import time
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

from .backends import LocalBackend, BasilicaBackend, URLBackend
from .infrastructure import ImageBuilder
from .core import EnvironmentWrapper, get_registry, InstancePool, InstanceInfo
from .utils.logger import logger
from .utils.exceptions import ValidationError, BackendError


def build_image_from_env(
    env_path: str,
    image_tag: str,
    nocache: bool = False,
    quiet: bool = False,
    buildargs: Optional[Dict[str, str]] = None,
    push: bool = False,
    registry: Optional[str] = None
) -> str:
    """
    Build Docker image from environment definition and optionally push to registry
    
    Args:
        env_path: Path to environment directory (must contain env.py and Dockerfile)
        image_tag: Image tag (e.g., "affine:latest")
        nocache: Don't use build cache
        quiet: Suppress build output
        buildargs: Docker build arguments (e.g., {"ENV_NAME": "webshop"})
        push: Push image to registry after build (default: False)
        registry: Registry URL (e.g., "docker.io/myuser"). If not specified, uses image_tag as-is
        
    Returns:
        Built image tag (with registry prefix if specified)
        
    Examples:
        >>> # Build only
        >>> build_image_from_env("environments/affine", "affine:latest")
        'affine:latest'
        
        >>> # Build and push to Docker Hub
        >>> build_image_from_env("environments/affine", "affine:latest",
        ...                      push=True, registry="docker.io/myuser")
        'docker.io/myuser/affine:latest'
        
        >>> # Build and push (image_tag already contains registry)
        >>> build_image_from_env("environments/affine", "myregistry.com/affine:latest", push=True)
        'myregistry.com/affine:latest'
    """
    try:
        logger.info(f"Building image '{image_tag}' from '{env_path}'")
        
        builder = ImageBuilder()
        image_id = builder.build_from_env(
            env_path=env_path,
            image_tag=image_tag,
            nocache=nocache,
            quiet=quiet,
            buildargs=buildargs
        )
        
        logger.info(f"Image '{image_tag}' built successfully")
        
        # Push to registry if requested
        if push:
            logger.info(f"Pushing image to registry...")
            builder.push_image(image_tag=image_tag, registry=registry)
            
            # Return final tag (with registry prefix if applicable)
            if registry:
                final_tag = f"{registry}/{image_tag}"
            else:
                final_tag = image_tag
            
            logger.info(f"Image pushed successfully: {final_tag}")
            return final_tag
        
        return image_tag
        
    except Exception as e:
        logger.error(f"Failed to build image: {e}")
        raise


def load_env(
    image: Optional[str] = None,
    mode: str = "docker",
    replicas: int = 1,
    hosts: Optional[List[str]] = None,
    load_balance: str = "random",
    container_name: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    env_type: Optional[str] = None,
    force_recreate: bool = False,
    pull: bool = False,
    mem_limit: Optional[str] = None,
    cpu_limit: Optional[str] = None,
    cleanup: bool = True,
    connect_only: bool = False,
    host_network: bool = False,
    host_port: Optional[int] = None,
    enable_logging: bool = False,
    log_file: Optional[str] = None,
    log_console: bool = True,
    **backend_kwargs
) -> EnvironmentWrapper:
    """
    Load and start an environment with multi-instance support
    
    Args:
        image: Docker image name (required unless connect_only=True)
        connect_only: If True, connect to existing container instead of creating new one
        mode: Execution mode - "docker" or "basilica"
        replicas: Number of instances to deploy (default: 1)
        hosts: List of Docker daemon addresses for deployment
               - None or ["localhost"]: Deploy all replicas locally
               - ["ssh://user@host1", "ssh://user@host2"]: Deploy to remote Docker daemons via SSH
               - Mix allowed: ["localhost", "ssh://user@host"]
        load_balance: Load balancing strategy - "random" or "round_robin" (default: "random")
        container_name: Optional container name prefix (local mode only)
        env_vars: Environment variables to pass to container(s)
        env_type: Override environment type detection ("function_based" or "http_based")
        force_recreate: If True, remove and recreate containers even if they exist (default: False)
        pull: If True, pull the image before deployment (default: False)
        mem_limit: Memory limit (e.g., "512m", "1g", "2g", "16Gi")
        cpu_limit: CPU limit (e.g., "1.0", "2.0" for Docker; "1000m", "4000m" for Basilica)
        cleanup: If True, automatically stop and remove container on exit (default: True)
                 If False, container will continue running after program exits
        host_network: If True, use host network mode (network_mode="host")
        host_port: Port to use when host_network=True (default: 8000). Useful to avoid port conflicts.
        enable_logging: If True, automatically start log streaming (Docker mode only, default: False)
        log_file: Log file path (optional, Docker mode only)
        log_console: Whether to print logs to console (default: True, Docker mode only)
        **backend_kwargs: Additional backend-specific parameters
        
    Returns:
        EnvironmentWrapper instance
        
    Examples:
        # Single local instance
        >>> env = load_env(image="affine:latest")
        
        # 3 local instances with load balancing
        >>> env = load_env(image="affine:latest", replicas=3)
        
        # 2 remote instances via SSH
        >>> env = load_env(
        ...     image="affine:latest",
        ...     replicas=2,
        ...     hosts=["ssh://user@host1", "ssh://user@host2"]
        ... )
        
        # Mixed: 1 local + 2 remote
        >>> env = load_env(
        ...     image="affine:latest",
        ...     replicas=3,
        ...     hosts=["localhost", "ssh://user@host1", "ssh://user@host2"]
        ... )
        
        # Keep container running after exit (for debugging or long-term use)
        >>> env = load_env(image="affine:latest", cleanup=False)
    """
    try:
        # Validate parameters
        if connect_only:
            if not container_name:
                raise ValidationError("container_name is required when connect_only=True")
            if replicas != 1:
                raise ValidationError("connect_only mode only supports single instance (replicas=1)")
        else:
            # URL mode doesn't require image parameter
            if mode != "url" and not image:
                raise ValidationError("image is required for docker and basilica modes")
        
        logger.debug(f"Loading '{image or container_name or 'url-service'}' in {mode} mode (replicas={replicas}, connect_only={connect_only})")
        
        if replicas < 1:
            raise ValidationError("replicas must be >= 1")
        
        if hosts and len(hosts) < replicas:
            raise ValidationError(
                f"Not enough hosts ({len(hosts)}) for replicas ({replicas}). "
                f"Either provide enough hosts or set hosts=None for local deployment."
            )
        
        # Single instance mode (backward compatible)
        if replicas == 1:
            return _load_single_instance(
                image=image,
                mode=mode,
                host=hosts[0] if hosts else None,
                container_name=container_name,
                env_vars=env_vars,
                env_type=env_type,
                force_recreate=force_recreate,
                pull=pull,
                mem_limit=mem_limit,
                cpu_limit=cpu_limit,
                cleanup=cleanup,
                connect_only=connect_only,
                host_network=host_network,
                host_port=host_port,
                enable_logging=enable_logging,
                log_file=log_file,
                log_console=log_console,
                **backend_kwargs
            )
        
        # Multi-instance mode
        return _load_multi_instance(
            image=image,
            mode=mode,
            replicas=replicas,
            hosts=hosts,
            load_balance=load_balance,
            container_name=container_name,
            env_vars=env_vars,
            env_type=env_type,
            force_recreate=force_recreate,
            pull=pull,
            mem_limit=mem_limit,
            cpu_limit=cpu_limit,
            cleanup=cleanup,
            host_network=host_network,
            host_port=host_port,
            **backend_kwargs
        )
        
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        raise


def _load_single_instance(
    image: Optional[str],
    mode: str,
    host: Optional[str],
    container_name: Optional[str],
    env_vars: Optional[Dict[str, str]],
    env_type: Optional[str],
    force_recreate: bool = False,
    pull: bool = False,
    mem_limit: Optional[str] = None,
    cpu_limit: Optional[str] = None,
    cleanup: bool = True,
    connect_only: bool = False,
    host_network: bool = False,
    host_port: Optional[int] = None,
    enable_logging: bool = False,
    log_file: Optional[str] = None,
    log_console: bool = True,
    **backend_kwargs
) -> EnvironmentWrapper:
    """Load a single instance"""
    
    # Create backend based on mode
    if mode == "docker":
        backend = LocalBackend(
            image=image,
            host=host,
            container_name=container_name,
            env_vars=env_vars,
            env_type_override=env_type,
            force_recreate=force_recreate,
            pull=pull,
            mem_limit=mem_limit,
            cpu_limit=cpu_limit,
            auto_cleanup=cleanup,
            connect_only=connect_only,
            host_network=host_network,
            host_port=host_port,
            **backend_kwargs
        )
    elif mode == "url":
        # URL mode for user-deployed environments
        if "base_url" not in backend_kwargs:
            raise ValidationError(
                "URL mode requires 'base_url' parameter. "
                "Example: base_url='http://your-service.com:8080'"
            )
        backend = URLBackend(
            **backend_kwargs
        )
    elif mode == "basilica":
        # Basilica mode - temporary pod per evaluation
        # Extract mode-specific parameters from backend_kwargs
        ttl_buffer = backend_kwargs.pop("ttl_buffer", 300)
        
        backend = BasilicaBackend(
            image=image,
            mem_limit=mem_limit,
            cpu_limit=cpu_limit,
            env_vars=env_vars,
            env_type_override=env_type,
            ttl_buffer=ttl_buffer,
            **backend_kwargs
        )
    else:
        raise ValidationError(
            f"Invalid mode: {mode}. Must be 'docker', 'basilica', or 'url'."
        )
    
    # Create wrapper
    wrapper = EnvironmentWrapper(backend=backend)
    
    # Register in global registry
    registry = get_registry()
    registry.register(backend.name, wrapper)
    
    # Start logging if enabled (Docker mode only)
    if enable_logging and mode == "docker":
        wrapper.start_logging(file=log_file, console=log_console)
    
    logger.debug(f"Single instance '{backend.name}' loaded successfully")
    return wrapper


def _load_multi_instance(
    image: str,
    mode: str,
    replicas: int,
    hosts: Optional[List[str]],
    load_balance: str,
    container_name: Optional[str],
    env_vars: Optional[Dict[str, str]],
    env_type: Optional[str],
    force_recreate: bool = False,
    pull: bool = False,
    mem_limit: Optional[str] = None,
    cpu_limit: Optional[str] = None,
    cleanup: bool = True,
    host_network: bool = False,
    host_port: Optional[int] = None,
    **backend_kwargs
) -> EnvironmentWrapper:
    """Load multiple instances with load balancing"""
    
    logger.info(f"Deploying {replicas} instances with {load_balance} load balancing")
    
    # Determine target hosts
    if not hosts:
        hosts = [None] * replicas  # None means localhost
    
    # Create instances concurrently
    instances = []
    
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Deploy all instances concurrently
        async def deploy_all():
            tasks = [
                _deploy_instance(
                    image=image,
                    mode=mode,
                    host=hosts[i],
                    instance_id=i,
                    container_name=container_name,
                    env_vars=env_vars,
                    env_type=env_type,
                    force_recreate=force_recreate,
                    pull=pull,
                    mem_limit=mem_limit,
                    cpu_limit=cpu_limit,
                    cleanup=cleanup,
                    host_network=host_network,
                    host_port=host_port + i if host_port else None,
                    **backend_kwargs
                )
                for i in range(replicas)
            ]
            return await asyncio.gather(*tasks)
        
        instances = loop.run_until_complete(deploy_all())
        
        logger.info(f"Successfully deployed {len(instances)} instances")
        
        # Generate meaningful pool name based on image and container name
        safe_image = image.split('/')[-1].replace(':', '-')
        name_prefix = container_name or safe_image
        pool_name = f"{name_prefix}-pool-{replicas}"
        
        # Create instance pool with custom name
        pool = InstancePool(
            instances=instances,
            load_balance_strategy=load_balance,
            pool_name=pool_name
        )
        
        # Create wrapper
        wrapper = EnvironmentWrapper(backend=pool)
        
        # Register in global registry
        registry = get_registry()
        registry.register(pool.name, wrapper)
        
        logger.debug(f"Multi-instance pool '{pool.name}' loaded successfully")
        return wrapper
        
    except Exception as e:
        # Cleanup any successfully deployed instances
        logger.error(f"Failed to deploy instances: {e}")
        
        if instances:
            logger.info("Cleaning up partially deployed instances")
            try:
                async def cleanup_all():
                    tasks = [inst.backend.cleanup() for inst in instances]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                loop.run_until_complete(cleanup_all())
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
        
        raise BackendError(f"Multi-instance deployment failed: {e}")


async def _deploy_instance(
    image: str,
    mode: str,
    host: Optional[str],
    instance_id: int,
    container_name: Optional[str],
    env_vars: Optional[Dict[str, str]],
    env_type: Optional[str],
    force_recreate: bool = False,
    pull: bool = False,
    mem_limit: Optional[str] = None,
    cpu_limit: Optional[str] = None,
    cleanup: bool = True,
    host_network: bool = False,
    host_port: Optional[int] = None,
    **backend_kwargs
) -> InstanceInfo:
    """Deploy a single instance (async)"""
    
    host_display = host or "localhost"
    logger.debug(f"Deploying instance {instance_id} on {host_display}")
    
    # Docker deployment (supports SSH remote via host parameter)
    if mode == "docker":
        # Generate unique container name (sanitize image name)
        safe_image = image.split('/')[-1].replace(':', '-')
        name_prefix = container_name or safe_image
        unique_name = f"{name_prefix}-{instance_id}"
        
        backend = LocalBackend(
            image=image,
            host=host,
            container_name=unique_name,
            env_vars=env_vars,
            env_type_override=env_type,
            force_recreate=force_recreate,
            pull=pull,
            mem_limit=mem_limit,
            cpu_limit=cpu_limit,
            auto_cleanup=cleanup,
            host_network=host_network,
            host_port=host_port,
            **backend_kwargs
        )
    else:
        raise ValidationError(f"Mode '{mode}' not supported for multi-instance")
    
    # Create InstanceInfo (use container internal IP as identifier)
    instance_info = InstanceInfo(
        host=host_display,
        port=8000,  # Internal port (not exposed)
        backend=backend,
    )
    
    logger.debug(f"Instance {instance_id} deployed: {instance_info}")
    return instance_info


def list_active_environments() -> list:
    """
    List all currently active environments
    
    Returns:
        List of environment IDs
        
    Example:
        >>> list_active_environments()
        ['affine-latest_1234567890', 'custom-v1_1234567891']
    """
    registry = get_registry()
    return registry.list_all()


def cleanup_all_environments() -> None:
    """
    Clean up all active environments
    
    Stops all containers and frees resources.
    Automatically called on program exit.
    
    Example:
        >>> cleanup_all_environments()
    """
    logger.info("Cleaning up all environments")
    registry = get_registry()
    registry.cleanup_all()


def get_environment(env_id: str) -> Optional[EnvironmentWrapper]:
    """
    Get an environment by ID
    
    Args:
        env_id: Environment identifier
        
    Returns:
        EnvironmentWrapper instance or None if not found
        
    Example:
        >>> env = get_environment('affine-latest_1234567890')
    """
    registry = get_registry()
    return registry.get(env_id)