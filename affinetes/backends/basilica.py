"""Basilica backend - Temporary pod creation for each evaluation task

This backend creates a new Basilica deployment (pod) for each evaluation task,
providing complete environment isolation. The pod is automatically destroyed
after task completion via TTL mechanism.

Key features:
- One pod per evaluate() call
- Automatic TTL-based cleanup
- Complete task isolation
- Suitable for heavy/stateful workloads (e.g., GAME environment)
"""

import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict
from urllib.parse import urlparse

from .base import AbstractBackend
from ..infrastructure import HTTPExecutor, EnvType
from ..utils.exceptions import BackendError
from ..utils.logger import logger

# Set AFFINETES_MAX_CONCURRENT_DEPLOYMENTS to limit concurrent SDK operations
_max_concurrent_env = os.getenv("AFFINETES_MAX_CONCURRENT_DEPLOYMENTS")
_sdk_executor = (
    ThreadPoolExecutor(
        max_workers=int(_max_concurrent_env),
        thread_name_prefix="basilica_sdk_"
    )
    if _max_concurrent_env
    else None  # None = use default executor (no limit)
)


class BasilicaBackend(AbstractBackend):
    """
    Basilica backend for temporary pod deployments

    Each call_method() creates a new deployment, waits for it to be ready,
    executes the method, and then deletes the deployment.

    Usage:
        >>> env = load_env(
        ...     mode="basilica",
        ...     image="affinefoundation/game:openspiel",
        ...     basilica_config={
        ...         "api_token": "xxx",
        ...         "cpu": "4000m",
        ...         "memory": "16Gi",
        ...         "ttl_buffer": 300,
        ...     }
        ... )
        >>> result = await env.evaluate(task_id=1, timeout=1800)
    """

    def __init__(
        self,
        image: str,
        mem_limit: Optional[str] = None,
        cpu_limit: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        env_type_override: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Basilica backend

        Args:
            image: Docker image name (e.g., "affinefoundation/game:openspiel")
            mem_limit: Memory limit (e.g., "16Gi", "8Gi") - Kubernetes memory request
            cpu_limit: CPU limit (e.g., "4000m", "2000m") - Kubernetes CPU format
            env_vars: Environment variables to pass to pod (e.g., {"CHUTES_API_KEY": "xxx", "UVICORN_WORKERS": "1"})
            env_type_override: Force environment type detection
            **kwargs: Additional backend parameters:
                - ttl_buffer: Time-to-live buffer in seconds (default: 300)
        """
        self.image = image
        self.kwargs = kwargs
        self._env_type = env_type_override

        # Get API token from environment
        self.api_token = os.getenv("BASILICA_API_TOKEN")
        if not self.api_token:
            raise BackendError(
                "Basilica API token not found. "
                "Set BASILICA_API_TOKEN environment variable"
            )

        # Resource configuration (use defaults if not provided)
        self.cpu = cpu_limit or "2000m"
        self.memory = mem_limit or "8Gi"
        self.ttl_buffer = kwargs.get("ttl_buffer", 300)

        # Environment variables to pass to pod
        self.env_vars = env_vars or {}
        # Set default UVICORN_WORKERS=1 if not specified
        if "UVICORN_WORKERS" not in self.env_vars:
            self.env_vars["UVICORN_WORKERS"] = "1"

        # Generate unique backend name
        safe_image = image.split('/')[-1].replace(':', '-')
        self.name = f"basilica-pod-{safe_image}-{int(time.time())}"

        logger.info(
            f"BasilicaBackend initialized: {image} "
            f"(cpu={self.cpu}, memory={self.memory}, ttl_buffer={self.ttl_buffer}s)"
        )

    def _generate_deployment_name(self, method_name: str, task_id: Optional[int] = None) -> str:
        """
        Generate unique deployment name

        Format: {image-safe}-{method}-{task_id}-{timestamp}
        Limited to 63 characters for Kubernetes compatibility

        Args:
            method_name: Method being called (e.g., "evaluate")
            task_id: Task ID if available

        Returns:
            Unique deployment name
        """
        # Sanitize image name
        safe_image = self.image.split('/')[-1].replace(':', '-').replace('_', '-')[:15]

        # Build name components
        timestamp = int(time.time())
        if task_id is not None:
            name = f"{safe_image}-{method_name[:8]}-t{task_id}-{timestamp}"
        else:
            name = f"{safe_image}-{method_name[:8]}-{timestamp}"

        # Ensure length limit
        return name[:63].lower()

    def _calculate_ttl(self, timeout: Optional[int] = None) -> int:
        """
        Calculate deployment TTL

        TTL = timeout + ttl_buffer (for cold start and cleanup)

        Args:
            timeout: Task timeout in seconds (default: 1800)

        Returns:
            TTL in seconds
        """
        timeout = timeout or 1800
        return timeout + self.ttl_buffer

    async def _create_deployment(
        self,
        deployment_name: str,
        ttl_seconds: int
    ) -> Any:
        """
        Create Basilica deployment asynchronously.

        Uses thread pool to run blocking SDK calls without blocking the event loop,
        enabling true concurrent deployment creation.

        Args:
            deployment_name: Unique deployment name
            ttl_seconds: Time-to-live in seconds

        Returns:
            Deployment object
        """
        try:
            from basilica import BasilicaClient, Deployment
        except ImportError:
            raise BackendError(
                "basilica-sdk not installed. Install with: pip install basilica-sdk>=0.10.0"
            )

        logger.info(f"Creating deployment: {deployment_name} (TTL: {ttl_seconds}s)")

        # Capture instance variables for closure
        api_token = self.api_token
        image = self.image
        cpu = self.cpu
        memory = self.memory
        env_vars = self.env_vars

        def _sync_create_and_wait() -> Any:
            """Synchronous SDK operations to run in thread pool."""
            os.environ["BASILICA_API_TOKEN"] = api_token
            client = BasilicaClient()

            response = client.create_deployment(
                instance_name=deployment_name,
                image=image,
                port=8000,
                cpu=cpu,
                memory=memory,
                ttl_seconds=ttl_seconds,
                public=True,
                env=env_vars,
            )

            logger.debug(f"Deployment created: {response.instance_name}")

            deployment = Deployment._from_response(client, response)
            logger.info(f"Waiting for deployment {deployment_name} to be ready...")

            # Wait for deployment - use 80% of TTL to allow time for task execution
            # No arbitrary cap; let the TTL drive the timeout
            wait_timeout = int(ttl_seconds * 0.8)
            deployment.wait_until_ready(timeout=wait_timeout, silent=True)
            deployment.refresh()

            return deployment

        try:
            loop = asyncio.get_running_loop()
            deployment = await loop.run_in_executor(_sdk_executor, _sync_create_and_wait)

            logger.info(f"Deployment ready: {deployment.url}")
            return deployment

        except Exception as e:
            logger.error(f"Failed to create deployment {deployment_name}: {e}")
            raise BackendError(f"Deployment creation failed: {e}")

    async def _delete_deployment(self, deployment_name: str) -> None:
        """
        Delete Basilica deployment asynchronously.

        Uses thread pool to avoid blocking the event loop during deletion.

        Args:
            deployment_name: Deployment name to delete
        """
        try:
            from basilica import BasilicaClient
        except ImportError:
            logger.warning("basilica-sdk not available, skipping deletion")
            return

        api_token = self.api_token

        def _sync_delete() -> None:
            """Synchronous deletion to run in thread pool."""
            os.environ["BASILICA_API_TOKEN"] = api_token
            client = BasilicaClient()
            logger.info(f"Deleting deployment: {deployment_name}")
            client.delete_deployment(deployment_name)
            logger.debug(f"Deployment deleted: {deployment_name}")

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_sdk_executor, _sync_delete)
        except Exception as e:
            logger.warning(f"Failed to delete deployment {deployment_name}: {e}")

    async def _detect_env_type(self, base_url: str) -> str:
        """
        Detect environment type by checking endpoints

        Args:
            base_url: Deployment base URL

        Returns:
            EnvType.FUNCTION_BASED or EnvType.HTTP_BASED
        """
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                # Check for function_based endpoint
                response = await client.get(f"{base_url}/methods")
                if response.status_code == 200:
                    logger.debug("Detected function_based environment")
                    return EnvType.FUNCTION_BASED
            except Exception:
                pass

            try:
                # Check for http_based endpoint
                response = await client.get(f"{base_url}/openapi.json")
                if response.status_code == 200:
                    logger.debug("Detected http_based environment")
                    return EnvType.HTTP_BASED
            except Exception:
                pass

            # Default to function_based
            logger.warning("Could not detect environment type, defaulting to function_based")
            return EnvType.FUNCTION_BASED

    async def _wait_for_http_ready(
        self,
        base_url: str,
        max_retries: int = 300,
        retry_delay: float = 2.0
    ) -> None:
        """
        Wait for HTTP server to be ready by polling /health endpoint.

        The Kubernetes deployment may be "ready" but the HTTP server inside
        the container may still be starting up. This function polls until
        the server responds.

        Args:
            base_url: Deployment base URL
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        import httpx

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        logger.info(f"HTTP server ready after {attempt + 1} attempts ({(attempt + 1) * retry_delay:.0f}s)")
                        return
            except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                if attempt < max_retries - 1:
                    logger.debug(f"HTTP not ready (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    raise BackendError(
                        f"HTTP server not ready after {max_retries} attempts: {e}"
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Health check failed (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    raise BackendError(f"Health check failed: {e}")

    async def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call method on temporary pod

        Creates deployment → executes method → deletes deployment

        Args:
            method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Method result
        """
        # Extract task_id and timeout from kwargs
        task_id = kwargs.get("task_id")
        timeout = kwargs.get("timeout", 1800)

        # Generate deployment name and TTL
        deployment_name = self._generate_deployment_name(method_name, task_id)
        ttl_seconds = self._calculate_ttl(timeout)

        deployment = None
        http_executor = None

        try:
            # Create deployment
            deployment = await self._create_deployment(deployment_name, ttl_seconds)
            base_url = deployment.url

            # Wait for HTTP server to be ready (handles container startup delay)
            await self._wait_for_http_ready(base_url)

            # Detect environment type if not overridden
            if not self._env_type:
                self._env_type = await self._detect_env_type(base_url)

            # Parse URL for HTTPExecutor
            parsed = urlparse(base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            # Create HTTP executor
            http_executor = HTTPExecutor(
                container_ip=host,
                container_port=port,
                env_type=self._env_type,
            )
            http_executor.base_url = base_url

            # Execute method
            logger.debug(f"Calling method '{method_name}' on {base_url}")
            result = await http_executor.call_method(method_name, *args, **kwargs)

            logger.info(
                f"Method '{method_name}' completed successfully "
                f"(deployment: {deployment_name})"
            )
            return result

        except Exception as e:
            logger.error(
                f"Method '{method_name}' failed on deployment {deployment_name}: {e}"
            )
            raise BackendError(f"Method execution failed: {e}")

        finally:
            # Cleanup HTTP executor
            if http_executor:
                await http_executor.close()

            # Delete deployment (async, don't wait)
            if deployment:
                # Note: TTL will auto-delete, but we clean up immediately to save resources
                asyncio.create_task(self._delete_deployment(deployment.name))

    async def list_methods(self) -> list:
        """
        List available methods

        Note: This creates a temporary deployment just for listing methods.
        It's recommended to cache method lists or use documentation instead.

        Returns:
            List of method information
        """
        logger.warning(
            "list_methods() creates a temporary deployment. "
            "Consider using documentation for method information."
        )

        deployment_name = self._generate_deployment_name("list_methods")
        ttl_seconds = 300  # Short TTL for listing

        deployment = None
        http_executor = None

        try:
            deployment = await self._create_deployment(deployment_name, ttl_seconds)
            base_url = deployment.url

            if not self._env_type:
                self._env_type = await self._detect_env_type(base_url)

            parsed = urlparse(base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            http_executor = HTTPExecutor(
                container_ip=host,
                container_port=port,
                env_type=self._env_type,
            )
            http_executor.base_url = base_url

            return await http_executor.list_methods()

        finally:
            if http_executor:
                await http_executor.close()
            if deployment:
                asyncio.create_task(self._delete_deployment(deployment_name))

    async def health_check(self) -> bool:
        """
        Health check

        For pod backend, we always return True since deployments are created on-demand.

        Returns:
            True
        """
        return True

    async def cleanup(self) -> None:
        """
        Cleanup backend

        For pod backend, no persistent resources to clean up.
        Individual pods are cleaned up after each call_method().
        """
        logger.debug(f"BasilicaBackend cleanup: {self.name}")

    def is_ready(self) -> bool:
        """
        Check if backend is ready

        Pod backend is always ready (creates pods on-demand).

        Returns:
            True
        """
        return True
