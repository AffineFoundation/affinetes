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
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict, Set
from urllib.parse import urlparse

import httpx

from .base import AbstractBackend
from ..infrastructure import HTTPExecutor, EnvType
from ..utils.exceptions import BackendError
from ..utils.logger import logger


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

        # Import and create SDK client once (thread-safe for concurrent use)
        try:
            from basilica import BasilicaClient, Deployment
        except ImportError:
            raise BackendError(
                "basilica-sdk not installed. Install with: pip install basilica-sdk>=0.19.0"
            )
        self._client = BasilicaClient(api_key=self.api_token)
        self._Deployment = Deployment

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

        # Large thread pool for blocking SDK calls -- threads are I/O-bound
        # (network waits), so high count is safe. No client-side throttle;
        # Basilica's API and K8s scheduler handle admission control.
        self._executor = ThreadPoolExecutor(
            max_workers=512,
            thread_name_prefix="basilica_sdk_",
        )

        # Track in-flight cleanup tasks to await them on shutdown
        self._cleanup_tasks: Set[asyncio.Task] = set()

        # Guard env_type detection so it runs exactly once across concurrent tasks
        self._env_type_lock = asyncio.Lock()

        logger.info(
            f"BasilicaBackend initialized: {image} "
            f"(cpu={self.cpu}, memory={self.memory}, ttl_buffer={self.ttl_buffer}s)"
        )

    def _generate_deployment_name(self, method_name: str, task_id: Optional[int] = None) -> str:
        """
        Generate unique deployment name

        Format: {image-safe}-{method}-{task_id}-{uuid_suffix}
        Limited to 63 characters for Kubernetes compatibility.
        Uses uuid4 hex suffix to avoid collisions when spawning many tasks
        in the same second.

        Args:
            method_name: Method being called (e.g., "evaluate")
            task_id: Task ID if available

        Returns:
            Unique deployment name
        """
        safe_image = self.image.split('/')[-1].replace(':', '-').replace('_', '-')[:15]
        suffix = uuid.uuid4().hex[:8]

        if task_id is not None:
            name = f"{safe_image}-{method_name[:8]}-t{task_id}-{suffix}"
        else:
            name = f"{safe_image}-{method_name[:8]}-{suffix}"

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
        logger.info(f"Creating deployment: {deployment_name} (TTL: {ttl_seconds}s)")

        # Capture instance variables for closure
        client = self._client
        DeploymentCls = self._Deployment
        image = self.image
        cpu = self.cpu
        memory = self.memory
        env_vars = self.env_vars

        def _sync_create_and_wait() -> Any:
            """Synchronous SDK operations to run in thread pool."""
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

            deployment = DeploymentCls._from_response(client, response)
            logger.info(f"Waiting for deployment {deployment_name} to be ready...")

            # Wait for deployment - use 80% of TTL to allow time for task execution
            wait_timeout = int(ttl_seconds * 0.8)
            try:
                deployment.wait_until_ready(timeout=wait_timeout, silent=True)
                deployment.refresh()
            except Exception:
                # Clean up orphaned deployment before propagating
                try:
                    client.delete_deployment(deployment_name)
                    logger.info(f"Cleaned up orphaned deployment: {deployment_name}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup orphaned deployment {deployment_name}: {cleanup_err}")
                raise

            return deployment

        try:
            loop = asyncio.get_running_loop()
            deployment = await loop.run_in_executor(self._executor, _sync_create_and_wait)

            logger.info(f"Deployment ready: {deployment.url}")
            return deployment

        except Exception as e:
            logger.error(f"Failed to create deployment {deployment_name}: {e}")
            # Async-level cleanup for edge cases (thread pool rejection, event loop cancellation)
            try:
                await self._delete_deployment(deployment_name)
            except Exception as cleanup_err:
                logger.warning(f"Async cleanup failed for {deployment_name}: {cleanup_err}")
            raise BackendError(f"Deployment creation failed: {e}")

    async def _delete_deployment(
        self,
        deployment_name: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """
        Delete Basilica deployment asynchronously with retry.

        Retries with exponential backoff to handle transient API failures.
        If all retries fail, the deployment's TTL will still clean it up.

        Args:
            deployment_name: Deployment name to delete
            max_retries: Maximum retry attempts (default: 3)
            base_delay: Base delay in seconds, doubled each retry (default: 1.0)
        """
        client = self._client

        def _sync_delete() -> None:
            """Synchronous deletion to run in thread pool."""
            client.delete_deployment(deployment_name)

        loop = asyncio.get_running_loop()
        for attempt in range(max_retries):
            try:
                await loop.run_in_executor(self._executor, _sync_delete)
                logger.info(f"Deployment deleted: {deployment_name}")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to delete deployment {deployment_name} after {max_retries} attempts: {e}")
                    return
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Delete attempt {attempt + 1}/{max_retries} failed for {deployment_name}: {e}, retrying in {delay:.0f}s")
                await asyncio.sleep(delay)

    def _create_http_executor(self, base_url: str) -> HTTPExecutor:
        """Create an HTTPExecutor from a deployment URL."""
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        executor = HTTPExecutor(
            container_ip=host,
            container_port=port,
            env_type=self._env_type,
        )
        executor.base_url = base_url
        return executor

    async def _detect_env_type(self, base_url: str) -> str:
        """
        Detect environment type by checking endpoints.

        function_based servers expose GET /methods (injected template).
        http_based servers expose GET /openapi.json but NOT /methods.

        Note: /openapi.json alone is ambiguous -- FastAPI serves it for all apps,
        including the function_based template. We must check /methods first and
        only fall back to /openapi.json if /methods is absent (not errored).

        Args:
            base_url: Deployment base URL

        Returns:
            EnvType.FUNCTION_BASED or EnvType.HTTP_BASED
        """
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(f"{base_url}/methods")
                if response.status_code == 200:
                    logger.debug("Detected function_based environment")
                    return EnvType.FUNCTION_BASED
                # /methods exists but returned non-200 -- server is up but
                # doesn't have the function_based template. Check http_based.
                if response.status_code == 404:
                    response = await client.get(f"{base_url}/openapi.json")
                    if response.status_code == 200:
                        logger.debug("Detected http_based environment")
                        return EnvType.HTTP_BASED
            except Exception as e:
                logger.warning(f"Env type detection failed: {e}")

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
        async with httpx.AsyncClient(timeout=10) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        logger.info(f"HTTP server ready after {attempt + 1} attempts ({(attempt + 1) * retry_delay:.0f}s)")
                        return
                except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                    if attempt == max_retries - 1:
                        raise BackendError(
                            f"HTTP server not ready after {max_retries} attempts: {e}"
                        )
                    logger.debug(f"HTTP not ready (attempt {attempt + 1}/{max_retries}): {e}")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise BackendError(f"Health check failed: {e}")
                    logger.debug(f"Health check failed (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)

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
        t_start = time.monotonic()

        try:
            # Create deployment
            deployment = await self._create_deployment(deployment_name, ttl_seconds)
            base_url = deployment.url
            t_deploy = time.monotonic()

            # Wait for HTTP server to be ready (handles container startup delay)
            await self._wait_for_http_ready(base_url)
            t_ready = time.monotonic()

            # Detect environment type once (guarded by lock for concurrent tasks)
            async with self._env_type_lock:
                if not self._env_type:
                    self._env_type = await self._detect_env_type(base_url)

            # Create HTTP executor from deployment URL
            http_executor = self._create_http_executor(base_url)

            # Execute method
            logger.debug(f"Calling method '{method_name}' on {base_url}")
            result = await http_executor.call_method(method_name, *args, **kwargs)
            t_done = time.monotonic()

            logger.info(
                f"[{deployment_name}] deploy={t_deploy - t_start:.1f}s "
                f"ready={t_ready - t_deploy:.1f}s "
                f"exec={t_done - t_ready:.1f}s "
                f"total={t_done - t_start:.1f}s"
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

            # Delete deployment (async, tracked for cleanup)
            if deployment:
                task = asyncio.create_task(self._delete_deployment(deployment.name))
                self._cleanup_tasks.add(task)
                task.add_done_callback(self._cleanup_tasks.discard)

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

            async with self._env_type_lock:
                if not self._env_type:
                    self._env_type = await self._detect_env_type(base_url)

            http_executor = self._create_http_executor(base_url)

            return await http_executor.list_methods()

        finally:
            if http_executor:
                await http_executor.close()
            if deployment:
                task = asyncio.create_task(self._delete_deployment(deployment_name))
                self._cleanup_tasks.add(task)
                task.add_done_callback(self._cleanup_tasks.discard)

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

        Awaits all in-flight pod deletion tasks and shuts down the thread pool.
        """
        if self._cleanup_tasks:
            logger.debug(f"Awaiting {len(self._cleanup_tasks)} pending pod deletions...")
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self._cleanup_tasks.clear()

        self._executor.shutdown(wait=False)
        logger.debug(f"BasilicaBackend cleanup complete: {self.name}")

    def is_ready(self) -> bool:
        """
        Check if backend is ready

        Pod backend is always ready (creates pods on-demand).

        Returns:
            True
        """
        return True
