"""URL backend - Remote HTTP-based execution for user-deployed environments"""

import httpx
import time
import asyncio
from typing import Optional, Any

from .base import AbstractBackend
from ..infrastructure import HTTPExecutor, EnvType
from ..utils.exceptions import BackendError
from ..utils.logger import logger


class URLBackend(AbstractBackend):
    """
    URL backend for connecting to user-deployed environment services
    
    This backend allows users to deploy their own environment services and
    connect to them via HTTP URL. Unlike LocalBackend which manages Docker
    containers, URLBackend connects to already-running services that users
    have deployed themselves.
    
    Supports two environment types:
    1. function_based: Uses /methods and /call endpoints
    2. http_based: Uses direct endpoints like /evaluate, /create, etc.
    
    The backend automatically detects the environment type by checking available endpoints.
    
    Usage:
        >>> env = load_env(
        ...     mode="url",
        ...     base_url="http://your-service.com:8080"
        ... )
        >>> result = await env.evaluate(task_type="sat", num_samples=1)
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 600,
        verify_ssl: bool = True,
        env_type_override: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize URL backend
        
        Args:
            base_url: Environment service base URL (e.g., "http://your-service.com:8080")
            timeout: Request timeout in seconds (default: 600)
            verify_ssl: Verify SSL certificates for HTTPS connections (default: True)
            env_type_override: Force environment type (EnvType.FUNCTION_BASED or EnvType.HTTP_BASED)
            **kwargs: Additional configuration
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.config = kwargs
        self._env_type = env_type_override
        self._http_executor = None
        
        # Generate unique name for this backend
        # Extract hostname from URL for a meaningful name
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname or "unknown"
            port = f":{parsed.port}" if parsed.port else ""
            self.name = f"url-{hostname}{port}-{int(time.time())}"
        except Exception:
            self.name = f"url-{int(time.time())}"
        
        logger.info(f"URLBackend initialized: {self.base_url}")
        
        # Detect environment type and setup executor
        self._setup_executor()
    
    def _setup_executor(self):
        """Setup HTTP executor with environment type detection"""
        try:
            # Get event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Detect environment type if not overridden
            if not self._env_type:
                self._env_type = loop.run_until_complete(self._detect_env_type())
                logger.info(f"Detected environment type: {self._env_type}")
            else:
                logger.info(f"Using override environment type: {self._env_type}")
            
            # Parse URL to get host and port
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            
            # Create HTTP executor
            self._http_executor = HTTPExecutor(
                container_ip=host,
                container_port=port,
                env_type=self._env_type
            )
            
            # Override base_url to use our full URL (including scheme)
            self._http_executor.base_url = self.base_url
            
        except Exception as e:
            raise BackendError(f"Failed to setup URL backend: {e}")
    
    async def _detect_env_type(self) -> str:
        """Detect environment type by checking available endpoints
        
        Returns:
            EnvType.FUNCTION_BASED or EnvType.HTTP_BASED
        """
        # Create temporary client for detection
        async with httpx.AsyncClient(
            timeout=10,
            verify=self.verify_ssl
        ) as client:
            try:
                # Check for function_based endpoint
                response = await client.get(f"{self.base_url}/methods")
                if response.status_code == 200:
                    logger.debug("Found /methods endpoint - function_based environment")
                    return EnvType.FUNCTION_BASED
            except Exception:
                pass
            
            try:
                # Check for http_based endpoint (OpenAPI schema)
                response = await client.get(f"{self.base_url}/openapi.json")
                if response.status_code == 200:
                    logger.debug("Found /openapi.json endpoint - http_based environment")
                    return EnvType.HTTP_BASED
            except Exception:
                pass
            
            # Default to function_based if cannot detect
            logger.warning("Could not detect environment type, defaulting to function_based")
            return EnvType.FUNCTION_BASED
    
    async def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call method on remote environment service
        
        Uses HTTPExecutor which handles both environment types automatically.
        
        Args:
            method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method result
        """
        try:
            return await self._http_executor.call_method(
                method_name,
                *args,
                **kwargs
            )
        except Exception as e:
            raise BackendError(f"Method call failed: {e}")
    
    async def list_methods(self) -> list:
        """
        List available methods from remote environment
        
        Uses HTTPExecutor which handles both environment types automatically.
        
        Returns:
            List of method information
        """
        try:
            return await self._http_executor.list_methods()
        except Exception as e:
            raise BackendError(f"Failed to list methods: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if remote environment is healthy
        
        Returns:
            True if healthy
        """
        try:
            return await self._http_executor.health_check()
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Close HTTP client (no service cleanup needed)"""
        logger.info(f"Closing URL backend: {self.name}")
        if self._http_executor:
            await self._http_executor.close()
    
    def is_ready(self) -> bool:
        """
        Check if backend is ready
        
        Returns:
            True if executor is initialized
        """
        return self._http_executor is not None