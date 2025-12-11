"""Local backend - Docker + async HTTP execution"""

import time
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

import nest_asyncio
nest_asyncio.apply()

from .base import AbstractBackend
from ..infrastructure import DockerManager, HTTPExecutor, EnvType
from ..infrastructure.ssh_tunnel import SSHTunnelManager
from ..utils.exceptions import BackendError, SetupError
from ..utils.logger import logger
from ..utils.config import Config


class LocalBackend(AbstractBackend):
    """
    Local execution backend using Docker containers and HTTP
    
    Lifecycle:
    1. __init__: Start Docker container with HTTP server
    2. call_method(): Execute methods via HTTP API
    3. cleanup(): Stop container
    """
    
    def __init__(
        self,
        image: Optional[str] = None,
        host: Optional[str] = None,
        container_name: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        env_type_override: Optional[str] = None,
        force_recreate: bool = False,
        pull: bool = False,
        mem_limit: Optional[str] = None,
        auto_cleanup: bool = True,
        connect_only: bool = False,
        **docker_kwargs
    ):
        """
        Initialize backend - starts Docker container or connects to existing one
        
        Args:
            image: Docker image name (required unless connect_only=True)
            host: Docker daemon address (None/"localhost" for local, "ssh://user@host" for remote)
            container_name: Container name (required if connect_only=True)
            env_vars: Environment variables to pass to container
            env_type_override: Override environment type detection
            force_recreate: If True, remove existing container and create new one
            pull: If True, pull image before starting container
            mem_limit: Memory limit (e.g., "512m", "1g", "2g")
            auto_cleanup: If True, automatically stop and remove container on cleanup (default: True)
                         If False, container will continue running after cleanup
            connect_only: If True, only connect to existing container without creating new one
            **docker_kwargs: Additional Docker container options
        """
        self.image = image
        self.host = host
        self._connect_only = connect_only
        
        # Validate parameters
        if connect_only:
            if not container_name:
                raise ValueError("container_name is required when connect_only=True")
            self.name = container_name
        else:
            if not image:
                raise ValueError("image is required when connect_only=False")
            # Sanitize image name for container naming (remove / and :)
            safe_image = image.split('/')[-1].replace(':', '-')
            self.name = container_name or f"{safe_image}"
        
        self._container = None
        self._docker_manager = None
        self._http_executor = None
        self._is_setup = False
        self._env_type = None
        self._env_type_override = env_type_override
        self._force_recreate = force_recreate
        self._pull = pull
        self._mem_limit = mem_limit
        self._auto_cleanup = auto_cleanup
        
        # SSH tunnel for remote access
        self._is_remote = host and host.startswith("ssh://")
        self._ssh_tunnel_manager = None
        
        # Track container restart: store StartedAt timestamp for restart detection
        self._container_started_at = None
        
        # Cache runtime environment to avoid repeated detection
        self._runtime_env: Optional[str] = None
        
        # Connect to existing container or start new one
        if connect_only:
            self._connect_to_container()
        else:
            self._start_container(env_vars=env_vars, **docker_kwargs)
    
    def _connect_to_container(self) -> None:
        """Connect to existing Docker container without creating new one"""
        try:
            logger.debug(f"Connecting to existing container '{self.name}' on host '{self.host or 'localhost'}'")
            
            # Initialize Docker manager
            self._docker_manager = DockerManager(host=self.host)
            
            # Get existing container
            try:
                self._container = self._docker_manager.client.containers.get(self.name)
            except Exception as e:
                raise BackendError(f"Container '{self.name}' not found: {e}")
            
            # Check container status
            self._container.reload()
            if self._container.status != 'running':
                raise BackendError(f"Container '{self.name}' is not running (status: {self._container.status})")
            
            # Get environment type from container labels
            if self._env_type_override:
                self._env_type = self._env_type_override
                logger.info(f"Environment type (manual): {self._env_type}")
            else:
                labels = self._container.labels or {}
                self._env_type = labels.get("affinetes.env.type", EnvType.FUNCTION_BASED)
                logger.info(f"Environment type (from container): {self._env_type}")
            
            # Initialize connection address
            local_host, local_port = self._initialize_connection_address()
            
            # Create HTTP executor with configurable timeout
            self._http_executor = HTTPExecutor(
                container_ip=local_host,
                container_port=local_port,
                env_type=self._env_type,
            )
            
            # Record container start time for restart detection
            self._container.reload()
            self._container_started_at = self._container.attrs["State"]["StartedAt"]
            logger.debug(f"Container '{self.name}' started at: {self._container_started_at}")
            
            # Verify HTTP server is ready
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if not loop.run_until_complete(self._wait_for_http_ready(timeout=10)):
                raise BackendError(
                    f"HTTP server not responding. "
                    f"Container '{self.name}' may not be ready or misconfigured."
                )
            
            self._is_setup = True
            logger.debug(f"Successfully connected to container '{self.name}'")
            
        except Exception as e:
            # Cleanup on failure
            if self._ssh_tunnel_manager:
                try:
                    self._ssh_tunnel_manager.cleanup()
                except:
                    pass
            raise BackendError(f"Failed to connect to container: {e}")
    
    def _needs_restart_detection(self) -> bool:
        """Determine if restart detection is needed for current deployment mode
        
        Returns:
            True if restart detection is needed, False otherwise
        """
        if self._is_remote:
            # Remote: SSH tunnel bound to IP, must detect restarts
            return True
        
        # DOOD: Docker DNS resolves container name, no detection needed
        # DIND/Host: IP-based access, detection needed (though IP changes are rare)
        return self._runtime_env != "dood"
    
    def _check_container_restart(self) -> bool:
        """Check if container has restarted by comparing StartedAt timestamp
        
        Only performs check if restart detection is needed for current deployment mode.
        
        Returns:
            True if container restarted, False otherwise
        """
        # Skip check if not needed (performance optimization)
        if not self._needs_restart_detection():
            return False
        
        if not self._container or not self._container_started_at:
            return False
        
        try:
            self._container.reload()
            current_started_at = self._container.attrs["State"]["StartedAt"]
            
            if current_started_at != self._container_started_at:
                logger.warning(
                    f"Container '{self.name}' restart detected: "
                    f"{self._container_started_at} → {current_started_at}"
                )
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to check container restart: {e}")
            return False
    
    def _initialize_connection_address(self) -> Tuple[str, int]:
        """Initialize connection address based on deployment mode
        
        Returns:
            (local_host, local_port) tuple
        """
        if self._is_remote:
            # Remote: create SSH tunnel using container name
            logger.debug("Remote connection detected, creating SSH tunnel")
            self._ssh_tunnel_manager = SSHTunnelManager(self.host)
            local_host, local_port = self._ssh_tunnel_manager.create_tunnel(
                remote_host=self.name,
                remote_port=8000
            )
            logger.info(f"Accessing via SSH tunnel: {local_host}:{local_port}")
            return local_host, local_port
        
        # Local deployment: detect and cache runtime environment
        if not self._runtime_env:
            self._runtime_env = self._detect_runtime_environment()
        
        if self._runtime_env == "dood":
            # DOOD: Use container name as hostname (Docker DNS)
            local_host = self.name
            local_port = 8000
            logger.info(f"DOOD mode, accessing via container name: {local_host}:{local_port}")
        else:
            # Host or DIND: Use container IP
            container_ip = self._docker_manager.get_container_ip(self._container)
            local_host = container_ip
            local_port = 8000
            logger.info(f"{self._runtime_env.upper()} mode, accessing via IP: {local_host}:{local_port}")
        
        return local_host, local_port
    
    def _handle_container_restart(self) -> Tuple[str, int]:
        """Handle container restart: update timestamp and recreate connection if needed
        
        Returns:
            (local_host, local_port) tuple
        """
        logger.info(f"Handling container restart for '{self.name}'")
        
        # Update stored timestamp
        self._container.reload()
        self._container_started_at = self._container.attrs["State"]["StartedAt"]
        logger.info(f"Updated container start time: {self._container_started_at}")
        
        # Remote: recreate SSH tunnel (IP may have changed)
        if self._is_remote:
            if self._ssh_tunnel_manager:
                try:
                    self._ssh_tunnel_manager.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up old tunnel: {e}")
            
            self._ssh_tunnel_manager = SSHTunnelManager(self.host)
            local_host, local_port = self._ssh_tunnel_manager.create_tunnel(
                remote_host=self.name,
                remote_port=8000
            )
            logger.info(f"SSH tunnel recreated: {local_host}:{local_port}")
            return local_host, local_port
        
        # Local DIND/Host: resolve new container IP
        # (DOOD mode won't reach here due to _needs_restart_detection())
        new_ip = self._docker_manager.get_container_ip(self._container)
        local_host = new_ip
        local_port = 8000
        logger.info(f"{self._runtime_env.upper()} mode: updated to new IP {local_host}:{local_port}")
        
        return local_host, local_port
    
    def _start_container(self, env_vars: Optional[Dict[str, str]] = None, **docker_kwargs) -> None:
        """Start Docker container with HTTP server
        
        Args:
            env_vars: Environment variables to pass to container
            **docker_kwargs: Additional Docker options
        """
        try:
            logger.debug(f"Starting container for image '{self.image}' on host '{self.host or 'localhost'}'")
            
            # Initialize Docker manager with host support (must be done before env_type detection)
            self._docker_manager = DockerManager(host=self.host)
            
            # Pull image if requested (must be done BEFORE env_type detection)
            if self._pull:
                self._docker_manager.pull_image(self.image)
            
            # Get environment type
            if self._env_type_override:
                self._env_type = self._env_type_override
                logger.info(f"Environment type (manual): {self._env_type}")
            else:
                self._env_type = self._get_env_type()
                logger.info(f"Environment type (detected): {self._env_type}")
            
            # Merge environment variables
            if env_vars:
                if "environment" in docker_kwargs:
                    docker_kwargs["environment"].update(env_vars)
                else:
                    docker_kwargs["environment"] = env_vars
            
            # Prepare container configuration
            container_config = {
                "image": self.image,
                "name": self.name,
                "detach": True,
                "restart_policy": {"Name": "always"},
                "force_recreate": self._force_recreate,
                "mem_limit": self._mem_limit,
                **docker_kwargs
            }
            
            # Network handling: Only needed for local DOOD deployment
            # Remote deployment uses SSH tunnel, so no network configuration needed
            if not self._is_remote:
                # Detect runtime environment
                runtime_env = self._detect_runtime_environment()
                
                if runtime_env == "dood":
                    # DOOD: Need to join the same network as affinetes container
                    network_name = self._ensure_docker_network()
                    container_config["network"] = network_name
                    logger.info(f"DOOD mode detected, connecting environment container to network: {network_name}")
                elif runtime_env == "dind":
                    # DIND: Use default bridge network, no special configuration needed
                    logger.info("DIND mode detected, using default bridge network")
                else:
                    # Host: No network configuration needed
                    logger.info("Host mode detected, using default network")
            
            # Start container
            self._container = self._docker_manager.start_container(**container_config)
            
            # Initialize connection address
            local_host, local_port = self._initialize_connection_address()
            
            # Create HTTP executor with accessible address
            # Use longer timeout for long-running tasks (default 1800s = 30 minutes)
            self._http_executor = HTTPExecutor(
                container_ip=local_host,
                container_port=local_port,
                env_type=self._env_type,
            )
            
            # Wait for HTTP server to be ready
            # Increased timeout to handle concurrent deployments and slow container initialization
            timeout = 180 if self._env_type == EnvType.HTTP_BASED else 120
            access_info = f"{local_host}:{local_port}"
            logger.debug(f"Waiting for HTTP server at {access_info} (timeout={timeout}s)")
            
            # Run async health check in sync context
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if not loop.run_until_complete(self._wait_for_http_ready(timeout=timeout)):
                raise BackendError(
                    f"HTTP server did not start within {timeout}s. "
                    "Check container logs for errors."
                )
            
            # Record container start time for restart detection
            self._container.reload()
            self._container_started_at = self._container.attrs["State"]["StartedAt"]
            logger.debug(f"Container '{self.name}' started at: {self._container_started_at}")
            
            # Mark as setup - HTTP backend is ready after container starts
            self._is_setup = True
            logger.debug("Container started and HTTP server ready")
            
        except Exception as e:
            # Cleanup on failure
            if self._ssh_tunnel_manager:
                try:
                    self._ssh_tunnel_manager.cleanup()
                except:
                    pass
            if self._container:
                try:
                    self._docker_manager.stop_container(self._container)
                except:
                    pass
            raise BackendError(f"Failed to start container: {e}")
    
    def _is_running_in_docker(self) -> bool:
        """Check if running inside a Docker container"""
        try:
            p1 = open("/proc/1/comm").read().strip().lower()
            if p1 not in ("systemd", "init"):
                return True
        except:
            pass
        return False
    
    def _detect_runtime_environment(self) -> str:
        """
        Detect runtime environment type
        
        Returns:
            "host" - Running on host machine
            "dood" - Docker-out-of-Docker (mounted docker.sock from host)
            "dind" - Docker-in-Docker (independent Docker daemon)
        """
        # Check if running in Docker container
        if not self._is_running_in_docker():
            logger.debug("Runtime environment: host")
            return "host"
        
        # Running in container - check if dockerd process exists
        # If dockerd exists → DIND, otherwise → DOOD
        try:
            import os
            for pid in os.listdir('/proc'):
                if not pid.isdigit():
                    continue
                try:
                    with open(f'/proc/{pid}/comm', 'r') as f:
                        comm = f.read().strip()
                        if comm == 'dockerd':
                            logger.debug("Runtime environment: dind (dockerd process found)")
                            return "dind"
                except:
                    continue
        except Exception as e:
            logger.debug(f"Failed to check for dockerd process: {e}")
        
        # No dockerd found → DOOD
        logger.debug("Runtime environment: dood (no dockerd process)")
        return "dood"

    def _ensure_docker_network(self) -> str:
        """Get the network name that affinetes container is connected to
        
        Strategy: Parse container ID from mountinfo using regex pattern matching
        This works for all Docker environments and is resilient to path changes.
        """
        import re
        
        try:
            # Regex pattern to match 64-character hex container IDs in mountinfo
            # Matches IDs in paths like:
            # - /var/lib/docker/containers/<id>/...
            # - /docker/containers/<id>/...
            # - Any other path containing containers/<id>/
            container_id_pattern = re.compile(r'/containers/([0-9a-f]{64})/')
            
            container_id = None
            with open("/proc/self/mountinfo", "r") as f:
                for line in f:
                    match = container_id_pattern.search(line)
                    if match:
                        container_id = match.group(1)
                        logger.debug(f"Found container ID from mountinfo: {container_id[:12]}...")
                        break
            
            if not container_id:
                logger.warning("Could not extract container ID from mountinfo")
                return "bridge"
            
            # Query Docker API with full container ID
            try:
                current_container = self._docker_manager.client.containers.get(container_id)
                logger.debug(f"Found container: {current_container.name}")
                
                # Extract network information
                current_container.reload()
                networks = current_container.attrs["NetworkSettings"]["Networks"]
                network_names = list(networks.keys())
                logger.info(f"Container '{current_container.name}' networks: {network_names}")
            except Exception as container_err:
                logger.debug(f"Container {container_id[:12]}... not found or inaccessible: {container_err}")
                logger.info("Using default bridge network")
                return "bridge"
            
            # Return first non-default network, or first network if all are default
            for net_name in network_names:
                if net_name not in ["host", "none"]:
                    logger.info(f"Environment containers will use network: {net_name}")
                    return net_name
            
            if network_names:
                logger.info(f"Using network: {network_names[0]}")
                return network_names[0]
            
        except FileNotFoundError:
            logger.warning("Not running in Docker (mountinfo not found)")
        except Exception as e:
            logger.warning(f"Failed to detect network: {e}")
        
        # Fallback when not running in Docker
        logger.warning("Using default bridge network")
        return "bridge"
    
    def _get_env_type(self) -> str:
        """Get environment type from image labels"""
        try:
            img = self._docker_manager.client.images.get(self.image)
            labels = img.labels or {}
            env_type = labels.get("affinetes.env.type", EnvType.FUNCTION_BASED)
            logger.debug(f"Detected env_type from image labels: {env_type}")
            return env_type
        except Exception as e:
            logger.warning(f"Failed to get env type from image: {e}, defaulting to function_based")
            return EnvType.FUNCTION_BASED
    
    async def _wait_for_http_ready(self, timeout: int = 60) -> bool:
        """Wait for HTTP server to be ready (async)"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                if await self._http_executor.health_check():
                    return True
            except Exception as e:
                logger.debug(f"Health check failed: {e}")
            await asyncio.sleep(1)
        return False
    
    async def call_method(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Call a method from env.py via HTTP (async)
        
        Proactively checks for container restart before each call and recreates
        connection if needed. This ensures requests always target the correct container.
        
        Args:
            method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method result
        """
        try:
            # Check if container restarted (proactive detection)
            if self._check_container_restart():
                logger.info(f"Container restart detected before calling '{method_name}', updating connection")
                new_host, new_port = self._handle_container_restart()
                # Update HTTP executor base URL
                self._http_executor.base_url = f"http://{new_host}:{new_port}"
                logger.info(f"Updated HTTPExecutor base URL to: {self._http_executor.base_url}")
                # Wait briefly for container to stabilize
                await asyncio.sleep(2)
            
            # Execute method call (without retry logic - connection is now correct)
            return await self._http_executor.call_method(
                method_name,
                *args,
                **kwargs
            )
        except Exception as e:
            # Preserve full exception chain for debugging
            raise BackendError(f"Method call failed: {e}") from e
    
    async def list_methods(self) -> list:
        """
        List all available methods from env.py (async)
        
        Returns:
            List of method names
        """
        try:
            return await self._http_executor.list_methods()
        except Exception as e:
            raise BackendError(f"Failed to list methods: {e}")
    
    async def cleanup(self) -> None:
        """Stop container and close HTTP client (async)"""
        logger.debug(f"Cleaning up backend for container {self.name}")
        
        # Close HTTP client
        if self._http_executor:
            try:
                await self._http_executor.close()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")
            finally:
                self._http_executor = None
        
        # Close SSH tunnel if exists
        if self._ssh_tunnel_manager:
            try:
                self._ssh_tunnel_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error closing SSH tunnel: {e}")
            finally:
                self._ssh_tunnel_manager = None
        
        # Stop container
        if self._container and self._docker_manager:
            try:
                self._docker_manager.stop_container(self._container)
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            finally:
                self._container = None
        
        self._is_setup = False
        logger.debug("Backend cleanup completed")
    
    def is_ready(self) -> bool:
        """
        Check if backend is ready for method calls
        
        Returns:
            True if setup completed
        """
        return self._is_setup
    
    async def health_check(self) -> bool:
        """
        Check if backend is healthy and responsive
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._is_setup or not self._http_executor:
            return False
        
        try:
            # Check if container is still running
            if self._container:
                self._container.reload()
                if self._container.status != 'running':
                    logger.warning(f"Container {self.name} is not running (status: {self._container.status})")
                    return False
            
            # Check HTTP endpoint
            return await self._http_executor.health_check()
        
        except Exception as e:
            logger.debug(f"Health check failed for {self.name}: {e}")
            return False
    
    def get_container_logs(self, tail: int = 100) -> str:
        """
        Get container logs for debugging
        
        Args:
            tail: Number of lines to return
            
        Returns:
            Log output
        """
        if not self._container:
            return ""
        
        try:
            logs = self._container.logs(tail=tail, timestamps=True)
            return logs.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to get container logs: {e}")
            return ""
