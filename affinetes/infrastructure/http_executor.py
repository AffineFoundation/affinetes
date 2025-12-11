"""HTTP-based remote execution with async support"""

import asyncio
import httpx
from typing import Any, Optional

from ..utils.exceptions import ExecutionError
from ..utils.logger import logger
from .env_detector import EnvType


class HTTPExecutor:
    """Unified async HTTP executor for both environment types"""
    
    def __init__(
        self,
        container_ip: str,
        container_port: int = 8000,
        env_type: str = None,
    ):
        """
        Args:
            container_ip: Container internal IP address (e.g., 172.17.0.2)
            container_port: Container internal port (default: 8000)
            env_type: EnvType.FUNCTION_BASED or EnvType.HTTP_BASED
            timeout: Request timeout in seconds (read timeout, connect timeout is 10s)
        """
        self.base_url = f"http://{container_ip}:{container_port}"
        self.env_type = env_type
        # Separate connect and read timeouts to avoid long hangs on connection
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,  # 30s for connection establishment (SSH tunnel needs more time)
                read=None,     # No read timeout - allow long-running tasks
                write=30.0,    # 30s for sending request
                pool=30.0      # 30s for acquiring connection from pool
            ),
            limits=httpx.Limits(
                max_connections=2000,
                max_keepalive_connections=200  # Increased from 20 to 200 to support high concurrency
            )
        )
        logger.debug(f"HTTPExecutor initialized: {self.base_url} (type: {env_type}, connect_timeout=30s)")
    
    async def call_method(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Call method via HTTP (async)
        
        Routes:
        - function_based: POST /call with {"method": "...", "args": [...], "kwargs": {...}}
        - http_based: POST /{method_name} with direct kwargs
        
        Args:
            method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Note:
            Container restart detection and connection management is handled by
            LocalBackend.call_method() which proactively checks container status
            before delegating to this executor.
        """
        try:
            if self.env_type == EnvType.FUNCTION_BASED:
                # Generic endpoint for function-based
                logger.debug(f"Calling function-based method: {method_name}")
                response = await self.client.post(
                    f"{self.base_url}/call",
                    json={
                        "method": method_name,
                        "args": list(args),
                        "kwargs": kwargs
                    }
                )
            else:  # HTTP_BASED
                # Direct endpoint for http-based
                logger.debug(f"Calling http-based endpoint: /{method_name}")
                response = await self.client.post(
                    f"{self.base_url}/{method_name}",
                    json=kwargs
                )
            
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if isinstance(data, dict):
                if "status" in data:
                    # MethodResponse format (function_based)
                    if data["status"] != "success":
                        raise ExecutionError(f"Remote execution failed: {data}")
                    return data.get("result")
                else:
                    # Direct result (http_based)
                    return data
            else:
                return data
                
        except httpx.HTTPStatusError as e:
            raise ExecutionError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise ExecutionError(
                f"Failed to call method '{method_name}': {type(e).__name__}: {e}"
            ) from e
    
    async def list_methods(self) -> list:
        """List available methods with detailed information (async)"""
        try:
            if self.env_type == EnvType.FUNCTION_BASED:
                response = await self.client.get(f"{self.base_url}/methods")
                data = response.json()
                return data.get("methods", [])
            else:
                # For http_based, parse OpenAPI schema
                response = await self.client.get(f"{self.base_url}/openapi.json")
                schema = response.json()
                return self._parse_openapi_schema(schema)
        except Exception as e:
            logger.warning(f"Failed to list methods: {e}")
            return []
    
    def _parse_openapi_schema(self, schema: dict) -> list:
        """Parse OpenAPI schema to extract endpoint information"""
        endpoints = []
        paths = schema.get("paths", {})
        components = schema.get("components", {}).get("schemas", {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                    continue
                
                endpoint = {
                    "path": path,
                    "method": method.upper(),
                    "summary": details.get("summary", ""),
                    "description": details.get("description", ""),
                    "parameters": []
                }
                
                # Parse query parameters
                for param in details.get("parameters", []):
                    param_info = {
                        "name": param.get("name"),
                        "in": param.get("in"),
                        "required": param.get("required", False),
                        "type": param.get("schema", {}).get("type", "unknown")
                    }
                    endpoint["parameters"].append(param_info)
                
                # Parse request body
                request_body = details.get("requestBody")
                if request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    body_schema = json_content.get("schema", {})
                    
                    # Resolve schema reference
                    if "$ref" in body_schema:
                        ref_name = body_schema["$ref"].split("/")[-1]
                        body_schema = components.get(ref_name, {})
                    elif "allOf" in body_schema:
                        # Handle allOf references
                        for item in body_schema["allOf"]:
                            if "$ref" in item:
                                ref_name = item["$ref"].split("/")[-1]
                                body_schema = components.get(ref_name, {})
                                break
                    
                    # Extract properties
                    properties = body_schema.get("properties", {})
                    required_fields = body_schema.get("required", [])
                    
                    for prop_name, prop_schema in properties.items():
                        param_info = {
                            "name": prop_name,
                            "in": "body",
                            "required": prop_name in required_fields,
                            "type": prop_schema.get("type", "unknown"),
                            "default": prop_schema.get("default")
                        }
                        endpoint["parameters"].append(param_info)
                
                endpoints.append(endpoint)
        
        return endpoints
    
    async def health_check(self) -> bool:
        """Check server health (async)"""
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """Close HTTP client (async)"""
        try:
            # Check if event loop is still running
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # Event loop already closed, close client synchronously without await
                # This happens during garbage collection after asyncio.run() completes
                return
            await self.client.aclose()
        except RuntimeError as e:
            # Event loop might be None or closed
            if "Event loop is closed" not in str(e):
                raise
            # Silently ignore if event loop is closed