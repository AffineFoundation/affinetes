"""Instance pool for managing multiple environment instances"""

import asyncio
import time
from typing import List, Optional, Any

from .load_balancer import LoadBalancer, InstanceInfo
from ..backends.base import AbstractBackend
from ..utils.logger import logger
from ..utils.exceptions import BackendError


class InstancePool:
    """Manages multiple environment instances with load balancing"""
    
    def __init__(
        self,
        instances: List[InstanceInfo],
        load_balance_strategy: str = "random",
        pool_name: Optional[str] = None
    ):
        """
        Initialize instance pool
        
        Args:
            instances: List of InstanceInfo objects
            load_balance_strategy: Load balancing strategy ("random" or "round_robin")
            pool_name: Optional custom pool name (auto-generated if not provided)
        """
        if not instances:
            raise BackendError("Cannot create InstancePool with empty instances list")
        
        self._instances = instances
        self._load_balancer = LoadBalancer(strategy=load_balance_strategy)
        
        # Pool metadata - generate unique name if not provided
        if pool_name:
            self.name = pool_name
        else:
            # Generate unique name using timestamp to avoid collisions
            import time
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            self.name = f"pool-{len(instances)}-{timestamp}"
        
        logger.info(
            f"InstancePool '{self.name}' created with {len(instances)} instances, "
            f"strategy: {load_balance_strategy}"
        )
        
        # Log instance details
        for i, inst in enumerate(instances):
            logger.debug(f"  Instance {i}: {inst}")
    
    async def call_method(
        self,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Call method on a selected instance (load-balanced)
        
        Args:
            method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method result from selected instance
        """
        # Select instance using load balancer
        instance = self._load_balancer.select_instance(self._instances)
        
        logger.debug(
            f"Routing '{method_name}' to instance {instance.host}:{instance.port}"
        )
        
        # Call method on selected backend
        result = await instance.backend.call_method(
            method_name,
            *args,
            **kwargs
        )
        
        # Update statistics
        instance.request_count += 1
        
        return result
    
    async def list_methods(self) -> list:
        """
        List methods from any healthy instance
        
        Returns:
            List of available methods
        """
        # Use first healthy instance to get method list
        instance = self._load_balancer.select_instance(self._instances)
        
        try:
            return await instance.backend.list_methods()
        except Exception as e:
            raise BackendError(f"Failed to list methods: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup all instances in the pool"""
        logger.info(f"Cleaning up instance pool ({len(self._instances)} instances)")
        
        # Cleanup all instances concurrently
        cleanup_tasks = [
            inst.backend.cleanup()
            for inst in self._instances
        ]
        
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Log any cleanup failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to cleanup instance {self._instances[i]}: {result}"
                )
        
        logger.info("Instance pool cleanup completed")
    
    def is_ready(self) -> bool:
        return True
    
    def get_total_count(self) -> int:
        """Get total number of instances"""
        return len(self._instances)
    
    def get_instances(self) -> List[InstanceInfo]:
        """Get list of all instances"""
        return self._instances.copy()
    
    def get_stats(self) -> dict:
        """
        Get pool statistics
        
        Returns:
            Dictionary with pool statistics
        """
        total_requests = sum(inst.request_count for inst in self._instances)
        
        return {
            "total_instances": len(self._instances),
            "total_requests": total_requests,
            "instances": [
                {
                    "host": inst.host,
                    "port": inst.port,
                    "requests": inst.request_count
                }
                for inst in self._instances
            ]
        }
    
    def __repr__(self) -> str:
        total = self.get_total_count()
        return f"<InstancePool {total} healthy instances>"