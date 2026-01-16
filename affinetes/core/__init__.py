_LAZY_EXPORTS = {
    # environment management
    "EnvironmentWrapper",
    "EnvironmentRegistry",
    "get_registry",
    "LoadBalancer",
    "InstanceInfo",
    "InstancePool",
    # OpenEnv schemas
    "ResetRequest",
    "StepRequest",
    "OpenEnvResponse",
    # helpers
    "llm_chat",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'affinetes.core' has no attribute {name!r}")

    if name in {"EnvironmentWrapper"}:
        from .wrapper import EnvironmentWrapper
        return EnvironmentWrapper
    if name in {"EnvironmentRegistry", "get_registry"}:
        from .registry import EnvironmentRegistry, get_registry
        return EnvironmentRegistry if name == "EnvironmentRegistry" else get_registry
    if name in {"LoadBalancer", "InstanceInfo"}:
        from .load_balancer import LoadBalancer, InstanceInfo
        return LoadBalancer if name == "LoadBalancer" else InstanceInfo
    if name in {"InstancePool"}:
        from .instance_pool import InstancePool
        return InstancePool

    if name in {"ResetRequest", "StepRequest", "OpenEnvResponse"}:
        from .openenv import ResetRequest, StepRequest, OpenEnvResponse
        return {"ResetRequest": ResetRequest, "StepRequest": StepRequest, "OpenEnvResponse": OpenEnvResponse}[name]

    if name == "llm_chat":
        from .llm_chat import llm_chat
        return llm_chat

    raise AttributeError(f"module 'affinetes.core' has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)