"""Fixer Agent Factory"""

from typing import Literal
from .base import BaseFixerAgent, FixerConfig, FixerResult

AgentType = Literal["miniswe", "ridge", "swe-agent", "codex"]


def create_fixer_agent(agent_type: AgentType, config: FixerConfig) -> BaseFixerAgent:
    """Create a fixer agent of the specified type"""
    if agent_type == "miniswe":
        from .miniswe import MiniSWEFixerAgent
        return MiniSWEFixerAgent(config)
    elif agent_type == "ridge":
        from .ridge import RidgeFixerAgent
        return RidgeFixerAgent(config)
    elif agent_type == "swe-agent":
        from .stub import SWEAgentFixerAgent
        return SWEAgentFixerAgent(config)
    elif agent_type == "codex":
        from .stub import CodexFixerAgent
        return CodexFixerAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


__all__ = ["create_fixer_agent", "BaseFixerAgent", "FixerConfig", "FixerResult", "AgentType"]
