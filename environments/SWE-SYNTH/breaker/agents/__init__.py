"""
Code Agent Implementations

Provides pluggable agent backends for bug injection.
"""

from .base import BaseCodeAgent, AgentConfig, AgentResult
from .miniswe import MiniSweAgent
from .ridge import RidgeCodeAgent

__all__ = [
    "BaseCodeAgent",
    "AgentConfig",
    "AgentResult",
    "MiniSweAgent",
    "RidgeCodeAgent",
]
