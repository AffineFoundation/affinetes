"""
Code Agent Implementations

Provides pluggable agent backends for bug injection.
"""

from .base import BaseCodeAgent, AgentConfig
from .miniswe import MiniSweAgent

__all__ = [
    "BaseCodeAgent",
    "AgentConfig",
    "MiniSweAgent",
]
