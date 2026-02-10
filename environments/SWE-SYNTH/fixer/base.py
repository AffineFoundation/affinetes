"""Abstract Base Class for Fixer Agents"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from . import config


@dataclass
class FixerConfig:
    """Configuration for fixer agents"""
    model: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    max_iterations: int = 100
    cost_limit: float = 3.0
    timeout: int = 300
    seed: Optional[int] = None
    cwd: str = "/app"

    # Ridge-specific
    ridge_agent_path: Optional[str] = None

    # Reserved for future agents
    swe_agent_config: Optional[Dict[str, Any]] = None
    external_agent_endpoint: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def get_ridge_agent_path(self) -> str:
        """Get Ridge agent path (param > env var > default)"""
        return self.ridge_agent_path or config.get_ridge_agent_path()


@dataclass
class FixerResult:
    """Result from fixer agent execution"""
    patch: str
    model_calls: int = 0
    model_cost: float = 0.0
    total_tokens: int = 0
    conversation: List[Any] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class BaseFixerAgent(ABC):
    """Abstract base class for fixer agents"""

    def __init__(self, config: FixerConfig):
        self.config = config

    @abstractmethod
    async def fix(
        self,
        problem_statement: str,
        docker_image: str,
        repo_path: Optional[str] = None,
        gold_patch: Optional[str] = None,
        bug_patch: Optional[str] = None,
        base_commit: Optional[str] = None,
    ) -> FixerResult:
        """
        Run the fixer agent to repair a bug.

        The agent sees code in state: base_commit + gold_patch + bug_patch
        and must produce a fix_patch to restore correct behavior.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (Docker containers, temp files, etc.)"""
        pass
