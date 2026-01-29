"""
Abstract Base Class for Code Agents

Defines the interface that all code agent implementations must follow.
This allows swapping different agent backends (mini-swe-agent, SWE-agent, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AgentConfig:
    """Configuration for code agents"""
    # Model settings
    model: str
    api_base: str
    api_key: str
    temperature: float = 0.7

    # Execution limits
    max_iterations: int = 30
    cost_limit: float = 5.0
    timeout: int = 300

    # Docker settings
    docker_image: str = ""
    cwd: str = "/app"

    # Additional model kwargs
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution"""
    # The code diff produced by the agent
    diff: str

    # Agent's output text (may contain structured data)
    output_text: str

    # Execution metrics
    steps: int
    cost: float

    # Status
    exit_status: str = ""
    success: bool = True
    error: Optional[str] = None


class BaseCodeAgent(ABC):
    """
    Abstract base class for code agents.

    Implementations can use different backends:
    - mini-swe-agent: Simple bash-based agent
    - SWE-agent: Princeton's agent with ACI
    - Custom tools: Structured Read/Edit/RunTests tools
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    async def run(
        self,
        task: str,
        setup_script: str,
        template_vars: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the agent to complete a task in Docker environment.

        Args:
            task: The task description/prompt for the agent
            setup_script: Shell script to run before agent starts
                         (sets up environment, applies patches, etc.)
            template_vars: Variables to substitute in prompt templates
            workspace_dir: Host directory to mount as /workspace (optional)

        Returns:
            AgentResult with diff, output text, and execution metrics
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up any resources (Docker containers, etc.)"""
        pass
