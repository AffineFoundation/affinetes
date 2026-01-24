"""
Mini-SWE-Agent Implementation

Uses the mini-swe-agent library for bash-based code agent execution.
"""

import os
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseCodeAgent, AgentConfig, AgentResult


class MiniSweAgent(BaseCodeAgent):
    """
    Code agent implementation using mini-swe-agent.

    The agent interacts with code through bash commands in a Docker container.
    """

    def __init__(self, config: AgentConfig, prompt_config: Dict[str, str] = None):
        """
        Initialize MiniSweAgent.

        Args:
            config: Agent configuration
            prompt_config: Prompt templates with keys:
                - system_template
                - instance_template
                - action_observation_template
                - format_error_template
        """
        super().__init__(config)
        self.prompt_config = prompt_config or {}
        self.env = None
        self.agent = None

    async def run(
        self,
        task: str,
        setup_script: str,
        template_vars: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Run the agent to complete a task.

        Args:
            task: Task description for the agent
            setup_script: Setup script to run before agent starts
            template_vars: Variables for prompt templates

        Returns:
            AgentResult with diff and execution metrics
        """
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        # Prepare model name for litellm
        model_name = self._prepare_model_name()

        # Setup model kwargs
        model_kwargs = {"temperature": self.config.temperature}
        model_kwargs.update(self.config.model_kwargs)

        # Handle API configuration
        is_anthropic = self.config.model.startswith(("claude", "anthropic/"))
        if is_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
        else:
            if self.config.api_base:
                model_kwargs["api_base"] = self.config.api_base
            model_kwargs["api_key"] = self.config.api_key

        # Initialize model
        model = LitellmModel(
            model_name=model_name,
            model_kwargs=model_kwargs,
            cost_tracking="ignore_errors",
        )

        # Initialize Docker environment
        container_lifetime = max(1800, self.config.timeout * 10)
        self.env = DockerEnvironment(
            image=self.config.docker_image,
            cwd=self.config.cwd,
            timeout=self.config.timeout,
            executable="docker",
            run_args=["--rm", "--entrypoint", ""],
            container_timeout=str(container_lifetime),
        )

        # Pull image
        print(f"Pulling image: {self.config.docker_image}")
        subprocess.run(
            ["docker", "pull", self.config.docker_image],
            capture_output=True,
            timeout=300
        )

        # Prepare agent config
        agent_config = {
            "system_template": self.prompt_config.get("system_template", ""),
            "instance_template": self.prompt_config.get("instance_template", ""),
            "action_observation_template": self.prompt_config.get(
                "action_observation_template", ""
            ),
            "format_error_template": self.prompt_config.get(
                "format_error_template", ""
            ),
            "step_limit": self.config.max_iterations,
            "cost_limit": self.config.cost_limit,
        }

        # Create agent
        self.agent = DefaultAgent(model, self.env, **agent_config)

        # Set template variables
        if template_vars:
            self.agent.extra_template_vars = template_vars

        result_text = ""
        diff = ""
        error = None
        exit_status = ""

        try:
            # Run setup script
            print("Setting up environment...")
            setup_output = self.env.execute(setup_script)
            print(f"Setup output: {setup_output.get('output', '')[:500]}")

            # Run agent
            print(f"Running agent with {self.config.model}...")
            loop = asyncio.get_event_loop()
            exit_status, result_text = await loop.run_in_executor(
                None,
                self.agent.run,
                task
            )
            print(f"Agent exit_status: {exit_status}")
            print(f"Agent steps: {model.n_calls}, cost: {model.cost}")

            # Extract diff from container
            diff = self._extract_diff()

        except Exception as e:
            import traceback
            error = traceback.format_exc()
            print(f"Error running agent: {e}")

        return AgentResult(
            diff=diff,
            output_text=result_text or "",
            steps=model.n_calls,
            cost=model.cost,
            exit_status=exit_status,
            success=bool(diff and diff.startswith("diff")),
            error=error,
        )

    def _prepare_model_name(self) -> str:
        """Prepare model name for litellm"""
        model = self.config.model
        if model.startswith(("openai/", "anthropic/", "azure/", "bedrock/")):
            return model
        elif model.startswith("claude"):
            return model
        else:
            return f"openai/{model}"

    def _extract_diff(self) -> str:
        """Extract code diff from Docker container"""
        if not self.env:
            return ""

        # Get diff of source files only (exclude logs, generated files)
        extensions = (
            "'*.js' '*.ts' '*.jsx' '*.tsx' '*.py' '*.java' '*.go' "
            "'*.c' '*.cpp' '*.h' '*.rs' '*.rb' '*.php' '*.cs' "
            "'*.swift' '*.kt' '*.scala' '*.vue' '*.svelte'"
        )
        patch_cmd = f"cd /app && git diff -- {extensions}"
        result = self.env.execute(patch_cmd)
        diff = result.get("output", "").strip()

        # Ensure diff ends with newline
        if diff and not diff.endswith('\n'):
            diff = diff + '\n'

        return diff

    def cleanup(self):
        """Clean up Docker environment"""
        if self.env:
            try:
                self.env.cleanup()
            except Exception as e:
                print(f"Cleanup error: {e}")
            self.env = None
