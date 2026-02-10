"""
Mini-SWE-Agent Implementation

Uses the mini-swe-agent library for bash-based code agent execution.
"""

import os
import sys
import asyncio
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseCodeAgent, AgentConfig, AgentResult
from utils import DIFF_EXTENSIONS

# Configure logging - only show INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
logging.getLogger("minisweagent").setLevel(logging.INFO)


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
        workspace_dir: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the agent to complete a task.

        Args:
            task: Task description for the agent
            setup_script: Setup script to run before agent starts
            template_vars: Variables for prompt templates
            workspace_dir: Host directory to mount as /workspace (optional)

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

        # Build run_args with optional workspace volume mount
        run_args = ["--rm", "--entrypoint", ""]
        if workspace_dir:
            run_args.extend(["-v", f"{workspace_dir}:/workspace"])

        self.env = DockerEnvironment(
            image=self.config.docker_image,
            cwd=self.config.cwd,
            timeout=self.config.timeout,
            executable="docker",
            run_args=run_args,
            container_timeout=str(container_lifetime),
        )

        # Pull image (must succeed, otherwise fail fast)
        print(f"Pulling image: {self.config.docker_image}")
        pull_result = subprocess.run(
            ["docker", "pull", self.config.docker_image],
            capture_output=True,
            timeout=300,
            text=True,
        )
        if pull_result.returncode != 0:
            raise RuntimeError(
                f"Failed to pull image {self.config.docker_image}: {pull_result.stderr}"
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

        patch_cmd = f"cd /app && git diff -- {DIFF_EXTENSIONS}"
        result = self.env.execute(patch_cmd)
        # Don't use strip() - it can remove trailing content
        # Only remove leading/trailing whitespace lines, preserve internal content
        diff = result.get("output", "")

        # Remove only leading whitespace, preserve trailing content
        diff = diff.lstrip()

        # Ensure diff ends with exactly one newline
        diff = diff.rstrip('\n') + '\n' if diff else ""

        # Validate patch format
        if diff and not self._validate_patch(diff):
            print("[BREAKER] Warning: Generated patch may be incomplete or corrupted")

        return diff

    def _validate_patch(self, patch: str) -> bool:
        """Validate that patch format is correct and complete.

        Checks that each hunk has the correct number of lines as declared
        in the hunk header.

        Returns:
            True if patch is valid, False otherwise.
        """
        import re

        if not patch or not patch.startswith("diff"):
            return False

        lines = patch.split('\n')
        i = 0
        valid = True

        while i < len(lines):
            line = lines[i]

            # Find hunk header
            if line.startswith('@@'):
                # Parse @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if not match:
                    i += 1
                    continue

                old_count = int(match.group(2)) if match.group(2) else 1
                new_count = int(match.group(4)) if match.group(4) else 1

                # Count actual lines in hunk
                actual_old = 0
                actual_new = 0
                i += 1

                while i < len(lines):
                    hunk_line = lines[i]
                    if not hunk_line:
                        # Empty line at end of patch
                        break
                    if hunk_line.startswith('diff ') or hunk_line.startswith('@@'):
                        # Next file or hunk
                        break
                    if hunk_line.startswith(' '):
                        actual_old += 1
                        actual_new += 1
                    elif hunk_line.startswith('-'):
                        actual_old += 1
                    elif hunk_line.startswith('+'):
                        actual_new += 1
                    elif hunk_line.startswith('\\'):
                        # "\ No newline at end of file" - doesn't count
                        pass
                    else:
                        # Unknown line format
                        break
                    i += 1

                # Check counts match
                if actual_old != old_count or actual_new != new_count:
                    print(f"[BREAKER] Patch validation failed: "
                          f"expected old={old_count}/new={new_count}, "
                          f"got old={actual_old}/new={actual_new}")
                    valid = False
            else:
                i += 1

        return valid

    def _print_agent_history(self):
        """Print agent execution history for debugging."""
        if not self.agent or not hasattr(self.agent, 'messages'):
            return

        print("\n" + "=" * 60)
        print("AGENT EXECUTION HISTORY")
        print("=" * 60)

        for i, msg in enumerate(self.agent.messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Skip system message (too long)
            if role == 'system':
                print(f"\n[{i}] SYSTEM: (prompt template, skipped)")
                continue

            print(f"\n[{i}] {role.upper()}:")

            # For assistant messages, extract the command
            if role == 'assistant':
                # Extract bash command
                import re
                commands = re.findall(r'```bash\s*\n(.*?)\n```', content, re.DOTALL)
                if commands:
                    print(f"  COMMAND: {commands[0][:200]}")
                else:
                    print(f"  {content[:300]}...")
            else:
                # For user messages (observations), show truncated output
                if len(content) > 500:
                    print(f"  {content[:500]}...")
                else:
                    print(f"  {content}")

        print("\n" + "=" * 60)

    def cleanup(self):
        """Clean up Docker environment"""
        if self.env:
            try:
                self.env.cleanup()
            except Exception as e:
                print(f"Cleanup error: {e}")
            self.env = None
