"""
Ridge Code Agent Implementation

Uses the Ridge Docker sandbox with LLM proxy for code agent execution.
"""

import os
import sys
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseCodeAgent, AgentConfig, AgentResult


# Ridge project path configuration (relative to SWE-SYNTH)
def _get_default_ridge_path() -> str:
    """Get default Ridge path relative to this file"""
    # breaker/agents/ridge.py -> ../.. -> SWE-SYNTH -> ridges
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ridges"))


def get_ridge_project_path() -> str:
    """Get Ridge project path (env var > default)"""
    return os.getenv("RIDGE_PROJECT_PATH", _get_default_ridge_path())


def get_ridge_agent_path() -> str:
    """Get Ridge agent path (env var > default)"""
    default = os.path.join(get_ridge_project_path(), "agents/agent01.py")
    return os.getenv("RIDGE_AGENT_PATH", default)


class RidgeCodeAgent(BaseCodeAgent):
    """
    Code agent implementation using Ridge's Docker sandbox.

    The agent:
    1. Sets up a local repository from Docker image
    2. Starts a proxy container for LLM calls
    3. Runs the Ridge agent in sandbox
    4. Extracts diff from the result
    """

    def __init__(self, config: AgentConfig, prompt_config: Dict[str, str] = None):
        """
        Initialize RidgeCodeAgent.

        Args:
            config: Agent configuration
            prompt_config: Prompt templates (not used by Ridge, kept for interface compatibility)
        """
        super().__init__(config)
        self.prompt_config = prompt_config or {}
        self._temp_dir = None
        self._proxy_started = False
        self._proxy_container_name = None
        self.env = None  # Keep for interface compatibility with BugInjector

    def _get_ridges_module(self):
        """Import ridges_evaluate module"""
        ridge_project = get_ridge_project_path()
        if ridge_project not in sys.path:
            sys.path.insert(0, ridge_project)
        import ridges_evaluate
        return ridges_evaluate

    def _extract_repo_from_docker(self, docker_image: str, temp_dir: str) -> Optional[str]:
        """Extract repository from SWE-bench Docker image"""
        try:
            # Pull image first
            print(f"[RIDGE] Pulling Docker image: {docker_image}")
            subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True, timeout=300
            )

            container_name = f"ridge-extract-{int(time.time() * 1000)}"
            local_repo_path = os.path.join(temp_dir, "repo")

            # Create container
            subprocess.run(
                ["docker", "create", "--name", container_name, docker_image, "true"],
                capture_output=True, check=True
            )

            # Copy /app from container
            result = subprocess.run(
                ["docker", "cp", f"{container_name}:/app", local_repo_path],
                capture_output=True, text=True
            )

            # Clean up container
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

            if result.returncode != 0:
                print(f"[RIDGE] Failed to copy repo from container: {result.stderr}")
                return None

            print(f"[RIDGE] Repository extracted to: {local_repo_path}")
            return local_repo_path

        except Exception as e:
            print(f"[RIDGE] Error extracting repo: {e}")
            return None

    async def run(
        self,
        task: str,
        setup_script: str,
        template_vars: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the Ridge agent to complete a task.

        Args:
            task: Task description for the agent
            setup_script: Setup script to run before agent starts
            template_vars: Variables for prompt templates
            workspace_dir: Host directory to mount (not used by Ridge, kept for interface)

        Returns:
            AgentResult with diff and execution metrics
        """
        ridges = None
        try:
            ridges = self._get_ridges_module()
            agent_path = get_ridge_agent_path()

            if not os.path.exists(agent_path):
                return AgentResult(
                    diff="",
                    output_text="",
                    steps=0,
                    cost=0.0,
                    success=False,
                    error=f"Ridge agent not found: {agent_path}"
                )

            # Create temp directory
            self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            temp_dir = self._temp_dir.name

            # Extract repo from Docker image
            local_repo_path = self._extract_repo_from_docker(self.config.docker_image, temp_dir)
            if not local_repo_path:
                return AgentResult(
                    diff="",
                    output_text="",
                    steps=0,
                    cost=0.0,
                    success=False,
                    error="Failed to extract repository from Docker image"
                )

            # Run setup script in the extracted repo
            actual_repo = local_repo_path
            if os.path.exists(os.path.join(local_repo_path, "app")):
                actual_repo = os.path.join(local_repo_path, "app")

            print(f"[RIDGE] Running setup script in: {actual_repo}")
            setup_result = subprocess.run(
                ["bash", "-c", setup_script],
                cwd=actual_repo,
                capture_output=True,
                text=True,
                timeout=300
            )
            print(f"[RIDGE] Setup output: {setup_result.stdout[:500]}")
            if setup_result.stderr:
                print(f"[RIDGE] Setup stderr: {setup_result.stderr[:500]}")

            # Build problem statement for Ridge agent
            problem_statement = self._build_problem_statement(task, template_vars)

            # Start proxy
            print("[RIDGE] Starting proxy container...")
            self._proxy_container_name = f"ridge-breaker-proxy-{int(time.time() * 1000)}"
            proxy_result = ridges.run_proxy_container(
                openai_api_key=self.config.api_key,
                openai_model=self.config.model,
                openai_base_url=self.config.api_base,
                temperature=self.config.temperature,
                seed=self.config.model_kwargs.get("seed"),
                port=8001,  # Use different port to avoid conflict with fixer
                container_name=self._proxy_container_name
            )

            if not proxy_result.get("success"):
                return AgentResult(
                    diff="",
                    output_text="",
                    steps=0,
                    cost=0.0,
                    success=False,
                    error=f"Failed to start proxy: {proxy_result.get('error')}"
                )

            self._proxy_started = True
            proxy_url = f"http://host.docker.internal:{proxy_result['port']}"
            print(f"[RIDGE] Proxy running at: {proxy_result['endpoint']}")

            # Run Ridge sandbox
            print(f"[RIDGE] Running agent sandbox...")
            result = ridges.run_ridges_sandbox(
                repo_path=actual_repo,
                agent_path=agent_path,
                problem_statement=problem_statement,
                sandbox_proxy_url=proxy_url,
                timeout=self.config.timeout,
            )

            # Process result
            if isinstance(result, dict):
                output = result.get("output", {})
                if isinstance(output, dict):
                    diff = output.get("patch", "")
                    output_text = output.get("bug_description", "")
                else:
                    diff = str(output)
                    output_text = str(output)

                return AgentResult(
                    diff=diff,
                    output_text=output_text,
                    steps=result.get("steps", 0),
                    cost=result.get("cost", 0.0),
                    exit_status="completed",
                    success=bool(diff and diff.startswith("diff")),
                    error=result.get("error"),
                )
            else:
                return AgentResult(
                    diff="",
                    output_text=str(result),
                    steps=0,
                    cost=0.0,
                    success=False,
                    error="Unexpected result format from Ridge sandbox"
                )

        except Exception as e:
            import traceback
            return AgentResult(
                diff="",
                output_text="",
                steps=0,
                cost=0.0,
                success=False,
                error=traceback.format_exc()
            )

        finally:
            self.cleanup()

    def _build_problem_statement(
        self,
        task: str,
        template_vars: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build problem statement for Ridge agent"""
        vars = template_vars or {}

        # Build a comprehensive prompt for bug injection
        prompt_parts = [
            "You are a BUG INJECTION agent. Your goal is to introduce a realistic bug that causes tests to FAIL.",
            "",
            f"Repository: {vars.get('repo', 'unknown')}",
            f"Bug types to inject: {vars.get('bug_types_str', 'any realistic bug')}",
            "",
        ]

        if vars.get('bug_descriptions'):
            prompt_parts.extend([
                "Bug type descriptions:",
                vars['bug_descriptions'],
                "",
            ])

        if vars.get('gold_patch'):
            prompt_parts.extend([
                "GOLD PATCH (already applied - code is correct, all tests pass):",
                "```diff",
                vars['gold_patch'][:4000],
                "```",
                "",
            ])

        if vars.get('target_tests_str'):
            prompt_parts.extend([
                "TARGET TESTS (your bug should cause at least one to FAIL):",
                vars['target_tests_str'],
                "",
            ])

        if vars.get('feedback'):
            prompt_parts.extend([
                "PREVIOUS ATTEMPT FEEDBACK:",
                vars['feedback'],
                "",
                "Please try a DIFFERENT approach this time.",
                "",
            ])

        prompt_parts.extend([
            "INSTRUCTIONS:",
            "1. Read the source files to understand the code",
            "2. Identify a good location to inject a bug",
            "3. Make a SUBTLE change that causes tests to fail",
            "4. Run tests to verify at least one test fails",
            "5. Finish as quickly as possible",
            "",
            "RULES:",
            "- Make SUBTLE changes (what a tired developer might write)",
            "- NO syntax errors",
            "- Focus on return values, conditionals, error handling",
            "- Your bug should cause 1-10 tests to fail, not catastrophic failure",
        ])

        return "\n".join(prompt_parts)

    def cleanup(self):
        """Clean up proxy container and temp directory"""
        if self._proxy_started and self._proxy_container_name:
            try:
                ridges = self._get_ridges_module()
                ridges.stop_proxy_container(self._proxy_container_name)
                print(f"[RIDGE] Proxy container stopped: {self._proxy_container_name}")
            except Exception as e:
                print(f"[RIDGE] Error stopping proxy: {e}")
            self._proxy_started = False

        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except (PermissionError, OSError) as e:
                print(f"[RIDGE] Warning: Could not cleanup temp directory: {e}")
            self._temp_dir = None
