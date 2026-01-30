"""
Bug Injector

Responsible for injecting bugs into correct code using a code agent.
Includes internal verification loop to ensure injected bug causes test failures.
"""

import base64
import json
import random
import re
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from .types import BreakerInput, InjectionResult, BreakerException
from .bug_types import format_bug_types_for_prompt
from .agents.base import BaseCodeAgent, AgentConfig


class BugInjector:
    """
    Injects bugs into correct code using a code agent.

    The injector:
    1. Sets up a Docker environment with gold_patch applied
    2. Runs a code agent to inject a bug
    3. Verifies the bug causes test failures
    4. Returns the bug patch and metadata
    """

    def __init__(
        self,
        agent: BaseCodeAgent,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize BugInjector.

        Args:
            agent: The code agent to use for bug injection
            config_path: Path to config.yaml (defaults to module's config.yaml)
        """
        self.agent = agent

        # Load prompt config
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.prompt_config = config.get("injector", {})

    async def inject(
        self,
        input: BreakerInput,
        feedback: Optional[str] = None,
    ) -> InjectionResult:
        """
        Inject a bug into the code.

        Args:
            input: BreakerInput with all necessary context
            feedback: Feedback from previous failed attempt (for retry)

        Returns:
            InjectionResult with bug patch and test results
        """
        # Create temp directory for workspace files (avoids command line length issues)
        # Use /tmp/breaker_workspaces which is mounted to host for Docker volume access
        import os
        workspace_base = "/tmp/breaker_workspaces"
        os.makedirs(workspace_base, exist_ok=True)
        workspace_dir = tempfile.mkdtemp(prefix="ws_", dir=workspace_base)

        try:
            # Write large files to workspace directory
            self._prepare_workspace(workspace_dir, input)

            # Build setup script (now much shorter, just runs init)
            setup_script = self._build_setup_script(input)

            # Build template variables for prompt
            template_vars = self._build_template_vars(input, feedback)

            # Update agent's prompt config
            if hasattr(self.agent, 'prompt_config'):
                self.agent.prompt_config = {
                    "system_template": self.prompt_config.get("system_template", ""),
                    "instance_template": self.prompt_config.get("instance_template", ""),
                    "action_observation_template": self.prompt_config.get(
                        "action_observation_template", ""
                    ),
                    "format_error_template": self.prompt_config.get(
                        "format_error_template", ""
                    ),
                }

            # Run the agent with workspace mounted
            result = await self.agent.run(
                task="Inject a bug that causes tests to fail",
                setup_script=setup_script,
                template_vars=template_vars,
                workspace_dir=workspace_dir,
            )

            if not result.diff or not result.diff.startswith("diff"):
                raise BreakerException(
                    f"Agent failed to produce valid diff. Output: {result.output_text[:500]}"
                )

            # Validate patch format is complete
            if not self._validate_patch_format(result.diff):
                raise BreakerException(
                    f"Generated patch is incomplete or corrupted. Patch: {result.diff[:500]}"
                )

            # Parse bug description from output
            bug_description = self._parse_bug_description(result.output_text)

            # Run tests to verify bug effectiveness
            test_result = self._run_tests_in_container(input)

            return InjectionResult(
                bug_patch=result.diff,
                bug_description=bug_description,
                failed_tests=test_result.get("failed", []),
                passed_tests=test_result.get("passed", []),
                error_output=test_result.get("error_output", ""),
                agent_steps=result.steps,
                agent_cost=result.cost,
            )

        finally:
            self.agent.cleanup()
            # Clean up temp workspace
            shutil.rmtree(workspace_dir, ignore_errors=True)

    def _prepare_workspace(self, workspace_dir: str, input: BreakerInput) -> None:
        """Write large files to workspace directory (mounted as volume)."""
        workspace = Path(workspace_dir)

        # Write gold patch
        (workspace / "gold_patch.diff").write_text(input.gold_patch)

        # Write test runner script
        (workspace / "run_script.sh").write_text(input.test_runner_script)

        # Write parser script
        (workspace / "parser.py").write_text(input.test_parser_script)

        # Write run_tests.sh wrapper
        run_tests_script = f"""#!/bin/bash
cd /app
{input.env_cmds}
{input.before_repo_set_cmd}
bash /workspace/run_script.sh {input.test_files} > /workspace/stdout.log 2> /workspace/stderr.log
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
echo "=== TEST RESULTS ==="
if [ -f /workspace/output.json ]; then
    python3 -c "
import json
with open('/workspace/output.json') as f:
    data = json.load(f)
passed = [t['name'] for t in data.get('tests', []) if t['status'] == 'PASSED']
failed = [t['name'] for t in data.get('tests', []) if t['status'] == 'FAILED']
print(f'PASSED: {{len(passed)}}')
print(f'FAILED: {{len(failed)}}')
if failed:
    print('Failed tests:')
    for t in failed[:10]:
        print(f'  - {{t}}')
"
else
    echo "No test output found"
fi
"""
        (workspace / "run_tests.sh").write_text(run_tests_script)

        # Write init script
        init_script = f"""#!/bin/bash
cd /app
git reset --hard {input.base_commit}
git checkout {input.base_commit}

# Configure git for commit
git config user.email "breaker@swe-synth.local"
git config user.name "SWE-SYNTH Breaker"

# Apply gold_patch (all tests should pass after this)
git apply -v /workspace/gold_patch.diff

# Commit gold_patch so git diff will only show agent's bug injection
git add -A
git commit -m "Apply gold patch - baseline for bug injection"
echo "Gold patch committed. Agent's changes will be tracked from here."
"""
        (workspace / "init.sh").write_text(init_script)

    def _build_setup_script(self, input: BreakerInput) -> str:
        """Build the setup script that runs before agent starts.

        Note: Large files are already mounted via volume, so this script is short.
        """
        setup_script = """#!/bin/bash
chmod +x /workspace/run_script.sh /workspace/run_tests.sh /workspace/init.sh
bash /workspace/init.sh
echo "Setup complete. Working directory: /app"
pwd && ls -la
"""
        return setup_script

    def _build_template_vars(
        self,
        input: BreakerInput,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build template variables for the agent prompt"""
        # Select target tests (prioritize fail_to_pass if available)
        target_tests = self._select_target_tests(input.test_cases, input.seed)

        return {
            "repo": input.repo,
            "bug_types_str": ", ".join(input.bug_types),
            "bug_descriptions": format_bug_types_for_prompt(input.bug_types),
            "gold_patch": input.gold_patch,
            "target_tests_str": "\n".join(f"- {t}" for t in target_tests),
            "test_patch_snippet": (input.test_patch or "")[:4000],
            "feedback": feedback,
        }

    def _select_target_tests(
        self,
        all_tests: List[str],
        seed: int,
        max_tests: int = 5,
    ) -> List[str]:
        """Select target tests for the agent to focus on"""
        if not all_tests:
            return []

        rng = random.Random(seed)
        num_tests = min(max_tests, len(all_tests))
        return rng.sample(all_tests, num_tests)

    def _validate_patch_format(self, patch: str) -> bool:
        """Validate that patch format is correct and complete.

        Checks that each hunk has the correct number of lines as declared
        in the hunk header. This catches truncated patches.

        Returns:
            True if patch is valid, False otherwise.
        """
        if not patch or not patch.startswith("diff"):
            return False

        lines = patch.split('\n')
        i = 0

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
                        break
                    if hunk_line.startswith('diff ') or hunk_line.startswith('@@'):
                        break
                    if hunk_line.startswith(' '):
                        actual_old += 1
                        actual_new += 1
                    elif hunk_line.startswith('-'):
                        actual_old += 1
                    elif hunk_line.startswith('+'):
                        actual_new += 1
                    elif hunk_line.startswith('\\'):
                        pass  # "\ No newline at end of file"
                    else:
                        break
                    i += 1

                # Check counts match
                if actual_old != old_count or actual_new != new_count:
                    print(f"[BREAKER] Patch validation failed: "
                          f"hunk declares old={old_count}/new={new_count}, "
                          f"actual old={actual_old}/new={actual_new}")
                    return False
            else:
                i += 1

        return True

    def _parse_bug_description(self, output_text: str) -> str:
        """Parse bug description from agent output"""
        if not output_text:
            return ""

        # Look for BUG_DESCRIPTION section
        match = re.search(
            r'BUG_DESCRIPTION:\s*(.+?)(?=\n[A-Z_]+:|$)',
            output_text,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        return ""

    def _run_tests_in_container(self, input: BreakerInput) -> Dict[str, Any]:
        """Run tests in the container and parse results"""
        if not self.agent.env:
            return {"failed": [], "passed": [], "error_output": ""}

        try:
            # Run tests
            result = self.agent.env.execute("bash /workspace/run_tests.sh")
            output = result.get("output", "")

            # Read parsed output
            json_result = self.agent.env.execute("cat /workspace/output.json 2>/dev/null || echo '{}'")
            json_str = json_result.get("output", "{}")

            try:
                test_data = json.loads(json_str)
                tests = test_data.get("tests", [])
                failed = [t["name"] for t in tests if t.get("status") == "FAILED"]
                passed = [t["name"] for t in tests if t.get("status") == "PASSED"]
            except json.JSONDecodeError:
                failed = []
                passed = []

            return {
                "failed": failed,
                "passed": passed,
                "error_output": output[:5000],
            }

        except Exception as e:
            print(f"Error running tests: {e}")
            return {"failed": [], "passed": [], "error_output": str(e)}


def create_injector(
    input: BreakerInput,
    config_path: Optional[Path] = None,
    agent_type: str = "miniswe",
) -> BugInjector:
    """
    Create a BugInjector with specified agent backend.

    Args:
        input: BreakerInput with model configuration
        config_path: Optional path to config.yaml
        agent_type: Agent type to use ("miniswe" or "ridge")

    Returns:
        Configured BugInjector
    """
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    injector_config = config.get("injector", {})

    # Create agent config
    agent_config = AgentConfig(
        model=input.model,
        api_base=input.api_base,
        api_key=input.api_key,
        temperature=input.temperature,
        max_iterations=input.max_iterations,
        cost_limit=input.cost_limit,
        timeout=input.timeout,
        docker_image=input.docker_image,
    )

    # Create prompt config
    prompt_config = {
        "system_template": injector_config.get("system_template", ""),
        "instance_template": injector_config.get("instance_template", ""),
        "action_observation_template": injector_config.get(
            "action_observation_template", ""
        ),
        "format_error_template": injector_config.get("format_error_template", ""),
    }

    # Create agent based on type
    if agent_type == "ridge":
        from .agents.ridge import RidgeCodeAgent
        agent = RidgeCodeAgent(agent_config, prompt_config)
    else:
        # Default to miniswe
        from .agents.miniswe import MiniSweAgent
        agent = MiniSweAgent(agent_config, prompt_config)

    return BugInjector(agent, config_path)
