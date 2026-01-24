"""
Bug Injector

Responsible for injecting bugs into correct code using a code agent.
Includes internal verification loop to ensure injected bug causes test failures.
"""

import base64
import json
import random
import re
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
        # Build setup script
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

        try:
            # Run the agent
            result = await self.agent.run(
                task="Inject a bug that causes tests to fail",
                setup_script=setup_script,
                template_vars=template_vars,
            )

            if not result.diff or not result.diff.startswith("diff"):
                raise BreakerException(
                    f"Agent failed to produce valid diff. Output: {result.output_text[:500]}"
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

    def _build_setup_script(self, input: BreakerInput) -> str:
        """Build the setup script that runs before agent starts"""
        # Encode patches and scripts for transfer
        gold_patch_b64 = base64.b64encode(
            input.gold_patch.encode('utf-8')
        ).decode('ascii')

        run_script_b64 = base64.b64encode(
            input.test_runner_script.encode('utf-8')
        ).decode('ascii')

        parser_script_b64 = base64.b64encode(
            input.test_parser_script.encode('utf-8')
        ).decode('ascii')

        # Build run_tests.sh wrapper
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
        run_tests_b64 = base64.b64encode(
            run_tests_script.encode('utf-8')
        ).decode('ascii')

        # Build init script
        init_script = f"""#!/bin/bash
mkdir -p /workspace
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
        init_script_b64 = base64.b64encode(
            init_script.encode('utf-8')
        ).decode('ascii')

        # Full setup script
        setup_script = f"""#!/bin/bash
mkdir -p /workspace
echo "{gold_patch_b64}" | base64 -d > /workspace/gold_patch.diff
echo "{run_script_b64}" | base64 -d > /workspace/run_script.sh
echo "{parser_script_b64}" | base64 -d > /workspace/parser.py
echo "{run_tests_b64}" | base64 -d > /workspace/run_tests.sh
echo "{init_script_b64}" | base64 -d > /workspace/init.sh
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
) -> BugInjector:
    """
    Create a BugInjector with default mini-swe-agent backend.

    Args:
        input: BreakerInput with model configuration
        config_path: Optional path to config.yaml

    Returns:
        Configured BugInjector
    """
    from .agents.miniswe import MiniSweAgent

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

    # Create agent with prompt config
    prompt_config = {
        "system_template": injector_config.get("system_template", ""),
        "instance_template": injector_config.get("instance_template", ""),
        "action_observation_template": injector_config.get(
            "action_observation_template", ""
        ),
        "format_error_template": injector_config.get("format_error_template", ""),
    }

    agent = MiniSweAgent(agent_config, prompt_config)

    return BugInjector(agent, config_path)
