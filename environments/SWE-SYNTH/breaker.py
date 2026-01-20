"""
SWE-SYNTH Breaker Module

Responsible for generating realistic bugs by injecting faults into correct code.

Two modes:
1. Prompt mode: Single LLM call to generate bug (faster but less reliable)
2. Agent mode: Code agent that iteratively injects and verifies bugs (more reliable)

Flow:
1. Start from gold_patch applied state (all tests pass)
2. Randomly select target tests to break
3. Generate bug that causes selected tests to fail
"""

import os
import re
import json
import time
import random
import asyncio
import subprocess
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List


# Bug types that Breaker can inject
BUG_TYPES = [
    "off-by-one",           # Boundary errors (< vs <=, > vs >=)
    "logic-inversion",      # Logic errors (and vs or, == vs !=)
    "wrong-variable",       # Using wrong variable
    "missing-check",        # Missing null/boundary check
    "type-error",           # Type conversion errors
    "wrong-operator",       # Arithmetic operator errors (+, -, *, /)
    "missing-return",       # Missing return statement
    "exception-swallow",    # Swallowing exceptions incorrectly
    "wrong-constant",       # Wrong constant/magic number
    "string-format",        # String formatting errors
    "order-dependency",     # Wrong order of operations
    "early-exit",           # Premature return/break/continue
    "wrong-default",        # Wrong default value
]


BUG_TYPE_DESCRIPTIONS = {
    "off-by-one": {
        "description": "Boundary errors like < vs <=, wrong loop bounds, index ±1 errors",
        "examples": [
            "for i in range(n) → for i in range(n-1)",
            "if x <= 0 → if x < 0",
            "array[i+1] → array[i]",
        ]
    },
    "logic-inversion": {
        "description": "Logic errors like and/or swap, condition negation, comparison flip",
        "examples": [
            "if a and b → if a or b",
            "if x == y → if x != y",
            "if not flag → if flag",
        ]
    },
    "wrong-variable": {
        "description": "Using incorrect but similar variable names",
        "examples": [
            "return result → return results",
            "self.value → self.values",
            "user_id → user_ids",
        ]
    },
    "missing-check": {
        "description": "Removing important null/boundary/type checks",
        "examples": [
            "if x is not None: use(x) → use(x)",
            "if len(arr) > 0: → (remove check)",
            "if isinstance(x, str): → (remove check)",
        ]
    },
    "type-error": {
        "description": "Type conversion or coercion errors",
        "examples": [
            "int(x) → str(x)",
            "float(x) → int(x)",
            "str(x) → x",
        ]
    },
    "wrong-operator": {
        "description": "Arithmetic or bitwise operator errors",
        "examples": [
            "a + b → a - b",
            "a * b → a / b",
            "a | b → a & b",
        ]
    },
    "missing-return": {
        "description": "Forgetting return statements",
        "examples": [
            "return result → result",
            "return True → (nothing)",
        ]
    },
    "exception-swallow": {
        "description": "Incorrectly handling or swallowing exceptions",
        "examples": [
            "except ValueError: raise → except ValueError: pass",
            "except Exception as e: log(e) → except: pass",
        ]
    },
    "wrong-constant": {
        "description": "Using wrong constant or magic number",
        "examples": [
            "timeout = 1000 → timeout = 100",
            "MAX_RETRIES = 3 → MAX_RETRIES = 0",
            "BUFFER_SIZE = 4096 → BUFFER_SIZE = 40",
        ]
    },
    "string-format": {
        "description": "String formatting or interpolation errors",
        "examples": [
            'f"{name}" → f"{names}"',
            '"{} {}".format(a, b) → "{} {}".format(b, a)',
            'f"Error: {msg}" → f"Error: {err}"',
        ]
    },
    "order-dependency": {
        "description": "Wrong order of operations or statements",
        "examples": [
            "validate(); process() → process(); validate()",
            "lock(); access() → access(); lock()",
            "init(); use() → use(); init()",
        ]
    },
    "early-exit": {
        "description": "Premature return, break, or continue",
        "examples": [
            "if error: log(); return → if error: return",
            "for item in items: process(item) → for item in items: break",
            "while condition: work(); check() → while condition: return",
        ]
    },
    "wrong-default": {
        "description": "Using wrong default value for parameters or variables",
        "examples": [
            "def f(x=None): → def f(x=0):",
            "count = 0 → count = 1",
            "enabled = True → enabled = False",
        ]
    },
}


def select_target_tests(
    all_tests: List[str],
    seed: int,
    min_tests: int = 1,
    max_tests: int = 3,
) -> List[str]:
    """
    Randomly select target tests to break.

    Args:
        all_tests: List of all passing tests
        seed: Random seed for reproducibility
        min_tests: Minimum number of tests to select
        max_tests: Maximum number of tests to select

    Returns:
        List of selected test names
    """
    if not all_tests:
        return []

    rng = random.Random(seed)

    # Determine number of tests to select
    num_tests = rng.randint(min_tests, min(max_tests, len(all_tests)))

    # Select tests
    selected = rng.sample(all_tests, num_tests)

    return selected


def build_breaker_prompt(
    repo: str,
    bug_types: List[str],
    gold_patch: str,
    problem_statement_original: str,
    repo_files: List[dict] = None,
    target_tests: List[str] = None,
    test_patch: str = None,
) -> str:
    """Build the prompt for Breaker model."""

    # Build description for all bug types
    bug_descriptions = []
    all_examples = []
    for bug_type in bug_types:
        bug_info = BUG_TYPE_DESCRIPTIONS.get(bug_type, {})
        desc = bug_info.get("description", bug_type)
        bug_descriptions.append(f"- {bug_type}: {desc}")
        all_examples.extend(bug_info.get("examples", []))

    bug_types_str = ", ".join(bug_types)
    descriptions_str = "\n".join(bug_descriptions)
    examples_str = "\n".join(f"  - {ex}" for ex in all_examples[:6])  # Limit examples

    # Build source files section
    files_section = ""
    if repo_files:
        files_section = "\n\nSOURCE FILES (choose one to inject bug):\n"
        for f in repo_files:
            files_section += f"\n--- {f['path']} ---\n```\n{f['content'][:2000]}\n```\n"

    # Build target tests section
    tests_section = ""
    if target_tests:
        tests_str = "\n".join(f"  - {t}" for t in target_tests[:5])
        tests_section = f"""

TARGET TESTS (your bug should cause at least one of these to fail):
{tests_str}"""

        # Include test code if available - only for target tests
        if test_patch:
            # Extract only the test code relevant to target_tests
            import re
            relevant_code = []
            for test_name in target_tests[:5]:
                # Search for function definition pattern (supports multiple languages)
                # Go: func TestXxx(
                # Python: def test_xxx(
                # JS: describe('xxx' / it('xxx' / test('xxx'
                patterns = [
                    rf'(func\s+{re.escape(test_name)}\s*\([^)]*\)\s*\{{[\s\S]*?^\}})',  # Go
                    rf'(def\s+{re.escape(test_name)}\s*\([^)]*\):[\s\S]*?)(?=\ndef\s|\nclass\s|\Z)',  # Python
                    rf'((?:describe|it|test)\s*\(\s*[\'\"]{re.escape(test_name)}[\'\"][\s\S]*?\}}\s*\))',  # JS
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, test_patch, re.MULTILINE)
                    if matches:
                        for match in matches:
                            if match not in relevant_code:
                                relevant_code.append(match)
                        break

            # If we found relevant code, use it; otherwise use truncated full patch
            if relevant_code:
                test_code = "\n\n".join(relevant_code)[:6000]
            else:
                test_code = test_patch[:4000]

            tests_section += f"""

TEST CODE (analyze what these tests check):
```
{test_code}
```

CRITICAL: Study the test assertions above. Your bug must cause at least one assertion to fail.
- Identify what values/behaviors the tests expect
- Inject a bug that produces WRONG values or behaviors
- Focus on the exact code paths being tested"""

    return f"""You are a code mutation expert. Inject a realistic bug combining these types: {bug_types_str}.

CONTEXT:
- Repository: {repo}
- Code is CORRECT (all tests pass). Your job is to introduce a bug that will cause tests to fail.

BUG TYPES TO COMBINE:
{descriptions_str}

Examples:
{examples_str}
{files_section}{tests_section}
REFERENCE PATCH (shows what was recently fixed - you can inject bug here or in the source files above):
```diff
{gold_patch}
```

INSTRUCTIONS:
1. Choose a location to inject the bug (either in the gold_patch area OR in one of the source files)
2. Create a bug_patch in unified diff format that introduces a bug combining: {bug_types_str}
3. Write a problem_statement describing symptoms (like a user bug report)

RULES:
- Make subtle, realistic changes (what a tired developer might write)
- Your bug_patch is applied AFTER the gold_patch
- NO syntax errors, NO deleting large code blocks
- The file path in your diff must match the actual file path
- If the specified bug types don't fit this code, use your judgment to inject a different realistic bug type
- The bug MUST cause at least one test to fail - otherwise it's not a valid bug

STRATEGY FOR EFFECTIVE BUGS:
1. Analyze what the gold_patch fixes - inject bugs in the SAME code path
2. Focus on return values, conditionals, and error handling - these affect test outcomes
3. Avoid cosmetic changes (comments, variable names) - they don't break tests
4. Prefer bugs in core logic that gets exercised by multiple code paths

PROBLEM_STATEMENT GUIDELINES:
Write a clear bug report that helps developers investigate, but does NOT directly reveal the fix:
- Describe the FEATURE/FUNCTIONALITY affected (e.g., "email confirmation", "user search", "data export")
- Describe WHAT GOES WRONG with specific details (e.g., "displays undefined instead of the expected value", "returns empty results", "throws TypeError")
- Describe WHEN/HOW it happens (e.g., "when resending confirmation too quickly", "when searching with special characters")
- Include observable symptoms like error messages, wrong output values, or unexpected behavior
- Do NOT mention specific line numbers, variable names, or the exact code change needed

GOOD example: "The email confirmation resend feature shows a malformed error message. When users try to resend too quickly, the rate-limit message displays 'undefined' or a raw variable path instead of showing the actual wait time in minutes. The error should show something like 'Please wait 10 minutes' but instead shows gibberish."

BAD example (too vague): "Error message is confusing."
BAD example (reveals answer): "Line 80 uses meta.config.emailConfirmInterval instead of emailInterval variable."

OUTPUT: Return ONLY a JSON object (no markdown, no extra text):
{{"bug_patch": "diff --git a/path/to/file.js b/path/to/file.js\\nindex abc..def 100644\\n--- a/path/to/file.js\\n+++ b/path/to/file.js\\n@@ -10,3 +10,3 @@\\n-    correct code\\n+    buggy code", "problem_statement": "Clear bug report describing the feature, symptoms, and context", "bug_description": "Technical description of the injected bug mechanism"}}

Generate the JSON now:"""

async def generate_bug(
    swe_instance: Dict[str, Any],
    bug_types: List[str],
    seed: int,
    breaker_model: str,
    breaker_base_url: str,
    breaker_api_key: str,
    temperature: float = 0.7,
    repo_files: List[dict] = None,
) -> Dict[str, Any]:
    """
    Generate a bug using Breaker model.

    Args:
        swe_instance: SWE-bench instance data
        bug_types: List of bug types to inject (1-3 types)
        seed: Random seed for generation
        breaker_model: Model name for bug generation
        breaker_base_url: API base URL
        breaker_api_key: API key
        temperature: Sampling temperature
        repo_files: List of source files for bug injection

    Returns:
        Bug instance dict with patch, problem_statement, metadata
    """
    from litellm import acompletion

    instance_id = swe_instance["instance_id"]
    repo = swe_instance.get("repo", "")
    base_commit = swe_instance.get("base_commit", "")

    # Get the gold patch
    gold_patch = swe_instance.get("patch", "")
    problem_statement_original = swe_instance.get("problem_statement", "")

    # Get ALL tests (after gold_patch, all should pass)
    fail_to_pass = swe_instance.get("FAIL_TO_PASS", swe_instance.get("fail_to_pass", "[]"))
    pass_to_pass = swe_instance.get("PASS_TO_PASS", swe_instance.get("pass_to_pass", "[]"))

    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = eval(fail_to_pass)
        except:
            fail_to_pass = []
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = eval(pass_to_pass)
        except:
            pass_to_pass = []

    # All tests pass after gold_patch
    all_passing_tests = list(set(fail_to_pass) | set(pass_to_pass))

    # Mixed selection: prioritize fail_to_pass (2x weight) but include pass_to_pass for diversity
    import random
    rng = random.Random(seed)

    # Build weighted pool: fail_to_pass tests appear twice (2x weight)
    weighted_pool = list(fail_to_pass) * 2 + list(pass_to_pass)
    rng.shuffle(weighted_pool)

    # Select unique tests from shuffled weighted pool
    target_tests = []
    seen = set()
    for t in weighted_pool:
        if t not in seen:
            target_tests.append(t)
            seen.add(t)
        if len(target_tests) >= 5:
            break

    # Get test patch (shows what tests check)
    test_patch = swe_instance.get("test_patch", "")

    # Build prompt with target tests and test code
    prompt = build_breaker_prompt(
        repo=repo,
        bug_types=bug_types,
        gold_patch=gold_patch,
        problem_statement_original=problem_statement_original,
        repo_files=repo_files,
        target_tests=target_tests,
        test_patch=test_patch,
    )

    import re

    response = await acompletion(
        model=f"openai/{breaker_model}" if not breaker_model.startswith("openai/") else breaker_model,
        messages=[{"role": "user", "content": prompt}],
        api_base=breaker_base_url,
        api_key=breaker_api_key,
        temperature=temperature,
        seed=seed,
        timeout=300,
    )

    content = response.choices[0].message.content

    # Clean control characters that break JSON parsing
    def clean_json_string(s):
        # Remove control characters except \n, \r, \t
        s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
        # Also escape unescaped control chars within string values
        # Replace literal newlines/tabs in string values with escaped versions
        def escape_in_strings(match):
            val = match.group(0)
            val = val.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return val
        # Match JSON string values and escape control chars within them
        s = re.sub(r'"(?:[^"\\]|\\.)*"', escape_in_strings, s)
        return s

    # Parse JSON from response (strict=False allows control chars in strings)
    bug_data = None

    # Method 1: Try ```json ... ``` block
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        try:
            json_str = clean_json_string(json_match.group(1))
            bug_data = json.loads(json_str, strict=False)
        except json.JSONDecodeError:
            pass

    # Method 2: Find JSON by matching balanced braces starting from "bug_patch"
    if bug_data is None:
        # Find the start of JSON object containing bug_patch
        idx = content.find('"bug_patch"')
        if idx != -1:
            # Search backward for opening brace
            start = content.rfind('{', 0, idx)
            if start != -1:
                # Find matching closing brace
                depth = 0
                end = start
                for i, c in enumerate(content[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                try:
                    json_str = clean_json_string(content[start:end])
                    bug_data = json.loads(json_str, strict=False)
                except json.JSONDecodeError:
                    pass

    # Method 3: Try parsing entire content as JSON
    if bug_data is None:
        json_str = clean_json_string(content)
        bug_data = json.loads(json_str, strict=False)

    # Validate required fields
    if not bug_data.get("bug_patch"):
        raise ValueError("Missing bug_patch in response")

    bug_instance = {
        "source": {
            "dataset": "SWE-bench_Pro",
            "swe_instance_id": instance_id,
            "base_commit": base_commit,
            "repo": repo,
        },
        "bug": {
            "bug_types": bug_types,
            "patch": bug_data.get("bug_patch", ""),
            "problem_statement": bug_data.get("problem_statement", ""),
            "description": bug_data.get("bug_description", ""),
            # target_tests will be discovered during verification
        },
        "generation": {
            "breaker_model": breaker_model,
            "seed": seed,
            "temperature": temperature,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "original": {
            "gold_patch": gold_patch,
            "problem_statement": problem_statement_original,
            "all_tests": all_passing_tests,
            "fail_to_pass": fail_to_pass,  # Tests that gold_patch fixes
        },
        "verification": {"verified": False},
    }

    return bug_instance


async def generate_bug_with_agent(
    swe_instance: Dict[str, Any],
    bug_types: List[str],
    seed: int,
    breaker_model: str,
    breaker_base_url: str,
    breaker_api_key: str,
    temperature: float = 0.7,
    max_iterations: int = 30,
    cost_limit: float = 5.0,
    timeout: int = 300,
    dockerhub_username: str = "jefzda",
    run_script: str = None,
    parser_script: str = None,
    env_cmds: str = "",
    test_files_str: str = "",
    before_repo_set_cmd: str = "",
) -> Dict[str, Any]:
    """
    Generate a bug using a code agent that can iteratively explore, inject, and verify.

    This is more reliable than single-prompt generation because the agent can:
    - Read and understand the code structure
    - Try different bug injection strategies
    - Verify that tests actually fail
    - Adjust based on test results

    Args:
        swe_instance: SWE-bench instance data
        bug_types: List of bug types to inject
        seed: Random seed for generation
        breaker_model: Model name for bug generation
        breaker_base_url: API base URL
        breaker_api_key: API key
        temperature: Sampling temperature
        max_iterations: Max agent steps
        cost_limit: Max cost for agent
        timeout: Timeout for commands
        dockerhub_username: Docker Hub username for images
        run_script: Test run script content
        parser_script: Test parser script content
        env_cmds: Environment setup commands
        test_files_str: Comma-separated test files
        before_repo_set_cmd: Command to run before repo setup

    Returns:
        Bug instance dict with patch, problem_statement, metadata
    """
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    instance_id = swe_instance["instance_id"]
    repo = swe_instance.get("repo", "")
    base_commit = swe_instance.get("base_commit", "")
    gold_patch = swe_instance.get("patch", "")
    test_patch = swe_instance.get("test_patch", "")

    # Get target tests
    fail_to_pass = swe_instance.get("FAIL_TO_PASS", swe_instance.get("fail_to_pass", "[]"))
    pass_to_pass = swe_instance.get("PASS_TO_PASS", swe_instance.get("pass_to_pass", "[]"))

    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = eval(fail_to_pass)
        except:
            fail_to_pass = []
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = eval(pass_to_pass)
        except:
            pass_to_pass = []

    # Select target tests (prioritize fail_to_pass)
    rng = random.Random(seed)
    weighted_pool = list(fail_to_pass) * 2 + list(pass_to_pass)
    rng.shuffle(weighted_pool)

    target_tests = []
    seen = set()
    for t in weighted_pool:
        if t not in seen:
            target_tests.append(t)
            seen.add(t)
        if len(target_tests) >= 5:
            break

    # Build bug type descriptions
    bug_descriptions = []
    for bug_type in bug_types:
        bug_info = BUG_TYPE_DESCRIPTIONS.get(bug_type, {})
        desc = bug_info.get("description", bug_type)
        examples = bug_info.get("examples", [])
        bug_descriptions.append(f"- {bug_type}: {desc}")
        for ex in examples[:2]:
            bug_descriptions.append(f"  Example: {ex}")

    # Prepare test patch snippet (truncated)
    test_patch_snippet = test_patch[:4000] if test_patch else ""

    # Get Docker image
    from env import get_dockerhub_image_uri
    image = get_dockerhub_image_uri(instance_id, dockerhub_username, repo)

    # Create run_tests.sh that the agent can use
    run_tests_script = f"""#!/bin/bash
cd /app
{env_cmds}
{before_repo_set_cmd}
bash /workspace/run_script.sh {test_files_str} > /workspace/stdout.log 2> /workspace/stderr.log
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

    # Prepare initialization script (runs before agent starts)
    init_script = f"""#!/bin/bash
mkdir -p /workspace
cd /app
git reset --hard {base_commit}
git checkout {base_commit}

# Apply gold_patch (all tests should pass after this)
git apply -v /workspace/gold_patch.diff

# Save current state as "clean" state
git stash
git stash drop 2>/dev/null || true
"""

    # Encode scripts for transfer
    gold_patch_b64 = base64.b64encode(gold_patch.encode('utf-8')).decode('ascii')
    run_script_b64 = base64.b64encode(run_script.encode('utf-8')).decode('ascii') if run_script else ""
    parser_script_b64 = base64.b64encode(parser_script.encode('utf-8')).decode('ascii') if parser_script else ""
    run_tests_b64 = base64.b64encode(run_tests_script.encode('utf-8')).decode('ascii')
    init_script_b64 = base64.b64encode(init_script.encode('utf-8')).decode('ascii')

    # Full setup script that runs first
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

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    breaker_config = config.get("breaker", {})

    # Initialize model
    if breaker_model.startswith(("openai/", "anthropic/", "azure/", "bedrock/")):
        litellm_model_name = breaker_model
    elif breaker_model.startswith("claude"):
        litellm_model_name = breaker_model
    else:
        litellm_model_name = f"openai/{breaker_model}"

    model_kwargs = {"temperature": temperature}
    if seed is not None:
        model_kwargs["seed"] = seed

    is_anthropic = breaker_model.startswith("claude") or breaker_model.startswith("anthropic/")
    if is_anthropic:
        os.environ["ANTHROPIC_API_KEY"] = breaker_api_key
    else:
        if breaker_base_url:
            model_kwargs["api_base"] = breaker_base_url
        model_kwargs["api_key"] = breaker_api_key

    model_obj = LitellmModel(
        model_name=litellm_model_name,
        model_kwargs=model_kwargs,
        cost_tracking="ignore_errors",
    )

    # Initialize Docker environment
    # Note: DockerEnvironment auto-generates container name, don't pass --name in run_args
    # Need --entrypoint "" to override image's /bin/bash entrypoint
    # Container needs to stay alive longer than agent might take
    # timeout is per-command, container_timeout is total container lifetime
    container_lifetime = max(1800, timeout * 10)  # At least 30 min
    env = DockerEnvironment(
        image=image,
        cwd="/app",
        timeout=timeout,
        executable="docker",
        run_args=["--rm", "--entrypoint", ""],
        container_timeout=str(container_lifetime),
    )

    # Pull image first
    print(f"Pulling image: {image}")
    subprocess.run(["docker", "pull", image], capture_output=True, timeout=300)

    # Prepare agent config
    agent_config = {
        "system_template": breaker_config.get("system_template", ""),
        "instance_template": breaker_config.get("instance_template", ""),
        "action_observation_template": breaker_config.get("action_observation_template", ""),
        "format_error_template": breaker_config.get("format_error_template", ""),
        "step_limit": max_iterations,
        "cost_limit": cost_limit,
    }

    # Create agent
    agent = DefaultAgent(model_obj, env, **agent_config)

    # Add extra template vars for the instance template
    agent.extra_template_vars = {
        "repo": repo,
        "bug_types_str": ", ".join(bug_types),
        "bug_descriptions": "\n".join(bug_descriptions),
        "gold_patch": gold_patch,
        "target_tests_str": "\n".join(f"- {t}" for t in target_tests),
        "test_patch_snippet": test_patch_snippet,
    }

    result_text = ""
    error = ""
    exit_status = ""

    try:
        # Run setup first
        print("Setting up breaker environment...")
        setup_output = env.execute(setup_script)
        print(f"Setup output: {setup_output.get('output', '')[:500]}")

        # Run agent
        print(f"Running breaker agent with {breaker_model}...")
        loop = asyncio.get_event_loop()
        exit_status, result_text = await loop.run_in_executor(
            None,
            agent.run,
            "Inject a bug that causes tests to fail"  # This is the task
        )
        print(f"Agent exit_status: {exit_status}")
        print(f"Agent result_text (first 1000 chars): {result_text[:1000] if result_text else 'EMPTY'}")
        print(f"Agent steps: {model_obj.n_calls}, cost: {model_obj.cost}")

        # Debug: print last few agent messages
        if agent.messages:
            print(f"=== Agent messages ({len(agent.messages)} total) ===")
            for i, msg in enumerate(agent.messages[-6:]):  # Last 6 messages
                role = msg.get('role', '?')
                content = str(msg.get('content', ''))[:300]
                print(f"[{i}] {role}: {content}...")

    except Exception as e:
        import traceback
        error = traceback.format_exc()
        print(f"Error running breaker agent: {e}")
        print(f"Full traceback: {error}")

    finally:
        try:
            env.cleanup()
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")

    # Parse result
    bug_patch = ""
    problem_statement = ""
    bug_description = ""

    if result_text:
        # Parse the structured output
        sections = {
            "BUG_PATCH": "",
            "PROBLEM_STATEMENT": "",
            "BUG_DESCRIPTION": "",
        }

        current_section = None
        lines = result_text.split("\n")

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("BUG_PATCH:"):
                current_section = "BUG_PATCH"
                # Check if there's content on the same line
                content = line_stripped[len("BUG_PATCH:"):].strip()
                if content:
                    sections[current_section] = content + "\n"
            elif line_stripped.startswith("PROBLEM_STATEMENT:"):
                current_section = "PROBLEM_STATEMENT"
                content = line_stripped[len("PROBLEM_STATEMENT:"):].strip()
                if content:
                    sections[current_section] = content + "\n"
            elif line_stripped.startswith("BUG_DESCRIPTION:"):
                current_section = "BUG_DESCRIPTION"
                content = line_stripped[len("BUG_DESCRIPTION:"):].strip()
                if content:
                    sections[current_section] = content + "\n"
            elif current_section:
                sections[current_section] += line + "\n"

        bug_patch = sections["BUG_PATCH"].strip()
        problem_statement = sections["PROBLEM_STATEMENT"].strip()
        bug_description = sections["BUG_DESCRIPTION"].strip()

    if not bug_patch:
        raise ValueError(f"Agent failed to generate bug patch. Result: {result_text[:500]}")

    # Build bug instance
    bug_instance = {
        "source": {
            "dataset": "SWE-bench_Pro",
            "swe_instance_id": instance_id,
            "base_commit": base_commit,
            "repo": repo,
        },
        "bug": {
            "bug_types": bug_types,
            "patch": bug_patch,
            "problem_statement": problem_statement,
            "description": bug_description,
        },
        "generation": {
            "breaker_model": breaker_model,
            "seed": seed,
            "temperature": temperature,
            "mode": "agent",
            "agent_steps": model_obj.n_calls,
            "agent_cost": model_obj.cost,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "original": {
            "gold_patch": gold_patch,
            "problem_statement": swe_instance.get("problem_statement", ""),
            "all_tests": list(set(fail_to_pass) | set(pass_to_pass)),
            "fail_to_pass": fail_to_pass,
        },
        "verification": {"verified": False},
    }

    if error:
        bug_instance["generation"]["error"] = error

    return bug_instance
