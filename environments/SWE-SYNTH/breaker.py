"""
SWE-SYNTH Breaker Module

Responsible for generating realistic bugs by injecting faults into correct code.

Flow:
1. Start from gold_patch applied state (all tests pass)
2. Randomly select target tests to break
3. Generate bug that causes selected tests to fail
"""

import os
import json
import time
import random
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
    bug_type: str,
    gold_patch: str,
    problem_statement_original: str,
    repo_files: List[dict] = None,
) -> str:
    """Build the prompt for Breaker model."""

    bug_info = BUG_TYPE_DESCRIPTIONS.get(bug_type, {})
    bug_description = bug_info.get("description", bug_type)
    bug_examples = bug_info.get("examples", [])
    examples_str = "\n".join(f"  - {ex}" for ex in bug_examples)

    # Build source files section
    files_section = ""
    if repo_files:
        files_section = "\n\nSOURCE FILES (choose one to inject bug):\n"
        for f in repo_files:
            files_section += f"\n--- {f['path']} ---\n```\n{f['content'][:2000]}\n```\n"

    return f"""You are a code mutation expert. Inject a realistic "{bug_type}" bug into the codebase.

CONTEXT:
- Repository: {repo}
- Code is CORRECT (all tests pass). Your job is to introduce a bug.

BUG TYPE: {bug_type}
{bug_description}
Examples: {examples_str}
{files_section}
REFERENCE PATCH (shows what was recently fixed - you can inject bug here or in the source files above):
```diff
{gold_patch}
```

INSTRUCTIONS:
1. Choose a location to inject the bug (either in the gold_patch area OR in one of the source files)
2. Create a bug_patch in unified diff format that introduces a {bug_type} bug
3. Write a problem_statement describing symptoms (like a user bug report)

RULES:
- Make subtle, realistic changes (what a tired developer might write)
- Your bug_patch is applied AFTER the gold_patch
- NO syntax errors, NO deleting large code blocks
- problem_statement describes SYMPTOMS users would see, not the bug itself
- The file path in your diff must match the actual file path

OUTPUT: Return ONLY a JSON object (no markdown, no extra text):
{{"bug_patch": "diff --git a/path/to/file.js b/path/to/file.js\\nindex abc..def 100644\\n--- a/path/to/file.js\\n+++ b/path/to/file.js\\n@@ -10,3 +10,3 @@\\n-    correct code\\n+    buggy code", "problem_statement": "User-facing bug description", "bug_description": "Technical description of the bug"}}

Generate the JSON now:"""

async def generate_bug(
    swe_instance: Dict[str, Any],
    bug_type: str,
    seed: int,
    breaker_model: str,
    breaker_base_url: str,
    breaker_api_key: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    repo_files: List[dict] = None,
) -> Dict[str, Any]:
    """
    Generate a bug using Breaker model.

    Args:
        swe_instance: SWE-bench instance data
        bug_type: Type of bug to inject
        seed: Random seed for generation
        breaker_model: Model name for bug generation
        breaker_base_url: API base URL
        breaker_api_key: API key
        temperature: Sampling temperature
        max_retries: Maximum retry attempts for LLM call (default 3)
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

    # Build prompt (no target_tests - we'll discover them through verification)
    prompt = build_breaker_prompt(
        repo=repo,
        bug_type=bug_type,
        gold_patch=gold_patch,
        problem_statement_original=problem_statement_original,
        repo_files=repo_files,
    )

    # Retry loop for LLM call
    import re
    last_error = None
    retry_errors = []

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} generating bug...")
            response = await acompletion(
                model=f"openai/{breaker_model}" if not breaker_model.startswith("openai/") else breaker_model,
                messages=[{"role": "user", "content": prompt}],
                api_base=breaker_base_url,
                api_key=breaker_api_key,
                temperature=temperature,
                seed=seed + attempt,  # Vary seed on retry
            )

            content = response.choices[0].message.content

            # Parse JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                bug_data = json.loads(json_match.group(1))
            else:
                # Try to find JSON object with bug_patch
                json_match = re.search(r'\{[^{}]*"bug_patch"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    bug_data = json.loads(json_match.group(0))
                else:
                    # Try parsing entire content as JSON
                    bug_data = json.loads(content)

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
                    "bug_type": bug_type,
                    "patch": bug_data.get("bug_patch", ""),
                    "problem_statement": bug_data.get("problem_statement", ""),
                    "description": bug_data.get("bug_description", ""),
                    # target_tests will be discovered during verification
                },
                "generation": {
                    "breaker_model": breaker_model,
                    "seed": seed + attempt,
                    "temperature": temperature,
                    "attempt": attempt + 1,
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

            print(f"Bug generated successfully on attempt {attempt + 1}")
            return bug_instance

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            last_error = str(e)
            retry_errors.append({"attempt": attempt + 1, "error": last_error})
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying...")
                continue

    # All retries failed, raise exception
    error_summary = "; ".join([f"Attempt {e['attempt']}: {e['error']}" for e in retry_errors])
    raise RuntimeError(f"Failed to generate bug after {max_retries} attempts: {error_summary}")
