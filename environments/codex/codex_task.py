"""Codex Django bug-fix task generator and evaluator

This module:
1. Uses an LLM to introduce bugs into Django codebase
2. Generates challenges for models to fix those bugs
3. Evaluates fixes by running Django test suite
"""

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import shutil
import httpx
import openai
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

sys.path.insert(0, '/app')
from models import Challenge, BugSpec

logger = logging.getLogger("codex_task")
handler = logging.StreamHandler(sys.stderr)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
logger.addHandler(handler)
logger.setLevel(os.environ.get("CODEX_LOG_LEVEL", "INFO"))

# Django paths
DJANGO_PATH = Path("/django")
DJANGO_TESTS_PATH = DJANGO_PATH / "tests"

# Target Django modules for bug injection (smaller files with good test coverage)
TARGET_MODULES = [
    "django/utils/functional.py",
    "django/utils/encoding.py",
    "django/utils/safestring.py",
]

# Specific functions to target for bug injection (function_name, file_path)
TARGET_FUNCTIONS = [
    ("slugify", "django/utils/text.py"),
    ("capfirst", "django/utils/text.py"),
    ("wrap", "django/utils/text.py"),
    ("normalize_newlines", "django/utils/text.py"),
    ("escape", "django/utils/html.py"),
    ("strip_tags", "django/utils/html.py"),
    ("escapejs", "django/utils/html.py"),
    ("validate_integer", "django/core/validators.py"),
    ("validate_ipv4_address", "django/core/validators.py"),
    ("base36_to_int", "django/utils/http.py"),
    ("int_to_base36", "django/utils/http.py"),
    ("urlencode", "django/utils/http.py"),
]

DEFAULT_BUG_INJECTION_TIMEOUT = 180
DEFAULT_TEST_TIMEOUT = 300


def get_file_content(file_path: Path) -> str:
    """Read file content safely"""
    try:
        return file_path.read_text()
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return ""


def save_file_content(file_path: Path, content: str) -> bool:
    """Save file content safely"""
    try:
        file_path.write_text(content)
        return True
    except Exception as e:
        logger.error(f"Failed to write {file_path}: {e}")
        return False


def reset_django_repo() -> bool:
    """Reset Django repo to clean state using git"""
    try:
        result = subprocess.run(
            ["git", "checkout", "."],
            cwd=DJANGO_PATH,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to reset Django repo: {e}")
        return False


def find_related_tests(module_path: str) -> List[str]:
    """
    Find tests related to a Django module
    
    Args:
        module_path: Path to Django module relative to Django root
    
    Returns:
        List of test module paths
    """
    tests = []
    
    # Map Django modules to their test directories
    module_to_test_mapping = {
        "django/core/validators.py": ["validators"],
        "django/utils/text.py": ["utils_tests"],
        "django/utils/dateformat.py": ["utils_tests"],
        "django/utils/html.py": ["utils_tests"],
        "django/utils/http.py": ["utils_tests"],
        "django/utils/functional.py": ["utils_tests"],
        "django/utils/encoding.py": ["utils_tests"],
        "django/utils/safestring.py": ["utils_tests"],
        "django/template/defaultfilters.py": ["template_tests"],
        "django/forms/fields.py": ["forms_tests"],
    }
    
    test_dirs = module_to_test_mapping.get(module_path, [])
    for test_dir in test_dirs:
        test_path = DJANGO_TESTS_PATH / test_dir
        if test_path.exists():
            tests.append(str(test_path))
    
    return tests


def run_django_tests(test_paths: List[str], timeout: int = 300) -> Tuple[bool, str]:
    """
    Run Django tests and return pass/fail status
    
    Args:
        test_paths: List of test paths to run
        timeout: Test timeout in seconds
    
    Returns:
        Tuple of (tests_passed, test_output)
    """
    if not test_paths:
        # Run a subset of core tests if no specific tests provided
        test_paths = ["utils_tests", "validators"]
    
    try:
        # Run Django test suite
        result = subprocess.run(
            [sys.executable, "tests/runtests.py", "--verbosity=2"] + test_paths,
            cwd=DJANGO_PATH,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout + "\n" + result.stderr
        
        # Check for test success
        # Django reports "OK" at the end if all tests pass
        tests_passed = result.returncode == 0 and "OK" in output
        
        return tests_passed, output
    except subprocess.TimeoutExpired:
        return False, f"Tests timed out after {timeout}s"
    except Exception as e:
        return False, f"Failed to run tests: {e}"


async def call_llm(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 120,
    temperature: float = 0.7,
) -> Tuple[str, Optional[Dict]]:
    """
    Call LLM API to get a response
    
    Args:
        prompt: The prompt to send
        api_key: API key
        base_url: API base URL
        model: Model name
        timeout: Request timeout
        temperature: Generation temperature
    
    Returns:
        Tuple of (response_text, usage_dict)
    """
    # Unset SSL cert env vars to use defaults
    os.environ.pop('SSL_CERT_FILE', None)
    os.environ.pop('REQUESTS_CA_BUNDLE', None)
    
    client = openai.AsyncOpenAI(
        base_url=base_url.rstrip('/'),
        api_key=api_key,
        timeout=httpx.Timeout(timeout),
        max_retries=2
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    
    content = response.choices[0].message.content
    usage = response.usage.model_dump() if response.usage else None
    
    return content.strip() if content else "", usage


class CodexTask:
    """Django bug-fix task generator and evaluator"""
    
    def __init__(
        self,
        django_path: Path = DJANGO_PATH,
        target_modules: List[str] = None,
    ):
        self.django_path = django_path
        self.target_modules = target_modules or TARGET_MODULES
        
        # Verify Django is available
        if not self.django_path.exists():
            raise RuntimeError(f"Django not found at {self.django_path}")
        
        logger.info(f"CodexTask initialized with Django at {self.django_path}")
    
    def _extract_function(self, content: str, func_name: str) -> Optional[Tuple[str, int, int]]:
        """
        Extract a function definition from file content using regex
        
        Returns: Tuple of (function_code, start_line, end_line) or None
        """
        lines = content.split('\n')
        in_function = False
        func_start = -1
        func_lines = []
        base_indent = 0
        
        for i, line in enumerate(lines):
            if not in_function:
                # Look for function definition
                stripped = line.lstrip()
                if stripped.startswith(f'def {func_name}('):
                    in_function = True
                    func_start = i
                    base_indent = len(line) - len(stripped)
                    func_lines.append(line)
            else:
                # Check if we're still in the function
                if not line.strip():  # Empty line
                    func_lines.append(line)
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                
                # If line is more indented than function base, it's part of function
                if current_indent > base_indent:
                    func_lines.append(line)
                # If it's a decorator or new definition at same/lower indent, we're done
                elif line.lstrip().startswith(('@', 'def ', 'class ')):
                    break
                else:
                    # Any other line at base indent or less ends the function
                    break
        
        if func_lines:
            return '\n'.join(func_lines), func_start, func_start + len(func_lines)
        return None
    
    async def inject_bug(
        self,
        api_key: str,
        base_url: str = "https://llm.chutes.ai/v1",
        model: str = "deepseek-ai/DeepSeek-V3",
        target_module: str = None,
        seed: int = None,
    ) -> Optional[BugSpec]:
        """
        Use LLM to introduce a bug into Django
        
        Args:
            api_key: Chutes API key
            base_url: Chutes API base URL
            model: Model to use for bug injection
            target_module: Specific module to target (random if None)
            seed: Random seed for reproducibility
        
        Returns:
            BugSpec describing the introduced bug, or None if failed
        """
        if seed is not None:
            random.seed(seed)
        
        # Reset Django to clean state
        reset_django_repo()
        
        # Select a random target function
        func_name, target_module = random.choice(TARGET_FUNCTIONS)
        
        target_path = self.django_path / target_module
        if not target_path.exists():
            logger.error(f"Target module not found: {target_path}")
            return None
        
        # Store original content
        original_content = get_file_content(target_path)
        if not original_content:
            return None
        
        # Extract the target function
        func_extract = self._extract_function(original_content, func_name)
        if not func_extract:
            logger.error(f"Could not extract function {func_name} from {target_module}")
            return None
        
        func_code, func_start, func_end = func_extract
        
        # Create bug injection prompt (focused on a single function)
        bug_prompt = f"""You are tasked with introducing a subtle bug into a Python function for testing purposes.

Here is the function `{func_name}` from Django's `{target_module}`:

```python
{func_code}
```

Your task: Introduce ONE subtle bug that will cause tests to fail.

Requirements:
1. The bug must be SUBTLE - not an obvious syntax error
2. The bug must be REALISTIC - something a developer might accidentally introduce
3. Make EXACTLY ONE small change

Examples of good bugs:
- Off-by-one errors in loops or slices
- Wrong comparison operators (< vs <=, == vs !=)
- Missing edge case handling
- Incorrect string operations
- Wrong variable used

Respond with ONLY the modified function wrapped in a Python code block.
Do NOT include any explanation, just the code block with the buggy function.

```python
{func_code[:50]}...
```"""

        # Call LLM to inject bug
        logger.info(f"Injecting bug into {func_name} in {target_module} using LLM")
        try:
            response, _ = await call_llm(
                prompt=bug_prompt,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=DEFAULT_BUG_INJECTION_TIMEOUT,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
        
        # Extract buggy function from response
        buggy_func = self._extract_code(response)
        if not buggy_func:
            logger.warning("Could not extract code from LLM response")
            logger.debug(f"Response was: {response[:500]}")
            return None
        
        # Verify a change was made
        if buggy_func.strip() == func_code.strip():
            logger.warning("LLM did not make any changes")
            return None
        
        # Reconstruct the full file with the buggy function
        lines = original_content.split('\n')
        buggy_lines = lines[:func_start] + buggy_func.split('\n') + lines[func_end:]
        buggy_content = '\n'.join(buggy_lines)
        
        # Verify it's valid Python by trying to compile
        try:
            compile(buggy_content, target_module, 'exec')
        except SyntaxError as e:
            logger.warning(f"Generated code has syntax errors: {e}")
            return None
        
        # Apply the buggy code
        if not save_file_content(target_path, buggy_content):
            return None
        
        # Find related tests
        related_tests = find_related_tests(target_module)
        
        # Create bug specification
        bug_spec = BugSpec(
            file_path=target_module,
            original_content=original_content,
            buggy_content=buggy_content,
            bug_description=f"Bug introduced in {func_name} in {target_module}",
            affected_tests=related_tests
        )
        
        logger.info(f"Bug injected successfully into {func_name}")
        return bug_spec
    
    async def generate(
        self,
        bug_spec: BugSpec,
        task_id: int = None,
    ) -> Challenge:
        """
        Generate a bug-fix challenge from a bug specification
        
        Args:
            bug_spec: The bug specification
            task_id: Optional task ID for tracking
        
        Returns:
            Challenge object with prompt and metadata
        """
        # Create the bug fix prompt
        prompt = f"""You are tasked with fixing a bug in the Django web framework.

A bug has been introduced in the following file:
`{bug_spec.file_path}`

Here is the current (buggy) content of the file:

```python
{bug_spec.buggy_content}
```

The bug is causing test failures. Your task is to:
1. Identify the bug in the code
2. Provide the corrected code

Please respond with the complete fixed file content wrapped in a Python code block.
Do not include any explanations outside the code block - just provide the fixed code.

```python
<complete fixed file>
```"""

        return Challenge(
            env="codex",
            prompt=prompt,
            extra={
                "task_id": task_id,
                "file_path": bug_spec.file_path,
                "original_content": bug_spec.original_content,
                "buggy_content": bug_spec.buggy_content,
                "bug_description": bug_spec.bug_description,
                "affected_tests": bug_spec.affected_tests,
            }
        )
    
    async def evaluate(
        self,
        response: str,
        challenge: Challenge,
        timeout: int = DEFAULT_TEST_TIMEOUT,
    ) -> Tuple[float, str]:
        """
        Evaluate a bug-fix response by running tests
        
        Args:
            response: The model's response with fixed code
            challenge: The original challenge
            timeout: Test timeout in seconds
        
        Returns:
            Tuple of (score, result_summary)
        """
        file_path = challenge.extra.get("file_path", "")
        original_content = challenge.extra.get("original_content", "")
        affected_tests = challenge.extra.get("affected_tests", [])
        
        if not file_path or not original_content:
            return 0.0, "Invalid challenge metadata"
        
        # Extract code from response
        fixed_code = self._extract_code(response)
        if not fixed_code:
            return 0.0, "No code found in response"
        
        target_path = self.django_path / file_path
        
        # Apply the fix
        if not save_file_content(target_path, fixed_code):
            return 0.0, "Failed to apply fix"
        
        try:
            # Run tests
            tests_passed, test_output = run_django_tests(
                affected_tests,
                timeout=timeout
            )
            
            if tests_passed:
                return 1.0, "1/1 - All tests passed"
            else:
                # Check if the fix at least restored original functionality
                if fixed_code.strip() == original_content.strip():
                    return 1.0, "1/1 - Code restored to original"
                
                return 0.0, f"0/1 - Tests failed:\n{test_output[-500:]}"
        
        finally:
            # Always reset to original state
            reset_django_repo()
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from model response"""
        # Remove thinking/reasoning tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        
        # Extract code from markdown code blocks
        code_block_match = re.search(
            r"```(?:python)?\n?(.*?)\n?```",
            response,
            flags=re.DOTALL
        )
        
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # If no code block, check if the entire response looks like Python
        lines = response.strip().split('\n')
        if lines and (
            lines[0].startswith(('import ', 'from ', '#', '"""', "'''")) or
            any(line.startswith('def ') or line.startswith('class ') for line in lines[:20])
        ):
            return response.strip()
        
        return None


async def main():
    """Test the CodexTask implementation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Django bug-fix task")
    parser.add_argument("--api-key", help="Chutes API key")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="Model to use")
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("CHUTES_API_KEY")
    if not api_key:
        print("Error: CHUTES_API_KEY required")
        sys.exit(1)
    
    task = CodexTask()
    
    # Test bug injection
    print("Injecting bug...")
    bug_spec = await task.inject_bug(
        api_key=api_key,
        model=args.model,
        seed=42
    )
    
    if bug_spec:
        print(f"Bug injected into: {bug_spec.file_path}")
        
        # Generate challenge
        challenge = await task.generate(bug_spec)
        print(f"\nChallenge prompt:\n{challenge.prompt[:500]}...")
    else:
        print("Bug injection failed")


if __name__ == "__main__":
    asyncio.run(main())
