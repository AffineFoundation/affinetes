from __future__ import annotations
import json
import os
import requests
import subprocess
import sys
import textwrap

import time
import traceback
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import logging
from uuid import uuid4
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel

run_id = None
agent_start_time = None
_current_tool_manager = None

# Global tracking variables for modified files (persists across workflow)
SOLUTION_FILES = []
MODIEFIED_FILE_PATHS = []
MODIFIED_FILES = []
total_inferenced_chars = 0
individual_inferenced_chars = 0

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
ACTUAL_MODEL = os.getenv("ACTUAL_MODEL", "unknown")
PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"
PROBLEM_TYPE_BUG_INJECTION = "BUG_INJECTION"

# Model class with individual timeout settings
class Model(BaseModel):
    name: str
    timeout: int

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

MAX_FIX_TASK_STEPS = 200
LATEST_OBSERVATIONS_TO_KEEP = 15
SUMMARIZE_BATCH_SIZE = 5
MAX_SUMMARY_RANGES = 6

# Model definitions with individual timeout settings
GLM_MODEL_NAME = Model(name="zai-org/GLM-4.6-FP8", timeout=150)
GLM_OLD_MODEL_NAME = Model(name="zai-org/GLM-4.5-FP8", timeout=150)
QWEN_MODEL_NAME = Model(name="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", timeout=100)
KIMI_MODEL_NAME = Model(name="moonshotai/Kimi-K2-Instruct", timeout=60)
DEEPSEEK_MODEL_NAME = Model(name="deepseek-ai/DeepSeek-V3-0324", timeout=50)

# Fallback mappings for model redundancy
KIMI_MODEL_NAME = QWEN_MODEL_NAME
DEEPSEEK_MODEL_NAME = GLM_MODEL_NAME
GLM_OLD_MODEL_NAME = GLM_MODEL_NAME

AGENT_MODELS = [model for model in [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME] for _ in range(2)]

FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent(
    """
Now let's start.
```
{problem_statement}
```
"""
)

DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent(
    """
You are making same mistakes.
Your previous response: 
{previous_response}

**Critical**:
1. Notice what you are going to do.
2. Find the reason the same mistake is repeated.
3. Don't make the same mistakes any more and make a real progress.
"""
)

DONT_REPEAT_TOOL_CALL = textwrap.dedent("""
**CRITICAL - DO NOT REPEAT THE SAME TOOL CALL**

You just called the following tool in the previous step:

Tool Name: {last_tool_name}
Tool Arguments: {last_tool_args}

**DO NOT call the same tool with the same arguments again.** This will cause redundant execution and waste steps.

Instead:
- If you need to explore more, use a different tool or different arguments
- If you need to read a different file, use get_file_content with a different file_path
- If you need to search for something else, use grep_search with different search_term or route
- Move forward with your analysis and make progress toward fixing the issue

Remember: Each step should make progress. Repeating the exact same tool call does not make progress.
""")

STOP_INSTRUCTION = textwrap.dedent(
    """
# 🎯 RESPONSE REQUIREMENTS
- DO NOT generate `observation:` - it will be provided by the system
- You can make MULTIPLE tool calls in one response using tool_call_1, tool_call_2, tool_call_3, etc.
- For efficiency: Batch related operations together (e.g., edit + test in ONE response)
- Format: next_thought: ... followed by one or more tool_call_N blocks
"""
)


FIX_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
Role: You are a senior bug-fix engineer working on an open-source repository.

You will be tasked to fix an issue from this repository.

Your thinking should be thorough and so it's fine if it's very long. You should think step by step before and after each action you decide to take.

You already have everything you need to solve this problem in the repository, even without internet connection.

Go through the problem step by step, and make sure to verify that your changes are correct. NEVER GIVE UP without having solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.

THE PROBLEM CAN DEFINITELY BE SOLVED WITHOUT THE INTERNET.
                                         
Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.                   

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

# Workflow

## High-Level Problem Solving Strategy

1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. **MANDATORY**: Generate test cases from root cause using `generate_test_cases_from_root_cause` BEFORE creating test files.
6. Debug as needed. Use debugging techniques to isolate and resolve issues.
7. Test frequently. Run tests after each change to verify correctness.
8. Iterate until the root cause is fixed and all tests pass.
9. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete.

Refer to the detailed sections below for more information on each step.
                                         
## 1. Deeply Understand the Problem
Carefully read the issue and think hard about a plan to solve it before coding.

## 2. Codebase Investigation
**CRITICAL: Find working examples first, then identify what's broken.**

- Search for key terms from the issue throughout the codebase
- Find similar functionality that WORKS correctly - this is your template
- Study how working code accomplishes what you need
- Locate the broken code using same keywords
- Look beyond surface symptoms - search in domains, helpers, utilities, base classes
- Trace to where mechanisms are actually DEFINED, not just where they're called
- Find the ROOT files where functionality is implemented

**Trace from final output backwards to root cause:**
- Start with working feature's final output, trace backwards to find generator
- Start with broken feature's final output, trace backwards to find what's missing or different
- Compare the paths: where do they diverge?
- Don't stop at the first file you find - keep tracing back to where the behavior originates

- Read and understand relevant code snippets
- Compare working vs broken code: what's different? Missing calls? Missing imports?
- Identify the root cause by finding what working code does that broken code doesn't
- Validate and update your understanding continuously as you gather more context

## 3. Root Cause Verification
**Before implementing any fix, verify you understand the root cause:**

**Trace the COMPLETE data flow for both working and broken:**
1. Find similar WORKING feature
2. Trace working feature through all stages from start to final output
3. Trace broken feature through all stages from start to final output
4. Find EXACT point where paths diverge

**Compare working vs broken at EACH stage:**
- What does working code do that broken code doesn't?
- What functions are called? What imports exist?
- Where does the behavior differ?
- Keep tracing backwards until you find the root cause

**Find root, not symptoms:**
- Don't patch surface symptoms - find the missing or different mechanism
- Trace all the way back to where the behavior originates
- The fix location may be far from where symptoms appear
- Compare: How does working feature accomplish the task? How does broken feature differ?

**Search comprehensively:**
- Is this pattern missing in multiple places? Search the whole repository
- Are there similar files/classes that need the same fix?
- Fix all instances, not just the one example in the issue

## 4. Develop a Detailed Plan
- Outline a specific, simple, and verifiable sequence of steps to fix the problem
- Break down the fix into small, incremental changes
- Think through all the steps you need to take to fix the problem

## 5. Making Code Changes
**Copy patterns from working code. Make minimal focused changes.**

- Before editing, always read the relevant file contents or section to ensure complete context
- If a patch is not applied correctly, attempt to reapply it
- **Use the EXACT same pattern as working code**: same functions, same imports, same structure
- Make small, testable, incremental changes that logically follow from your investigation
- **Search for similar locations**: Is this pattern needed elsewhere? Fix all instances if it's systemic
- Keep changes minimal and focused - don't refactor or change unrelated code

## 6. Debugging
**CRITICAL: Fix root cause, not symptoms. Search broadly across the repository.**

- Make code changes only if you have high confidence they can solve the problem
- When debugging, determine the ROOT CAUSE rather than addressing surface symptoms
- Don't just patch the calling code - trace back to where the mechanism is defined
- Trace from working feature backwards to find where behavior is implemented
- The fix location is often far from where the problem is first noticed

**Search across the entire repository:**
- Broadly search like domain logic files, helper/utility modules, base classes, configuration files, handler classes...
- Look beyond the obvious files mentioned in error messages

- Look for similar patterns that might need the same fix in multiple locations
- Debug for as long as needed to identify the root cause and identify a fix
- Use print statements, logs, or temporary code to inspect program state, including descriptive statements or error messages to understand what's happening
- To test hypotheses, you can also add test statements or functions
- Revisit your assumptions if unexpected behavior occurs.

## 6. Testing
- Run tests frequently using the available testing tools (for example, by calling the `run_code` tool).
- After each change, verify correctness by running relevant tests via the testing tool rather than invoking shell commands directly.
- If tests fail, analyze failures and revise your patch.
- Write additional tests if needed to capture important behaviors or edge cases.
- Ensure all tests pass before finalizing.

## 7. Final Verification
- Confirm the root cause is fixed.
- Review your solution for logic correctness and robustness.
                                         
## 8. Final Reflection and Additional Testing
- Reflect carefully on the original intent of the user and the problem statement.
- Think about potential edge cases or scenarios that may not be covered by existing tests.
- Write additional tests that would need to pass to fully validate the correctness of your solution.
- Run these new tests and ensure they all pass.
- Be aware that there are additional hidden tests that must also pass for the solution to be successful.
- Do not assume the task is complete just because the visible tests pass; continue refining until you are confident the fix is robust and comprehensive.

# Tool Documentation
You have access to the following tools:-
{tools_docs}
                                         
# Tool Usage Guidelines
- Use appropriate tools to gather context before making changes.
- If required parameters are missing, infer them from the problem statement and code.
- Use exact values provided by the user (especially in quotes).
- Don't make up values for or ask about optional parameters.
- Use `grep_search` to find all occurrences of an issue before fixing.

# Meta-Cognitive Checkpoints
Every 15 steps, you will receive a META-COGNITIVE CHECKPOINT that analyzes your recent activity and progress:
- **Progress Analysis**: Shows what tools you've used and whether you're making measurable progress
- **Pattern Detection**: Alerts you if you're stuck in repetitive behavior (e.g., using same tools repeatedly)
- **Mandatory Reflection**: You MUST address these reflection questions in your next_thought:
  1. Am I measurably closer to solving this problem than 15 steps ago?
  2. Is my current approach working, or am I stuck in a loop?
  3. What is the ONE most important thing to do next?

**How to respond to meta-cognitive prompts:**
- Honestly evaluate your progress with concrete evidence (not assumptions)
- If you haven't made progress, identify which assumption was WRONG
- If stuck in a pattern, CHANGE your approach (different files, different strategy, or rollback)
- Be specific about what you'll learn from your next action that you don't already know

**Critical**: These checkpoints exist to prevent wasted effort. Take them seriously and be willing to pivot when not making progress.

# Cognitive Tools for Knowledge Persistence

You have access to powerful cognitive tools designed to preserve knowledge across rollbacks and prevent retry loops:

## Strategy Memory

**Purpose**: Remember what approaches you've tried, even after rolling back changes.

**Tools**:
- **log_strategy(approach, reasoning)**: Record planned approach BEFORE implementing
  - Use when: About to make significant code changes
  - Example: "Update function in <file> at line <N>" because "this fixes the root cause"

- **mark_strategy_outcome(strategy_id, success, reason)**: Record whether it worked
  - Use when: After testing the strategy (tests pass/fail)
  - Example: Mark strategy #1 as failed: "Tests passed but broke edge case in rare input scenario"

- **list_attempted_strategies()**: Review all strategies and outcomes
  - Use when: After rollbacks (to see what doesn't work), during reflection, or when choosing next approach
  - Shows: Which strategies succeeded/failed/pending

**When to Use These Tools**:

1. **Before Making Changes** (Before edits):
   - Use `log_strategy` to record your planned approach

2. **After Testing** (After running tests):
   - Use `mark_strategy_outcome` to record whether strategy worked

3. **During Meta-Cognitive Checkpoints** (Every 15 steps):
   - Use `list_attempted_strategies` to avoid retrying failed approaches

4. **After Rollbacks**:
   - IMMEDIATELY use `list_attempted_strategies` to see what you tried
   - This prevents retry loops since file state resets but cognitive state persists

**Critical**: These tools create institutional memory that survives rollbacks. Use them consistently to avoid wasting effort.

## Hypothesis Tracking (Enhanced Feedback Loop)

**Purpose**: Track theories about the bug systematically and test them methodically.

**Tools**:
- **create_hypothesis(description, evidence)**: Record a theory about the root cause
  - Use when: You have a theory about what's causing the bug
  - Example: "Missing null check in parse_config" with evidence "Line 45 doesn't handle None input"

- **test_hypothesis(hypothesis_id, outcome, findings)**: Record whether hypothesis was confirmed/rejected
  - Outcomes: 'confirmed', 'rejected', or 'inconclusive'
  - Example: Mark hypothesis #1 as "confirmed" with findings "Added null check and tests pass"

- **list_hypotheses()**: Review all hypotheses and their status
  - Use when: Choosing which theory to investigate next, or after rollbacks

## Test Progress Tracking

**Purpose**: Monitor your testing progress and get actionable feedback on failures.

**Tools**:
- **get_test_progress()**: Get summary of test pass/fail rates and patterns
  - Shows: Current streak, common failure types, trend analysis

- **analyze_test_failure(test_output)**: Get detailed analysis of a failed test
  - Returns: Failure type, specific suggestions, pattern detection

**Enhanced Feedback Loop Workflow**:

1. **When investigating a bug**:
   - Create hypotheses about possible root causes
   - Test each hypothesis systematically
   - Record findings even if hypothesis is rejected

2. **After each test run**:
   - System automatically tracks test results
   - If consecutive failures detected, feedback will be injected automatically
   - Use `get_test_progress()` to see overall trends

3. **When stuck after multiple failures**:
   - Use `list_hypotheses()` to see what theories you've tested
   - Use `analyze_test_failure()` to understand the current failure
   - Consider if you need to create new hypotheses or revisit rejected ones

4. **Responding to feedback alerts**:
   - When you see "🔔 FEEDBACK LOOP ALERT", take it seriously
   - Review your approach and consider changing strategy
   - Don't ignore consecutive failure warnings

# Critical Requirements
- Fix must be backward compatible unless stated otherwise.
- Ensure changes are exhaustive and don't break other functionality.
- Don't edit test files directly - use the dedicated test generation tool when needed.
- Don't create new files unless absolutely necessary.
- Check both expected output in the problem statement AND in relevant test cases.

# Step Efficiency
You have a limited step budget (target: 10 steps, maximum: 20 steps). Prioritize simpler, faster solutions and make forward progress with each step. Test frequently to catch issues early. Don't over-investigate - once you understand the issue, implement the fix.

Here is the problem statement:
{problem_statement}

# Response Format Requirements
{format_prompt}
"""
)

FORMAT_PROMPT_FIX = textwrap.dedent(
    """
**CRITICAL: You can make MULTIPLE tool calls in ONE response for efficiency!**
## Response Formats
### Format 1: Multiple Tool Calls (RECOMMENDED for efficiency)
next_thought: [Your detailed reasoning]
tool_call_1:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_2:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_3:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
### Format 2: Single Tool Call (Legacy, less efficient)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}
## When to Use Multiple Tool Calls
**ALWAYS batch these operations:**
1. **Edit + Test**: After code edit, MUST test in same response
2. **Multiple Searches**: Batch all search operations together
3. **Multiple File Reads**: Read all needed files at once
4. **Multiple Tests**: Run all test files together
## Examples
✅ **Excellent - Edit and Test Together**:
next_thought: I'll fix the bug and immediately verify with tests
tool_call_1:
    tool_name: apply_code_edit
    tool_args: {"file_path": "abcd.py", "search": "old_code", "replace": "fixed_code"}
tool_call_2:
    tool_name: run_code
    tool_args: {"content": "test_content", "file_path": "file.js", "run_command": ["node", "file.js"]}
✅ **Good - Batch Multiple Searches**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'function problematic_func' ."}
tool_call_2:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'problematic_func(' ."}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "abcd.js"}
❌ **Bad - One tool per response (too slow)**:
Response 1:
next_thought: Let me edit the file
next_tool_name: apply_code_edit
next_tool_args: {"file_path": "aaa.py", ...}
Response 2 (next turn):
next_thought: Now let me test it
next_tool_name: run_code
...  # ← Should have been in previous response!
## Critical Rules
- Use multiple tool_call_N when possible (tool_call_1, tool_call_2, tool_call_3, ...)
- After any edit: MUST include test in same response
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
"""
)

FORMAT_PROMPT_CREATE = textwrap.dedent(
    """
**Default: Use single tool call format. Use multiple tool calls ONLY when searching multiple files at once for time efficiency.**

## Response Formats

### Format 1: Single Tool Call (DEFAULT - Use this for most operations)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}

### Format 2: Multiple Tool Calls (ONLY for multi-file searches)
next_thought: [Your detailed reasoning]
tool_call_1:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_2:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_3:
    tool_name: [exact tool name]
    tool_args: {valid JSON}

## When to Use Multiple Tool Calls

**ONLY use multiple tool calls when:**
- Searching multiple files at once (e.g., codebase_search on multiple files/directories simultaneously)

**Examples:**

✅ **Good - Multiple file searches (time efficient)**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'function function_name' ."}
tool_call_2:
    tool_name: grep_search
    tool_args: {"grep_search_command": "grep -r 'function_name(' ."}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "file_name.js"}

✅ **Good - Single tool call (default)**:
next_thought: I'll read this file to understand the code
next_tool_name: get_file_content
next_tool_args: {"file_path": "aaa.py"}

✅ **Good - Single tool to edit file**:
next_thought: I'll edit the file
next_tool_name: apply_code_edit
next_tool_args: {"file_path": "aaa.py", "search": "old_code", "replace": "new_code"}

✅ **Good - Single tool call to verify**:
next_thought: I'll run a command to verify the changes
next_tool_name: run_tests
next_tool_args: {"command": ["node", "file.js"], "timeout": 5}

## Critical Rules
- Default to single tool call format (next_tool_name, next_tool_args)
- Use multiple tool calls ONLY for parallel multi-file searches
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
"""
)


VERSION_COMPATIBILITY_FIX = """
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;
"""

class TestManager:
    def __init__(self, runner_hint: str | None = None, runner_mode_hint: str | None = None, file_ops: "FileOperationsUtil" = None):
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.file_ops = file_ops

    def run_code(self, content: str, file_path: str, generated_test_files: list, run_command: list[str]) -> str:
        if file_path.endswith((".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz")):
            content = VERSION_COMPATIBILITY_FIX + "\n\n" + content
        file_exists = os.path.exists(file_path) and os.path.isfile(file_path)
        self.file_ops.save(file_path, content)
        if file_path not in generated_test_files and not file_exists:
            generated_test_files.append(file_path)
        try:
            result = subprocess.run(run_command, capture_output=True, text=True, check=False, timeout=60)
            if result.returncode != 0:
                return f"Error running code: {result.stderr}"
            return f"{result.stdout}\n"
        except Exception as e:
            return f"Error: {e}"


class SearchManager:
    def search_in_all_files(self, grep_search_command: str) -> str:
        cmd = grep_search_command.lstrip()
        if not cmd.startswith("grep"):
            return f"Error: Invalid command. Expected a grep command but got: '{grep_search_command}'"
        try:
            result = subprocess.run(["bash", "-c", grep_search_command], capture_output=True, text=True, timeout=45)
        except Exception as e:
            return f"Error: Failed to execute grep command: {e}"
        if result.returncode > 1:
            error_msg = result.stderr.strip() or "Unknown error"
            return f"Error: Grep command failed with return code {result.returncode}: {error_msg}"
        output = result.stdout

        if not output.strip():
            return "No matches found for pattern in codebase."
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output

    def search_in_file(self, file_path: str, search_term: str) -> str:
        def extract_matches(filepath, term, max_output_lines=1000):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except Exception as e:
                return f"Error reading '{filepath}': {e}"

            # NOTE: Use literal substring matching. Using re.escape(term) and then searching for
            # that escaped string breaks common queries like "." -> "\\." which won't exist in source lines.
            match_lines = [i + 1 for i, line in enumerate(lines) if term in line]
            if not match_lines:
                return f"'{term}' not found in file '{filepath}'"

            context = 20
            seen = set()
            chunks = []
            for ln in match_lines:
                start = max(1, ln - context)
                end = min(len(lines), ln + context)
                rkey = (start, end)
                if rkey in seen:
                    continue
                seen.add(rkey)
                chunk = lines[start - 1 : end]
                chunks.append(f"(lines {start}-{end}):\n" + "\n".join(chunk))
            return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output


class EnhancedNetwork:
    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        tool_name_match = re.search(r"tool_name\s*:\s*([^\s]+)", block, re.IGNORECASE)
        if not tool_name_match:
            return None
        tool_name = tool_name_match.group(1).strip("\"'")
        args_match = re.search(r"tool_args\s*:\s*\{", block, re.IGNORECASE)
        if not args_match:
            return None
        args_start = args_match.end() - 1
        json_str = cls._extract_balanced_braces(block, args_start)
        if json_str:
            try:
                tool_args = json.loads(json_str)
                return {"tool_name": tool_name, "tool_args": tool_args}
            except json.JSONDecodeError:
                try:
                    tool_args = json.loads(json_str.replace("'", '"'))
                    return {"tool_name": tool_name, "tool_args": tool_args}
                except Exception:
                    pass
        return None

    @classmethod
    def _extract_balanced_braces(cls, text: str, start_pos: int) -> str | None:
        if start_pos >= len(text):
            return None
        brace_count, in_string, escape_next, start = 0, False, False, -1
        for i in range(start_pos, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    if start == -1:
                        start = i
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        return text[start : i + 1]
        return None

    @classmethod
    def get_cost_usage(cls) -> dict:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/usage?evaluation_run_id={run_id if run_id else str(uuid4())}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            usage_info = response.json()
            if isinstance(usage_info, dict):
                return usage_info
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
        except Exception:
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}

    @classmethod
    def inference(
        cls,
        messages: list[dict],
        model: str,
        run_id: str = str(uuid4()),
        temperature: float = 0.0,
    ) -> dict:
        models = [model] if isinstance(model, str) else model
        cleaned_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in {"system", "user", "assistant", "tool"} and (m.get("role") != "assistant" or m.get("content", "").strip())
        ]
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        result = cls._request_next_action_with_retry(cleaned_msgs, models=models, temperature=temperature)
        return result

    @classmethod
    def is_http_response(cls, raw_text: str):
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if "HTTP ERROR: Request failed for model" in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def make_request(
        cls,
        messages: list,
        model: Model,
        attempt: int = 0,
        temperature: float = 0.0,
        tool_mode: str = "none",
        tool_docs: list = [],
    ) -> tuple[str, list]:
        global run_id, agent_start_time, total_inferenced_chars, individual_inferenced_chars
        messages_str = json.dumps(messages, ensure_ascii=False)
        individual_inferenced_chars = len(messages_str)
        total_inferenced_chars += individual_inferenced_chars
        
        elapsed_time = time.time() - agent_start_time if agent_start_time else 0
        if elapsed_time > 1300:
            raise RuntimeError(f"Agent execution timeout after {elapsed_time} seconds")
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        attempts = max(1, attempt or 1)
        
        # Handle both Model class and string for backwards compatibility
        model_name = model.name if isinstance(model, Model) else model
        model_timeout = model.timeout if isinstance(model, Model) else 150
        
        request_data = {
            "evaluation_run_id": run_id if run_id else str(uuid4()),
            "messages": messages,
            "temperature": temperature,
            "model": model_name,
            "tool_mode": tool_mode,
            "tools": tool_docs,
        }
        headers = {"Content-Type": "application/json"}
        for i in range(attempts):
            try:
                start_time = time.time()
                print(f"⏳ Sending request using {ACTUAL_MODEL} and {model_timeout} seconds timeout")
                resp = requests.post(url, json=request_data, timeout=(30, model_timeout), headers=headers)
                resp.raise_for_status()
                print(f"✔ Request success using {ACTUAL_MODEL} and {time.time() - start_time:.2f} seconds elapsed!")
                try:
                    resp_json = resp.json()
                except JSONDecodeError as e:
                    if i >= attempts - 1:
                        raise ValueError(f"HTTP ERROR: Invalid JSON response for model {model_name} after {attempts} attempts: {e}")
                    continue
                try:
                    raw_text = resp_json["content"]
                    tool_calls = resp_json["tool_calls"]
                except Exception:
                    raise RuntimeError(f"HTTP ERROR: Response Parse Error timeout for model {model_name} after {attempts} attempts")
                if (tool_mode == "none" and not raw_text) or (tool_mode != "none" and not tool_calls):
                    raise RuntimeError(f"HTTP ERROR: NO RESPONSE FOUND Tool model {model_name} after {attempts} attempts")
                return raw_text, tool_calls
            except requests.exceptions.Timeout:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request timeout for model {model_name} after {attempts} attempts")
                time.sleep(1)
            except requests.exceptions.ConnectionError as e:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Connection error for model {model_name} after {attempts} attempts: {e}")
                time.sleep(1)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(f"HTTP ERROR 504: Gateway Timeout for model {model_name} after {attempts} attempts: {e}")
                    time.sleep(1)
                    continue
                error_msg = f"HTTP ERROR: HTTP ERROR {status_code} for model {model_name}"
                if i >= attempts - 1:
                    raise RuntimeError(f"{error_msg} after {attempts} attempts: {e}")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request failed for model {model_name} after {attempts} attempts: {e}")
                time.sleep(1)
        raise RuntimeError(f"HTTP ERROR: Failed to get response for model {model_name} after {attempts} attempts")

    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern = r",\s*".join(rf'"{k}": (.*)' for k in arguments)
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        return {k: match.group(i + 1).strip().strip('"').replace("\\n", "\n") for i, k in enumerate(arguments)}

    @classmethod
    def _request_next_action_with_retry(
        cls,
        messages: dict,
        models: list[str],
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> str:
        raw_text = None
        error_counter = cls.get_error_counter()
        next_thought = next_tool_name = next_tool_args = None
        total_attempts = 0
        current_model_idx = 0
        used_model = models[0] if models else None
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                current_model = models[min(current_model_idx, len(models) - 1)]
                used_model = current_model
                raw_text, _ = cls.make_request(messages, model=current_model, temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                is_504_error = "504" in error_body or "HTTP ERROR 504" in error_body or "Gateway Timeout" in error_body
                if "Agent execution timeout" in error_body:
                    raise RuntimeError(error_body)
                if is_504_error and current_model_idx < len(models) - 1:
                    current_model_idx += 1
                    time.sleep(3)
                    continue
                if attempt < max_retries - 1:
                    matched = False
                    for key in ["RATE_LIMIT_EXCEEDED", "RESERVED_TOKEN_PRESENT", "EMPTY_RESPONSE", "TIMEOUT", "Invalid JSON", "Invalid response"]:
                        if key in error_body:
                            attr_name = key if key in cls.ErrorType.__members__ else "INVALID_RESPONSE_FORMAT"
                            error_counter[attr_name] += 1
                            matched = True
                            break
                    if not matched:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    skip_http = any(
                        x in error_body
                        for x in [
                            "HTTP ERROR",
                            "RATE_LIMIT_EXCEEDED",
                            "RESERVED_TOKEN_PRESENT",
                            "EMPTY_RESPONSE",
                            "TIMEOUT",
                            "NETWORK_ERROR",
                            "HTTP ERROR 429",
                            "INCOMPLETE_RESPONSE",
                        ]
                    )
                    if not skip_http:
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(3)
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)
        return (
            next_thought,
            next_tool_name,
            next_tool_args,
            raw_text,
            total_attempts,
            error_counter,
            messages,
            used_model,
        )

    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        try:
            return Utils.load_json(next_tool_args.strip())
        except JSONDecodeError:
            try:
                schema_tool_name = tool_name[0] if isinstance(tool_name, list) and tool_name else tool_name
                return cls.parse_malformed_json(
                    EnhancedToolManager.get_tool_args_for_tool(schema_tool_name, required_only=True),
                    next_tool_args,
                )
            except (EnhancedToolManager.Error, Exception):
                raise Exception(f"Invalid JSON: {next_tool_args}")

    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub(r"['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub(r"['\"]*observation['\"]*:", "observation:", text_resp)
        text_resp = re.sub(r"['\"]*tool_call_['\"]*", "tool_call_", text_resp)
        if (
            "next_thought" not in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
            and text_resp.find("next_tool_name:") > 10
        ):
            text_resp = "next_thought: " + text_resp
        if (
            "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("'").strip('"').strip()
            # Overwrite with normalized line
            text_resp = re.sub(
                f"next_tool_name:['\" ]*{re.escape(next_tool_name)}['\" ]*",
                "next_tool_name: " + next_tool_name,
                text_resp,
            )
        return text_resp

    @classmethod
    def is_valid_response(cls, raw_text: str) -> tuple[bool, str | None]:
        if isinstance(raw_text, dict) and raw_text.get("error"):
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        stripped = raw_text.strip()
        lower = raw_text.lower()
        has_next_thought = "next_thought" in lower or "<next_thought>" in lower
        has_next_tool_name = "next_tool_name" in lower or "<next_tool_name>" in lower
        has_next_tool_args = "next_tool_args" in lower or "<next_tool_args>" in lower
        valid_ending = stripped.endswith("}") or stripped.endswith("}]") or stripped.endswith("</next_tool_args>") or stripped.endswith(">")
        if has_next_thought and has_next_tool_name and has_next_tool_args and not valid_ending:
            return False, cls.ErrorType.INCOMPLETE_RESPONSE.name
        if not raw_text:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        return cls.is_http_response(raw_text)

    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str | None, any, any, str | None]:
        error_msg = None
        text_resp = text_resp.strip()
        if "observation:" in text_resp.lower():
            text_resp = re.split(r"observation\s*:", text_resp, flags=re.IGNORECASE)[0].strip()
        text_resp = cls.sanitise_text_resp(text_resp)
        if "Infrastructure is at maximum capacity" in text_resp:
            return None, None, None, "HTTP ERROR Maximum Capacity"
        if "No instances available" in text_resp:
            return None, None, None, "HTTP ERROR NO INSTANCES AVAILABLE"
        next_thought = None
        for pat in [
            r"next_thought\s*:\s*(.*?)(?=\n(?:tool_call_|next_tool_name:|$))",
            r"next_thought\s*:\s*(.*?)(?=\ntool_call_)",
            r"next_thought\s*:\s*(.*?)(?=\nnext_tool_name:)",
            r"next_thought\s*:\s*(.*)",
        ]:
            match = re.search(pat, text_resp, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) > 2:
                    next_thought = candidate
                    break
        if not next_thought:
            next_thought = "Processing request"
        tool_call_matches = list(re.finditer(r"tool_call_(\d+)\s*:", text_resp, re.IGNORECASE))
        if tool_call_matches:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                start = match.end()
                end = tool_call_matches[i + 1].start() if i + 1 < len(tool_call_matches) else len(text_resp)
                block = text_resp[start:end].strip()
                call = cls._extract_tool_call_from_block(block)
                if call:
                    tool_calls.append(call)
            if not tool_calls:
                return next_thought, None, None, "Multi-tool format detected but no valid tool calls extracted"
            tool_names = [c["tool_name"] for c in tool_calls]
            tool_args_list = [c["tool_args"] for c in tool_calls]
            if len(tool_names) == 1:
                return next_thought, tool_names[0], tool_args_list[0], error_msg
            return next_thought, tool_names, tool_args_list, error_msg

        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp:
            name_idx = text_resp.find("next_tool_name:")
            args_idx = text_resp.find("next_tool_args:")
            if text_resp.find("next_thought:") < name_idx < args_idx:
                next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip()
                next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip()
                try:
                    if next_tool_name_raw.startswith("["):
                        next_tool_names = Utils.load_json(next_tool_name_raw)
                    else:
                        next_tool_names = [next_tool_name_raw]
                    parsed_args = cls.parse_next_tool_args(next_tool_names, next_tool_args_raw)
                    next_tool_args_list = parsed_args if isinstance(parsed_args, list) else [parsed_args for _ in next_tool_names]
                    if len(next_tool_names) == 1:
                        return next_thought, next_tool_names[0], next_tool_args_list[0], error_msg
                    return next_thought, next_tool_names, next_tool_args_list, error_msg
                except (JSONDecodeError, Exception) as e:
                    error_msg = f"Invalid JSON in tool args: {str(e)}"
                    return next_thought, None, None, error_msg

        if "next_thought:" not in text_resp:
            error_msg = "Invalid response. next_thought not found"
        elif "next_tool_name:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. No tool calls found (expected next_tool_name: or tool_call_N:)"
        elif "next_tool_args:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. next_tool_args not found"
        else:
            error_msg = "Invalid response format. Could not parse tool calls."
        return next_thought, None, None, error_msg

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {
                "role": "system",
                "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else.",
            },
            {"role": "user", "content": json_string},
        ]
        selected_model = QWEN_MODEL_NAME
        retry = 0
        while retry < 5:
            try:
                response, _ = cls.make_request(messages, model=selected_model)
                break
            except Exception:
                retry += 1
                remaining = [model for model in AGENT_MODELS if model != selected_model]
                if remaining:
                    selected_model = random.choice(remaining)
                time.sleep(1)
        try:
            response = response.replace("```json", "").strip("```")
            return json.loads(response)
        except Exception:
            return None

    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
        INCOMPLETE_RESPONSE = 10


class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings

    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        import re

        if isinstance(messages, list):
            text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
        else:
            text = messages

        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for token in tokens:
            if token.isspace():
                continue
            elif len(token) == 1:
                count += 1
            else:
                count += max(1, (len(token) + 2) // 3)
        return count

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:

                fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)


# Tree-sitter code parsing for function extraction
from tree_sitter import Parser
from tree_sitter_language_pack import get_language
_codeparse_util_language_cache = {}


class CodeParseUtil:
    """
    Code parsing utility using tree-sitter for language-aware code analysis.
    Supports extracting function bodies, skeleton structures, and detecting languages.
    """
    def __init__(self):
        self._parsers = {}
        
    def _classify_node_type(self, node) -> tuple[str, int | None]:
        node_type_str = node.type.lower()
        if "function" in node_type_str or "method" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("function", i)
            return ("function", None)
        elif "class" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("class", i)
            return ("class", None)
        return ("other", None)

    def _get_parser(self, language: str):
        if Parser is None or get_language is None:
            return None
        if language not in self._parsers:
            try:
                lang_obj = get_language(language)
                if lang_obj is None:
                    return None 
                parser = Parser(lang_obj)
                self._parsers[language] = parser
            except Exception as e:
                logger.warning(f"Error creating parser for {language}: {e}")
                return None
        return self._parsers[language]
    
    def _is_identifier_node(self, node) -> bool:
        return "identifier" in node.type.lower()

    def check_language(self, source: str, file_path: str | None = None) -> str | None:
        global _codeparse_util_language_cache
        if file_path and not os.path.exists(file_path) or not source or not source.strip():
            return None
        if file_path:
            file_path = os.path.abspath(file_path) if file_path else None
            if file_path and file_path in _codeparse_util_language_cache:
                return _codeparse_util_language_cache[file_path] 
        stripped_source = source.strip()
        sample = stripped_source if len(stripped_source) <= 1000 else f"{stripped_source[:500]}\n\n... [middle content omitted] ...\n\n{stripped_source[-500:]}"
        prompt = f"""Detect the programming language of the following code sample.
        Analyze the code and determine which programming language it is written in.
        Return ONLY the language name in lowercase.
        If you cannot determine the language, return "unknown".
        Code sample:
        ```
        {sample}
        ```
        Return ONLY the language name in lowercase, no other text or explanation."""
        retry = 0 
        messages = [{"role": "user", "content": prompt}] 
        models_to_try = [KIMI_MODEL_NAME, GLM_MODEL_NAME]
        while retry < 3:
            try:
                result, _ = EnhancedNetwork.make_request(messages=messages, model=models_to_try[retry % len(models_to_try)], attempt=1, temperature=0.0)
                cleaned = result.strip().lower()
                cleaned = cleaned.removeprefix("```").removesuffix("```").strip() 
                cleaned = cleaned.strip('"').strip("'").strip() 

                if cleaned and ' ' not in cleaned and cleaned.isalpha():
                    detected_language = cleaned if cleaned != 'unknown' else None
                else:
                    retry += 1
                    if retry < 3:
                        messages.append({"role": "assistant", "content": result})
                        messages.append({"role": "user", "content": "Please return ONLY the language name as a single word in lowercase. No other text."})
                        time.sleep(1)
                    continue
                if file_path:
                    _codeparse_util_language_cache[file_path] = detected_language
                return detected_language
            except Exception as e:
                logger.warning(f"Error detecting language with LLM (attempt {retry + 1}/3): {e}")
                retry += 1
                if retry < 3:
                    time.sleep(1)
                continue
        return None

    def _find_specific_function(self, node, source_lines: list[str], target_qualified: str, target_simple: str, class_name: str = "", parent_node = None) -> dict | None:
        if not node.children:
            return None
        node_type, name_child_index = self._classify_node_type(node)
        if node_type == "class":
            name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if not name and parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if name:
                new_class_name = f"{class_name}.{name}" if class_name else name
                for child in node.children:
                    result = self._find_specific_function(child, source_lines, target_qualified, target_simple, new_class_name, node)
                    if result is not None:
                        return result

        elif node_type == "function":
            name = internal_name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    internal_name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if not name:
                name = internal_name
            if name:
                qualified_name = f"{class_name}.{name}" if class_name else name
                is_qualified_target = '.' in target_qualified
                is_match = qualified_name == target_qualified or (not is_qualified_target and name == target_simple)
                if is_match:
                    at_start = node.start_point[0]
                    for i in range(at_start - 1, -1, -1):
                        if source_lines[i].strip().startswith('@'):
                            at_start = i
                        elif source_lines[i].strip():
                            break
                    return {'start_line': at_start + 1, 'end_line': node.end_point[0] + 1}
            for child in node.children:
                result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
                if result is not None:
                    return result
        for child in node.children:
            result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
            if result is not None:
                return result
        return None

    def get_function_body(self, file_path: str, function_name: str, add_line_numbers: bool = False) -> str:
        if not function_name or not os.path.exists(file_path):
            return ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""
        if not source or Parser is None:
            return ""
        try:
            source_bytes, source_lines = bytes(source, 'utf8'), source.splitlines()
            language = self.check_language(source, file_path=file_path)
            if not language:
                return ""
            parser = self._get_parser(language)
            if parser is None:
                return ""
            tree = parser.parse(source_bytes)
            target_qualified, target_simple = function_name, function_name.split('.')[-1]
            func_info = self._find_specific_function(tree.root_node, source_lines, target_qualified, target_simple)
            if func_info is None:
                return ""
            start_idx, end_idx = func_info['start_line'] - 1, func_info['end_line'] - 1 
            if 0 <= start_idx < len(source_lines) and 0 <= end_idx < len(source_lines): 
                body_lines = source_lines[start_idx:end_idx + 1]
                return '\n'.join(f"{start_idx + i + 1}| {line}" for i, line in enumerate(body_lines)) if add_line_numbers else '\n'.join(body_lines)
        except Exception as e:
            logger.warning(f"Error finding function {function_name} in {file_path}: {e}")
        return ""


class WorkflowOptimizer:
    """
    Optimizes workflow execution by tracking patterns and suggesting improvements.
    Works without additional LLM calls by using rule-based optimization.
    """
    
    def __init__(self):
        self.step_patterns = []
        self.successful_patterns = []
        self.failed_patterns = []
    
    def get_optimization_hint(self, step: int, recent_tool_calls: list) -> Optional[str]:
        """
        Get optimization hint based on current state.
        Returns hint string or None.
        """
        logger.debug(f"[WORKFLOW_OPTIMIZER] get_optimization_hint (step {step})")
        if not recent_tool_calls:
            return None
        
        # Check for inefficient patterns
        if len(recent_tool_calls) >= 3:
            tools_used = [call[0] for call in recent_tool_calls[-3:]]
            
            # Pattern: read, read, read -> suggest batch
            if all(t in ["get_file_content", "search_in_file"] for t in tools_used):
                return "Batch file operations together for efficiency"
            
            # Pattern: edit without immediate test
            if "apply_code_edit" in tools_used and "run_code" not in tools_used:
                return "Test code changes immediately after editing"
        
        return None

    def analyze_step_pattern(self, step: int, tool_calls: list[tuple[str, dict]], 
                            success: bool, observation_summary: str) -> Optional[dict]:
        """
        Analyze step pattern and return optimization suggestions.
        Returns dict with suggestions or None.
        """
        logger.debug(f"[WORKFLOW_OPTIMIZER] analyze_step_pattern (step {step})")
        if not tool_calls:
            return None
        
        suggestions = {}
        
        # Analyze tool call count
        if len(tool_calls) == 1:
            tool_name, tool_args = tool_calls[0]
            # Suggest batching if single read operation
            if tool_name in ["get_file_content", "search_in_file"]:
                suggestions['batching'] = "Consider reading multiple files in one step"
        
        # Check for edit without test
        edit_tools = [name for name, _ in tool_calls if name == "apply_code_edit"]
        test_tools = [name for name, _ in tool_calls if name == "run_code"]
        if edit_tools and not test_tools:
            suggestions['testing'] = "Consider adding test execution after code edit"
        
        # Check for multiple searches that could be combined
        search_tools = [name for name, _ in tool_calls if name in ["search_in_file", "grep_search"]]
        if len(search_tools) > 2:
            suggestions['search_optimization'] = "Multiple searches detected. Consider using grep_search for batch operations"
        
        return suggestions if suggestions else None
    
    def should_suggest_checkpoint(self, step: int, modified_files: set, 
                                 consecutive_failures: int) -> bool:
        """Determine if checkpoint should be suggested."""
        logger.debug(f"[WORKFLOW_OPTIMIZER] should_suggest_checkpoint (step {step})")
        # Suggest checkpoint after multiple file modifications
        if len(modified_files) >= 3 and step % 10 == 0:
            return True
        
        # Suggest checkpoint after failures
        if consecutive_failures >= 2:
            return True
        
        return False


class TestFeedbackAnalyzer:
    """
    Enhanced feedback loop analyzer for fix tasks.
    Tracks test results, analyzes patterns, and generates actionable feedback.
    """
    
    def __init__(self):
        self.test_history: List[Dict] = []
        self.failure_patterns: Dict[str, int] = {}
        self.consecutive_failures = 0
        self.consecutive_passes = 0
        self.last_test_result = None
        self.root_cause_candidates: List[str] = []
        self.fix_attempts: List[Dict] = []
        
    def record_test_result(self, test_output: str, tool_args: dict, step: int) -> Dict:
        """Record a test result and analyze it."""
        is_pass = self._is_test_passing(test_output)
        
        result = {
            "step": step,
            "passed": is_pass,
            "output": test_output[:2000] if test_output else "",
            "command": tool_args.get("command", []) if tool_args else [],
            "failure_type": self._categorize_failure(test_output) if not is_pass else None,
            "timestamp": time.time(),
        }
        
        self.test_history.append(result)
        self.last_test_result = result
        
        if is_pass:
            self.consecutive_passes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_passes = 0
            if result["failure_type"]:
                self.failure_patterns[result["failure_type"]] = \
                    self.failure_patterns.get(result["failure_type"], 0) + 1
        
        return result
    
    def _is_test_passing(self, output: str) -> bool:
        """Determine if test output indicates success."""
        if not output:
            return False
        output_lower = output.lower()
        
        # Check for failure indicators
        failure_indicators = [
            "fail", "error", "exception", "traceback", "assertion",
            "expected", "actual", "not equal", "!=", "does not match",
            "failed", "errors", "failures"
        ]
        
        # Check for success indicators
        success_indicators = [
            "passed", "ok", "success", "all tests passed", "0 failures",
            "0 errors", "tests passed"
        ]
        
        has_failure = any(indicator in output_lower for indicator in failure_indicators)
        has_success = any(indicator in output_lower for indicator in success_indicators)
        
        # If explicit success and no failures
        if has_success and not has_failure:
            return True
        
        # Check for pytest/unittest style output
        if "passed" in output_lower and "failed" not in output_lower:
            return True
            
        return not has_failure
    
    def _categorize_failure(self, output: str) -> Optional[str]:
        """Categorize the type of test failure."""
        if not output:
            return "empty_output"
        output_lower = output.lower()
        
        if "assertion" in output_lower or "assertionerror" in output_lower:
            return "assertion_error"
        elif "typeerror" in output_lower:
            return "type_error"
        elif "attributeerror" in output_lower:
            return "attribute_error"
        elif "nameerror" in output_lower:
            return "name_error"
        elif "import" in output_lower and "error" in output_lower:
            return "import_error"
        elif "syntax" in output_lower:
            return "syntax_error"
        elif "timeout" in output_lower:
            return "timeout"
        elif "index" in output_lower and "error" in output_lower:
            return "index_error"
        elif "key" in output_lower and "error" in output_lower:
            return "key_error"
        elif "value" in output_lower and "error" in output_lower:
            return "value_error"
        elif "exception" in output_lower or "traceback" in output_lower:
            return "exception"
        elif "fail" in output_lower:
            return "generic_failure"
        return "unknown"
    
    def get_feedback(self, step: int) -> Optional[str]:
        """Generate contextual feedback based on test history."""
        if not self.test_history:
            return None
        
        feedback_parts = []
        
        # Check for repeated failures
        if self.consecutive_failures >= 3:
            feedback_parts.append(
                f"⚠️ **CRITICAL: {self.consecutive_failures} consecutive test failures detected.**\n"
                "Consider:\n"
                "1. Re-examining your understanding of the root cause\n"
                "2. Rolling back recent changes and trying a different approach\n"
                "3. Using `list_attempted_strategies` to review what you've tried"
            )
        
        # Analyze failure patterns
        if self.failure_patterns:
            most_common = max(self.failure_patterns.items(), key=lambda x: x[1])
            if most_common[1] >= 2:
                feedback_parts.append(
                    f"📊 **Pattern detected**: '{most_common[0]}' has occurred {most_common[1]} times.\n"
                    f"Focus on resolving this specific error type."
                )
        
        # Check for oscillating behavior (pass-fail-pass-fail)
        if len(self.test_history) >= 4:
            recent = [r["passed"] for r in self.test_history[-4:]]
            if recent == [True, False, True, False] or recent == [False, True, False, True]:
                feedback_parts.append(
                    "🔄 **Oscillating test results detected.**\n"
                    "Your fix may be addressing one case while breaking another.\n"
                    "Consider a more comprehensive solution that handles all cases."
                )
        
        # Check for progress
        if len(self.test_history) >= 2:
            recent_passes = sum(1 for r in self.test_history[-5:] if r["passed"])
            older_passes = sum(1 for r in self.test_history[-10:-5] if r["passed"]) if len(self.test_history) >= 10 else 0
            
            if recent_passes > older_passes:
                feedback_parts.append("✅ **Positive trend**: Recent tests show improvement.")
            elif recent_passes < older_passes and older_passes > 0:
                feedback_parts.append(
                    "📉 **Regression detected**: Recent changes may have introduced new issues."
                )
        
        # Suggest based on last failure type
        if self.last_test_result and not self.last_test_result["passed"]:
            failure_type = self.last_test_result.get("failure_type")
            suggestions = self._get_failure_suggestions(failure_type)
            if suggestions:
                feedback_parts.append(f"💡 **Suggestion for {failure_type}**:\n{suggestions}")
        
        if feedback_parts:
            return "\n\n".join(feedback_parts)
        return None
    
    def _get_failure_suggestions(self, failure_type: str) -> Optional[str]:
        """Get specific suggestions for a failure type."""
        suggestions = {
            "assertion_error": "Check expected vs actual values. Verify your logic matches the expected behavior.",
            "type_error": "Verify parameter types and return types. Check for None values being passed.",
            "attribute_error": "Ensure the object has the expected attributes. Check for typos or missing initialization.",
            "name_error": "Check for undefined variables or missing imports.",
            "import_error": "Verify module paths and import statements. Check if dependencies exist.",
            "syntax_error": "Review the code for syntax issues. Check brackets, quotes, and indentation.",
            "index_error": "Check array/list bounds. Verify index calculations and edge cases.",
            "key_error": "Verify dictionary keys exist. Add appropriate checks or default values.",
            "value_error": "Check input value ranges and formats. Add validation if needed.",
            "timeout": "Optimize code performance or increase timeout. Check for infinite loops.",
        }
        return suggestions.get(failure_type)
    
    def record_fix_attempt(self, file_path: str, change_description: str, step: int):
        """Record a fix attempt for tracking."""
        self.fix_attempts.append({
            "step": step,
            "file": file_path,
            "description": change_description,
            "timestamp": time.time(),
            "tests_after": []
        })
    
    def get_progress_summary(self) -> str:
        """Get a summary of testing progress."""
        if not self.test_history:
            return "No tests run yet."
        
        total = len(self.test_history)
        passed = sum(1 for r in self.test_history if r["passed"])
        failed = total - passed
        
        summary = [
            f"📈 **Test Progress Summary**",
            f"- Total runs: {total}",
            f"- Passed: {passed} ({passed/total*100:.1f}%)" if total > 0 else "- Passed: 0",
            f"- Failed: {failed}",
            f"- Current streak: {self.consecutive_passes} passes" if self.consecutive_passes > 0 
                else f"- Current streak: {self.consecutive_failures} failures",
        ]
        
        if self.failure_patterns:
            summary.append("- Most common failures: " + 
                ", ".join(f"{k}({v})" for k, v in 
                    sorted(self.failure_patterns.items(), key=lambda x: -x[1])[:3]))
        
        return "\n".join(summary)


class EnhancedCOT:
    def __init__(self, latest_observations_to_keep=5, summarize_batch_size=10):
        self.thoughts = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        self.summaries = {}
        self.summarized_ranges = []

    def count_repeated_thoughts(self) -> int:
        """
        Count the number of consecutive repeated thoughts at the end of the COT.
        Returns 0 if no repetition, or the count of consecutive repeated thoughts.
        """
        if len(self.thoughts) < 2:
            return 0
        last_thought = self.thoughts[-1]
        last_tool_name = last_thought.next_tool_name
        last_tool_args = last_thought.next_tool_args
        count = 0
        for i in range(len(self.thoughts) - 1, -1, -1):
            thought = self.thoughts[i]
            if thought.next_tool_name == last_tool_name and thought.next_tool_args == last_tool_args:
                count += 1
            else:
                break
        return max(0, count - 1)

    def _get_summary_for_index(self, idx):
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None

    def _check_and_summarize_if_needed(self):
        total_thoughts = len(self.thoughts)
        cutoff_idx = total_thoughts - self.latest_observations_to_keep
        if cutoff_idx < self.summarize_batch_size:
            return
        unsummarized = 0
        for s, e in sorted(self.summarized_ranges):
            if s <= unsummarized < e:
                unsummarized = e
            elif s > unsummarized:
                break
        if unsummarized >= cutoff_idx:
            return
        summarize_start = unsummarized
        summarize_end = min(summarize_start + self.summarize_batch_size, cutoff_idx)
        batch_size = summarize_end - summarize_start
        if batch_size >= self.summarize_batch_size:
            range_key = (summarize_start, summarize_end)
            if range_key not in self.summaries:
                summary = self._summarize_messages_batch(summarize_start, summarize_end)
                if summary:
                    self.summaries[range_key] = summary
                    self.summarized_ranges.append(range_key)
                    self.summarized_ranges.sort()

    def _summarize_messages_batch(self, start_idx, end_idx):
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if getattr(thought, "is_deleted", False):
                continue
            assistant_part = (
                f"next_thought: {thought.next_thought}\n" f"next_tool_name: {thought.next_tool_name}\n" f"next_tool_args: {thought.next_tool_args}\n"
            )
            obs = thought.observation
            if isinstance(obs, (list, tuple)):
                try:
                    obs_render = json.dumps(list(obs), ensure_ascii=False)
                except Exception:
                    obs_render = str(obs)
            else:
                obs_render = str(obs) if obs else ""
            if len(obs_render) > 40000:
                obs_render = obs_render[:40000] + "... [truncated for summarization]"
            user_part = f"observation: {obs_render}"
            conversation_parts.append(
                {
                    "assistant": assistant_part,
                    "user": user_part,
                    "is_error": getattr(thought, "is_error", False),
                }
            )
        if not conversation_parts:
            return None
        conv_lines = []
        for idx, part in enumerate(conversation_parts, 1):
            conv_lines.append(f"\n--- Step {idx} ---")
            conv_lines.append(f"Assistant: {part['assistant']}")
            user_obs = part["user"]
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conv_lines.append(f"User: {user_obs}")
            if part.get("is_error"):
                conv_lines.append("[Error occurred]")
        conversation_text = "\n".join(conv_lines)
        summarization_prompt = textwrap.dedent(
            f"""
            You are summarizing a conversation history between an AI agent and its environment.
            Summarize the following conversation steps concisely, focusing on:
            1. Key actions taken (tools used, files modified, tests run)
            2. Important findings or errors encountered
            3. Progress made toward solving the problem
            4. Critical decisions or changes in approach
            Keep the summary concise (2-4 sentences per step) but preserve important details.
            Conversation to summarize:
            {conversation_text}
            Provide a concise summary:
        """
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversation history concisely.",
            },
            {"role": "user", "content": summarization_prompt},
        ]
        for _ in range(3):
            try:
                response, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
                return response.strip()
            except Exception:
                time.sleep(1)
        return None

    def is_thought_repeated(self):
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False

    def add_action(self, action):
        self.thoughts.append(action)
        if len(self.thoughts) >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True

    def to_str(self):
        messages = []
        last_summary_range = None

        allowed_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:]) if self.summarized_ranges else set()
        total = len(self.thoughts)
        keep_last = self.latest_observations_to_keep

        for i, thought in enumerate(self.thoughts):
            if getattr(thought, "is_deleted", False):
                continue

            recent = i >= total - keep_last

            if not recent:
                summary = self._get_summary_for_index(i)
                if summary:
                    found_range = False
                    for (start, end), _ in self.summaries.items():
                        if start <= i < end:
                            cur_range = (start, end)
                            if cur_range not in allowed_ranges:
                                found_range = True
                                break
                            if cur_range != last_summary_range:
                                messages.append(
                                    {"role": "system", "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]"}
                                )
                                last_summary_range = cur_range
                            found_range = True
                            break
                    if found_range:
                        continue
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n" f"next_tool_name:{thought.next_tool_name}\n" f"next_tool_args:{thought.next_tool_args}"
                )

                obs = thought.observation
                if isinstance(obs, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(obs), ensure_ascii=False)
                    except Exception:
                        obs_render = str(obs)
                else:
                    obs_render = str(obs) if obs else ""
                user_str = f"observation: {obs_render}"

                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
            else:
                if thought.is_error is None or i == total - 1:
                    assistant_str = (
                        f"next_thought:{thought.next_thought}\n"
                        f"next_tool_name:{thought.next_tool_name}\n"
                        f"next_tool_args:{thought.next_tool_args}"
                    )

                    obs = thought.observation
                    if isinstance(obs, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(obs), ensure_ascii=False)
                        except Exception:
                            obs_render = str(obs)
                    else:
                        obs_render = str(obs)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error is None and thought.is_error is not None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if obs is None:
                            obs_len = 0
                        elif isinstance(obs, (list, tuple)):
                            obs_len = len(obs)
                        else:
                            obs_len = len(str(obs).splitlines())
                        user_str = f"observation: error ocurred. detailed output omitted ({obs_len}) lines\n"
                    else:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if isinstance(obs, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(obs), ensure_ascii=False)
                            except Exception:
                                obs_render = str(obs)
                        else:
                            obs_render = str(obs)
                        user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})

        return messages

    class Action:
        def __init__(
            self,
            next_thought: str,
            next_tool_name: str,
            next_tool_args: dict,
            observation,
            is_error: bool = False,
            raw_response: str = None,
            total_attempts: int = 0,
            inference_error_counter: dict = None,
            request_data: list = None,
        ):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False


class FileSystemManager:
    def __init__(self):
        pass

    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
        search_in_file_callback=None,
    ) -> str:

        def add_line_numbers_to_content(content: str, start_line: int = 1) -> str:
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:6}|{line}")
            return "\n".join(numbered_lines)

        if search_term and search_in_file_callback:
            return search_in_file_callback(file_path, search_term)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if search_start_line is not None or search_end_line is not None:

                lines = f.readlines()
                start_idx = max(0, (search_start_line or 1) - 1)
                end_idx = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start_idx:end_idx])
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, start_idx + 1)
                else:
                    result = content
            else:
                content = f.read()
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, 1)
                else:
                    result = content

        return Utils.limit_strings(result, n=limit) if limit != -1 else result

    def list_directory_structure(self, directory_path: str, max_depth: int = 0) -> str:
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."

        ignore = {".git", "__pycache__", ".pytest_cache", "node_modules", ".tox", ".venv", "venv", ".eggs"}

        def tree(path: str, prefix: str = "", depth: int = 0, current_max_depth: int = 0) -> list[str]:
            if depth > current_max_depth:
                return []

            try:
                items = sorted(os.listdir(path))
            except (PermissionError, OSError) as e:
                return [f"{prefix}[Error reading directory: {str(e)}]"]

            dirs = [
                i for i in items if os.path.isdir(os.path.join(path, i)) and not i.startswith(".") and i not in ignore and not i.endswith(".egg-info")
            ]

            files = [i for i in items if os.path.isfile(os.path.join(path, i)) and not i.startswith(".")]

            lines: list[str] = []

            for idx, d in enumerate(dirs):
                is_last = (idx == len(dirs) - 1) and not files
                branch = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")

                lines.append(f"{prefix}{branch}{d}/")
                lines.extend(tree(os.path.join(path, d), new_prefix, depth + 1, current_max_depth))

            for idx, f in enumerate(files):
                is_last = idx == len(files) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{f}")

            return lines

        def count_tokens(text: str) -> int:
            try:
                if "Utils" in globals() and hasattr(Utils, "count_tokens"):
                    return Utils.count_tokens(text)
            except (NameError, AttributeError):
                pass
            return len(text) // 4

        MAX_TOKENS = 3000
        current_depth = max_depth
        while current_depth >= 0:
            entries = tree(directory_path, "", 0, current_depth)
            result = f"Directory structure (depth={current_depth}):\n{directory_path}/\n" + "\n".join(entries)

            token_count = count_tokens(result)

            if token_count <= MAX_TOKENS:
                if current_depth < max_depth:
                    result += (
                        f"\n\n[Note: Requested depth {max_depth} exceeded token limit. Showing depth {current_depth} instead ({token_count} tokens).]"
                    )
                return result

            if current_depth == 0:
                result += f"\n\n[Warning: Result exceeds token limit ({token_count} tokens > {MAX_TOKENS} tokens). Consider using a more specific directory_path.]"
                return result
            current_depth -= 1
        entries = tree(directory_path, "", 0, 0)
        result = f"Directory structure (depth=0):\n{directory_path}/\n" + "\n".join(entries)
        return result


class FileOperationsUtil:
    def __init__(self, new_files_created: list):
        self.new_files_created = new_files_created
        self.file_system_manager = None
        self.search_manager = None

    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
    ) -> str:
        search_callback = lambda fp, st: self.search_manager.search_in_file(fp, st)
        return self.file_system_manager.get_file_content(
            file_path=file_path,
            search_start_line=search_start_line,
            search_end_line=search_end_line,
            search_term=search_term,
            limit=limit,
            add_line_numbers=add_line_numbers,
            search_in_file_callback=search_callback,
        )

    def save(self, file_path: str, content: str) -> str:
        with open(file_path, "w") as file:
            file.write(content)
        self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"

    def set_managers(self, file_system_manager, search_manager):
        """Set manager references after initialization to avoid circular dependencies."""
        self.file_system_manager = file_system_manager
        self.search_manager = search_manager


class CodeEditManager:
    def __init__(self, file_ops: "FileOperationsUtil" = None):
        self.file_ops = file_ops

    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        def add_context_to_similar_match(original_content: str, formatted_match: str, context_lines: int = 2) -> str:
            """Add context lines around a similar match for better understanding."""
            lines = original_content.split("\n")
            match_lines = formatted_match.split("\n")
            if len(match_lines) < 2:
                return formatted_match
            actual_content_lines = match_lines[1:]
            actual_content = "\n".join(actual_content_lines)
            best_match_start = -1
            best_similarity = 0

            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i : i + len(actual_content_lines)]
                candidate_content = "\n".join(candidate_lines)
                import difflib

                similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            if best_match_start == -1:
                return formatted_match

            start_line = max(0, best_match_start - context_lines)
            end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
            return f"{description}\n" + "\n".join(context_lines_list)

        def find_most_similar_content(original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
            """Find the most similar content chunks to the search string."""
            import difflib

            lines = original_content.split("\n")
            chunks = []
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
            search_lines = search_string.split("\n")
            target_chunk_size = max(3, len(search_lines))
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i : i + target_chunk_size]
                chunk_content = "\n".join(chunk_lines).strip()
                if chunk_content:
                    chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
                if ratio > 0.3:
                    similarities.append((ratio, chunk_desc, chunk_content))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]

        if search == replace:
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."
        original = self.file_ops.get_file_content(file_path, limit=-1)
        match original.count(search):
            case 0:
                similar_matches = find_most_similar_content(original, search, 1)
                error_msg = f"Error: search string not found in file {file_path}."
                if similar_matches:
                    error_msg += f"\n\nMost similar snippet found (you may need to adjust your search string):"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        content_with_context = add_context_to_similar_match(original, content, context_lines=2)
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{content_with_context}"
                else:
                    error_msg += " No similar content found. Please check the file content and provide the exact code you want to replace."
                return error_msg
            case 1:
                new_content = original.replace(search, replace)
                try:
                    self.file_ops.save(file_path, new_content)

                    replace_pos = new_content.find(replace)
                    if replace_pos != -1:
                        lines = new_content.split("\n")
                        chars_so_far = 0
                        replace_line_start = 0
                        for i, line in enumerate(lines):
                            if chars_so_far + len(line) >= replace_pos:
                                replace_line_start = i
                                break
                            chars_so_far += len(line) + 1  # +1 for newline
                        replace_lines_count = replace.count("\n") + 1
                        replace_line_end = replace_line_start + replace_lines_count - 1
                        start_line = max(0, replace_line_start - 10)
                        end_line = min(len(lines), replace_line_start + 10)
                        context_lines = []
                        for i in range(start_line, end_line):
                            line_num = i + 1
                            if replace_line_start <= i <= replace_line_end:
                                prefix = ">>> "
                            else:
                                prefix = "    "
                            context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
                        context = "\n".join(context_lines)
                        return f"ok, code edit applied successfully. Here is the edited code (lines {start_line+1}-{end_line}):\n\n{context}"
                    else:
                        return "ok, code edit applied successfully"
                except Exception as e:
                    return f"Error: syntax error in file {file_path}. {str(e)}"
            case num_hits:
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."


class EnhancedToolManager:
    TOOL_LIST = {}

    def __init__(self, **kwargs):
        pass

    def get_tool_docs(self) -> str:
        return "\n\n".join([json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()])

    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.default is param.empty and param.kind in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            ):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description,
                }
                continue
            elif "str" in type_hint:
                json_type = "string"
            elif "int" in type_hint:
                json_type = "integer"
            elif "float" in type_hint:
                json_type = "number"
            elif "bool" in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description,
            }
        parameters = {"type": "object", "properties": properties, "required": required}
        tool_schemas = {
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters,
        }
        return tool_schemas

    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] = self.tool_invocations.get(fn.__name__, 0) + 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {j: 0 for j in self.Error.ErrorType.__members__}
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True
        return wrapper

    @classmethod
    def get_tool_args_for_tool(cls, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in cls.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(cls.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        else:
            return cls.TOOL_LIST[tool_name]["input_schema"]["required"]

    @classmethod
    def get_final_git_patch(cls) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(cls, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Get only modified files (not newly created/untracked files)
            ls = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)

            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    def create_modified_files_patch(
        self,
    ) -> str:
        """
        Create a patch containing only modified tracked files. Does NOT modify the repository state.
        """
        try:
            result = subprocess.run(["git", "diff", "--diff-filter=M"], capture_output=True, text=True, timeout=30, check=True)
            return result.stdout
        except Exception as e:
            return f"Error creating modified files patch: {e}"

    class Error(Exception):

        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14

        def __init__(self, error_type: ErrorType, message: str):
            self.error_type = error_type
            self.message = message


class FixTaskEnhancedToolManager(EnhancedToolManager):

    def __init__(
        self,
        available_tools: Optional[list[str]] = [],
        runner_hint: str | None = None,
        runner_mode_hint: str | None = None,
        initial_checkpoint=None,
        problem_statement: str = None,
        should_review: bool = True,
        is_fix_task: bool = False,
    ):
        self.new_files_created = []
        self.available_tools = available_tools
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.generated_test_files = []
        self.initial_checkpoint = initial_checkpoint
        self.observation_dir = ".observation"
        self.problem_statement = problem_statement
        self.repo_dir = "."
        self.saved_observation_counter = 0
        self.is_fix_task = is_fix_task
        # Initialize strategy tracking
        self.strategy_counter = 0
        self.strategies = []
        if should_review:
            self.is_reviewed = False
            self.file_by_file_reviewed = False
        else:
            self.is_reviewed = True
            self.file_by_file_reviewed = True
        os.makedirs(self.observation_dir, exist_ok=True)
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        self.search_manager = SearchManager()
        self.file_system_manager = FileSystemManager()
        self.test_manager = TestManager(
            runner_hint=runner_hint,
            runner_mode_hint=runner_mode_hint,
            file_ops=self.file_ops,
        )
        self.code_edit_manager = CodeEditManager(file_ops=self.file_ops)
        self.code_parser = CodeParseUtil()
        self.workflow_optimizer = WorkflowOptimizer()
        self.thought_history: list[dict[str, Any]] = []
        self.branches: dict[str, list[dict[str, Any]]] = {}
        self.file_ops.set_managers(self.file_system_manager, self.search_manager)
        self.TOOL_LIST = {}
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()}
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
        self.finish_called_count = 0
        
        # Enhanced feedback loop for fix tasks
        self.hypothesis_counter = 0
        self.hypotheses: List[Dict] = []
        self.test_feedback_analyzer = TestFeedbackAnalyzer()
        self._current_step = 0
        self._cot_snapshot_cache = []

    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        return self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)

    @EnhancedToolManager.tool
    def list_directory_structure(self, directory_path: str = ".", max_depth: int = 1) -> str:
        """
        Lists the directory structure of the repository
        Arguments:
            directory_path: the directory path to list (default: ".")
            max_depth: maximum depth to traverse (default: 1)
        """
        return self.file_system_manager.list_directory_structure(directory_path=directory_path, max_depth=max_depth)

    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str, run_command: List[str]) -> str:
        """
        Runs any code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.
            run_command: command to run the file (i.e., ["python", "file.py"] or ["node", "file.js"] etc)
        """
        return self.test_manager.run_code(
            content=content,
            file_path=file_path,
            generated_test_files=self.generated_test_files,
            run_command=run_command,
        )

    @EnhancedToolManager.tool
    def run_tests(self, command: List[str], timeout: int = 5) -> str:
        """
        Runs tests with strict timeout.
        Arguments:
            command: list of command line arguments,
            timeout: timeout in seconds (default: 5)
        Output:
            Standard output or error output of the command.
        """
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Test run timed out."
        except Exception as e:
            return f"Test execution error: {e}"

    @EnhancedToolManager.tool
    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
    ) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self.file_ops.get_file_content(
            file_path,
            search_start_line,
            search_end_line,
            search_term,
            add_line_numbers=True,
            limit=1000,
        )

    def get_final_git_patch(self) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Get only modified files (not newly created/untracked files)
            ls = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)

            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    def _save_large_observation(self, observation: str, tool_name: str) -> str:
        self.saved_observation_counter += 1
        filename = f"observation_{self.saved_observation_counter}_{tool_name}_{int(time.time())}.txt"
        if not os.path.exists(self.observation_dir):
            os.makedirs(self.observation_dir, exist_ok=True)
        file_path = os.path.join(self.observation_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(observation)
            return file_path
        except Exception as e:
            return f"Error: Failed to save observation: {e}"
    
    
    @EnhancedToolManager.tool
    def generate_test_cases_from_root_cause(self, root_cause_code: str, file_path: str = None, function_name: str = None) -> str:
        """
        Generates comprehensive test cases based on the problem statement and the identified root cause code section.
        Call this tool when you have identified the main root cause code part that needs to be fixed.
        The generated test cases will be saved and automatically referenced when you create test files using generate_test_file.
        Arguments:
            root_cause_code: The code section identified as the root cause of the issue (required)
            file_path: Optional file path where the root cause code is located (helps provide context)
            function_name: Optional function name where the root cause code is located (helps provide context)
        Output:
            A structured markdown document containing test cases with descriptions, inputs/setup, expected results, and reasons for each test case
        """
        if not self.problem_statement:
            return "Error: Problem statement not available. Cannot generate test cases."
        
        TEST_CASE_GENERATION_PROMPT = textwrap.dedent("""
        You are an expert test case generator. Your task is to generate comprehensive test cases based on a problem statement and the root cause code section.
        
        Analyze the problem statement and the root cause code to generate test cases that:
        1. Verify the bug exists (reproduction test)
        2. Verify the fix works correctly
        3. Cover edge cases related to the root cause
        4. Test boundary conditions
        
        For each test case, provide:
        - Test case description: What the test case does
        - Input/Setup: What inputs or setup are needed
        - Expected result: What should happen when the code is correct
        - Reason: Why this test case is important for verifying the root cause fix

        **NOTE**: Don't ONLY consider the primary issue in the problem statement.
        You should consider all, every possible edge cases.
        Invalid or wrong test cases should be also generated to test thoroughly.
        For those invalid or wrong cases, you should correctly handle error or edge case.
        
        Format your response as a structured markdown document with clear sections for each test case.
        Be specific and actionable. Focus on test cases that directly relate to the root cause identified.
        """)
        
        retry = 0
        selected_model = QWEN_MODEL_NAME
        root_cause_context = root_cause_code
        if file_path:
            root_cause_context += f"\n\nFile: {file_path}"
        if function_name:
            root_cause_context += f"\n\nFunction: {function_name}"
        
        while retry < 10:
            try:
                messages = [
                    {"role": "system", "content": TEST_CASE_GENERATION_PROMPT},
                    {
                        "role": "user",
                        "content": f"Problem Statement:\n{self.problem_statement}\n\nRoot Cause Code:\n{root_cause_context}\n\nGenerate comprehensive test cases for this root cause."
                    }
                ]
                test_cases, _ = EnhancedNetwork.make_request(
                    messages, model=selected_model, attempt=1, temperature=0.0
                )
                # Store the generated test cases
                self.generated_test_cases = test_cases
                print(f"[GENERATE_TEST_CASES_FROM_ROOT_CAUSE] Test cases generated successfully and saved: {test_cases}")
                return f"Test cases generated successfully and saved.\n\n{test_cases}"
            except Exception as e:
                logger.error(f"Error generating test cases: {e}")
                retry += 1
                if retry < 10:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    if other_models:
                        selected_model = random.choice(other_models)
                    time.sleep(1)
                else:
                    return f"Error: Failed to generate test cases after {retry} attempts: {str(e)}"
        return "Error: Failed to generate test cases"

    @EnhancedToolManager.tool
    def grep_search(self, grep_search_command: str) -> str:
        """
        Performs grep search across all files in the codebase
        Arguments:
            grep_search_command: grep search command to locate (e.g., "grep <your grep command>").
        Output:
            locations where pattern was found with file paths and line numbers
        """
        return self.search_manager.search_in_all_files(grep_search_command)

    @EnhancedToolManager.tool
    def search_in_file(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        """
        return self.search_manager.search_in_file(file_path=file_path, search_term=search_term)

    @EnhancedToolManager.tool
    def get_function_body(self, file_path: str, function_name: str) -> str:
        """
        Retrieves the complete body of a function from a file, including decorators.
        Arguments:
            file_path: filesystem path to target file.
            function_name: name of the function to retrieve (supports both qualified names like "ClassName.method_name" and simple names like "method_name").
        Output:
            The complete function body including decorators, or empty string if function not found.
        """
        if not hasattr(self, 'code_parser'):
            self.code_parser = CodeParseUtil()
        return self.code_parser.get_function_body(file_path, function_name, add_line_numbers=True)

    @EnhancedToolManager.tool
    def find_symbol_references(self, symbol_identifier: str) -> str:
        """
        Discovers all code locations where a specific function, class, method, or variable is referenced.
        Provides contextual information around each usage to understand how the symbol is being used.
        Particularly valuable before modifying or refactoring code elements.
        Works across all programming languages and file types.
        Arguments:
            symbol_identifier: exact name of the function, class, method, or variable to locate
        Output:
            comprehensive listing of files and line numbers with surrounding context for each reference
        """
        try:
            cmd = f"grep -rn --binary-files=without-match '{symbol_identifier}' . | head -100"
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
            refs = result.stdout.strip()
            
            if not refs:
                return f"No references discovered for symbol '{symbol_identifier}' in the codebase."
            
            lines = refs.split('\n')
            if len(lines) > 50:
                summary = f"Found {len(lines)} references for '{symbol_identifier}' (showing first 50):\n\n"
                return summary + '\n'.join(lines[:50]) + f"\n\n... and {len(lines) - 50} more references (refine search if needed)"
            
            return f"References for '{symbol_identifier}' ({len(lines)} found):\n{refs}"
        except subprocess.TimeoutExpired:
            return f"Search timeout: Symbol '{symbol_identifier}' search took too long. Try a more specific identifier."
        except Exception as e:
            return f"Error locating symbol references: {str(e)}"

    @EnhancedToolManager.tool
    def run_shell_cmd(self, command: str) -> str:
        '''
        Runs shell commands for the repository. This tool executes shell commands directly.
        Arguments:
            command: A shell command to be run.
        Output:
            The stdout results of the command. Your working directory is the root of the project.
        '''
        if not command:
            return "Error: No command provided."
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=150
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command '{command}' timed out after 150 seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"

    @EnhancedToolManager.tool
    def think(self, thought: str) -> str:
        """ Use the tool to think about something. It will not make any changes to the repository. Use it when reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be correct and most effective. Alternatively, if you receive some test results, you can call this tool to brainstorm ways to fix the failing tests.
        Arguments:
            thought: Your thoughts.
        Output:
            Confirmation that the thought has been logged.
        """
        return "ok"

    @EnhancedToolManager.tool
    def log_strategy(self, approach: str, reasoning: str) -> str:
        """Record a high-level strategy before attempting it.

        Use this BEFORE making significant code changes to log your planned approach. This creates
        a history that persists across rollbacks, preventing you from retrying failed strategies.

        Arguments:
            approach: Brief description of the approach
            reasoning: Why you think this will work

        Output:
            Confirmation with strategy ID for later reference.
        """
        self.strategy_counter += 1
        strategy = {
            "id": self.strategy_counter,
            "approach": approach,
            "reasoning": reasoning,
            "success": None,
            "reason": None,
            "timestamp": time.time(),
            "created_step": len(getattr(self, "tool_invocations", {})),
        }
        self.strategies.append(strategy)

        return f"Strategy #{self.strategy_counter} logged: {approach}\nReasoning: {reasoning}\nUse mark_strategy_outcome to record results."

    @EnhancedToolManager.tool
    def mark_strategy_outcome(self, strategy_id: int, success: bool, reason: str) -> str:
        """Record whether a strategy worked.

        After attempting a strategy, record the outcome. This is crucial for institutional memory,
        especially when using rollbacks - you'll know what you already tried even after reverting changes.

        Arguments:
            strategy_id: ID from log_strategy (e.g., 1, 2, 3)
            success: True if approach worked (tests passed, bug fixed), False otherwise
            reason: Why it succeeded/failed (e.g., "Tests passed but introduced new bug in edge case")

        Output:
            Updated strategy status.
        """
        for strat in self.strategies:
            if strat["id"] == strategy_id:
                strat["success"] = success
                strat["reason"] = reason
                strat["completed_step"] = len(getattr(self, "tool_invocations", {}))
                status = "SUCCEEDED" if success else "FAILED"
                return f"Strategy #{strategy_id} marked as {status}\nReason: {reason}"

        return f"Error: Strategy #{strategy_id} not found"

    @EnhancedToolManager.tool
    def list_attempted_strategies(self) -> str:
        """View all strategies tried, with outcomes.

        Use this to review what approaches you've already attempted. Critical for:
        - Avoiding retry loops (especially after rollbacks)
        - Understanding what doesn't work
        - Building on partially successful strategies

        Arguments:
            None

        Output:
            Formatted list of all strategies with outcomes.
        """
        if not self.strategies:
            return "No strategies recorded yet. Use log_strategy before attempting significant changes."

        output = ["=== STRATEGY HISTORY ===\n"]

        succeeded = [s for s in self.strategies if s["success"] is True]
        failed = [s for s in self.strategies if s["success"] is False]
        pending = [s for s in self.strategies if s["success"] is None]

        output.append(
            f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(pending)} pending\n"
        )

        for status, strategies in [
            ("SUCCEEDED", succeeded),
            ("FAILED", failed),
            ("PENDING", pending),
        ]:
            if strategies:
                output.append(f"\n{status}:")
                for s in strategies:
                    output.append(f"\n  [{s['id']}] {s['approach']}")
                    output.append(f"      Reasoning: {s['reasoning']}")
                    if s['reason']:
                        output.append(f"      Outcome: {s['reason']}")

        return "\n".join(output)

    @EnhancedToolManager.tool
    def create_hypothesis(self, description: str, evidence: str) -> str:
        """Create a hypothesis about the bug's root cause.

        Use this when you have a theory about what's causing the issue. This creates
        a trackable hypothesis that persists across rollbacks - critical for systematic debugging.

        Arguments:
            description: What you think is causing the bug (e.g., "Missing null check in parse_config")
            evidence: What evidence supports this theory (e.g., "Line 45 doesn't handle None input")

        Output:
            Confirmation with hypothesis ID for tracking.
        """
        self.hypothesis_counter += 1
        hypothesis = {
            "id": self.hypothesis_counter,
            "description": description,
            "evidence": evidence,
            "status": "untested",
            "findings": None,
            "created_step": self._current_step,
            "tested_step": None,
            "timestamp": time.time(),
        }
        self.hypotheses.append(hypothesis)

        return f"Hypothesis #{self.hypothesis_counter} created: {description}\nEvidence: {evidence}\nStatus: untested\nUse test_hypothesis to record findings after testing."

    @EnhancedToolManager.tool
    def test_hypothesis(self, hypothesis_id: int, outcome: str, findings: str) -> str:
        """Record the result of testing a hypothesis.

        After investigating a hypothesis (running tests, examining code, etc.), record whether
        it was confirmed, rejected, or inconclusive. This builds institutional memory.

        Arguments:
            hypothesis_id: ID from create_hypothesis (e.g., 1, 2, 3)
            outcome: One of 'confirmed', 'rejected', or 'inconclusive'
            findings: What you discovered (e.g., "Confirmed: null check is missing, added it and tests pass")

        Output:
            Updated hypothesis status.
        """
        if outcome not in ["confirmed", "rejected", "inconclusive"]:
            return f"Error: outcome must be 'confirmed', 'rejected', or 'inconclusive', got '{outcome}'"

        for hyp in self.hypotheses:
            if hyp["id"] == hypothesis_id:
                hyp["status"] = outcome
                hyp["findings"] = findings
                hyp["tested_step"] = self._current_step
                status_emoji = {"confirmed": "✅", "rejected": "❌", "inconclusive": "❓"}.get(outcome, "")
                return f"{status_emoji} Hypothesis #{hypothesis_id} marked as {outcome.upper()}\nFindings: {findings}"

        return f"Error: Hypothesis #{hypothesis_id} not found"

    @EnhancedToolManager.tool
    def list_hypotheses(self) -> str:
        """View all hypotheses with their test status.

        Use this to review what theories you've already considered and tested. Especially useful:
        - After a rollback (to see what you learned before rolling back)
        - When stuck (to avoid retrying rejected hypotheses)
        - During metacognitive reflection checkpoints

        Arguments:
            None

        Output:
            Formatted list of all hypotheses with status and findings.
        """
        if not self.hypotheses:
            return "No hypotheses recorded yet. Use create_hypothesis to log theories about the bug."

        output = ["=== HYPOTHESIS TRACKER ===\n"]

        untested = [h for h in self.hypotheses if h["status"] == "untested"]
        confirmed = [h for h in self.hypotheses if h["status"] == "confirmed"]
        rejected = [h for h in self.hypotheses if h["status"] == "rejected"]
        inconclusive = [h for h in self.hypotheses if h["status"] == "inconclusive"]

        output.append(
            f"Summary: {len(confirmed)} confirmed, {len(rejected)} rejected, {len(inconclusive)} inconclusive, {len(untested)} untested\n"
        )

        for status, hypotheses in [
            ("✅ CONFIRMED", confirmed),
            ("❌ REJECTED", rejected),
            ("❓ INCONCLUSIVE", inconclusive),
            ("🔍 UNTESTED", untested),
        ]:
            if hypotheses:
                output.append(f"\n{status}:")
                for h in hypotheses:
                    output.append(f"\n  [{h['id']}] {h['description']}")
                    output.append(f"      Evidence: {h['evidence']}")
                    if h["findings"]:
                        output.append(f"      Findings: {h['findings']}")

        return "\n".join(output)

    @EnhancedToolManager.tool
    def get_test_progress(self) -> str:
        """Get a summary of your testing progress and feedback.

        Use this to understand how your fix attempts are progressing. Shows:
        - Test pass/fail statistics
        - Current streak (consecutive passes or failures)
        - Common failure patterns
        - Actionable suggestions

        Arguments:
            None

        Output:
            Test progress summary with feedback.
        """
        summary = self.test_feedback_analyzer.get_progress_summary()
        feedback = self.test_feedback_analyzer.get_feedback(self._current_step)
        
        if feedback:
            return f"{summary}\n\n{feedback}"
        return summary

    @EnhancedToolManager.tool
    def analyze_test_failure(self, test_output: str) -> str:
        """Analyze a test failure to understand what went wrong.

        Use this after a test fails to get structured analysis and suggestions.
        Helps identify the failure type and provides targeted guidance.

        Arguments:
            test_output: The output from the failed test run

        Output:
            Analysis of the failure with suggestions.
        """
        if not test_output:
            return "Error: No test output provided for analysis."
        
        # Record the result
        result = self.test_feedback_analyzer.record_test_result(
            test_output, {}, self._current_step
        )
        
        analysis_parts = ["=== TEST FAILURE ANALYSIS ===\n"]
        
        if result["passed"]:
            analysis_parts.append("✅ **Result**: Test appears to have PASSED\n")
        else:
            analysis_parts.append(f"❌ **Result**: Test FAILED\n")
            analysis_parts.append(f"**Failure Type**: {result['failure_type']}\n")
            
            # Get specific suggestions
            suggestion = self.test_feedback_analyzer._get_failure_suggestions(result['failure_type'])
            if suggestion:
                analysis_parts.append(f"**Suggestion**: {suggestion}\n")
        
        # Add pattern analysis
        feedback = self.test_feedback_analyzer.get_feedback(self._current_step)
        if feedback:
            analysis_parts.append(f"\n{feedback}")
        
        # Extract key error info
        lines = test_output.split('\n')
        error_lines = [l for l in lines if any(x in l.lower() for x in ['error', 'fail', 'assert', 'traceback', 'exception'])]
        if error_lines:
            analysis_parts.append("\n**Key Error Lines**:")
            for line in error_lines[:5]:
                analysis_parts.append(f"  • {line.strip()[:150]}")
        
        return "\n".join(analysis_parts)

    def _inject_test_feedback(self, observation: str, tool_name: str, tool_args: dict) -> str:
        """Internal method to inject feedback after test runs."""
        if tool_name not in ["run_tests", "run_code"]:
            return observation
        
        # Record the test result
        self.test_feedback_analyzer.record_test_result(
            observation, tool_args, self._current_step
        )
        
        # Get feedback if there are issues
        feedback = self.test_feedback_analyzer.get_feedback(self._current_step)
        
        if feedback and self.test_feedback_analyzer.consecutive_failures >= 2:
            return f"{observation}\n\n---\n🔔 **FEEDBACK LOOP ALERT**\n{feedback}"
        
        return observation

    @EnhancedToolManager.tool
    def finish_bug_injection(self, problem_statement: str) -> str:
        """
        Signals completion of the bug injection task and it should return problem statement of the bug you generated.
        Problem statement shouldn't include where and how you injected bugs. It should be quiz that we will provide to the software engineers to solve.
        It shouldn't mention the direct targeted tests as well. It should be generalized description. Make problem statement hard so that it is too easy for software engineers to solve. But it must include all the issues that needs to be fixed.
        **Never mention the clue where and how you injected bugs.**

        Arguments:
            problem_statement: The problem statement that explains the bugs happened. It must include all the bugs.
        Output:
            The problem statement that explains the bugs happened.
        """
        return problem_statement


    @EnhancedToolManager.tool
    def finish(self):
        """
        Signals completion of the current workflow execution. Validates patch application before finishing.
        Arguments:
            None
        Output:
            Review patch prompt with validation results, or "finish" if called 5+ times (for fix tasks) or 1+ times (for create tasks)
        """
        return "finish"

    @EnhancedToolManager.tool
    def finish_find_files_to_fix(self, files: List[str]):
        """
        Signals completion of the file finding workflow execution
        Arguments:
            files: The list of files to fix.
        """
        self.files_to_fix = files
        return files


def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
) -> str:
    global run_id, _current_tool_manager
    run_id = run_id_1
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "generate_test_cases_from_root_cause",
            "list_directory_structure",
            "get_file_content",
            "get_function_body",
            "find_symbol_references",
            "grep_search",
            "search_in_file",
            "apply_code_edit",
            "run_code",
            "run_tests",
            "run_shell_cmd",
            "think",
            # Strategy tracking
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            # Enhanced feedback loop - hypothesis tracking
            "create_hypothesis",
            "test_hypothesis",
            "list_hypotheses",
            # Enhanced feedback loop - test progress
            "get_test_progress",
            "analyze_test_failure",
            "finish",
        ],
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
        is_fix_task=True,
    )
    _current_tool_manager = tool_manager
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX,
    )
    enhanced_problem = problem_statement
    if enhancement:
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem)
    patch, _ = execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        [QWEN_MODEL_NAME, KIMI_MODEL_NAME],
        log_prefix="FIX_MAIN_AGENT",
    )
    return patch

    
def get_problem_type(problem_statement: str, enhancement: str) -> str:
    retry = 0
    PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
        """
        You are a helpful Problem Classifier to find a Task Name from PROJECT DESCRIPTION and project structure.
        Classify development tasks as one of:
        - CREATE: If the PROJECT DESCRIPTION is about creating a new functionality from scratch.
        - FIX: If the PROJECT DESCRIPTION is about fixing a bug, error, or issue in existing code.
        - BUG_INJECTION: If the PROJECT DESCRIPTION is about introducing a bug, making code break, or causing tests to fail (opposite of fixing).
        Output ONLY: "CREATE", "FIX", or "BUG_INJECTION"
        """
    )
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [{"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT}, {"role": "user", "content": f"{problem_statement}\n# Enhanced Problem: \n{enhancement}"}]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model)
            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX, PROBLEM_TYPE_BUG_INJECTION]:
                retry += 1
            else:
                return response
        except Exception as e:
            retry += 1
            if retry > 4:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return PROBLEM_TYPE_FIX

def check_problem_type(problem_statement):
    type_count = {
        PROBLEM_TYPE_CREATE: 0,
        PROBLEM_TYPE_FIX: 0,
        PROBLEM_TYPE_BUG_INJECTION: 0
    }
    enhancement = ""
    for _ in range(3):
        problem_type = get_problem_type(problem_statement, enhancement)
        type_count[problem_type] += 1
    
    # Return the type with the highest count
    max_type = max(type_count, key=type_count.get)
    return max_type

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    global DEFAULT_PROXY_URL, run_id, agent_start_time
    agent_start_time = time.time()
    run_id = os.getenv("EVALUATION_RUN_ID", "")
    default_timeout = int(os.getenv("TIMEOUT", "1500"))
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    set_env_for_agent()

    timeout = default_timeout
    result = None
    exception_occurred = None
    task_completed = threading.Event()

    def run_task():
        nonlocal result, exception_occurred
        enhancement = ""  # Initialize to avoid NameError in exception handler
        try:
            global _current_tool_manager
            print(f"checking problem type...")
            problem_type = check_problem_type(input_dict.get("problem_statement", ""))
            print(f"Problem Type: {problem_type}")
            # time.sleep(100)

            _current_tool_manager = EnhancedToolManager()
            if problem_type != PROBLEM_TYPE_BUG_INJECTION:
                result = process_fix_task(input_dict, enhancement)
            else:
                result = process_bug_injection_task(input_dict, timeout)
        except Exception as e:
            logger.error(f"Error in agent_main: {e}, {traceback.format_exc()}")
            exception_occurred = e
            try:
                time.sleep(1)
                result = process_fix_task(input_dict, enhancement)
            except Exception as e2:
                exception_occurred = e2
        finally:
            task_completed.set()

    task_thread = threading.Thread(target=run_task, daemon=True)
    task_thread.start()
    task_thread.join(timeout=timeout)

    timed_out = task_thread.is_alive()
    if timed_out:
        logger.warning(f"Task execution timed out after {timeout} seconds, killing thread")

    global _current_tool_manager
    if _current_tool_manager is not None and result is None:
        try:
            final_patch = _current_tool_manager.get_final_git_patch()
            if final_patch:
                result = final_patch
        except Exception as e:
            logger.warning(f"Failed to get final patch from tool manager: {e}")
        finally:
            _current_tool_manager = None

    try:
        subprocess.Popen(["git", "reset", "--hard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    print(f"[AGENT MAIN] Result: {result}")
    return result if result else ""

def clean_code_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response

def process_bug_injection_task(problem_statement: str, timeout: int):
    global run_id, _current_tool_manager, run_id
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "list_directory_structure",
            "get_file_content",
            "get_function_body",
            "find_symbol_references",
            "grep_search",
            "search_in_file",
            "apply_code_edit",
            "run_code",
            "run_tests",
            "run_shell_cmd",
            "finish_bug_injection"
        ],
        initial_checkpoint=None,
        problem_statement=problem_statement,
        should_review=False,
        is_fix_task=True,
    )
    _current_tool_manager = tool_manager

    BUG_INJECTION_TASK_PROMPT = textwrap.dedent(
        """
        Your role is to inject certain type of bugs into a code that causes the given tests that are currently passing to fails.
        Your primiary role includes 2 main aspects:
        - Find the best place to inject the bugs to make the given tests failing and inject bugs.
        - Generate problem statement that explains the bugs happened so that the the other software engineers can fix the whole bugs with the problem statement and repository only.

        Follow these steps:
        - Read the given test files to understand their purpsoes.
        - Read the relevant files that those tests includes.
        - Inject some bugs into those relevant files.
        - Run the test to see if test actually fails and see the full failure error message
        - Based on those errors, you should generate problem statement, the quiz that we should provide to software engineers that will fix the bugs with that problem statement only.

        Important Rules:
        **- Problem statement shouldn't include where and how you injected bugs. It should be quiz that we will provide to the software engineers to solve.**
        **- Never mention the clue where and how you injected bugs.**
        
        # Tool Documentation
        You have access to the following tools:-
        {tools_docs}

        Here is the bug injection requirements:
        {problem_statement}

        # Response Format Requirements
        {format_prompt}
        """
    )

    system_prompt = BUG_INJECTION_TASK_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX,
    )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    bug_description, patch, _ = execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        50,
        timeout,
        [QWEN_MODEL_NAME],
        log_prefix="FIX_MAIN_AGENT",
        finish_tool_name="finish_bug_injection"
    )

    return {
        "bug_description": bug_description,
        "patch": patch
    }
    

def process_fix_task(input_dict: Dict[str, Any], enhancement: str):
    global run_id, agent_start_time
    problem_text = input_dict.get("problem_statement")
    patch_text = ""
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    set_env_for_agent()
    cwd = os.getcwd()

    try:
        elapsed_time = time.time() - agent_start_time
        patch_text = fix_task_solve_workflow(problem_text, timeout=1300 - elapsed_time, run_id_1=run_id, enhancement=enhancement, should_review=True)
        os.system("git reset --hard")
    except Exception as e:
        logger.error(f"Error in process_fix_task: {e}, {traceback.format_exc()}")
    finally:
        os.chdir(cwd)
    return patch_text

def execute_agent_workflow(
    cot: EnhancedCOT,
    tool_manager: EnhancedToolManager,
    system_prompt: str,
    instance_prompt: str,
    n_max_steps: int,
    timeout: int,
    models: List[str],
    log_prefix: str = "AGENT",
    finish_tool_name="finish",
    reject_observation_token_threshold: int = 50000,
    save_observation_to_file_token_threshold: int = 4000,
) -> tuple[str, bool]:
    global run_id
    logger.info(f"{log_prefix} Starting agent execution... ")
    start_time = time.time()
    raw_text = ""
    total_attempts = 0
    error_counter = {}
    next_thought = None
    next_tool_name = None
    next_tool_args = None
    modified_files = set()
    files_with_syntax_errors = set()
    current_model_index = 0

    def _safe_call_tool(tool_manager: EnhancedToolManager, tool_name: str, tool_args):
        """
        General safety layer:
        - If get_tool returns an error string, return it as observation (don't crash)
        - Drop unexpected kwargs (prevents occasional 'unexpected keyword argument' failures)
        - Coerce common "list[str]" args passed as a single string (e.g. "ls -l")
        """
        tool_fn = tool_manager.get_tool(tool_name)
        if isinstance(tool_fn, str):
            return tool_fn

        if tool_args is None or tool_args == {}:
            return tool_fn()
        if not isinstance(tool_args, dict):
            # If model returned something weird, don't crash
            return tool_fn()

        # Filter kwargs to what the tool actually accepts
        try:
            sig = inspect.signature(tool_fn)
            allowed = set(sig.parameters.keys())
            allowed.discard("self")
        except Exception:
            allowed = set(tool_args.keys())

        cleaned = {k: v for k, v in tool_args.items() if k in allowed}

        # Light coercion for list-like params if the model returns a single string
        try:
            for k in list(cleaned.keys()):
                v = cleaned[k]
                p = sig.parameters.get(k)
                ann = str(getattr(p, "annotation", ""))
                if v is not None and isinstance(v, str) and ("List" in ann or "list" in ann):
                    # Turn "ls -l" into ["ls", "-l"]
                    cleaned[k] = v.split() if v.strip() else []
        except Exception:
            pass

        return tool_fn(**cleaned) if cleaned else tool_fn()

    for step in range(n_max_steps):
        selected_model = models[current_model_index]
        elapsed_time = time.time() - start_time
        logger.info("=" * 40 + f"[{log_prefix}] Step {step}" + "=" * 40)
        cost_usage = EnhancedNetwork.get_cost_usage()
        logger.info(
            f"[{log_prefix}] Elapsed time: {elapsed_time}/{timeout} seconds, Usage: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
        )
        if cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", 0):
            logger.warning(f"[{log_prefix}] Usage exceeded limit: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD")
            break
        if time.time() - start_time > timeout:
            logger.info(f"[{log_prefix}] Global timeout reached")
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[],
                )
            )
            break
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})

        if cot.is_thought_repeated():
            logger.info(f"[TEMPERATURE] Thought repeated {cot.repeated_thoughts} times")
            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_thought:{last_thought.next_thought}\n next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )
            temperature = 0.5
            if cot.repeated_thoughts >= 2:
                model_idx = (cot.repeated_thoughts - 2) % len(models)
                selected_model = models[model_idx]
        else:
            temperature = 0.0

        try:
            inference_start_time = time.time()
            models_to_try = [selected_model] + [m for m in models if m != selected_model]
            (
                next_thought,
                next_tool_name,
                next_tool_args,
                raw_text,
                total_attempts,
                error_counter,
                messages,
                used_model,
            ) = EnhancedNetwork.inference(messages, model=models_to_try, run_id=run_id, temperature=temperature)
            selected_model = used_model
            inference_duration = time.time() - inference_start_time
        except Exception as e:
            inference_duration = 0
            logger.error(f"[{log_prefix}] Inference error: {e}")
            is_timeout_error = "Agent execution timeout" in str(e)
            if is_timeout_error:
                cot.add_action(
                    EnhancedCOT.Action(
                        next_thought="global timeout reached",
                        next_tool_name="",
                        next_tool_args={},
                        observation="",
                        is_error=True,
                        inference_error_counter={},
                        request_data=[],
                    )
                )
                return tool_manager.get_final_git_patch(), False

        tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
        tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]

        logger.info(f"[{log_prefix}] Used model: {selected_model}, Inference time: {inference_duration:.2f}s")
        logger.info(f"[{log_prefix}] Next thought: {next_thought}\n\n")
        logger.info(f"[{log_prefix}] About to execute {len(tool_names_list)} tool call(s): {tool_names_list}\n")
        logger.info(f"[{log_prefix}] Tool arguments: {json.dumps(tool_args_list, indent=4)}\n\n")

        tool_manager._current_step = step
        tool_manager._cot_snapshot_cache = [
            {
                "thought": t.next_thought,
                "tool": t.next_tool_name,
                "args": str(t.next_tool_args)[:200],
                "success": not t.is_error,
            }
            for t in cot.thoughts[-10:]
        ]
        all_observations = []
        all_successful = True
        for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
            try:
                if '"' in tool_name or "'" in tool_name:
                    tool_name = tool_name.replace('"', "").replace("'", "")
                observation = _safe_call_tool(tool_manager, tool_name, tool_args)
                if tool_name == "apply_code_edit" and tool_args and "file_path" in tool_args:
                    file_path = tool_args["file_path"]
                    if "ok, code edit applied successfully" in str(observation).lower():
                        modified_files.add(file_path)
                    elif "syntax error" in str(observation).lower():
                        files_with_syntax_errors.add(file_path)
                
                # Enhanced feedback loop: inject feedback after test runs
                if tool_name in ["run_tests", "run_code"] and hasattr(tool_manager, '_inject_test_feedback'):
                    observation = tool_manager._inject_test_feedback(str(observation), tool_name, tool_args)
                
                estimated_tokens = Utils.count_tokens(str(observation))
                if estimated_tokens > reject_observation_token_threshold:
                    observation = f"Error: Tool output from '{tool_name}' exceeded token limit ({estimated_tokens} tokens > 50000 tokens limit). The response is too large to process. Please use more specific queries, target smaller file ranges, or break the request into smaller operations."
                elif estimated_tokens > save_observation_to_file_token_threshold:
                    observation_path = tool_manager._save_large_observation(str(observation), tool_name)
                    observation = f"Tool output from `{tool_name}` exceeded token limit ({estimated_tokens} tokens > 4000 tokens limit). The full output has been saved to: {observation_path}. You can read this file using the get_file_content tool if needed, but specify the start and end line numbers to read the file."
                all_observations.append(observation)
            except EnhancedToolManager.Error as e:
                error_msg = f"Tool {idx+1} ({tool_name}) error: {e.message}"
                all_observations.append(error_msg)
                all_successful = False
            except Exception as e:
                import traceback

                error_traceback = traceback.format_exc()
                error_msg = f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                all_observations.append(error_msg)
                all_successful = False

        if len(all_observations) == 1:
            combined_observation = all_observations[0]
        else:
            combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                [f"Tool {i+1} ({tool_names_list[i]}):\n{obs}" for i, obs in enumerate(all_observations)]
            )

        logger.info(f"[{log_prefix}] Combined observation: {combined_observation}\n\n")
        cot.add_action(
            EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,  # Keep original format (list or single)
                next_tool_args=next_tool_args,
                observation=combined_observation,
                is_error=not all_successful,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages,
            )
        )
        if finish_tool_name in tool_names_list:
            if finish_tool_name == "finish_find_files_to_fix":
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        return obs, False
            if finish_tool_name == "finish_bug_injection":
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        return obs, tool_manager.get_final_git_patch(), False
            elif finish_tool_name == "finish":
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        if obs != "finish":
                            break
                        return tool_manager.get_final_git_patch(), True
    return tool_manager.get_final_git_patch(), False


def set_env_for_agent():
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
    if Path(os.getcwd() + "/lib").exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.error(f"ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)


def enhance_problem_statement(problem_statement: str) -> str:
    ENHANCEMENT_PROMPT = textwrap.dedent(
        """
        You are an expert at analyzing problem statements and extracting key information.
        Analyze the given problem statement and extract the following structured information:
        1. **Problem Summary** (1-2 sentences): What needs to be fixed or implemented?
        2. **Current Behavior**: What is happening now? (Include error messages, unexpected outputs, etc.)
        3. **Expected Behavior**: What should happen instead?
        4. **Reproduction Steps** (if applicable): Clear steps to reproduce the issue
        5. **Success Criteria**: How will we know the problem is solved?
            - What tests should pass?
            - What behavior should change?
            - What outputs should be different?
        6. **Key Requirements**:
            - Must-have functionality
            - Constraints to respect (backwards compatibility, performance, etc.)
            - Files/functions likely involved
        7. **Important Notes**:
            - Edge cases to consider
            - Potential pitfalls
            - Related functionality that might be affected
        If any section is not applicable or cannot be determined from the problem statement, write "Not specified" for that section.
        Format your response as markdown with clear section headers.
        Be concise but complete. Extract information that's present, don't invent details.
        """
    )
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": ENHANCEMENT_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n\n{problem_statement}",
                },
            ]
            enhanced, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            return enhanced
        except Exception as e:
            retry += 1
            other_models = [model for model in AGENT_MODELS if model != selected_model]
            selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return ""
