"""Shared utilities for SWE-INFINITE environment."""

import json
import re

# Source code file extensions for git diff filtering
DIFF_EXTENSIONS = (
    "'*.js' '*.ts' '*.jsx' '*.tsx' '*.py' '*.java' '*.go' "
    "'*.c' '*.cpp' '*.h' '*.rs' '*.rb' '*.php' '*.cs' "
    "'*.swift' '*.kt' '*.scala' '*.vue' '*.svelte'"
)

# Git history sanitization script (prevent cheating via git log)
SANITIZE_GIT_SCRIPT = """
cd /app
git config user.email "agent@swe-infinite.local"
git config user.name "SWE-INFINITE Agent"
git checkout --orphan sanitized_branch
git add -A
git commit -m "Initial state"
git branch -D main 2>/dev/null || git branch -D master 2>/dev/null || true
git branch -m main
rm -rf .git/logs
rm -rf .git/refs/original
git reflog expire --expire=now --all 2>/dev/null || true
git gc --prune=now 2>/dev/null || true
echo "Git history sanitized"
"""

# Normalize file timestamps to prevent fingerprinting via mtime
NORMALIZE_TIMESTAMPS_SCRIPT = """
cd /app
find . -not -path './.git/*' -not -path './node_modules/*' -not -path './.venv/*' -not -path './vendor/*' -exec touch -t 202001010000 {} + 2>/dev/null || true
echo "Timestamps normalized"
"""

# Commands that fingerprint the codebase instead of solving the task
_BLACKLISTED_PATTERNS = [
    re.compile(r'\bsha256sum\b'),
    re.compile(r'\bmd5sum\b'),
    re.compile(r'\bsha1sum\b'),
    re.compile(r'\bsha512sum\b'),
    re.compile(r'find\b.*-mmin\b'),
    re.compile(r'find\b.*-mtime\b'),
    re.compile(r'find\b.*-newer\b'),
]


def is_blacklisted_command(cmd: str) -> bool:
    """Return True if the command matches a known fingerprinting/cheating pattern."""
    for pattern in _BLACKLISTED_PATTERNS:
        if pattern.search(cmd):
            return True
    return False


# ===== Multi-language test output parsers =====
# Adapted from affine-swe-infinite/src/validators/test_validator.py

# pytest -v --tb=no output line format
_PYTEST_RE = re.compile(r"^([\w/.\-]+(?:::\w[\w\[\].\-]*)+)\s+(PASSED|FAILED|ERROR)")

# cargo test verbose output line format
_CARGO_RE = re.compile(r"^test\s+([\w:]+)\s+\.\.\.\s+(ok|FAILED|ignored)")

# Minitest verbose output line format
_MINITEST_RE = re.compile(r"^(.+?)\s+=\s+[\d.]+\s+s\s+=\s+([.FES])\s*$")


def parse_test_output(
    stdout: str,
    stderr: str,
    language: str,
    test_command: str,
) -> tuple[set[str], set[str]]:
    """Parse test output into (passed, failed) sets.

    Matches upstream affine-swe-infinite logic:
    - JSON-based parsers (jest, mocha, rspec): stdout only
    - Others (go, rust, minitest, pytest): stdout + stderr combined
    """
    framework = _detect_framework(language, test_command)
    if framework in ("jest", "mocha", "rspec"):
        text = stdout
    else:
        text = stdout + "\n" + stderr
    passed, failed = _PARSERS[framework](text)
    return set(passed), set(failed)


def _detect_framework(language: str, test_command: str) -> str:
    """Detect test framework from language and test command."""
    cmd_lower = test_command.lower()
    if language in ("javascript", "typescript"):
        if "mocha" in cmd_lower:
            return "mocha"
        if "vitest" in cmd_lower:
            return "jest"  # vitest uses jest-compatible JSON
        return "jest"
    if language == "go":
        return "go"
    if language == "rust":
        return "cargo"
    if language == "ruby":
        if "rspec" in cmd_lower:
            return "rspec"
        return "minitest"
    return "pytest"


def _parse_pytest(stdout: str) -> tuple[list[str], list[str]]:
    passed, failed = [], []
    for line in stdout.splitlines():
        m = _PYTEST_RE.match(line.strip())
        if m:
            test_id, status = m.group(1), m.group(2)
            (passed if status == "PASSED" else failed).append(test_id)
    return passed, failed


def _parse_jest(stdout: str) -> tuple[list[str], list[str]]:
    """Parse Jest/Vitest --json output."""
    passed, failed = [], []
    start = stdout.find("{")
    if start == -1:
        return passed, failed
    try:
        data = json.loads(stdout[start:])
    except Exception:
        end = stdout.rfind("}")
        if end == -1 or end <= start:
            return passed, failed
        try:
            data = json.loads(stdout[start:end + 1])
        except Exception:
            return passed, failed

    for suite in data.get("testResults", []):
        file_path = suite.get("testFilePath", "")
        if file_path.startswith("/app/"):
            file_path = file_path[len("/app/"):]
        for t in suite.get("assertionResults") or suite.get("testResults", []):
            full_name = t.get("fullName") or t.get("title", "")
            test_id = f"{file_path}::{full_name}"
            status = t.get("status", "")
            if status == "passed":
                passed.append(test_id)
            elif status == "failed":
                failed.append(test_id)
    return passed, failed


def _parse_mocha(stdout: str) -> tuple[list[str], list[str]]:
    """Parse Mocha --reporter json output."""
    passed, failed = [], []
    start = stdout.find("{")
    if start == -1:
        return passed, failed
    try:
        data = json.loads(stdout[start:])
    except Exception:
        return passed, failed
    for t in data.get("passes", []):
        passed.append(t.get("fullTitle", t.get("title", "")))
    for t in data.get("failures", []):
        failed.append(t.get("fullTitle", t.get("title", "")))
    return passed, failed


def _parse_go(stdout: str) -> tuple[list[str], list[str]]:
    """Parse `go test -json` output."""
    passed, failed = [], []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        action = event.get("Action", "")
        test_name = event.get("Test", "")
        package = event.get("Package", "")
        if not test_name:
            continue
        test_id = f"{package}::{test_name}"
        if action == "pass":
            passed.append(test_id)
        elif action == "fail":
            failed.append(test_id)
    return passed, failed


def _parse_cargo(stdout: str) -> tuple[list[str], list[str]]:
    """Parse `cargo test` verbose output."""
    passed, failed = [], []
    for line in stdout.splitlines():
        m = _CARGO_RE.match(line.strip())
        if m:
            test_id, status = m.group(1), m.group(2)
            if status == "ok":
                passed.append(test_id)
            elif status == "FAILED":
                failed.append(test_id)
    return passed, failed


def _parse_rspec(stdout: str) -> tuple[list[str], list[str]]:
    """Parse `rspec -f json` output."""
    passed, failed = [], []
    start = stdout.find("{")
    if start == -1:
        return passed, failed
    try:
        data = json.loads(stdout[start:])
    except Exception:
        end = stdout.rfind("}")
        if end == -1 or end <= start:
            return passed, failed
        try:
            data = json.loads(stdout[start:end + 1])
        except Exception:
            return passed, failed
    for ex in data.get("examples", []):
        test_id = ex.get("full_description", ex.get("description", ""))
        file_path = ex.get("file_path", "")
        if file_path:
            test_id = f"{file_path}::{test_id}"
        status = ex.get("status", "")
        if status == "passed":
            passed.append(test_id)
        elif status == "failed":
            failed.append(test_id)
    return passed, failed


def _parse_minitest(stdout: str) -> tuple[list[str], list[str]]:
    """Parse Minitest verbose output."""
    passed, failed = [], []
    for line in stdout.splitlines():
        m = _MINITEST_RE.match(line.strip())
        if m:
            test_id, status = m.group(1), m.group(2)
            if status == ".":
                passed.append(test_id)
            elif status in ("F", "E"):
                failed.append(test_id)
    return passed, failed


_PARSERS = {
    "pytest": _parse_pytest,
    "jest": _parse_jest,
    "mocha": _parse_mocha,
    "go": _parse_go,
    "cargo": _parse_cargo,
    "rspec": _parse_rspec,
    "minitest": _parse_minitest,
}
