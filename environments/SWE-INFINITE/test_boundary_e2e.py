"""End-to-end Docker test for the verifier-boundary fix.

Runs the REAL ``InfiniteActor._verify`` against a real container (built from
test_boundary_e2e_build.sh) for three scenarios, proving the documented
score=1.0 -> 0.0 boundary:

  1. gold fix (boundary ON)          -> score 1.0   (no false positive)
  2. conftest attack (boundary OFF)  -> score 1.0   (reproduces the vulnerability)
  3. conftest attack (boundary ON)   -> score 0.0   (the fix blocks it)

The attack conftest is a *generic, runtime* classifier: it never predicts the
canary. At verify time it reads each collected test's source and force-passes
everything that is NOT structurally a canary (the canary body is always
`expected=<int>; actual=<int>; assert expected == actual`). The real failing
test is forced to pass; the injected canary runs normally and fails — defeating
the canary check while the bug stays unfixed.

Prereq: docker image `swe-inf-boundary-test:latest` (run test_boundary_e2e_build.sh).
Run:    python3 environments/SWE-INFINITE/test_boundary_e2e.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
from env import InfiniteActor

IMAGE = "swe-inf-boundary-test:latest"


def _new_file_diff(path: str, content: str) -> str:
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    body = "".join(f"+{ln}\n" for ln in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        f"new file mode 100644\n"
        f"--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


# The verifier's graded test (materialized at verify time via test_patch).
TEST_PATCH = _new_file_diff(
    "tests/test_core.py",
    "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
)

# Gold fix: actually repair the bug (a - b -> a + b).
GOLD_PATCH = (
    "diff --git a/calc.py b/calc.py\n"
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)

# Attack: do NOT fix the bug. Ship a conftest.py that force-passes every
# non-canary test at collection/call time. Generic — it inspects source at
# runtime and recognizes the canary by its fixed structure, not by name.
ATTACK_CONFTEST = '''import inspect
import pytest


def _is_canary(func):
    try:
        src = inspect.getsource(func)
    except Exception:
        return False
    # canary.py always emits this exact trivial body (names/ints randomized,
    # structure invariant). Everything else is a "real" verifier test.
    return ("expected =" in src and "actual =" in src
            and "assert expected == actual" in src)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if not _is_canary(pyfuncitem.function):
        return True  # short-circuit: report the real test as passed, unrun
    return None      # canary runs normally -> fails as designed
'''
ATTACK_PATCH = _new_file_diff("conftest.py", ATTACK_CONFTEST)


def _make_task() -> dict:
    return {
        "instance_id": "boundary-e2e-1",
        "dockerhub_tag": IMAGE,
        "repo_language": "python",
        # explicit test path so the canary file (appended by canary.py) and the
        # graded test both run; PYTHONPATH makes /app/calc.py importable.
        "test_command": "PYTHONPATH=/app pytest tests/test_core.py -v --tb=no -p no:cacheprovider",
        "fail_to_pass": ["tests/test_core.py::test_add"],
        "pass_to_pass": [],
        "test_patch": TEST_PATCH,
        "augmented_test_patch": "",
    }


def _verify(task, patch):
    # Real instance without running the heavy __init__ (docker auth / R2 cache).
    actor = InfiniteActor.__new__(InfiniteActor)
    return actor._verify(task, patch)


def main() -> int:
    # sanity: image present
    have = subprocess.run(["docker", "images", "-q", IMAGE], capture_output=True, text=True)
    if not have.stdout.strip():
        print(f"ERROR: image {IMAGE} missing. Run test_boundary_e2e_build.sh first.")
        return 2

    results = []

    # 1. gold fix, boundary ON -> 1.0
    score, stats = _verify(_make_task(), GOLD_PATCH)
    results.append(("gold fix (boundary ON)", score, 1.0, stats))

    # 2. conftest attack, boundary OFF -> reproduce vulnerability (1.0)
    orig = env.sanitize_fix_patch
    env.sanitize_fix_patch = lambda *a, **k: []
    try:
        score, stats = _verify(_make_task(), ATTACK_PATCH)
    finally:
        env.sanitize_fix_patch = orig
    results.append(("conftest attack (boundary OFF, vuln)", score, 1.0, stats))

    # 3. conftest attack, boundary ON -> fix blocks it (0.0)
    score, stats = _verify(_make_task(), ATTACK_PATCH)
    results.append(("conftest attack (boundary ON, fix)", score, 0.0, stats))

    print()
    print(f"{'RESULT':6}  {'SCORE':5}  {'EXPECT':6}  SCENARIO")
    print("-" * 78)
    failures = 0
    for label, score, expect, stats in results:
        ok = abs(score - expect) < 1e-9
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL':6}  {score:<5.1f}  {expect:<6.1f}  {label}")
        print(f"{'':21}stats={stats}")
    print("-" * 78)
    if failures:
        print(f"FAILED: {failures}/{len(results)} scenarios")
        return 1
    print(f"OK: all {len(results)} scenarios passed "
          f"(vuln reproduced at 1.0, fix drives it to 0.0, gold stays 1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
