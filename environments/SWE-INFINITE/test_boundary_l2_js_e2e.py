"""End-to-end Docker test for Layer 2 (JavaScript / Jest oracle).

JS tasks have NO canary (canary.py supports only go/python/ruby), so A+B are the
only other defenses — and they do not stop an in-process attack hidden in a
regular implementation source file (src/calc.js, allowed by the boundary filter)
that neuters `expect` at require time so every assertion silently passes. The
Jest oracle (NODE_OPTIONS preload -> reporter + setupFilesAfterEnv) catches it by
detecting that `expect` was replaced inside the test vm.

Prereq: docker image swe-inf-boundary-test-js:latest (test_boundary_l2_js_build.sh).
Run:    python3 environments/SWE-INFINITE/test_boundary_l2_js_e2e.py
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
from env import InfiniteActor

IMAGE = "swe-inf-boundary-test-js:latest"


def _new_file_diff(path, content):
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    body = "".join(f"+{l}\n" for l in lines)
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\n"
        f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"
    )


def _replace_calc(new_content):
    old = ["function add(a, b) {", "  return a - b;", "}", "module.exports = { add };"]
    new = [l for l in new_content.split("\n")]
    if new and new[-1] == "":
        new = new[:-1]
    minus = "".join(f"-{l}\n" for l in old)
    plus = "".join(f"+{l}\n" for l in new)
    return (
        "diff --git a/src/calc.js b/src/calc.js\n--- a/src/calc.js\n+++ b/src/calc.js\n"
        f"@@ -1,{len(old)} +1,{len(new)} @@\n{minus}{plus}"
    )


TEST_PATCH = _new_file_diff(
    "__tests__/calc.test.js",
    "const { add } = require('../src/calc');\n"
    "test('adds two numbers', () => {\n"
    "  expect(add(1, 2)).toBe(3);\n"
    "});\n",
)

GOLD_PATCH = _replace_calc(
    "function add(a, b) {\n  return a + b;\n}\nmodule.exports = { add };\n"
)

# Attack: keep the bug; neuter global.expect at require time so the assertion
# never throws -> jest reports the test as passed.
ATTACK_PATCH = _replace_calc(
    "global.expect = function (_actual) {\n"
    "  return new Proxy({}, { get: () => () => {} });\n"
    "};\n"
    "function add(a, b) {\n  return a - b;\n}\nmodule.exports = { add };\n"
)


def _make_task():
    return {
        "instance_id": "boundary-l2-js-1",
        "dockerhub_tag": IMAGE,
        "repo_language": "javascript",
        "test_command": "jest --json",
        "fail_to_pass": ["__tests__/calc.test.js::adds two numbers"],
        "pass_to_pass": [],
        "test_patch": TEST_PATCH,
        "augmented_test_patch": "",
    }


def _verify(task, patch):
    return InfiniteActor.__new__(InfiniteActor)._verify(task, patch)


def _run_attack(disable_layer2):
    if disable_layer2:
        os.environ["SWE_INFINITE_DISABLE_ORACLE"] = "1"
    try:
        return _verify(_make_task(), ATTACK_PATCH)
    finally:
        os.environ.pop("SWE_INFINITE_DISABLE_ORACLE", None)


def main():
    if not subprocess.run(["docker", "images", "-q", IMAGE],
                          capture_output=True, text=True).stdout.strip():
        print(f"ERROR: image {IMAGE} missing. Run test_boundary_l2_js_build.sh first.")
        return 2

    gold_score, gold_stats = _verify(_make_task(), GOLD_PATCH)
    s_off, st_off = _run_attack(disable_layer2=True)
    s_on, st_on = _run_attack(disable_layer2=False)

    print()
    print("JS in-process expect-neuter hidden in src/calc.js (no canary for JS):")
    print("-" * 78)
    print(f"gold fix (Layer 2 ON):              score={gold_score}  stats={gold_stats}")
    print(f"expect-neuter attack (Layer 2 OFF): score={s_off}  stats={st_off}")
    print(f"expect-neuter attack (Layer 2 ON):  score={s_on}  stats={st_on}")
    print("-" * 78)

    gold_ok = abs(gold_score - 1.0) < 1e-9
    off_bypass = abs(s_off - 1.0) < 1e-9
    on_caught = abs(s_on - 0.0) < 1e-9 and st_on.get("error") == "runner_integrity_violation"

    print()
    if gold_ok and off_bypass and on_caught:
        print("PASS: gold=1.0 (no false positive); A+B MISS the in-source expect-neuter "
              "(1.0, no canary for JS);")
        print(f"      Layer 2 CATCHES it (0.0, reasons={st_on.get('reasons')}).")
        return 0
    print(f"FAIL: gold_ok={gold_ok} off_bypass={off_bypass} on_caught={on_caught}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
