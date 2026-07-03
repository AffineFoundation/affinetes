"""End-to-end Docker test for Go Layer 2 (trusted-TestMain aggregate cross-check).

Go is statically compiled, so the runner cannot be monkeypatched at runtime —
the dynamic-language attack does not exist. boundary.py (A) already forbids the
miner from adding a TestMain / *_test.go. The residual in-process vector is a
package init() that fabricates the `go test -json` stream.

Layer 2 materializes a trusted TestMain in the test package; it wraps m.Run()
and writes the REAL aggregate exit code out-of-band (init() cannot forge the
framework's verdict — Go has no monkeypatching). env.py cross-checks: stdout
claiming all-pass while the real run failed is fabrication.

  gold fix                       -> 1.0   (no false positive)
  init() fabricates all-pass (L2 OFF) -> 1.0   (go test -json stdout fooled)
  init() fabricates all-pass (L2 ON)  -> 0.0   (runner_integrity_violation)

CAVEAT: the aggregate only catches BLANKET fabrication / output suppression.
SELECTIVE fabrication (fake the graded test pass, leave the canary failing) keeps
the aggregate consistent and is NOT caught here — it needs trusted per-test
capture, which Go's TestMain does not expose. See the summary.

Prereq: image swe-inf-boundary-go:latest (test_boundary_go_e2e_build.sh).
Run:    python3 environments/SWE-INFINITE/test_boundary_go_e2e.py
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
from env import InfiniteActor

IMAGE = "swe-inf-boundary-go:latest"
GRADED = "example.com/calc/calc::TestAdd"

TEST_PATCH = (
    "diff --git a/calc/calc_test.go b/calc/calc_test.go\n"
    "new file mode 100644\n--- /dev/null\n+++ b/calc/calc_test.go\n"
    "@@ -0,0 +1,5 @@\n"
    "+package calc\n"
    "+import \"testing\"\n"
    "+func TestAdd(t *testing.T) {\n"
    "+\tif Add(2, 3) != 5 { t.Errorf(\"got %d\", Add(2, 3)) }\n"
    "+}\n"
)

GOLD_PATCH = (
    "diff --git a/calc/calc.go b/calc/calc.go\n"
    "--- a/calc/calc.go\n+++ b/calc/calc.go\n"
    "@@ -1,5 +1,5 @@\n"
    " package calc\n"
    " \n"
    " func Add(a, b int) int {\n"
    "-\treturn a - b\n"
    "+\treturn a + b\n"
    " }\n"
)

# init() suppresses the real -json stream and fabricates an all-pass for the
# graded test (bug left unfixed).
_SUBVERT = [
    "package calc",
    "",
    "import (",
    "\t\"fmt\"",
    "\t\"os\"",
    ")",
    "",
    "func init() {",
    "\treal := os.Stdout",
    "\tif dn, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0); err == nil {",
    "\t\tos.Stdout = dn",
    "\t}",
    "\tfmt.Fprintln(real, \"=== RUN   TestAdd\")",
    "\tfmt.Fprintln(real, \"--- PASS: TestAdd (0.00s)\")",
    "\tfmt.Fprintln(real, \"PASS\")",
    "}",
    "",
    "func Add(a, b int) int {",
    "\treturn a - b",
    "}",
]
ATTACK_PATCH = (
    "diff --git a/calc/calc.go b/calc/calc.go\n"
    "--- a/calc/calc.go\n+++ b/calc/calc.go\n"
    f"@@ -1,5 +1,{len(_SUBVERT)} @@\n"
    "-package calc\n"
    "-\n"
    "-func Add(a, b int) int {\n"
    "-\treturn a - b\n"
    "-}\n"
    + "".join(f"+{l}\n" for l in _SUBVERT)
)


def _make_task():
    return {
        "instance_id": "boundary-go-1",
        "dockerhub_tag": IMAGE,
        "repo_language": "go",
        "test_command": "go test -json ./...",
        "fail_to_pass": [GRADED],
        "pass_to_pass": [],
        "test_patch": TEST_PATCH,
        "augmented_test_patch": "",
    }


def _verify(task, patch):
    return InfiniteActor.__new__(InfiniteActor)._verify(task, patch)


def _run_attack(disable_layer2, disable_canary):
    """Run the fabrication attack. The aggregate cross-check overlaps with the
    canary for blanket fabrication, so to isolate Layer 2's INDEPENDENT value we
    can disable the canary (generate_canary -> None) and show the aggregate is
    the backstop that still catches it."""
    real_gen = env.generate_canary
    if disable_canary:
        env.generate_canary = lambda *a, **k: None
    if disable_layer2:
        os.environ["SWE_INFINITE_DISABLE_ORACLE"] = "1"
    try:
        return _verify(_make_task(), ATTACK_PATCH)
    finally:
        env.generate_canary = real_gen
        os.environ.pop("SWE_INFINITE_DISABLE_ORACLE", None)


def main() -> int:
    if not subprocess.run(["docker", "images", "-q", IMAGE],
                          capture_output=True, text=True).stdout.strip():
        print(f"ERROR: image {IMAGE} missing. Run test_boundary_go_e2e_build.sh first.")
        return 2

    # gold with everything on -> proves no false positive from the injected TestMain.
    s_gold, st_gold = _verify(_make_task(), GOLD_PATCH)
    # Isolate the aggregate's independent value: canary OFF in both rows below.
    s_base, st_base = _run_attack(disable_layer2=True, disable_canary=True)   # baseline: both off
    s_l2, st_l2 = _run_attack(disable_layer2=False, disable_canary=True)      # aggregate only

    print()
    print("Go — init() fabricates the go test -json stream (boundary A+B ON):")
    print("-" * 78)
    print(f"gold fix (everything ON):                    score={s_gold}  stats={st_gold}")
    print(f"blanket fab (canary OFF, Layer 2 OFF):       score={s_base}  stats={st_base}")
    print(f"blanket fab (canary OFF, Layer 2 ON):        score={s_l2}  stats={st_l2}")
    print("-" * 78)

    gold_ok = abs(s_gold - 1.0) < 1e-9
    base_bypass = abs(s_base - 1.0) < 1e-9
    l2_caught = abs(s_l2 - 0.0) < 1e-9 and st_l2.get("error") == "runner_integrity_violation"

    print()
    if gold_ok and base_bypass and l2_caught:
        print("PASS: gold=1.0 (no false positive); with the canary disabled, blanket "
              "fabrication fools go test -json (1.0), and Layer 2's trusted-TestMain "
              f"aggregate is the backstop that catches it (0.0, reasons={st_l2.get('reasons')}).")
        print("NOTE: for blanket fabrication this overlaps the canary; SELECTIVE "
              "fabrication (fake graded pass, leave canary failing) evades the aggregate.")
        return 0
    print(f"FAIL: gold_ok={gold_ok} base_bypass={base_bypass} l2_caught={l2_caught}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
