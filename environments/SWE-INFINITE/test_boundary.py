"""Unit tests for the verifier-boundary fix patch sanitizer (boundary.py).

These are deterministic, Docker-free tests of the static half of the defense:
given a miner fix_patch and the set of verifier-owned test paths, does
``sanitize_fix_patch`` reject the three documented attack vectors (and direct
test-file edits) while leaving legitimate source-only fixes untouched?

Run: python3 environments/SWE-INFINITE/test_boundary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boundary import sanitize_fix_patch, extract_patch_paths


# A legitimate source-only hunk reused across attack cases: the attack patches
# below pair this real fix with a malicious control-surface file.
_SRC_FIX = (
    "diff --git a/src/calc/core.py b/src/calc/core.py\n"
    "--- a/src/calc/core.py\n"
    "+++ b/src/calc/core.py\n"
    "@@ -10,7 +10,7 @@ def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)

# ---- Attack vector 1: Python conftest.py / pytest.py runner hook ----
ATTACK_PY_CONFTEST = _SRC_FIX + (
    "diff --git a/conftest.py b/conftest.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/conftest.py\n"
    "@@ -0,0 +1,6 @@\n"
    "+import pytest\n"
    "+def pytest_collection_modifyitems(config, items):\n"
    "+    for it in items:\n"
    "+        if _looks_real(it):\n"
    "+            it.obj = lambda *a, **k: None\n"
)

ATTACK_PY_SHIM = _SRC_FIX + (
    "diff --git a/pytest.py b/pytest.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/pytest.py\n"
    "@@ -0,0 +1,3 @@\n"
    "+# shadows the installed pytest; rewrites non-canary tests then delegates\n"
    "+import runpy, sys\n"
)

ATTACK_PY_SITECUSTOMIZE = _SRC_FIX + (
    "diff --git a/sitecustomize.py b/sitecustomize.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/sitecustomize.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+import sys  # auto-imported at interpreter startup\n"
)

ATTACK_PY_PTH = _SRC_FIX + (
    "diff --git a/inject.pth b/inject.pth\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/inject.pth\n"
    "@@ -0,0 +1 @@\n"
    "+import os; os.system('...')\n"
)

# ---- Attack vector 2: Go main_test.go with TestMain ----
ATTACK_GO_TESTMAIN = (
    "diff --git a/server/handler.go b/server/handler.go\n"
    "--- a/server/handler.go\n"
    "+++ b/server/handler.go\n"
    "@@ -1,1 +1,1 @@\n"
    "-broken\n"
    "+fixed\n"
    "diff --git a/server/main_test.go b/server/main_test.go\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/server/main_test.go\n"
    "@@ -0,0 +1,5 @@\n"
    "+package server\n"
    "+import \"testing\"\n"
    "+func TestMain(m *testing.M) { /* emit fabricated -json events */ }\n"
)

# ---- Attack vector 3: Ruby .rspec / spec_helper.rb startup hook ----
ATTACK_RB_RSPEC = (
    "diff --git a/lib/widget.rb b/lib/widget.rb\n"
    "--- a/lib/widget.rb\n"
    "+++ b/lib/widget.rb\n"
    "@@ -1,1 +1,1 @@\n"
    "-bad\n"
    "+good\n"
    "diff --git a/.rspec b/.rspec\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/.rspec\n"
    "@@ -0,0 +1 @@\n"
    "+--require ./hijack\n"
)

ATTACK_RB_SPEC_HELPER = _SRC_FIX.replace("src/calc/core.py", "lib/widget.rb") + (
    "diff --git a/spec/spec_helper.rb b/spec/spec_helper.rb\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/spec/spec_helper.rb\n"
    "@@ -0,0 +1,2 @@\n"
    "+RSpec.configure { |c| c.before(:suite) { rewrite_specs } }\n"
)

ATTACK_RB_SUPPORT_HOOK = (
    "diff --git a/spec/support/hijack.rb b/spec/support/hijack.rb\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/spec/support/hijack.rb\n"
    "@@ -0,0 +1 @@\n"
    "+# auto-required by spec_helper\n"
)

# ---- Attack vector 4: direct edit of the verifier's graded test file ----
ATTACK_DIRECT_TEST_EDIT = (
    "diff --git a/tests/test_core.py b/tests/test_core.py\n"
    "--- a/tests/test_core.py\n"
    "+++ b/tests/test_core.py\n"
    "@@ -5,1 +5,1 @@ def test_add():\n"
    "-    assert add(1, 2) == 3\n"
    "+    assert True\n"
)

# new *_test.go / *_spec.rb in fix even without TestMain text — still a new
# test file the fix must never introduce.
ATTACK_GO_NEW_TESTFILE = (
    "diff --git a/pkg/extra_test.go b/pkg/extra_test.go\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/pkg/extra_test.go\n"
    "@@ -0,0 +1 @@\n"
    "+package pkg\n"
)

# ---- Legitimate fixes (must NOT be blocked) ----
GOLD_PY_SOURCE = _SRC_FIX
GOLD_PY_NEW_MODULE = (
    "diff --git a/src/calc/extra.py b/src/calc/extra.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/src/calc/extra.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def helper():\n"
    "+    return 1\n"
)
GOLD_GO_SOURCE = (
    "diff --git a/server/handler.go b/server/handler.go\n"
    "--- a/server/handler.go\n"
    "+++ b/server/handler.go\n"
    "@@ -1,1 +1,1 @@\n"
    "-broken\n"
    "+fixed\n"
)
GOLD_RB_SOURCE = (
    "diff --git a/lib/widget.rb b/lib/widget.rb\n"
    "--- a/lib/widget.rb\n"
    "+++ b/lib/widget.rb\n"
    "@@ -1,1 +1,1 @@\n"
    "-bad\n"
    "+good\n"
)
# pyproject.toml dep edit is a real, common fix and is intentionally allowed
# (config-section hardening is left to runtime; see boundary.py docstring).
GOLD_PYPROJECT_DEP = (
    "diff --git a/pyproject.toml b/pyproject.toml\n"
    "--- a/pyproject.toml\n"
    "+++ b/pyproject.toml\n"
    "@@ -10,1 +10,2 @@ dependencies = [\n"
    "+    \"requests>=2.0\",\n"
)
# a source file whose basename merely *contains* a denied word must pass
GOLD_FALSE_FRIEND = (
    "diff --git a/src/contest.py b/src/contest.py\n"   # not conftest.py
    "--- a/src/contest.py\n"
    "+++ b/src/contest.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-x\n"
    "+y\n"
    "diff --git a/src/latest.py b/src/latest.py\n"      # ends in 'test' substring, not _test.py
    "--- a/src/latest.py\n"
    "+++ b/src/latest.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-x\n"
    "+y\n"
)
# rename of a source file (allowed); rename INTO a control surface (blocked)
GOLD_RENAME_SOURCE = (
    "diff --git a/src/old.py b/src/new.py\n"
    "similarity index 100%\n"
    "rename from src/old.py\n"
    "rename to src/new.py\n"
)
ATTACK_RENAME_INTO_CONFTEST = (
    "diff --git a/src/util.py b/conftest.py\n"
    "similarity index 90%\n"
    "rename from src/util.py\n"
    "rename to conftest.py\n"
)

# verifier paths derived from a typical test_patch (modifies an existing test
# file) — used to prove direct-edit detection keys off the real path set.
VERIFIER_PATHS = {"tests/test_core.py", "tests/test_api.py"}


# (label, fix_patch, verifier_paths, expect_blocked, reason_substr)
CASES = [
    ("py: conftest.py hook",        ATTACK_PY_CONFTEST,       set(),          True,  "conftest.py"),
    ("py: pytest.py shim",          ATTACK_PY_SHIM,           set(),          True,  "pytest.py"),
    ("py: sitecustomize.py",        ATTACK_PY_SITECUSTOMIZE,  set(),          True,  "sitecustomize.py"),
    ("py: .pth import hook",        ATTACK_PY_PTH,            set(),          True,  "inject.pth"),
    ("go: main_test.go TestMain",   ATTACK_GO_TESTMAIN,       set(),          True,  "main_test.go"),
    ("go: new *_test.go file",      ATTACK_GO_NEW_TESTFILE,   set(),          True,  "extra_test.go"),
    ("rb: .rspec hook",             ATTACK_RB_RSPEC,          set(),          True,  ".rspec"),
    ("rb: spec_helper.rb",          ATTACK_RB_SPEC_HELPER,    set(),          True,  "spec_helper.rb"),
    ("rb: spec/support hook",       ATTACK_RB_SUPPORT_HOOK,   set(),          True,  "support"),
    ("direct edit of graded test",  ATTACK_DIRECT_TEST_EDIT,  VERIFIER_PATHS, True,  "verifier test path"),
    ("rename src -> conftest.py",   ATTACK_RENAME_INTO_CONFTEST, set(),       True,  "conftest.py"),

    ("gold: py source only",        GOLD_PY_SOURCE,           VERIFIER_PATHS, False, None),
    ("gold: py new module",         GOLD_PY_NEW_MODULE,       VERIFIER_PATHS, False, None),
    ("gold: go source only",        GOLD_GO_SOURCE,           set(),          False, None),
    ("gold: rb source only",        GOLD_RB_SOURCE,           set(),          False, None),
    ("gold: pyproject dep edit",    GOLD_PYPROJECT_DEP,       set(),          False, None),
    ("gold: false-friend names",    GOLD_FALSE_FRIEND,        set(),          False, None),
    ("gold: rename source file",    GOLD_RENAME_SOURCE,       set(),          False, None),
]


def main() -> int:
    failures = 0
    print(f"{'RESULT':6}  {'EXPECT':9}  CASE")
    print("-" * 70)
    for label, patch, vpaths, expect_blocked, reason_substr in CASES:
        violations = sanitize_fix_patch(patch, vpaths)
        blocked = bool(violations)
        ok = blocked == expect_blocked
        if ok and expect_blocked and reason_substr:
            ok = any(reason_substr in v for v in violations)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        exp = "BLOCK" if expect_blocked else "ALLOW"
        detail = ""
        if blocked:
            detail = "  -> " + "; ".join(violations)
        print(f"{status:6}  {exp:9}  {label}{detail}")

    print("-" * 70)
    # spot-check the path extractor directly
    paths = extract_patch_paths(ATTACK_GO_TESTMAIN)
    assert paths == {"server/handler.go", "server/main_test.go"}, paths
    assert "/dev/null" not in paths
    print(f"extract_patch_paths(go attack) = {sorted(paths)}  [ok]")

    print()
    if failures:
        print(f"FAILED: {failures}/{len(CASES)} cases")
        return 1
    print(f"OK: all {len(CASES)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
