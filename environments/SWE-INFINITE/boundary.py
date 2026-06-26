"""Verifier-boundary enforcement for miner fix patches.

The miner's fix_patch is applied inside the verification container before the
test runner executes. If the patch is allowed to touch the verifier's own test
files or any test-runner control surface (conftest.py, sitecustomize.py, a Go
``TestMain`` file, an RSpec ``.rspec``/``spec_helper.rb`` hook, ...), the miner
can force the graded tests to pass while leaving the injected canaries failing —
defeating the canary check, which only proves the canaries themselves failed.

A legitimate SWE-bench fix only edits *implementation* source; the tests always
arrive separately via test_patch / augmented_test_patch. We therefore reject any
fix_patch that touches:

  1. a verifier-owned path (any file the test patches add or modify), or
  2. a generic test-runner control surface (see ``_CONTROL_*`` below).

``sanitize_fix_patch`` returns the list of violations (empty == clean). The
caller rejects the submission outright rather than silently stripping hunks, so
that cheating attempts surface as score=0 with an explicit reason.

This is the static half of the defense. The dynamic half (re-materializing the
verifier tests *after* the fix patch, so a missed edit cannot survive) lives in
``env.InfiniteActor._verify``.
"""

import fnmatch
import re
from typing import List, Optional, Set


# Exact basenames that are pure test-runner control surfaces. A real source fix
# never needs to create or modify these.
_CONTROL_BASENAMES = frozenset({
    # Python
    "conftest.py",
    "sitecustomize.py",
    "usercustomize.py",
    "pytest.py",
    "pytest.ini",
    "tox.ini",
    # Ruby
    ".rspec",
    ".rspec-local",
    "spec_helper.rb",
    "rails_helper.rb",
    "test_helper.rb",
    "minitest_helper.rb",
    "Rakefile",
    "rakefile",
    # Go
    "go.mod",
    "go.sum",
    # JS
    "jest.config.js",
    "jest.config.ts",
    "vitest.config.js",
    "vitest.config.ts",
    ".mocharc.js",
    ".mocharc.json",
    ".mocharc.yml",
})

# fnmatch patterns applied to the basename. These match both *new* test/spec
# files (a fresh ``*_test.go`` with TestMain hijacks the whole Go package; a new
# ``*_spec.rb`` can install a global RSpec hook) and pytest plugin / path-hook
# files that the framework auto-loads.
_CONTROL_PATTERNS = (
    "*.pth",            # site-packages path hook -> arbitrary import on startup
    "pytest_*.py",      # pytest plugin module (auto-discovered via entry points)
    "*_test.go",        # Go test file (TestMain controls the test binary)
    "*_test.py",        # python test file
    "test_*.py",
    "*_test.rb",        # minitest
    "test_*.rb",
    "*_spec.rb",        # rspec
    "*.test.js", "*.test.ts", "*.test.jsx", "*.test.tsx",
    "*.spec.js", "*.spec.ts", "*.spec.jsx", "*.spec.tsx",
)


def _normalize(path: str) -> Optional[str]:
    """Normalize a raw diff path. Returns None for /dev/null or empty."""
    if not path:
        return None
    path = path.strip()
    # git quotes paths containing special chars: "a/weird\tname"
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        path = path[1:-1]
    # a diff path may carry a trailing tab + timestamp on the same line
    path = path.split("\t", 1)[0].strip()
    if path in ("/dev/null", ""):
        return None
    # strip a/ or b/ prefix used by git diffs
    if path.startswith(("a/", "b/")):
        path = path[2:]
    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    return path or None


def extract_patch_paths(diff: str) -> Set[str]:
    """Collect every normalized file path a unified diff touches.

    Reads ``---``/``+++`` header lines plus ``rename``/``copy`` lines so that
    renamed or copied targets are not missed. ``/dev/null`` (new/deleted side)
    is ignored.
    """
    paths: Set[str] = set()
    if not diff:
        return paths
    for line in diff.splitlines():
        raw: Optional[str] = None
        if line.startswith("--- ") or line.startswith("+++ "):
            raw = line[4:]
        elif line.startswith("rename from ") or line.startswith("rename to "):
            raw = line.split(" ", 2)[2]
        elif line.startswith("copy from ") or line.startswith("copy to "):
            raw = line.split(" ", 2)[2]
        elif line.startswith("diff --git "):
            # fallback: "diff --git a/x b/y" (only when paths are space-free)
            rest = line[len("diff --git "):]
            m = re.match(r'^(?:a/)?(\S+)\s+(?:b/)?(\S+)$', rest)
            if m:
                for g in m.groups():
                    n = _normalize(g)
                    if n:
                        paths.add(n)
            continue
        if raw is not None:
            n = _normalize(raw)
            if n:
                paths.add(n)
    return paths


def _has_segment(path: str, segments: Set[str]) -> bool:
    return any(part in segments for part in path.split("/"))


def control_surface_reason(path: str) -> Optional[str]:
    """Return a reason string if ``path`` is a test-runner control surface."""
    if ".." in path.split("/"):
        return f"path traversal in '{path}'"
    base = path.rsplit("/", 1)[-1]
    if base in _CONTROL_BASENAMES:
        return f"control file '{path}'"
    for pat in _CONTROL_PATTERNS:
        if fnmatch.fnmatch(base, pat):
            return f"test/runner file '{path}' (matches {pat})"
    # rspec support hooks: spec/support/**/*.rb is auto-required by spec_helper
    if base.endswith(".rb") and _segment_pair(path, "spec", "support"):
        return f"rspec support hook '{path}'"
    return None


def _segment_pair(path: str, first: str, second: str) -> bool:
    parts = path.split("/")
    for i in range(len(parts) - 1):
        if parts[i] == first and parts[i + 1] == second:
            return True
    return False


def sanitize_fix_patch(fix_patch: str, verifier_paths: Set[str]) -> List[str]:
    """Return a list of boundary violations for ``fix_patch`` (empty == clean).

    A violation is any touched path that is either a verifier-owned test path or
    a generic test-runner control surface.
    """
    violations: List[str] = []
    verifier_norm = {p for p in (_normalize(v) for v in verifier_paths) if p}
    for path in sorted(extract_patch_paths(fix_patch)):
        if path in verifier_norm:
            violations.append(f"modifies verifier test path '{path}'")
            continue
        reason = control_surface_reason(path)
        if reason:
            violations.append(reason)
    return violations
