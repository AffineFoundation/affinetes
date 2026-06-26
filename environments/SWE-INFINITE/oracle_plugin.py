"""Trusted out-of-repo pytest oracle plugin (verifier Layer 2).

Boundary filter (A) + materialize-after (B) stop the *control-surface* attacks
(conftest.py / pytest.py / TestMain / .rspec). They do NOT stop subversion code
hidden in a *regular implementation source file* the fix legitimately edits
(e.g. `calc.py`): pytest imports that file during collection, so its top-level
code runs in-process and can monkeypatch the runner or fabricate the result
stream to force outcomes.

This plugin is the verifier's in-process oracle. It is shipped from OUTSIDE the
repo and loaded via `-p _swe_oracle` (PYTEST_ADDOPTS) so the miner's patch — which
boundary.py already forbids from adding conftest/plugins — cannot shadow it. It:

  1. captures the real os.write + a private result fd (opened in pytest_configure)
     so test outcomes are reported out-of-band; the verifier reads this channel,
     not stdout, defeating fabricated/suppressed terminal output;
  2. snapshots the real runner entrypoint (`Function.runtest`) in pytest_configure
     — after all (trusted) plugins are registered but BEFORE any test or
     implementation module is imported during collection — and checks at session
     end that it was not replaced, flagging in-process runner tampering;
  3. emits one JSON line per test (real outcome) + a final integrity verdict.

Integrity uses only the runtest-identity signal, which has ~zero false positives:
a legitimate test run never rebinds `_pytest.python.Function.runtest`. We
deliberately do NOT flag an active sys.settrace/setprofile (coverage.py,
debuggers and profilers set those legitimately) nor disable them (that would
break coverage measurement).

It does NOT defend against a fix that returns an __eq__-overloaded object or
hard-codes answers for specific inputs ("special-casing"); those need held-out
behavioral tests, not runner hardening. Scope is runner/outcome subversion.
"""
import json
import os

# --- capture trusted primitives at import (before model code loads) ---
_REAL_WRITE = os.write
_REAL_OPEN = os.open
_OUT_FD = -1

try:
    import _pytest.python as _pp
except Exception:  # pragma: no cover - pytest internals moved
    _pp = None

# Snapshotted in pytest_configure (after plugins, before collection imports).
_ORIG_RUNTEST = None


def _emit(obj):
    if _OUT_FD < 0:
        return
    try:
        _REAL_WRITE(_OUT_FD, (json.dumps(obj) + "\n").encode("utf-8"))
    except Exception:
        pass


def pytest_configure(config):
    global _OUT_FD, _ORIG_RUNTEST
    path = os.environ.get("_SWE_ORACLE_OUT", "/workspace/oracle.jsonl")
    try:
        _OUT_FD = _REAL_OPEN(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    except Exception:
        _OUT_FD = -1
    try:
        _ORIG_RUNTEST = _pp.Function.runtest if _pp is not None else None
    except Exception:
        _ORIG_RUNTEST = None
    _emit({"event": "configure"})


def _integrity():
    """Return (ok, reasons). Only the runtest-identity signal (zero FP)."""
    reasons = []
    if _pp is None or _ORIG_RUNTEST is None:
        reasons.append("pytest_internals_unavailable")
        return False, reasons
    try:
        if _pp.Function.runtest is not _ORIG_RUNTEST:
            reasons.append("Function.runtest_replaced")
    except Exception:
        reasons.append("Function.runtest_inaccessible")
    return (len(reasons) == 0), reasons


def pytest_runtest_logreport(report):
    # Record the authoritative per-test outcome out-of-band. 'call' is the test
    # body; a non-passing 'setup' is a collection/setup error worth recording.
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _emit({
            "event": "test",
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
        })


def pytest_sessionfinish(session, exitstatus):
    ok, reasons = _integrity()
    _emit({"event": "finish", "integrity_ok": ok, "reasons": reasons})
    try:
        if _OUT_FD >= 0:
            os.close(_OUT_FD)
    except Exception:
        pass
