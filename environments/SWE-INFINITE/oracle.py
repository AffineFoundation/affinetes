"""Per-language trusted oracle injection (verifier Layer 2).

Boundary filter (A) + materialize-after (B) stop control-surface attacks
(conftest.py / pytest.py / TestMain / .rspec) across all languages. Layer 2
targets the residual that A+B cannot: in-process runner subversion hidden in a
*regular implementation source file* the fix legitimately edits (monkeypatching
the runner, fabricating/suppressing the result stream).

Each language loads a trusted plugin/reporter from OUTSIDE the repo (the miner's
patch cannot add plugins — boundary.py forbids it — and cannot shadow a path it
does not control). Every language writes the SAME JSONL contract to
``/workspace/oracle.jsonl`` so the verifier grades uniformly:

    {"event": "test",   "nodeid": "<id>", "outcome": "passed"|"failed"}
    {"event": "finish", "integrity_ok": true|false, "reasons": [...]}

``build_oracle`` returns the bash that installs the plugin and points the runner
at it (empty string => no oracle for this language; caller falls back to stdout
parsing). The shared emit (cat the result file between ORACLE markers) is added
by the caller.
"""

import base64
import re
from pathlib import Path

_DIR = Path(__file__).parent
ORACLE_OUT = "/workspace/oracle.jsonl"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _read(name: str) -> str:
    return (_DIR / name).read_text()


def build_oracle(language: str, test_command: str,
                 test_patch: str = "", augmented_test_patch: str = "") -> str:
    """Return bash to install+wire the trusted oracle, or "" if unsupported."""
    lang = (language or "").lower().strip()
    if lang == "python":
        return _python()
    if lang == "ruby":
        return _ruby(test_command)
    if lang in ("javascript", "typescript"):
        return _javascript(test_command)
    if lang == "go":
        return _go(test_patch, augmented_test_patch)
    return ""


def _python() -> str:
    plugin_b64 = _b64(_read("oracle_plugin.py"))
    return (
        'echo "' + plugin_b64 + '" | base64 -d > /workspace/_swe_oracle.py\n'
        'ORACLE_PY=$(command -v python3 || command -v python)\n'
        'if [ -n "$ORACLE_PY" ]; then\n'
        '  SP=$("$ORACLE_PY" -c \'import sysconfig; print(sysconfig.get_paths()["purelib"])\' 2>/dev/null)\n'
        '  if [ -n "$SP" ] && [ -d "$SP" ]; then cp /workspace/_swe_oracle.py "$SP/_swe_oracle.py"; fi\n'
        'fi\n'
        'export _SWE_ORACLE_OUT=' + ORACLE_OUT + '\n'
        'export PYTEST_ADDOPTS="-p _swe_oracle ${PYTEST_ADDOPTS:-}"\n'
    )


def _ruby(test_command: str) -> str:
    # RSpec honors the SPEC_OPTS env var; we require our reporter from outside
    # the repo and add it as an extra formatter (writes to its own out-of-band
    # file, leaving any existing -f json on stdout intact). Minitest is not
    # covered here (different plugin mechanism).
    if "rspec" not in (test_command or "").lower():
        return ""
    plugin_b64 = _b64(_read("rspec_oracle.rb"))
    return (
        'mkdir -p /opt/oracle\n'
        'echo "' + plugin_b64 + '" | base64 -d > /opt/oracle/rspec_oracle.rb\n'
        'export _SWE_ORACLE_OUT=' + ORACLE_OUT + '\n'
        'export SPEC_OPTS="-r /opt/oracle/rspec_oracle.rb -f RSpecOracleFormatter ${SPEC_OPTS:-}"\n'
    )


def _javascript(test_command: str) -> str:
    # Jest has no env-injection point for reporters/setup (NODE_OPTIONS=--require
    # only reaches the main process, not the per-file test vm where `expect`
    # lives — verified empirically). So a NODE_OPTIONS preload, running in the
    # jest CLI process, pushes our reporter (outcomes, main process) and
    # setupFilesAfterEnv (runs inside the vm; snapshots expect/it/test before the
    # impl source loads, flags replacement at teardown) onto jest's argv.
    if "jest" not in (test_command or "").lower():
        return ""
    files = {
        "/opt/oracle/jest_oracle.js": _read("jest_oracle.js"),
        "/opt/oracle/jest_setup.js": _read("jest_setup.js"),
        "/opt/oracle/jest_reporter.js": _read("jest_reporter.js"),
    }
    setup = "mkdir -p /opt/oracle\n"
    for path, content in files.items():
        setup += 'echo "' + _b64(content) + '" | base64 -d > ' + path + "\n"
    setup += 'export _SWE_ORACLE_OUT=' + ORACLE_OUT + '\n'
    setup += 'export NODE_OPTIONS="--require /opt/oracle/jest_oracle.js ${NODE_OPTIONS:-}"\n'
    return setup


def _go(test_patch: str, augmented_test_patch: str) -> str:
    # Go is statically compiled — the runner cannot be monkeypatched at runtime.
    # The residual in-process vector is a package init() that fabricates/suppresses
    # the `go test -json` stream. Defense: materialize a trusted TestMain in the
    # test package; it wraps m.Run() and writes the REAL aggregate exit code
    # out-of-band (init() cannot forge m.Run()'s return without monkeypatching,
    # which Go does not allow). env.py cross-checks: stdout claiming all-pass while
    # go_exit != 0 is fabrication. Skipped if the package already defines TestMain
    # (cannot have two) — boundary.py forbids the MINER from adding one, so any
    # existing TestMain is the verifier's own test_patch.
    patches = (augmented_test_patch or "") + "\n" + (test_patch or "")
    if re.search(r'func\s+TestMain\s*\(', patches):
        return ""
    fm = re.search(r'^\+\+\+ b/(\S+_test\.go)$', patches, re.M)
    if not fm:
        return ""
    test_file = fm.group(1)
    dir_path = test_file.rsplit("/", 1)[0] if "/" in test_file else ""
    pm = re.search(r'^\+?\s*package\s+(\w+)', patches, re.M)
    if not pm:
        return ""
    pkg = pm.group(1)

    oracle_go = (
        f"package {pkg}\n\n"
        'import (\n\t"fmt"\n\t"os"\n\t"testing"\n)\n\n'
        "func TestMain(m *testing.M) {\n"
        "\tcode := m.Run()\n"
        '\tif p := os.Getenv("_SWE_ORACLE_OUT"); p != "" {\n'
        "\t\tif f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {\n"
        '\t\t\tfmt.Fprintf(f, "{\\"event\\":\\"finish\\",\\"go_exit\\":%d}\\n", code)\n'
        "\t\t\tf.Close()\n"
        "\t\t}\n"
        "\t}\n"
        "\tos.Exit(code)\n"
        "}\n"
    )
    app_dir = f"/app/{dir_path}" if dir_path else "/app"
    path = f"{app_dir.rstrip('/')}/zz_swe_oracle_test.go"
    return (
        'mkdir -p ' + app_dir + '\n'
        'echo "' + _b64(oracle_go) + '" | base64 -d > ' + path + '\n'
        'export _SWE_ORACLE_OUT=' + ORACLE_OUT + '\n'
    )
