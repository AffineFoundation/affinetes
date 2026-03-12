"""SWE-agent style scenarios for logprobs extraction.

Each scenario simulates a minisweagent multi-turn debugging session:
  - system prompt (agent instructions)
  - bug report (instance prompt)
  - prior conversation turns (THOUGHT + bash + observation pairs)

The model is expected to generate the next THOUGHT + bash action,
which is where we extract logprobs.
"""

# ---------- System prompt (matches SWE-SYNTH minisweagent style) ----------

SYSTEM_PROMPT = """\
You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

# ---------- Instance template ----------

INSTANCE_TEMPLATE = """\
<pr_description>
Consider the following PR description:
{bug_report}
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.

IMPORTANT: This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.

For each response:
1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute

## Important Boundaries
- MODIFY: Regular source code files in /app (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)
- NEVER add or modify unit tests. Your job is ONLY to fix the source code.

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a simple script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`

## Command Execution Rules
Each response should include:
1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

Commands must be specified in a single bash code block:

```bash
your_command_here
```

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.

## Submission
When you've completed your work, issue exactly the following command:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```
</instructions>"""

# ---------- Bug reports (diverse repos & languages) ----------

BUG_REPORTS = [
    # 0: Python - Django ORM
    {
        "repo": "django/django",
        "language": "python",
        "report": (
            "QuerySet.bulk_update() silently skips fields when the model uses "
            "multi-table inheritance. When updating a child model's fields that "
            "are defined on the parent, the generated SQL UPDATE only targets "
            "the child table, leaving parent fields unchanged. This causes data "
            "inconsistency when the application assumes all specified fields "
            "were updated."
        ),
        "files": [
            "django/db/models/query.py",
            "django/db/models/sql/subqueries.py",
        ],
    },
    # 1: Python - Flask request handling
    {
        "repo": "pallets/flask",
        "language": "python",
        "report": (
            "Flask's `send_file` returns HTTP 200 instead of 206 when a Range "
            "header is present and the file is served from a `BytesIO` object. "
            "This breaks resumable downloads and media streaming for clients "
            "that rely on partial content responses."
        ),
        "files": [
            "src/flask/helpers.py",
            "src/flask/wrappers.py",
        ],
    },
    # 2: Python - pandas groupby
    {
        "repo": "pandas-dev/pandas",
        "language": "python",
        "report": (
            "DataFrame.groupby().transform('cumsum') produces incorrect results "
            "when the DataFrame contains nullable integer columns (Int64). The "
            "cumulative sum resets to zero at each NA value instead of propagating "
            "the running total, which differs from the behavior with float columns."
        ),
        "files": [
            "pandas/core/groupby/groupby.py",
            "pandas/core/arrays/masked.py",
        ],
    },
    # 3: JavaScript - Express middleware
    {
        "repo": "expressjs/express",
        "language": "javascript",
        "report": (
            "Express `res.redirect()` does not URL-encode special characters in "
            "the Location header when given a relative path containing Unicode. "
            "This results in malformed headers that cause HTTP 500 errors on "
            "certain reverse proxy configurations (nginx, Cloudflare)."
        ),
        "files": [
            "lib/response.js",
            "lib/utils.js",
        ],
    },
    # 4: Python - requests library
    {
        "repo": "psf/requests",
        "language": "python",
        "report": (
            "Session.send() ignores the `proxies` parameter when redirects occur. "
            "After the first redirect, the session falls back to environment proxy "
            "settings instead of using the explicitly provided proxy dict. This "
            "causes connection failures in environments where direct access is "
            "blocked by firewall rules."
        ),
        "files": [
            "requests/sessions.py",
            "requests/adapters.py",
        ],
    },
    # 5: TypeScript - Next.js routing
    {
        "repo": "vercel/next.js",
        "language": "typescript",
        "report": (
            "Dynamic API routes with catch-all segments return 404 when the URL "
            "contains encoded slashes (%2F). The route matcher decodes the path "
            "before matching, causing the catch-all parameter to be split into "
            "multiple segments instead of preserving the encoded slash."
        ),
        "files": [
            "packages/next/src/server/router.ts",
            "packages/next/src/shared/lib/router/utils/route-matcher.ts",
        ],
    },
    # 6: Python - SQLAlchemy
    {
        "repo": "sqlalchemy/sqlalchemy",
        "language": "python",
        "report": (
            "Using `session.merge()` on a detached object that has a composite "
            "primary key raises an IntegrityError when the object already exists "
            "in the database. The merge operation generates an INSERT instead of "
            "an UPDATE because the identity map lookup fails to match on the "
            "composite key tuple."
        ),
        "files": [
            "lib/sqlalchemy/orm/session.py",
            "lib/sqlalchemy/orm/identity.py",
        ],
    },
    # 7: Go - HTTP server
    {
        "repo": "golang/go",
        "language": "go",
        "report": (
            "net/http.Server graceful shutdown with `Shutdown()` does not wait "
            "for WebSocket connections upgraded via `Hijack()`. Active WebSocket "
            "connections are forcefully closed even when the shutdown context has "
            "not expired, causing data loss for real-time applications."
        ),
        "files": [
            "src/net/http/server.go",
            "src/net/http/transport.go",
        ],
    },
    # 8: Python - NumPy
    {
        "repo": "numpy/numpy",
        "language": "python",
        "report": (
            "`np.einsum` with `optimize=True` produces wrong results for "
            "contractions involving more than 4 tensors when one tensor has "
            "a singleton dimension. The greedy path optimizer incorrectly "
            "broadcasts the singleton, leading to shape mismatch in the "
            "intermediate contraction."
        ),
        "files": [
            "numpy/core/einsumfunc.py",
            "numpy/core/_multiarray_umath.py",
        ],
    },
    # 9: Rust - Tokio runtime
    {
        "repo": "tokio-rs/tokio",
        "language": "rust",
        "report": (
            "`tokio::time::sleep_until` panics with 'instant is too far in the "
            "future' when the deadline exceeds `Instant::now() + Duration::MAX / 2` "
            "on 32-bit targets. The timer wheel entry calculation overflows on "
            "platforms where usize is 32 bits."
        ),
        "files": [
            "tokio/src/time/driver/entry.rs",
            "tokio/src/time/driver/wheel/mod.rs",
        ],
    },
    # 10: Python - FastAPI
    {
        "repo": "tiangolo/fastapi",
        "language": "python",
        "report": (
            "FastAPI dependency injection fails to resolve `Annotated` type hints "
            "when a dependency returns an `Optional[Model]`. The dependency solver "
            "strips the `Optional` wrapper and attempts to validate `None` against "
            "the inner Pydantic model, raising a ValidationError instead of "
            "passing `None` through."
        ),
        "files": [
            "fastapi/dependencies/utils.py",
            "fastapi/routing.py",
        ],
    },
    # 11: Python - pytest
    {
        "repo": "pytest-dev/pytest",
        "language": "python",
        "report": (
            "`pytest.raises` context manager does not respect `match` parameter "
            "when the exception message contains newlines. The regex match is "
            "applied with `re.search` in single-line mode, so patterns with `.` "
            "fail to match across line boundaries in the exception string."
        ),
        "files": [
            "src/_pytest/python_api.py",
            "src/_pytest/_code/code.py",
        ],
    },
    # 12: JavaScript - React
    {
        "repo": "facebook/react",
        "language": "javascript",
        "report": (
            "`useEffect` cleanup function is called twice on component unmount "
            "when Strict Mode is enabled and the effect depends on a ref object. "
            "The double invocation causes errors when the cleanup performs "
            "non-idempotent operations like removing event listeners that were "
            "already removed."
        ),
        "files": [
            "packages/react-reconciler/src/ReactFiberCommitWork.js",
            "packages/react-reconciler/src/ReactFiberHooks.js",
        ],
    },
    # 13: Python - Celery
    {
        "repo": "celery/celery",
        "language": "python",
        "report": (
            "Celery chord callback is never executed when one of the header tasks "
            "is retried with `self.retry(countdown=N)`. The chord counter in the "
            "result backend decrements on the original task failure but does not "
            "re-increment when the retry succeeds, so the counter never reaches "
            "zero."
        ),
        "files": [
            "celery/backends/base.py",
            "celery/app/task.py",
        ],
    },
    # 14: Python - scikit-learn
    {
        "repo": "scikit-learn/scikit-learn",
        "language": "python",
        "report": (
            "`Pipeline.set_output(transform='pandas')` causes `ColumnTransformer` "
            "to lose column names when a transformer returns fewer columns than "
            "expected. The output DataFrame has auto-generated integer column "
            "names instead of the feature names from `get_feature_names_out()`."
        ),
        "files": [
            "sklearn/compose/_column_transformer.py",
            "sklearn/utils/_set_output.py",
        ],
    },
]

# ---------- Prior conversation turns (THOUGHT + bash + observation) ----------
# Each is a list of (thought, command, observation) tuples representing
# the agent's prior actions before the current decision point.

CONVERSATION_TURNS = [
    # Style 0: Initial exploration
    [
        (
            "I need to understand the repository structure first. Let me check "
            "what files are in the current directory.",
            "find /app -type f -name '*.py' | head -30",
            "/app/setup.py\n/app/src/__init__.py\n/app/src/core.py\n"
            "/app/src/utils.py\n/app/src/handlers.py\n/app/tests/test_core.py\n"
            "/app/tests/test_utils.py",
        ),
    ],
    # Style 1: After reading a file
    [
        (
            "Let me start by examining the repository structure to find "
            "relevant files.",
            "ls -la /app/src/",
            "total 48\ndrwxr-xr-x 3 root root 4096 Jan 15 10:00 .\n"
            "-rw-r--r-- 1 root root  245 Jan 15 10:00 __init__.py\n"
            "-rw-r--r-- 1 root root 3421 Jan 15 10:00 core.py\n"
            "-rw-r--r-- 1 root root 1892 Jan 15 10:00 utils.py\n"
            "-rw-r--r-- 1 root root 2764 Jan 15 10:00 handlers.py",
        ),
        (
            "Now let me read the main module to understand the code structure "
            "and find where the bug might be.",
            "cat /app/src/core.py",
            'class DataProcessor:\n    def __init__(self, config):\n'
            '        self.config = config\n        self._cache = {}\n\n'
            '    def process(self, data, **kwargs):\n'
            '        """Process input data according to config."""\n'
            '        if not isinstance(data, (list, tuple)):\n'
            '            data = [data]\n'
            '        results = []\n'
            '        for item in data:\n'
            '            result = self._transform(item, **kwargs)\n'
            '            results.append(result)\n'
            '        return results\n',
        ),
    ],
    # Style 2: Reproduction attempt
    [
        (
            "I should first understand the bug. Let me look at the relevant "
            "source file mentioned in the issue.",
            "grep -n 'def bulk_update' /app/src/models/query.py",
            "342:    def bulk_update(self, objs, fields, batch_size=None):",
        ),
        (
            "Let me read the bulk_update implementation to understand the "
            "current logic.",
            "nl -ba /app/src/models/query.py | sed -n '342,400p'",
            "   342\t    def bulk_update(self, objs, fields, batch_size=None):\n"
            "   343\t        if batch_size is not None and batch_size < 0:\n"
            "   344\t            raise ValueError('batch_size must be positive')\n"
            "   345\t        if not fields:\n"
            "   346\t            raise ValueError('fields is required')\n"
            "   347\t        objs = tuple(objs)\n"
            "   348\t        if not objs:\n"
            "   349\t            return 0\n"
            "   350\t        for obj in objs:\n"
            "   351\t            obj._prepare_related_fields_for_save()\n",
        ),
        (
            "Now I'll create a reproduction script to confirm the bug before "
            "attempting any fix.",
            "cat <<'EOF' > /app/reproduce.py\nimport sys\nsys.path.insert(0, '/app')\n"
            "from src.models import TestModel\n\n"
            "obj = TestModel(pk=1, name='test', value=42)\n"
            "try:\n    result = TestModel.objects.bulk_update([obj], ['name'])\n"
            "    print(f'Updated {result} rows')\n"
            "except Exception as e:\n    print(f'Error: {e}')\nEOF",
            "Updated 0 rows",
        ),
    ],
    # Style 3: After identifying the bug
    [
        (
            "Let me search for the relevant function in the codebase.",
            "grep -rn 'def send_file' /app/src/",
            "/app/src/flask/helpers.py:187:def send_file(\n"
            "/app/src/flask/helpers.py:290:def send_from_directory(",
        ),
        (
            "Let me read the send_file function to understand how it handles "
            "Range headers.",
            "nl -ba /app/src/flask/helpers.py | sed -n '187,260p'",
            "   187\tdef send_file(\n"
            "   188\t    path_or_file,\n"
            "   189\t    mimetype=None,\n"
            "   190\t    as_attachment=False,\n"
            "   191\t    download_name=None,\n"
            "   192\t    conditional=True,\n"
            "   193\t    etag=True,\n"
            "   194\t    max_age=None,\n"
            "   195\t):\n"
            "   196\t    if isinstance(path_or_file, (str, os.PathLike)):\n"
            "   197\t        path = pathlib.Path(path_or_file)\n"
            "   198\t    else:\n"
            "   199\t        path = None\n",
        ),
        (
            "I see the issue. When `path` is None (BytesIO case), the Range "
            "header processing is skipped. Let me check the conditional "
            "response logic.",
            "grep -n 'range\\|Range\\|partial\\|206' /app/src/flask/helpers.py",
            "245:        if conditional:\n"
            "246:            data = wrap_file(request.environ, data)\n"
            "247:            rv = Response(data, mimetype=mimetype)\n"
            "248:            rv.make_conditional(request)\n",
        ),
        (
            "The issue is clear: `make_conditional` only handles Range "
            "for file paths, not BytesIO. I need to fix the logic to also "
            "handle BytesIO objects. Let me prepare the fix.",
            "nl -ba /app/src/flask/helpers.py | sed -n '240,260p'",
            "   240\t    if path is not None:\n"
            "   241\t        data = path.open('rb')\n"
            "   242\t        if etag:\n"
            "   243\t            rv_etag = _generate_etag(path)\n"
            "   244\t    else:\n"
            "   245\t        data = path_or_file\n"
            "   246\t\n"
            "   247\t    if conditional:\n"
            "   248\t        data = wrap_file(request.environ, data)\n"
            "   249\t        rv = Response(data, mimetype=mimetype)\n"
            "   250\t        rv.make_conditional(request)\n",
        ),
    ],
    # Style 4: Applying a fix
    [
        (
            "Based on my analysis, the bug is in the groupby transform method. "
            "Let me look at the specific function.",
            "grep -n 'def _transform_fast' /app/pandas/core/groupby/groupby.py",
            "1245:    def _transform_fast(self, result):",
        ),
        (
            "Let me read the transform method implementation.",
            "nl -ba /app/pandas/core/groupby/groupby.py | sed -n '1245,1290p'",
            "  1245\t    def _transform_fast(self, result):\n"
            "  1246\t        ids, _, ngroups = self.grouper.group_info\n"
            "  1247\t        cast = self._transform_should_cast(result)\n"
            "  1248\t        out = algorithms.take_nd(result._values, ids)\n"
            "  1249\t        if cast:\n"
            "  1250\t            out = maybe_downcast_to_dtype(out, result.dtype)\n"
            "  1251\t        return out\n",
        ),
        (
            "I see the problem. The `take_nd` function doesn't handle masked "
            "arrays properly for cumulative operations. Let me check how "
            "the masked array handles NA propagation.",
            "grep -n 'def _cumsum\\|def cumsum\\|cumulative_sum' "
            "/app/pandas/core/arrays/masked.py",
            "456:    def _reduce(self, name, *, skipna=True, **kwargs):\n"
            "521:    def cumsum(self, *, skipna=True, **kwargs):",
        ),
        (
            "Let me read the cumsum implementation in the masked array.",
            "nl -ba /app/pandas/core/arrays/masked.py | sed -n '521,555p'",
            "   521\t    def cumsum(self, *, skipna=True, **kwargs):\n"
            "   522\t        nv.validate_cumsum((), kwargs)\n"
            "   523\t        values = self._data.copy()\n"
            "   524\t        mask = self._mask.copy()\n"
            "   525\t        if skipna:\n"
            "   526\t            values[mask] = 0\n"
            "   527\t        result = values.cumsum()\n"
            "   528\t        return type(self)(result, mask)\n",
        ),
        (
            "Found it! Line 526 sets masked values to 0 before cumsum, which "
            "is correct. But the issue is that when transform calls this, it "
            "resets per group. I need to fix the groupby transform to properly "
            "handle masked arrays. Let me apply the fix.",
            "sed -i '1248s/result._values/result._values if not hasattr(result, \"_mask\") else result/' "
            "/app/pandas/core/groupby/groupby.py",
            "",
        ),
    ],
    # Style 5: Short exploration
    [
        (
            "Let me understand the project layout first.",
            "ls /app/",
            "LICENSE\nREADME.md\npackage.json\nsrc\ntests\ntsconfig.json",
        ),
    ],
    # Style 6: Deep investigation with grep
    [
        (
            "I need to trace how redirects are handled in the session. "
            "Let me search for the redirect handling code.",
            "grep -n 'redirect\\|resolve_redirect\\|rebuild_proxies' "
            "/app/requests/sessions.py",
            "112:    def resolve_redirects(self, resp, req, stream=False,\n"
            "118:        while resp.is_redirect:\n"
            "145:            proxies = self.rebuild_proxies(prepared, proxies)\n"
            "312:    def rebuild_proxies(self, prepared, proxies):",
        ),
        (
            "Let me read the resolve_redirects method closely, especially "
            "around how proxies are rebuilt.",
            "nl -ba /app/requests/sessions.py | sed -n '112,160p'",
            "   112\t    def resolve_redirects(self, resp, req, stream=False,\n"
            "   113\t                          timeout=None, verify=True, cert=None,\n"
            "   114\t                          proxies=None):\n"
            "   115\t        i = 0\n"
            "   116\t        hist = []\n"
            "   117\t\n"
            "   118\t        while resp.is_redirect:\n"
            "   119\t            prepared = req.copy()\n"
            "   120\t            extract_cookies_to_jar(prepared._cookies, req, resp.raw)\n"
            "   121\t            prepared.url = self.get_redirect_target(resp)\n"
            "   122\t\n"
            "   123\t            prepared = self.rebuild_method(prepared, resp)\n"
            "   124\t            prepared.body = None\n"
            "   125\t\n"
            "   126\t            headers = prepared.headers\n"
            "   127\t            try:\n"
            "   128\t                del headers['Content-Length']\n"
            "   129\t            except KeyError:\n"
            "   130\t                pass\n",
        ),
    ],
    # Style 7: Verification step
    [
        (
            "I've applied the fix. Let me verify it works by running "
            "the reproduction script.",
            "cd /app && python reproduce.py",
            "Test 1: PASSED - bulk_update correctly updates parent fields\n"
            "Test 2: PASSED - child fields also updated\n"
            "All tests passed!",
        ),
        (
            "The reproduction script confirms the fix works. Now let me run "
            "the full test suite to make sure nothing else is broken.",
            "cd /app && python -m pytest tests/ -x -q 2>&1 | tail -20",
            "tests/test_core.py ........ [40%]\n"
            "tests/test_utils.py ....... [75%]\n"
            "tests/test_handlers.py .... [95%]\n"
            "tests/test_edge.py .      [100%]\n\n"
            "24 passed in 3.42s",
        ),
    ],
]

# ---------- Observation templates (for formatting prior turn outputs) ----------

OBSERVATION_TEMPLATE = """\
<returncode>{returncode}</returncode>
<output>
{output}
</output>"""

# ---------- Action (agent turn) formatting ----------

ACTION_TEMPLATE = """\
THOUGHT: {thought}

```bash
{command}
```"""
