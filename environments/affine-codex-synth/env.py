"""Codex synthetic bug-injection environment."""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
import openai


class Actor:
    """Bug-injection + repair evaluation actor."""

    @staticmethod
    def _parse_bool(value: Optional[Any]) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")

        self.generator_model = os.getenv("BUG_INJECT_MODEL", "deepseek-ai/DeepSeek-V3")
        self.generator_base_url = os.getenv("BUG_INJECT_BASE_URL", "https://llm.chutes.ai/v1")
        self.generator_temperature = float(os.getenv("BUG_INJECT_TEMPERATURE", "0.2"))
        self.generator_timeout = int(os.getenv("BUG_INJECT_TIMEOUT", "120"))
        self.generator_max_tokens = int(os.getenv("BUG_INJECT_MAX_TOKENS", "1200"))
        self.generator_retries = int(os.getenv("BUG_INJECT_RETRIES", "2"))

        self.codex_bin = os.getenv("CODEX_BIN", "codex")
        self.workspace_root = Path(os.getenv("WORKSPACE_ROOT", "/workspace"))
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.keep_workspace = bool(self._parse_bool(os.getenv("KEEP_WORKSPACE")))
        self.test_timeout = int(os.getenv("TEST_TIMEOUT", "60"))

    def _deterministic_seed(self, task_id: int) -> int:
        seed_bytes = hashlib.sha256(f"codex-synth:{task_id}".encode("utf-8")).digest()[:8]
        return int.from_bytes(seed_bytes, byteorder="big") % (2**32)

    def _hash_file(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    async def _llm_chat(
        self, prompt: str, seed: Optional[int], api_key: Optional[str]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not api_key:
            raise ValueError("CHUTES_API_KEY is required for bug injection")

        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

        client = openai.AsyncOpenAI(
            base_url=self.generator_base_url.rstrip("/"),
            api_key=api_key,
            timeout=httpx.Timeout(self.generator_timeout),
            max_retries=0,
        )

        params = {
            "model": self.generator_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You generate bug-injected synthetic programming tasks as JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.generator_temperature,
            "max_tokens": self.generator_max_tokens,
        }
        if seed is not None:
            params["seed"] = seed

        response = await client.chat.completions.create(**params)
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ValueError("Bug-injector returned empty content")

        usage = response.usage.model_dump() if response.usage else None
        return content.strip(), usage

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            match = re.search(r"```(?:json)?\n(.*)```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in bug-injector output")
        return json.loads(cleaned[start : end + 1])

    def _validate_task_schema(self, task: Dict[str, Any]) -> None:
        for key in ("title", "description", "files", "fixes"):
            if key not in task:
                raise ValueError(f"Missing required key: {key}")
        if not isinstance(task["files"], dict) or not isinstance(task["fixes"], dict):
            raise ValueError("files and fixes must be objects")
        required_files = {"solution.py", "tests/test_solution.py"}
        if not required_files.issubset(task["files"]):
            raise ValueError("files must include solution.py and tests/test_solution.py")
        if "solution.py" not in task["fixes"]:
            raise ValueError("fixes must include solution.py")

    def _write_files(self, root: Path, files: Dict[str, str]) -> None:
        for rel_path, content in files.items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _run_pytest(self, root: Path, timeout: int) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"pytest timed out after {timeout}s"

        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output

    def _validate_generated_task(self, task: Dict[str, Any]) -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory(dir=self.workspace_root) as tmp_dir:
            root = Path(tmp_dir)
            self._write_files(root, task["files"])
            buggy_ok, buggy_output = self._run_pytest(root, self.test_timeout)
            if buggy_ok:
                return False, "Buggy code passed tests unexpectedly"

            self._write_files(root, task["fixes"])
            fixed_ok, fixed_output = self._run_pytest(root, self.test_timeout)
            if not fixed_ok:
                return False, f"Fixed code failed tests: {self._tail(fixed_output)}"

        return True, ""

    def _bug_injector_prompt(self, seed: int) -> str:
        return f"""Create a simple single-file Python programming task with a buggy solution.

Seed: {seed}

Return JSON only with this schema:
{{
  "title": "...",
  "description": "...",
  "files": {{
    "solution.py": "... buggy Python solution ...",
    "tests/test_solution.py": "... pytest tests ..."
  }},
  "fixes": {{
    "solution.py": "... corrected solution ..."
  }}
}}

Constraints:
- The problem statement must be in "description".
- The solution must expose solve(data: str) -> str.
- Tests must import solution and call solve() directly.
- Use only Python standard library.
- Tests must fail on the buggy solution and pass on the fixed one.
- Provide at least 4 tests.
- Keep each file under 120 lines.
- Output JSON only, no markdown.
"""

    async def _generate_task(
        self, task_id: int, seed: int, api_key: Optional[str]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        if not api_key:
            fallback = self._fallback_task(task_id)
            return fallback, None, "CHUTES_API_KEY is required for bug injection"

        last_error = None
        for attempt in range(self.generator_retries + 1):
            attempt_seed = seed + attempt
            try:
                prompt = self._bug_injector_prompt(attempt_seed)
                content, usage = await self._llm_chat(prompt, attempt_seed, api_key)
                task = self._extract_json(content)
                self._validate_task_schema(task)
                ok, reason = self._validate_generated_task(task)
                if ok:
                    return task, usage, None
                last_error = reason
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        fallback = self._fallback_task(task_id)
        return fallback, None, last_error

    def _fallback_task(self, task_id: int) -> Dict[str, Any]:
        description = (
            "Given a line of integers separated by spaces, output the sum of the numbers "
            "that are divisible by 3. If there are no numbers, output 0.\n"
            "\n"
            "Implement solve(data: str) -> str which takes the full stdin text and returns "
            "the output string (with no trailing whitespace)."
        )
        buggy = """\
def solve(data: str) -> str:
    nums = [int(x) for x in data.split()] if data.strip() else []
    total = sum(n for n in nums if n % 3 == 1)
    return str(total)


if __name__ == "__main__":
    import sys

    print(solve(sys.stdin.read()))
"""
        fixed = buggy.replace("n % 3 == 1", "n % 3 == 0")
        tests = """\
import solution


def test_mixed_numbers():
    data = "1 2 3 4 5 6\\n"
    assert solution.solve(data) == "9"


def test_single_divisible():
    data = "3\\n"
    assert solution.solve(data) == "3"


def test_none_divisible():
    data = "4 5 7\\n"
    assert solution.solve(data) == "0"


def test_negative_numbers():
    data = "-3 -6 2\\n"
    assert solution.solve(data) == "-9"
"""
        return {
            "title": "Sum multiples of three",
            "description": description,
            "files": {
                "solution.py": buggy,
                "tests/test_solution.py": tests,
            },
            "fixes": {"solution.py": fixed},
        }

    def _write_codex_config(self, codex_home: Path, model: str, base_url: str) -> None:
        codex_home.mkdir(parents=True, exist_ok=True)
        config = "\n".join(
            [
                f'model = {json.dumps(model)}',
                'model_provider = "chutes"',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                "",
                "[model_providers.chutes]",
                'name = "Chutes"',
                f'base_url = {json.dumps(base_url)}',
                'env_key = "CHUTES_API_KEY"',
                'wire_api = "chat"',
                "",
            ]
        )
        (codex_home / "config.toml").write_text(config, encoding="utf-8")

    def _run_codex(
        self,
        workdir: Path,
        prompt: str,
        timeout: int,
        codex_home: Path,
        api_key: Optional[str],
    ) -> Tuple[int, str]:
        cmd = [
            self.codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--cd",
            str(workdir),
            "--dangerously-bypass-approvals-and-sandbox",
            prompt,
        ]
        env = os.environ.copy()
        env["CODEX_HOME"] = str(codex_home)
        if api_key:
            env["CHUTES_API_KEY"] = api_key
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output

    def _tail(self, text: str, limit: int = 4000) -> str:
        if len(text) <= limit:
            return text
        return text[-limit:]

    def _extract_patch(self, text: str) -> Optional[str]:
        start = text.find("*** Begin Patch")
        if start == -1:
            return None
        end = text.find("*** End Patch", start)
        if end == -1:
            return None
        patch = text[start : end + len("*** End Patch")]
        if "\\n" in patch:
            patch = (
                patch.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\'", "'")
            )
        return patch

    def _extract_solution(self, text: str) -> Optional[str]:
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        for block in blocks:
            if "def solve" in block:
                return block.strip()
        return None

    def _resolve_patch_path(self, raw_path: str, root: Path) -> Optional[Path]:
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = root / raw_path
        try:
            resolved = candidate.resolve()
            root_resolved = root.resolve()
        except OSError:
            return None
        if root_resolved in resolved.parents or resolved == root_resolved:
            return resolved
        return None

    def _apply_hunks(self, path: Path, hunks: list[list[str]]) -> bool:
        original_text = path.read_text(encoding="utf-8")
        has_trailing_newline = original_text.endswith("\n")
        original = original_text.splitlines()

        for hunk in hunks:
            pattern = [line[1:] for line in hunk if line.startswith((" ", "-"))]
            if not pattern:
                return False
            start_idx = None
            for idx in range(len(original) - len(pattern) + 1):
                if original[idx : idx + len(pattern)] == pattern:
                    start_idx = idx
                    break
            if start_idx is None:
                return False

            new_lines = original[:start_idx]
            cursor = start_idx
            for line in hunk:
                if line.startswith(" "):
                    new_lines.append(original[cursor])
                    cursor += 1
                elif line.startswith("-"):
                    cursor += 1
                elif line.startswith("+"):
                    new_lines.append(line[1:])
            new_lines.extend(original[cursor:])
            original = new_lines

        new_text = "\n".join(original)
        if has_trailing_newline:
            new_text += "\n"
        path.write_text(new_text, encoding="utf-8")
        return True

    def _apply_patch(self, patch: str, root: Path) -> bool:
        lines = patch.splitlines()
        applied_any = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("*** Update File:"):
                raw_path = line.split(":", 1)[1].strip()
                target = self._resolve_patch_path(raw_path, root)
                i += 1
                hunk_lines = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    hunk_lines.append(lines[i])
                    i += 1
                if not target or not target.exists():
                    continue
                hunks = []
                current = []
                for hline in hunk_lines:
                    if hline.startswith("@@"):
                        if current:
                            hunks.append(current)
                            current = []
                        continue
                    if hline.startswith((" ", "+", "-")):
                        current.append(hline)
                if current:
                    hunks.append(current)
                if hunks and self._apply_hunks(target, hunks):
                    applied_any = True
                continue
            i += 1
        return applied_any

    async def evaluate(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        temperature: Optional[float] = None,
        timeout: int = 900,
        seed: Optional[int] = None,
        task_id: Optional[int] = None,
        api_key: Optional[str] = None,
        keep_workspace: Optional[bool] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        start = time.time()
        task_id = task_id if task_id is not None else random.randint(0, 2**31 - 1)
        seed = seed if seed is not None else self._deterministic_seed(task_id)

        keep_workspace_value = self.keep_workspace
        parsed_keep_workspace = self._parse_bool(keep_workspace)
        if parsed_keep_workspace is not None:
            keep_workspace_value = parsed_keep_workspace

        current_api_key = api_key or self.api_key
        task, usage, generator_error = await self._generate_task(task_id, seed, current_api_key)
        workdir = Path(
            tempfile.mkdtemp(prefix=f"task_{task_id}_", dir=self.workspace_root)
        )
        codex_home = workdir / ".codex"

        result: Dict[str, Any] = {
            "task_name": "affine:codex-synth",
            "score": 0.0,
            "success": False,
            "time_taken": 0.0,
            "extra": {
                "task_id": task_id,
                "seed": seed,
                "temperature": temperature,
                "keep_workspace": keep_workspace_value,
                "workspace_dir": str(workdir),
                "task_title": task.get("title"),
                "generator_model": self.generator_model,
                "generator_base_url": self.generator_base_url,
                "generator_usage": usage,
                "generator_error": generator_error,
            },
        }

        try:
            self._write_files(workdir, task["files"])
            (workdir / "README.md").write_text(task["description"], encoding="utf-8")
            self._write_codex_config(codex_home, model, base_url)
            solution_path = workdir / "solution.py"
            solution_hash_before = self._hash_file(solution_path) if solution_path.exists() else ""

            prompt = (
                "Fix the failing tests for the Python solution.\n\n"
                f"{task['description']}\n\n"
                "Rules:\n"
                "- Do not change tests.\n"
                "- Only edit solution.py.\n"
                "- Run pytest to confirm.\n"
                "- Do NOT use tool calls.\n"
                "- Return either:\n"
                "  1) A patch in this exact format, or\n"
                "  2) The full contents of solution.py inside a ```python``` code block.\n"
                "\n"
                "Patch format:\n"
                "*** Begin Patch\n"
                "*** Update File: solution.py\n"
                "@@\n"
                "- old line\n"
                "+ new line\n"
                "*** End Patch\n"
            )

            codex_rc, codex_output = self._run_codex(
                workdir, prompt, timeout, codex_home, current_api_key
            )
            result["extra"]["codex_returncode"] = codex_rc
            result["extra"]["codex_output"] = self._tail(codex_output)
            result["extra"]["solution_hash_before"] = solution_hash_before

            solution_hash_after = self._hash_file(solution_path) if solution_path.exists() else ""
            result["extra"]["solution_hash_after"] = solution_hash_after
            patch_applied = False
            patch_source = None

            if solution_hash_before == solution_hash_after:
                patch = self._extract_patch(codex_output)
                if patch:
                    patch_applied = self._apply_patch(patch, workdir)
                    patch_source = "codex_output"
                if not patch_applied:
                    solution_text = self._extract_solution(codex_output)
                    if solution_text:
                        solution_path.write_text(solution_text + "\n", encoding="utf-8")
                        patch_applied = True
                        patch_source = "full_file"

            result["extra"]["patch_applied"] = patch_applied
            if patch_source:
                result["extra"]["patch_source"] = patch_source

            tests_ok, test_output = self._run_pytest(workdir, min(self.test_timeout, timeout))
            result["extra"]["test_output"] = self._tail(test_output)

            result["score"] = 1.0 if tests_ok else 0.0
            result["success"] = tests_ok
            if not tests_ok:
                result["error"] = "tests_failed"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            result["time_taken"] = time.time() - start
            if not keep_workspace_value:
                shutil.rmtree(workdir, ignore_errors=True)

        return result
