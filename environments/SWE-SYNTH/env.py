"""
SWE-SYNTH Environment

A synthetic SWE-bench evaluation environment with dynamic bug generation:
1. Breaker model injects bugs into correct code
2. Fixer model attempts to fix the injected bugs
3. Results are cached for reproducibility

Flow:
    task_id → decode params → check cache → [generate if miss] → fixer repair → verify
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset

# Import breaker and cache modules
from breaker import BUG_TYPES, generate_bug
from cache import TwoLevelCache

# Import mini-swe-agent (installed via pip)
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel


# Timeout constants (in seconds)
DOCKER_PULL_TIMEOUT = 300
GENERATE_BUG_TIMEOUT = 300
VERIFY_BUG_TIMEOUT = 300
FIXER_TIMEOUT = 1800
VERIFY_FIX_TIMEOUT = 1800

# Bug generation: (generate + verify) * max_retries
BUG_GENERATION_MAX_RETRIES = 10
BUG_GENERATION_TIMEOUT = (GENERATE_BUG_TIMEOUT + VERIFY_BUG_TIMEOUT) * BUG_GENERATION_MAX_RETRIES  # 6000s

# Total timeout = bug generation + fix + verify fix + buffer
TOTAL_TIMEOUT = (
    BUG_GENERATION_TIMEOUT +                     # generate bug (worst case)
    FIXER_TIMEOUT +                              # fix bug
    DOCKER_PULL_TIMEOUT + VERIFY_FIX_TIMEOUT +   # verify fix
    10                                           # buffer
)


def get_dockerhub_image_uri(uid: str, dockerhub_username: str, repo_name: str) -> str:
    """Generate Docker Hub image URI matching SWE-bench naming scheme."""
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = 'element-web'
    elif 'element-hq' in repo_name.lower() and 'element-web' in repo_name.lower():
        repo_name_only = 'element'
        if hsh.endswith('-vnan'):
            hsh = hsh[:-5]
    elif hsh.endswith('-vnan'):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"{dockerhub_username}/sweap-images:{tag}"


class SynthActor:
    """
    SWE-SYNTH evaluation actor.

    Implements:
    1. Dynamic bug generation via Breaker model
    2. Bug fixing via Fixer model
    3. Caching for reproducibility
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "/tmp/swe-synth-cache",
        dockerhub_username: str = "jefzda",
        dockerfiles_dir: str = "/app/dockerfiles",
        run_scripts_dir: str = "/app/run_scripts",
        # R2 cache configuration (from environment variables)
        r2_endpoint_url: str = os.getenv("R2_ENDPOINT_URL"),
        r2_bucket: str = os.getenv("R2_BUCKET"),
        r2_access_key_id: str = os.getenv("R2_ACCESS_KEY_ID"),
        r2_secret_access_key: str = os.getenv("R2_SECRET_ACCESS_KEY"),
        r2_prefix: str = "bugs",
        r2_public_read_url: str = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev",
    ):
        """
        Initialize SWE-SYNTH actor.

        Args:
            api_key: API key for LLM (optional, can also use environment variables)
            cache_dir: Directory for local cache
            dockerhub_username: Docker Hub username for images
            dockerfiles_dir: Path to SWE-bench dockerfiles
            run_scripts_dir: Path to SWE-bench run scripts
            r2_endpoint_url: R2 endpoint URL for writes (private, requires auth)
            r2_bucket: R2 bucket name
            r2_access_key_id: R2 access key ID
            r2_secret_access_key: R2 secret access key
            r2_prefix: Prefix for R2 cache keys
            r2_public_read_url: R2 public CDN URL for reads (faster, no auth)
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.dockerhub_username = dockerhub_username
        self.dockerfiles_dir = dockerfiles_dir
        self.run_scripts_dir = run_scripts_dir

        # Initialize two-level cache
        self.cache = TwoLevelCache(
            local_cache_dir=cache_dir,
            r2_endpoint_url=r2_endpoint_url,
            r2_bucket=r2_bucket,
            r2_access_key_id=r2_access_key_id,
            r2_secret_access_key=r2_secret_access_key,
            r2_prefix=r2_prefix,
            r2_public_read_url=r2_public_read_url,
        )

        # Load SWE-bench Pro dataset
        print("Loading SWE-bench Pro dataset...")
        dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        sorted_instances = sorted(dataset, key=lambda x: x["instance_id"])
        self.swe_instances = {idx: inst for idx, inst in enumerate(sorted_instances)}
        print(f"Loaded {len(self.swe_instances)} SWE-bench Pro instances")

        self.num_swe_instances = len(self.swe_instances)
        self.num_bug_types = len(BUG_TYPES)

    def _decode_task_id(self, task_id: int) -> Dict[str, Any]:
        """
        Decode task_id into deterministic parameters.

        Encoding: task_id maps to (swe_instance, bug_types, seed)
        - swe_idx = task_id % num_swe_instances
        - seed = task_id
        - bug_types = randomly selected 1-3 types using seed
        """
        import random

        swe_idx = task_id % self.num_swe_instances
        seed = task_id

        # Use seed to randomly select 1-3 bug types
        rng = random.Random(seed)
        num_types = rng.randint(1, 3)
        bug_types = rng.sample(BUG_TYPES, num_types)

        return {
            "swe_instance_idx": swe_idx,
            "swe_instance": self.swe_instances[swe_idx],
            "bug_types": bug_types,
            "seed": seed,
        }


    def _load_instance_script(self, instance_id: str, script_name: str) -> Optional[str]:
        """Load instance-specific script."""
        script_path = Path(self.run_scripts_dir) / instance_id / script_name
        if not script_path.exists():
            return None
        with open(script_path, 'r') as f:
            return f.read()

    def _load_dockerfile(self, instance_id: str, dockerfile_type: str) -> str:
        """Load Dockerfile content."""
        dockerfile_path = f"{self.dockerfiles_dir}/{dockerfile_type}_dockerfile/{instance_id}/Dockerfile"
        with open(dockerfile_path) as fp:
            return fp.read()

    def _extract_env_commands(self, base_dockerfile: str, instance_dockerfile: str) -> str:
        """Extract ENV commands from Dockerfiles."""
        env_cmds = []
        for dockerfile_content in [base_dockerfile, instance_dockerfile]:
            for line in dockerfile_content.split("\n"):
                line = line.strip()
                if line.startswith("ENV"):
                    env_cmd = line.replace("ENV", "export", 1)
                    env_cmds.append(env_cmd)
        return "\n".join(env_cmds)

    def _fetch_repo_files(
        self,
        instance_id: str,
        repo: str,
        base_commit: str,
        gold_patch: str,
        seed: int,
        max_files: int = 5,
        extensions: tuple = (".py", ".js", ".ts", ".java", ".go", ".rb", ".php"),
    ) -> list[dict]:
        """
        Fetch source files from the repo for bug injection.
        Prioritizes files modified by gold_patch.

        Returns list of {"path": str, "content": str}
        """
        import random
        import re

        image = get_dockerhub_image_uri(instance_id, self.dockerhub_username, repo)

        # Extract files modified by gold_patch
        patch_files = []
        for line in gold_patch.split('\n'):
            if line.startswith('diff --git'):
                # Extract file path from "diff --git a/path b/path"
                match = re.search(r'diff --git a/(.+?) b/', line)
                if match:
                    filepath = match.group(1)
                    if not any(p in filepath.lower() for p in ['test', 'spec']):
                        patch_files.append('./' + filepath)

        print(f"Files in gold_patch: {patch_files}")

        # Script to read the patch files (after applying gold_patch)
        read_script = f"""#!/bin/bash
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /dev/stdin <<'PATCH'
{gold_patch}
PATCH

"""
        # Read each file from gold_patch
        for f in patch_files[:max_files]:
            read_script += f'echo "===FILE:{f}==="\n'
            read_script += f'cat "{f}" 2>/dev/null | head -300\n'
            read_script += f'echo "===ENDFILE==="\n'

        try:
            print(f"Pulling image: {image}")
            subprocess.run(["docker", "pull", image], capture_output=True, timeout=300)

            print(f"Fetching {len(patch_files)} files from gold_patch...")
            result = subprocess.run(
                ["docker", "run", "--rm", "-i", "--entrypoint", "/bin/bash", image],
                input=read_script,
                capture_output=True,
                timeout=300,
                text=True
            )

            # Parse file contents
            repo_files = []
            stdout = result.stdout
            for filepath in patch_files[:max_files]:
                marker_start = f"===FILE:{filepath}==="
                marker_end = "===ENDFILE==="
                if marker_start in stdout:
                    start = stdout.index(marker_start) + len(marker_start)
                    end = stdout.index(marker_end, start)
                    content = stdout[start:end].strip()
                    if content:
                        repo_files.append({"path": filepath, "content": content})

            print(f"Fetched {len(repo_files)} file contents")
            return repo_files

        except Exception as e:
            print(f"Failed to fetch repo files: {e}")
            return []

    def _verify_bug(
        self,
        bug_instance: Dict[str, Any],
        swe_instance: Dict[str, Any],
        max_failed_ratio: float = 0.2,  # Max 20% of tests can fail
        max_failed_count: int = 10,     # Or max 10 tests
    ) -> Dict[str, Any]:
        """
        Verify if the generated bug is valid:
        - Some tests should FAIL after bug injection (these become target_tests)
        - Not too many tests should fail (bug should be targeted, not catastrophic)

        Returns:
            verification result dict with 'valid', 'target_tests', etc.
        """
        import base64

        source = bug_instance["source"]
        instance_id = source["swe_instance_id"]
        repo = source["repo"]
        base_commit = source["base_commit"]

        gold_patch = bug_instance["original"]["gold_patch"]
        bug_patch = bug_instance["bug"].get("patch", "")
        fail_to_pass_original = bug_instance["original"].get("fail_to_pass", [])

        # Handle case where bug_patch might be dict or other type
        if isinstance(bug_patch, dict):
            bug_patch = bug_patch.get("patch", "") or ""
        if not bug_patch or not str(bug_patch).strip():
            return {"valid": False, "error": "Empty bug patch"}

        # Get all tests
        fail_to_pass = swe_instance.get("FAIL_TO_PASS", swe_instance.get("fail_to_pass", "[]"))
        pass_to_pass = swe_instance.get("PASS_TO_PASS", swe_instance.get("pass_to_pass", "[]"))

        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = eval(fail_to_pass)
            except:
                fail_to_pass = []
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = eval(pass_to_pass)
            except:
                pass_to_pass = []

        all_tests = set(fail_to_pass) | set(pass_to_pass)
        total_tests = len(all_tests)

        # Load scripts
        run_script = self._load_instance_script(instance_id, "run_script.sh")
        parser_script = self._load_instance_script(instance_id, "parser.py")

        if not run_script or not parser_script:
            return {"valid": False, "error": "Missing test scripts", "skipped": True}

        # Build verification script
        try:
            base_dockerfile = self._load_dockerfile(instance_id, "base")
            instance_dockerfile = self._load_dockerfile(instance_id, "instance")
            env_cmds = self._extract_env_commands(base_dockerfile, instance_dockerfile)
        except Exception:
            env_cmds = ""

        before_repo_set_cmd = swe_instance.get("before_repo_set_cmd", "").strip()
        if before_repo_set_cmd:
            before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1]

        selected_test_files = swe_instance.get("selected_test_files_to_run", "[]")
        if isinstance(selected_test_files, str):
            try:
                selected_test_files = eval(selected_test_files)
            except:
                selected_test_files = []
        test_files_str = ",".join(selected_test_files) if selected_test_files else ""

        # Flow: base_commit → gold_patch → bug_patch (no fix)
        entryscript = f"""
{env_cmds}
cd /app
git reset --hard {base_commit}
git checkout {base_commit}

# Step 1: Apply gold_patch to get correct code (all tests pass)
git apply -v /workspace/gold_patch.diff

# Step 2: Apply bug_patch to inject the bug
git apply -v /workspace/bug_patch.diff || echo "Bug patch apply failed"
{before_repo_set_cmd}

# Run tests
bash /workspace/run_script.sh {test_files_str} > /workspace/stdout.log 2> /workspace/stderr.log
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""

        gold_patch_b64 = base64.b64encode(gold_patch.encode('utf-8')).decode('ascii')
        bug_patch_b64 = base64.b64encode(bug_patch.encode('utf-8')).decode('ascii')
        run_script_b64 = base64.b64encode(run_script.encode('utf-8')).decode('ascii')
        parser_script_b64 = base64.b64encode(parser_script.encode('utf-8')).decode('ascii')
        entryscript_b64 = base64.b64encode(entryscript.encode('utf-8')).decode('ascii')

        full_script = f"""#!/bin/bash
mkdir -p /workspace
echo "{gold_patch_b64}" | base64 -d > /workspace/gold_patch.diff
echo "{bug_patch_b64}" | base64 -d > /workspace/bug_patch.diff
echo "{run_script_b64}" | base64 -d > /workspace/run_script.sh
echo "{parser_script_b64}" | base64 -d > /workspace/parser.py
echo "{entryscript_b64}" | base64 -d > /workspace/entryscript.sh
chmod +x /workspace/run_script.sh /workspace/entryscript.sh
bash /workspace/entryscript.sh
if [ -f /workspace/output.json ]; then
    echo "===SWESYNTH_OUTPUT_BEGIN==="
    cat /workspace/output.json
    echo "===SWESYNTH_OUTPUT_END==="
fi
"""

        # Get Docker image and run
        image = get_dockerhub_image_uri(instance_id, self.dockerhub_username, repo)

        try:
            print(f"Pulling image: {image}")
            pull_result = subprocess.run(
                ["docker", "pull", image],
                check=False, capture_output=True, timeout=300, text=True
            )
            if pull_result.returncode != 0:
                print(f"Pull warning: {pull_result.stderr[:200] if pull_result.stderr else 'unknown'}")

            print(f"Running verification container...")
            result = subprocess.run(
                ["docker", "run", "--rm", "-i", "--entrypoint", "/bin/bash", image],
                input=full_script,
                capture_output=True,
                timeout=300,
                text=True
            )

            stdout = result.stdout

            begin_marker = "===SWESYNTH_OUTPUT_BEGIN==="
            end_marker = "===SWESYNTH_OUTPUT_END==="

            if begin_marker not in stdout or end_marker not in stdout:
                print(f"[DEBUG] Verification failed - no output markers")
                print(f"[DEBUG] stdout (last 1000 chars): {stdout[-1000:] if stdout else 'empty'}")
                print(f"[DEBUG] stderr (last 1000 chars): {result.stderr[-1000:] if result.stderr else 'empty'}")
                return {
                    "valid": False,
                    "error": "No test output markers",
                    "stderr": result.stderr[:500] if result.stderr else "",
                    "stdout_tail": stdout[-500:] if stdout else "",
                    "skipped": True,
                }

            json_start = stdout.index(begin_marker) + len(begin_marker)
            json_end = stdout.index(end_marker)
            json_str = stdout[json_start:json_end].strip()

            output = json.loads(json_str)

            passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}
            failed_tests = {x["name"] for x in output["tests"] if x["status"] == "FAILED"}

            # New logic: discover target_tests from actual failures
            num_failed = len(failed_tests)
            num_passed = len(passed_tests)

            # Validation rules:
            # 1. Must have at least 1 failed test (bug has effect)
            # 2. Failed tests shouldn't exceed max(max_failed_ratio, max_failed_count)
            has_failures = num_failed > 0
            max_allowed = max(max_failed_count, int(total_tests * max_failed_ratio)) if total_tests > 0 else max_failed_count
            not_too_many_failures = num_failed <= max(max_allowed, 1)  # At least allow 1

            valid = has_failures and not_too_many_failures

            # The failed tests become our target_tests
            target_tests = list(failed_tests)

            # Update bug_instance with discovered target_tests
            bug_instance["bug"]["target_tests"] = target_tests

            verification_result = {
                "valid": valid,
                "verified": True,
                "target_tests": target_tests,
                "num_failed": num_failed,
                "num_passed": num_passed,
                "total_tests": total_tests,
                "max_allowed_failures": max_allowed,
            }

            if not has_failures:
                verification_result["error"] = "No tests failed - bug has no effect"
            elif not not_too_many_failures:
                verification_result["error"] = f"Too many tests failed ({num_failed} > {max_allowed})"

            return verification_result

        except subprocess.TimeoutExpired:
            return {"valid": False, "error": "Verification timeout", "skipped": True}
        except Exception as e:
            import traceback
            return {"valid": False, "error": str(e), "trace": traceback.format_exc(), "skipped": True}

    async def _fix_bug(
        self,
        bug_instance: Dict[str, Any],
        fixer_model: str,
        fixer_base_url: str,
        fixer_api_key: str,
        timeout: int,
        max_iterations: int,
        cost_limit: float,
        temperature: float,
        seed: Optional[int] = None,
    ) -> tuple[Optional[str], Dict[str, Any], list]:
        """
        Use Fixer model to repair the bug.

        Returns:
            Tuple of (patch, metadata, conversation)
        """
        try:
            import yaml

            source = bug_instance["source"]
            instance_id = source["swe_instance_id"]
            repo = source["repo"]

            problem_statement = bug_instance["bug"]["problem_statement"]

            # Get Docker image
            image = get_dockerhub_image_uri(instance_id, self.dockerhub_username, repo)
            print(f"Fixing bug using image: {image}")

            # Initialize model
            litellm_model_name = f"openai/{fixer_model}" if not fixer_model.startswith("openai/") else fixer_model
            model_kwargs = {
                "api_base": fixer_base_url,
                "api_key": fixer_api_key,
                "temperature": temperature,
            }
            if seed is not None:
                model_kwargs["seed"] = seed

            model_obj = LitellmModel(
                model_name=litellm_model_name,
                model_kwargs=model_kwargs,
                cost_tracking="ignore_errors",
            )

            # Initialize Docker environment
            container_name = f"swe-synth-fixer-{int(time.time() * 1000)}"
            env = DockerEnvironment(
                image=image,
                cwd="/app",
                timeout=timeout,
                executable="docker",
                run_args=["--rm", "--entrypoint", "", "--name", container_name],
                container_timeout=str(timeout),
            )

            # Load agent config
            config_path = Path(__file__).parent / "config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                agent_config = config.get("agent", {}).copy()
            else:
                agent_config = {}

            agent_config["step_limit"] = max_iterations
            agent_config["cost_limit"] = cost_limit

            # Create and run agent
            agent = DefaultAgent(model_obj, env, **agent_config)

            patch = ""
            error = ""

            try:
                loop = asyncio.get_event_loop()
                _, result = await loop.run_in_executor(
                    None,
                    agent.run,
                    problem_statement
                )
                patch = result

            except Exception as e:
                import traceback
                error = traceback.format_exc()
                print(f"Error running fixer agent: {e}")

            finally:
                try:
                    env.cleanup()
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

            # Extract usage stats
            total_tokens = 0
            clean_conversation = []

            for msg in agent.messages:
                if isinstance(msg, dict) and "extra" in msg:
                    msg_extra = msg.get("extra", {})
                    if isinstance(msg_extra, dict):
                        msg_usage = msg_extra.get("usage") or (
                            msg_extra.get("response", {}).get("usage")
                        )
                        if msg_usage:
                            total_tokens += msg_usage.get("total_tokens", 0)
                    clean_msg = {k: v for k, v in msg.items() if k != "extra"}
                    clean_conversation.append(clean_msg)
                else:
                    clean_conversation.append(msg)

            metadata = {
                "model_calls": agent.model.n_calls,
                "model_cost": agent.model.cost,
                "total_tokens": total_tokens,
            }
            if error:
                metadata["error"] = error

            return patch, metadata, clean_conversation

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in _fix_bug: {e}")
            return None, {"error": error_trace}, []

    def _verify_fix(
        self,
        bug_instance: Dict[str, Any],
        fix_patch: str,
    ) -> tuple[float, Dict[str, Any]]:
        """
        Verify if the fix patch resolves the bug.

        Returns:
            Tuple of (score, test_stats)
        """
        if not fix_patch or not fix_patch.strip():
            return 0.0, {"error": "no patch"}

        try:
            source = bug_instance["source"]
            instance_id = source["swe_instance_id"]
            repo = source["repo"]
            base_commit = source["base_commit"]

            # Get original instance for test info
            swe_instance = None
            for inst in self.swe_instances.values():
                if inst["instance_id"] == instance_id:
                    swe_instance = inst
                    break

            if not swe_instance:
                return 0.0, {"error": f"Instance not found: {instance_id}"}

            # Get test requirements
            fail_to_pass = swe_instance.get("FAIL_TO_PASS", swe_instance.get("fail_to_pass", "[]"))
            pass_to_pass = swe_instance.get("PASS_TO_PASS", swe_instance.get("pass_to_pass", "[]"))

            if isinstance(fail_to_pass, str):
                try:
                    fail_to_pass = eval(fail_to_pass)
                except:
                    fail_to_pass = []
            if isinstance(pass_to_pass, str):
                try:
                    pass_to_pass = eval(pass_to_pass)
                except:
                    pass_to_pass = []

            f2p = set(fail_to_pass)
            p2p = set(pass_to_pass)
            required_tests = f2p | p2p

            if not required_tests:
                return 0.0, {"error": "No required tests"}

            # Load scripts
            run_script = self._load_instance_script(instance_id, "run_script.sh")
            parser_script = self._load_instance_script(instance_id, "parser.py")

            if not run_script or not parser_script:
                return 0.0, {"error": "Missing scripts"}

            # Build verification script
            try:
                base_dockerfile = self._load_dockerfile(instance_id, "base")
                instance_dockerfile = self._load_dockerfile(instance_id, "instance")
                env_cmds = self._extract_env_commands(base_dockerfile, instance_dockerfile)
            except Exception:
                env_cmds = ""

            before_repo_set_cmd = swe_instance.get("before_repo_set_cmd", "").strip()
            if before_repo_set_cmd:
                before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1]

            selected_test_files = swe_instance.get("selected_test_files_to_run", "[]")
            if isinstance(selected_test_files, str):
                try:
                    selected_test_files = eval(selected_test_files)
                except:
                    selected_test_files = []
            test_files_str = ",".join(selected_test_files) if selected_test_files else ""

            import base64

            # Get patches
            gold_patch = bug_instance["original"]["gold_patch"]
            bug_patch = bug_instance["bug"]["patch"]

            # Flow: base_commit → gold_patch → bug_patch → fix_patch
            entryscript = f"""
{env_cmds}
cd /app
git reset --hard {base_commit}
git checkout {base_commit}

# Step 1: Apply gold_patch to get correct code (all tests pass)
git apply -v /workspace/gold_patch.diff

# Step 2: Apply bug_patch to inject the bug (target tests should fail)
git apply -v /workspace/bug_patch.diff || true

# Step 3: Apply fix_patch from Fixer (should restore correctness)
git apply -v /workspace/fix_patch.diff
{before_repo_set_cmd}

# Run tests
bash /workspace/run_script.sh {test_files_str} > /workspace/stdout.log 2> /workspace/stderr.log
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""

            gold_patch_b64 = base64.b64encode(gold_patch.encode('utf-8')).decode('ascii')
            bug_patch_b64 = base64.b64encode(bug_patch.encode('utf-8')).decode('ascii')
            fix_patch_b64 = base64.b64encode(fix_patch.encode('utf-8')).decode('ascii')
            run_script_b64 = base64.b64encode(run_script.encode('utf-8')).decode('ascii')
            parser_script_b64 = base64.b64encode(parser_script.encode('utf-8')).decode('ascii')
            entryscript_b64 = base64.b64encode(entryscript.encode('utf-8')).decode('ascii')

            full_script = f"""#!/bin/bash
mkdir -p /workspace
echo "{gold_patch_b64}" | base64 -d > /workspace/gold_patch.diff
echo "{bug_patch_b64}" | base64 -d > /workspace/bug_patch.diff
echo "{fix_patch_b64}" | base64 -d > /workspace/fix_patch.diff
echo "{run_script_b64}" | base64 -d > /workspace/run_script.sh
echo "{parser_script_b64}" | base64 -d > /workspace/parser.py
echo "{entryscript_b64}" | base64 -d > /workspace/entryscript.sh
chmod +x /workspace/run_script.sh /workspace/entryscript.sh
bash /workspace/entryscript.sh
if [ -f /workspace/output.json ]; then
    echo "===SWESYNTH_OUTPUT_BEGIN==="
    cat /workspace/output.json
    echo "===SWESYNTH_OUTPUT_END==="
fi
"""

            # Get Docker image and run
            image = get_dockerhub_image_uri(instance_id, self.dockerhub_username, repo)

            print(f"Pulling image for verify_fix: {image}")
            subprocess.run(
                ["docker", "pull", image],
                check=False, capture_output=True, timeout=DOCKER_PULL_TIMEOUT
            )

            print(f"Running verification container (timeout={VERIFY_FIX_TIMEOUT}s)...")
            result = subprocess.run(
                ["docker", "run", "--rm", "-i", "--entrypoint", "/bin/bash", image],
                input=full_script,
                capture_output=True,
                timeout=VERIFY_FIX_TIMEOUT,
                text=True
            )
            print("Verification container completed.")

            stdout = result.stdout

            begin_marker = "===SWESYNTH_OUTPUT_BEGIN==="
            end_marker = "===SWESYNTH_OUTPUT_END==="

            if begin_marker not in stdout or end_marker not in stdout:
                return 0.0, {"error": "No output markers", "stderr": result.stderr[:500]}

            json_start = stdout.index(begin_marker) + len(begin_marker)
            json_end = stdout.index(end_marker)
            json_str = stdout[json_start:json_end].strip()

            output = json.loads(json_str)

            passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}

            # Get target tests from bug_instance (tests that Breaker tried to break)
            target_tests = set(bug_instance["bug"].get("target_tests", []))

            # Success criteria: target_tests must pass (bug was fixed)
            # Also check all tests for completeness
            all_required = f2p | p2p
            target_passed = target_tests <= passed_tests
            all_passed = all_required <= passed_tests

            target_passed_count = len(target_tests & passed_tests)
            total_target = len(target_tests)
            all_passed_count = len(all_required & passed_tests)
            total_all = len(all_required)

            test_stats = {
                "target_tests": list(target_tests),
                "target_result": f"{target_passed_count}/{total_target}",
                "all_result": f"{all_passed_count}/{total_all}",
                "target_passed": target_passed,
                "all_passed": all_passed,
            }

            # Score based on target tests (primary) and all tests (bonus)
            if target_passed and all_passed:
                return 1.0, test_stats
            elif target_passed:
                # Fixed target but broke something else
                test_stats["missing_tests"] = list(all_required - passed_tests)
                return 0.5, test_stats
            else:
                # Failed to fix target tests
                test_stats["missing_target_tests"] = list(target_tests - passed_tests)
                return 0.0, test_stats

        except subprocess.TimeoutExpired:
            return 0.0, {"error": "timeout"}
        except Exception as e:
            import traceback
            return 0.0, {"error": traceback.format_exc()}

    async def evaluate(
        self,
        task_id: int,
        model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        timeout: int = 1800,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_iterations: int = 30,
        cost_limit: float = 10.0,
        skip_cache: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete evaluation: generate bug (if needed) + fix it.

        Args:
            task_id: Deterministically maps to (swe_instance, bug_types, seed)
            model: Model for fixing bugs
            base_url: API base URL
            api_key: API key (uses env var CHUTES_API_KEY if not provided)
            timeout: Timeout for commands
            temperature: Model temperature for fixer
            seed: Random seed for LLM inference
            max_iterations: Max agent steps for fixer
            cost_limit: Max cost for fixer
            skip_cache: Force regenerate bug even if cached

        Returns:
            Result dict with score and metadata
        """
        start = time.time()

        # Use provided api_key or fall back to instance api_key
        chutes_api_key = api_key or self.api_key
        if not chutes_api_key:
            raise ValueError("api_key required (pass to evaluate() or set CHUTES_API_KEY env var)")

        # Fixer uses user-provided model
        fixer_model = model
        fixer_base_url = base_url
        fixer_api_key = chutes_api_key

        # Breaker model is fixed (not user-configurable)
        breaker_model = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE"
        breaker_base_url = "https://llm.chutes.ai/v1"
        breaker_api_key = chutes_api_key

        # Decode task_id
        params = self._decode_task_id(task_id)
        print(f"Task {task_id} -> SWE instance {params['swe_instance_idx']}, bug types: {params['bug_types']}")

        # Check cache (L1: local, L2: R2)
        bug_from_cache = False
        bug_instance = None

        if not skip_cache:
            bug_instance = self.cache.load(task_id)
            if bug_instance:
                bug_from_cache = True
                print(f"Loaded bug from cache: task_{task_id}")

        if not bug_from_cache:
            # Acquire distributed lock before generating
            print(f"Acquiring lock for task {task_id}...")
            lock_acquired = self.cache.acquire_lock(task_id)

            if not lock_acquired:
                # Lock held by another machine, wait for it to complete
                print(f"Lock not acquired, waiting for another machine to complete...")
                wait_interval = 120  # seconds
                max_retries = 10

                for retry in range(max_retries):
                    time.sleep(wait_interval)
                    print(f"Checking cache (attempt {retry + 1}/{max_retries})...")

                    bug_instance = self.cache.load(task_id)
                    if bug_instance:
                        bug_from_cache = True
                        print(f"Bug generated by another machine: task_{task_id}")
                        break

                if not bug_from_cache:
                    raise RuntimeError(
                        f"Timed out waiting for another machine to generate bug for task {task_id}"
                    )

            if not bug_from_cache:
                try:
                    # Double-check cache after acquiring lock
                    bug_instance = self.cache.load(task_id)
                    if bug_instance:
                        bug_from_cache = True
                        print(f"Bug was generated by another machine: task_{task_id}")
                    else:
                        # Fetch repo files for bug injection (only once)
                        swe_inst = params["swe_instance"]
                        repo_files = self._fetch_repo_files(
                            instance_id=swe_inst["instance_id"],
                            repo=swe_inst.get("repo", ""),
                            base_commit=swe_inst.get("base_commit", ""),
                            gold_patch=swe_inst.get("patch", ""),
                            seed=params["seed"],
                            max_files=5,
                        )

                        # Generate and verify bug (retry up to max times)
                        bug_verified = False
                        last_verification = None

                        for verify_attempt in range(BUG_GENERATION_MAX_RETRIES):
                            # Generate bug using breaker module
                            print(f"Generating bug with {breaker_model} (verify attempt {verify_attempt + 1}/{BUG_GENERATION_MAX_RETRIES})...")
                            bug_instance = await generate_bug(
                                swe_instance=params["swe_instance"],
                                bug_types=params["bug_types"],
                                seed=params["seed"] + verify_attempt * 100,  # Vary seed on retry
                                breaker_model=breaker_model,
                                breaker_base_url=breaker_base_url,
                                breaker_api_key=breaker_api_key,
                                repo_files=repo_files,  # Pass fetched files
                            )

                            # Verify the generated bug
                            print(f"Verifying bug...")
                            verification = self._verify_bug(bug_instance, params["swe_instance"])
                            bug_instance["verification"] = verification
                            last_verification = verification

                            if verification.get("valid"):
                                num_failed = verification.get("num_failed", 0)
                                print(f"Bug verified: {num_failed} tests fail (target_tests discovered)")
                                bug_verified = True
                                break
                            elif verification.get("skipped"):
                                # Verification couldn't run (missing scripts, etc.), accept the bug
                                print(f"Bug verification skipped: {verification.get('error', 'unknown')}")
                                bug_verified = True
                                break
                            else:
                                # Verification failed
                                print(f"Bug verification failed: {verification}")
                                if verify_attempt < BUG_GENERATION_MAX_RETRIES - 1:
                                    print(f"Retrying bug generation...")

                        # All retries failed - raise error
                        if not bug_verified:
                            raise RuntimeError(
                                f"Failed to generate valid bug after {BUG_GENERATION_MAX_RETRIES} attempts. "
                                f"Last verification: {last_verification}"
                            )

                        # Save with conflict resolution
                        # If another machine saved while we were generating, use theirs
                        saved, existing = self.cache.save_if_not_exists(task_id, bug_instance)
                        if saved:
                            print(f"Bug cached: task_{task_id}")
                        else:
                            # Conflict: another machine finished first, use their result
                            print(f"Conflict detected, using existing bug: task_{task_id}")
                            bug_instance = existing
                            bug_from_cache = True
                finally:
                    self.cache.release_lock(task_id)

        # Fix bug
        print(f"Fixing bug with {fixer_model}...")
        fix_patch, fixer_metadata, conversation = await self._fix_bug(
            bug_instance,
            fixer_model,
            fixer_base_url,
            fixer_api_key,
            timeout,
            max_iterations,
            cost_limit,
            temperature,
            seed,
        )

        # Verify fix
        print("Verifying fix...")
        score, test_stats = self._verify_fix(bug_instance, fix_patch)

        result = {
            "task_name": "swe-synth",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "task_id": task_id,
                "bug_id": f"{bug_instance['source']['swe_instance_id']}/{'+'.join(params['bug_types'])}_{params['seed']}",
                "bug_from_cache": bug_from_cache,
                "bug_types": params["bug_types"],
                "swe_instance_id": bug_instance["source"]["swe_instance_id"],
                "problem_statement": bug_instance["bug"]["problem_statement"],
                "fix_patch": fix_patch,
                "conversation": conversation,
                **fixer_metadata,
                **test_stats,
            }
        }

        return result


# Alias for framework compatibility
Actor = SynthActor
