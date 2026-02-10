"""
Breaker Service - Independent task generation service

Continuously generates bug injection tasks and stores them to R2.
Tasks are generated sequentially starting from task_id=0.

Supports horizontal scaling with distributed locking:
- Workers claim tasks atomically via R2 lock files
- Failed tasks are retried by other workers
- Guarantees sequential task_id with no gaps

Usage:
    python -m breaker.service --max-tasks 100
    python -m breaker.service --start-from 500 --batch-size 10

Environment variables:
    CHUTES_API_KEY: API key for LLM
    R2_ENDPOINT_URL, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""

import os
import sys
import logging

# Disable verbose logging BEFORE any imports
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["HTTPX_LOG_LEVEL"] = "ERROR"


class NoDebugFilter(logging.Filter):
    """Filter out all DEBUG level messages."""
    def filter(self, record):
        return record.levelno >= logging.INFO


# Apply filter to root logger to block ALL debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
for handler in logging.root.handlers:
    handler.addFilter(NoDebugFilter())

# Also set root logger level
logging.root.setLevel(logging.INFO)

import json
import time
import random
import asyncio
import argparse
import socket
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import boto3
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Silence noisy loggers AFTER imports
_noisy_loggers = [
    "httpcore", "httpx", "botocore", "boto3", "urllib3",
    "filelock", "fsspec", "litellm", "LiteLLM",
    "huggingface_hub", "datasets", "aiohttp", "asyncio",
    "httpcore.http11", "httpcore.connection",
    "minisweagent", "minisweagent.environment",
]
for _name in _noisy_loggers:
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.ERROR)
    _logger.handlers = []
    _logger.propagate = False
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import load_dataset

from .types import BreakerInput, BreakerOutput, BreakerException
from .orchestrator import run_breaker
from .bug_types import BUG_TYPES

# Claim expires after 30 minutes (task generation timeout)
CLAIM_TIMEOUT = 1800


def log(msg: str) -> None:
    """Print log message with timestamp."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def cleanup_docker_resources(current_image: str = None, max_images: int = 10) -> None:
    """Clean up Docker resources to free disk space after each task.

    Args:
        current_image: The image used in current task (will be kept)
        max_images: Maximum number of sweap-images to keep (default 10)
    """
    try:
        # Remove stopped containers
        subprocess.run(
            ["docker", "container", "prune", "-f"],
            capture_output=True, timeout=60
        )
        # Remove dangling images (untagged)
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            capture_output=True, timeout=60
        )

        # Clean up old sweap-images to prevent disk space exhaustion
        # Keep only the most recent N images
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
                 "--filter", "reference=*/sweap-images:*"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                images = []
                for line in result.stdout.strip().split('\n'):
                    if line and '\t' in line:
                        image, created = line.split('\t', 1)
                        images.append((image, created))

                # Sort by creation time (newest first)
                images.sort(key=lambda x: x[1], reverse=True)

                # Remove images beyond max_images limit (keep current_image if specified)
                images_to_remove = []
                kept = 0
                for image, _ in images:
                    if current_image and image == current_image:
                        continue  # Always keep current image
                    if kept < max_images:
                        kept += 1
                    else:
                        images_to_remove.append(image)

                # Remove old images
                for image in images_to_remove:
                    subprocess.run(
                        ["docker", "rmi", "-f", image],
                        capture_output=True, timeout=60
                    )

                if images_to_remove:
                    log(f"Cleaned up {len(images_to_remove)} old sweap-images")
        except Exception as e:
            log(f"Warning: Failed to clean sweap-images: {e}")

        # Clean up Docker build cache
        subprocess.run(
            ["docker", "builder", "prune", "-f", "--filter", "until=24h"],
            capture_output=True, timeout=60
        )

        # Clean up old breaker workspaces
        workspace_base = Path("/tmp/breaker_workspaces")
        if workspace_base.exists():
            cutoff = time.time() - 3600  # 1 hour old
            for ws in workspace_base.iterdir():
                try:
                    if ws.is_dir() and ws.stat().st_mtime < cutoff:
                        shutil.rmtree(ws, ignore_errors=True)
                except Exception:
                    pass
    except Exception as e:
        log(f"Cleanup warning: {e}")


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


class BreakerService:
    """
    Independent breaker service for continuous task generation.

    Responsibilities:
    1. Load SWE-bench Pro dataset
    2. Generate tasks sequentially (task_id from 0)
    3. Store generated tasks to R2
    4. Maintain metadata file for tracking progress
    """

    def __init__(
        self,
        r2_endpoint_url: str,
        r2_bucket: str,
        r2_access_key_id: str,
        r2_secret_access_key: str,
        r2_prefix: str = "bugs",
        model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
        api_base: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        agent_type: str = "miniswe",
        dockerhub_username: str = "jefzda",
        run_scripts_dir: str = "/app/run_scripts",
        dockerfiles_dir: str = "/app/dockerfiles",
    ):
        self.r2_endpoint_url = r2_endpoint_url
        self.r2_bucket = r2_bucket
        self.r2_prefix = r2_prefix

        self.model = model
        self.api_base = api_base
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.agent_type = agent_type

        self.dockerhub_username = dockerhub_username
        self.run_scripts_dir = run_scripts_dir
        self.dockerfiles_dir = dockerfiles_dir

        # Machine identifier
        self.machine_id = f"{socket.gethostname()}_{os.getpid()}_{int(time.time())}"

        # Initialize S3 client for R2
        self.s3 = boto3.client(
            's3',
            endpoint_url=r2_endpoint_url,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 3}
            )
        )

        # Load SWE-bench Pro dataset
        print("Loading SWE-bench Pro dataset...")
        dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        sorted_instances = sorted(dataset, key=lambda x: x["instance_id"])

        # Filter to only instances with available Docker images
        print("Fetching available Docker images from Docker Hub...")
        available_tags = self._fetch_dockerhub_tags()
        print(f"Found {len(available_tags)} available images on Docker Hub")

        filtered_instances = []
        for inst in sorted_instances:
            instance_id = inst["instance_id"]
            repo = inst.get("repo", "")
            tag = self._get_image_tag(instance_id, repo)
            if tag in available_tags:
                filtered_instances.append(inst)

        self.swe_instances = {idx: inst for idx, inst in enumerate(filtered_instances)}
        self.num_swe_instances = len(self.swe_instances)
        print(f"Filtered to {self.num_swe_instances} instances with available Docker images")

    def _fetch_dockerhub_tags(self) -> set:
        """Fetch all available image tags from Docker Hub."""
        import httpx

        all_tags = set()
        page = 1
        base_url = f"https://hub.docker.com/v2/repositories/{self.dockerhub_username}/sweap-images/tags"

        with httpx.Client(timeout=30) as client:
            while True:
                try:
                    resp = client.get(f"{base_url}?page_size=100&page={page}")
                    if resp.status_code != 200:
                        break
                    data = resp.json()
                    results = data.get('results', [])
                    if not results:
                        break
                    all_tags.update(t['name'] for t in results)
                    if not data.get('next'):
                        break
                    page += 1
                except Exception as e:
                    print(f"Error fetching Docker Hub tags (page {page}): {e}")
                    break

        return all_tags

    def _get_image_tag(self, instance_id: str, repo: str) -> str:
        """Generate Docker image tag for an instance (without registry prefix)."""
        if not repo or "/" not in repo:
            return ""
        repo_base, repo_name_only = repo.lower().split("/")
        hsh = instance_id.replace("instance_", "")

        # Special cases matching get_dockerhub_image_uri
        if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
            repo_name_only = 'element-web'
        elif 'element-hq' in repo.lower() and 'element-web' in repo.lower():
            repo_name_only = 'element'
            if hsh.endswith('-vnan'):
                hsh = hsh[:-5]
        elif hsh.endswith('-vnan'):
            hsh = hsh[:-5]

        tag = f"{repo_base}.{repo_name_only}-{hsh}"
        if len(tag) > 128:
            tag = tag[:128]
        return tag

    def _get_task_key(self, task_id: int) -> str:
        # Zero-padded to 11 digits for lexicographic ordering (supports up to 100B tasks)
        return f"{self.r2_prefix}/task_{task_id:011d}.json"

    def _get_claim_key(self, task_id: int) -> str:
        return f"{self.r2_prefix}/claims/task_{task_id:011d}.claim"

    def _get_metadata_key(self) -> str:
        return f"{self.r2_prefix}/metadata.json"

    # ==================== Distributed Locking ====================

    def _get_claim(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get claim info for a task, return None if no claim."""
        try:
            response = self.s3.get_object(
                Bucket=self.r2_bucket,
                Key=self._get_claim_key(task_id)
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def _write_claim(self, task_id: int) -> Dict[str, Any]:
        """Write claim file for a task."""
        claim_data = {
            "machine_id": self.machine_id,
            "claimed_at": time.time(),
            "expires_at": time.time() + CLAIM_TIMEOUT,
        }
        self.s3.put_object(
            Bucket=self.r2_bucket,
            Key=self._get_claim_key(task_id),
            Body=json.dumps(claim_data).encode('utf-8'),
            ContentType='application/json'
        )
        return claim_data

    def _release_claim(self, task_id: int) -> None:
        """Release claim for a task (only if we own it)."""
        try:
            claim = self._get_claim(task_id)
            if claim and claim.get('machine_id') == self.machine_id:
                self.s3.delete_object(
                    Bucket=self.r2_bucket,
                    Key=self._get_claim_key(task_id)
                )
        except ClientError:
            pass

    def try_claim_task(self, task_id: int) -> bool:
        """
        Try to claim a task for processing.

        Returns True if:
        - Task already completed (no work needed)
        - Successfully claimed the task

        Returns False if:
        - Task is claimed by another worker (not expired)
        """
        # Already completed?
        if self.task_exists(task_id):
            return True

        # Check existing claim
        claim = self._get_claim(task_id)
        if claim:
            # Claim exists, check if expired
            if claim.get('expires_at', 0) > time.time():
                # Claim is valid and held by another machine
                if claim.get('machine_id') != self.machine_id:
                    return False
                # We already hold the claim
                return True
            # Claim expired, we can take over

        # Try to claim
        self._write_claim(task_id)

        # Small delay for consistency (eventual consistency of S3)
        time.sleep(0.3)

        # Verify we got the claim
        current_claim = self._get_claim(task_id)
        if current_claim and current_claim.get('machine_id') == self.machine_id:
            return True

        # Someone else got it
        return False

    def list_completed_tasks(self, start_id: int, end_id: int) -> set:
        """
        Batch query which tasks in [start_id, end_id) are completed.

        Uses S3 list_objects with StartAfter for efficient range queries.
        Task keys are zero-padded (task_00000123.json) for lexicographic ordering.

        Returns:
            Set of completed task_ids in the range
        """
        completed = set()
        prefix = f"{self.r2_prefix}/task_"

        # StartAfter: skip all keys before start_id (exclusive, so use start_id - 1)
        start_after = f"{self.r2_prefix}/task_{(start_id - 1):011d}.json" if start_id > 0 else None
        # We'll stop when we see task_id >= end_id

        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            paginate_args = {'Bucket': self.r2_bucket, 'Prefix': prefix}
            if start_after:
                paginate_args['StartAfter'] = start_after

            for page in paginator.paginate(**paginate_args):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.json'):
                        try:
                            # Extract task_id from "bugs/task_00000000123.json"
                            task_id_str = key[len(prefix):-5]
                            # Only match 11-digit zero-padded format (skip legacy keys like task_1.json)
                            if len(task_id_str) != 11:
                                continue
                            task_id = int(task_id_str)

                            if task_id >= end_id:
                                # Past our range, stop scanning
                                return completed
                            if task_id >= start_id:
                                completed.add(task_id)
                        except ValueError:
                            continue
        except ClientError:
            pass

        return completed

    def list_active_claims(self, start_id: int, end_id: int) -> Dict[int, Dict[str, Any]]:
        """
        Batch query active claims in [start_id, end_id).

        Uses StartAfter for efficient range queries.

        Returns:
            Dict mapping task_id to claim info (only non-expired claims by others)
        """
        claims = {}
        prefix = f"{self.r2_prefix}/claims/task_"
        now = time.time()

        start_after = f"{self.r2_prefix}/claims/task_{(start_id - 1):011d}.claim" if start_id > 0 else None

        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            paginate_args = {'Bucket': self.r2_bucket, 'Prefix': prefix}
            if start_after:
                paginate_args['StartAfter'] = start_after

            for page in paginator.paginate(**paginate_args):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.claim'):
                        try:
                            task_id_str = key[len(prefix):-6]
                            # Only match 11-digit zero-padded format
                            if len(task_id_str) != 11:
                                continue
                            task_id = int(task_id_str)

                            if task_id >= end_id:
                                return claims
                            if task_id >= start_id:
                                claim = self._get_claim(task_id)
                                if claim and claim.get('expires_at', 0) > now:
                                    if claim.get('machine_id') != self.machine_id:
                                        claims[task_id] = claim
                        except ValueError:
                            continue
        except ClientError:
            pass

        return claims

    def find_next_task(self, start_from: int, max_scan: int = 100) -> Optional[int]:
        """
        Find next available task_id to work on.

        Always starts from completed_up_to + 1 to ensure failed tasks get retried.
        Uses batch queries for efficiency:
        1. Get completed_up_to from metadata
        2. List all completed tasks in range
        3. List all active claims in range
        4. Find first gap (task that doesn't exist and has no claim)

        Args:
            start_from: Minimum task_id to scan from (used as lower bound)
            max_scan: Maximum number of task_ids to scan

        Returns:
            task_id to work on, or None if nothing available
        """
        # Always start from completed_up_to + 1 to catch failed tasks
        metadata = self.load_metadata()
        completed_up_to = metadata["tasks"].get("completed_up_to", -1)
        actual_start = max(start_from, completed_up_to + 1)
        end_id = actual_start + max_scan

        # Batch query completed tasks and active claims
        completed = self.list_completed_tasks(actual_start, end_id)
        active_claims = self.list_active_claims(actual_start, end_id)

        # Find first task that doesn't exist and has no active claim
        for task_id in range(actual_start, end_id):
            if task_id in completed:
                continue
            if task_id in active_claims:
                continue
            return task_id

        return None

    # ==================== Metadata Management ====================

    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata from R2, create if not exists."""
        try:
            response = self.s3.get_object(
                Bucket=self.r2_bucket,
                Key=self._get_metadata_key()
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return self._create_initial_metadata()
            raise

    def _create_initial_metadata(self) -> Dict[str, Any]:
        """Create initial metadata structure."""
        return {
            "version": 2,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tasks": {
                # All tasks in [0, completed_up_to] are guaranteed to exist
                # Use this to know "how many tasks are available"
                "completed_up_to": -1,
            }
        }

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to R2 (best effort, eventual consistency)."""
        metadata["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            self.s3.put_object(
                Bucket=self.r2_bucket,
                Key=self._get_metadata_key(),
                Body=json.dumps(metadata, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
        except ClientError:
            pass  # Best effort, will be corrected on next update

    def task_exists(self, task_id: int) -> bool:
        """Check if task already exists in R2."""
        try:
            self.s3.head_object(
                Bucket=self.r2_bucket,
                Key=self._get_task_key(task_id)
            )
            return True
        except ClientError:
            return False

    def save_task(self, task_id: int, data: Dict[str, Any]) -> bool:
        """Save task to R2. Returns True if successful."""
        key = self._get_task_key(task_id)
        try:
            self.s3.put_object(
                Bucket=self.r2_bucket,
                Key=key,
                Body=json.dumps(data, indent=2, default=str).encode('utf-8'),
                ContentType='application/json'
            )
            log(f"[Task {task_id}] ✓ Uploaded to R2: {key}")
            return True
        except Exception as e:
            log(f"[Task {task_id}] ✗ Upload failed: {e}")
            return False

    def decode_task_id(self, task_id: int) -> Dict[str, Any]:
        """
        Decode task_id into deterministic parameters.

        Encoding: task_id maps to (swe_instance, bug_types, seed)
        - swe_idx = task_id % num_swe_instances
        - seed = task_id
        - bug_types = randomly selected 1-3 types using seed
        """
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

    def _build_breaker_input(self, params: Dict[str, Any]) -> BreakerInput:
        """Build BreakerInput from decoded task parameters."""
        swe_inst = params["swe_instance"]
        instance_id = swe_inst["instance_id"]

        # Load scripts
        run_script = self._load_instance_script(instance_id, "run_script.sh")
        parser_script = self._load_instance_script(instance_id, "parser.py")

        if not run_script or not parser_script:
            raise RuntimeError(f"Missing run_script or parser_script for instance {instance_id}")

        # Prepare env commands
        try:
            base_dockerfile = self._load_dockerfile(instance_id, "base")
            instance_dockerfile = self._load_dockerfile(instance_id, "instance")
            env_cmds = self._extract_env_commands(base_dockerfile, instance_dockerfile)
        except Exception:
            env_cmds = ""

        before_repo_set_cmd = swe_inst.get("before_repo_set_cmd", "").strip()
        if before_repo_set_cmd:
            before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1]

        selected_test_files = swe_inst.get("selected_test_files_to_run", "[]")
        if isinstance(selected_test_files, str):
            try:
                selected_test_files = eval(selected_test_files)
            except:
                selected_test_files = []
        test_files_str = ",".join(selected_test_files) if selected_test_files else ""

        # Get all tests
        fail_to_pass = swe_inst.get("FAIL_TO_PASS", swe_inst.get("fail_to_pass", "[]"))
        pass_to_pass = swe_inst.get("PASS_TO_PASS", swe_inst.get("pass_to_pass", "[]"))
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
        all_tests = list(set(fail_to_pass) | set(pass_to_pass))

        # Get Docker image
        docker_image = get_dockerhub_image_uri(
            instance_id, self.dockerhub_username, swe_inst.get("repo", "")
        )

        return BreakerInput(
            docker_image=docker_image,
            base_commit=swe_inst.get("base_commit", ""),
            repo=swe_inst.get("repo", ""),
            instance_id=instance_id,
            gold_patch=swe_inst.get("patch", ""),
            test_cases=all_tests,
            test_runner_script=run_script,
            test_parser_script=parser_script,
            test_files=test_files_str,
            bug_types=params["bug_types"],
            seed=params["seed"],
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            test_patch=swe_inst.get("test_patch", ""),
            problem_statement_original=swe_inst.get("problem_statement", ""),
            env_cmds=env_cmds,
            before_repo_set_cmd=before_repo_set_cmd,
            temperature=0.7,
            max_iterations=50,
            cost_limit=5.0,
            timeout=300,
        )

    async def _try_generate_with_params(
        self,
        task_id: int,
        params: Dict[str, Any],
        max_retries: int,
        attempt_label: str = "",
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Try to generate a task with given parameters.

        Returns:
            (success, data) where data includes actual_source_task_id if fallback was used
        """
        swe_inst = params["swe_instance"]
        instance_id = swe_inst["instance_id"]
        bug_types = params['bug_types']

        log(f"[Task {task_id}]{attempt_label} Instance: {instance_id}")
        log(f"[Task {task_id}]{attempt_label} Bug types: {bug_types}")

        try:
            breaker_input = self._build_breaker_input(params)
        except RuntimeError as e:
            log(f"[Task {task_id}]{attempt_label} Skipped (no scripts): {e}")
            return False, None

        try:
            output = await run_breaker(
                breaker_input,
                max_retries=max_retries,
                agent_type=self.agent_type,
            )
            log(f"[Task {task_id}]{attempt_label} Generation successful")
            result_dict = output.to_dict()
            result_dict["_docker_image"] = breaker_input.docker_image
            return True, result_dict
        except BreakerException as e:
            log(f"[Task {task_id}]{attempt_label} Generation failed: {e}")
            return False, None

    async def generate_task(
        self,
        task_id: int,
        max_retries: int = 5,
        fallback_offset: int = 100000,
        max_fallback_attempts: int = 5,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Generate a single task with fallback to different instances.

        If the original instance fails, tries alternative instances by adding
        fallback_offset to the task_id (which maps to a different SWE-bench instance).
        The result is still saved under the original task_id.

        Returns:
            (success, data):
            - (True, data) if generated successfully
            - (True, None) if task already exists (skipped)
            - (False, None) if generation failed after all attempts
        """
        if self.task_exists(task_id):
            log(f"[Task {task_id}] Already exists, skipping")
            return True, None

        log(f"[Task {task_id}] Starting generation")

        # Try original task_id first
        params = self.decode_task_id(task_id)
        success, result = await self._try_generate_with_params(
            task_id, params, max_retries, attempt_label=""
        )

        if success and result is not None:
            result["source_task_id"] = task_id
            return True, result

        # Fallback: try alternative instances
        for fallback_idx in range(1, max_fallback_attempts + 1):
            alternative_task_id = task_id + fallback_offset * fallback_idx
            log(f"[Task {task_id}] Fallback {fallback_idx}: trying source task_id={alternative_task_id}")

            params = self.decode_task_id(alternative_task_id)
            success, result = await self._try_generate_with_params(
                task_id, params, max_retries, attempt_label=f" [fallback {fallback_idx}]"
            )

            if success and result is not None:
                result["source_task_id"] = alternative_task_id
                log(f"[Task {task_id}] Fallback succeeded with source task_id={alternative_task_id}")
                return True, result

        log(f"[Task {task_id}] All attempts failed (original + {max_fallback_attempts} fallbacks)")
        return False, None

    def _update_metadata(self, task_id: int, success: bool) -> None:
        """
        Update metadata (best effort, eventual consistency).

        Only updates on success. completed_up_to is recalculated from R2 state.
        """
        if not success:
            return

        try:
            metadata = self.load_metadata()

            # Recalculate completed_up_to from actual R2 state
            # This self-heals any inconsistencies
            self._update_completed_up_to(metadata, task_id)

            self.save_metadata(metadata)
        except Exception as e:
            print(f"Warning: metadata update failed: {e}")

    def _update_completed_up_to(self, metadata: Dict[str, Any], hint_task_id: int) -> None:
        """
        Update completed_up_to by scanning for the highest continuous range.

        completed_up_to = N means tasks [0, N] are all completed.

        Args:
            metadata: Metadata dict to update
            hint_task_id: Recently completed task_id, used as scan boundary hint
        """
        current = metadata["tasks"].get("completed_up_to", -1)

        # If we just completed current+1, we can extend without scanning
        if hint_task_id == current + 1:
            metadata["tasks"]["completed_up_to"] = hint_task_id
            return

        # Otherwise, scan from current+1 to find continuous range
        scan_start = current + 1
        scan_end = min(scan_start + 200, hint_task_id + 1)

        completed = self.list_completed_tasks(scan_start, scan_end)

        # Find highest continuous task_id
        new_completed_up_to = current
        for tid in range(scan_start, scan_end):
            if tid in completed:
                new_completed_up_to = tid
            else:
                break  # Gap found

        if new_completed_up_to > current:
            metadata["tasks"]["completed_up_to"] = new_completed_up_to

    async def run(
        self,
        start_from: Optional[int] = None,
        batch_size: int = 10,
        max_tasks: Optional[int] = None,
    ):
        """
        Main service loop with distributed locking.

        Each worker:
        1. Finds next available task (unclaimed or expired claim)
        2. Claims the task atomically
        3. Generates the task
        4. Saves result and releases claim
        5. Updates metadata

        Args:
            start_from: Override starting task_id (uses metadata if None)
            batch_size: Number of tasks per batch (for logging)
            max_tasks: Maximum tasks to generate (None = infinite)
        """
        metadata = self.load_metadata()

        # Determine starting point
        if start_from is not None:
            scan_from = start_from
        else:
            # Start from completed_up_to + 1 (skip known completed range)
            scan_from = metadata["tasks"].get("completed_up_to", -1) + 1

        tasks_processed = 0
        tasks_generated = 0
        tasks_failed = 0
        consecutive_no_work = 0

        log("=" * 50)
        log("Starting breaker service")
        log(f"Machine ID: {self.machine_id}")
        log(f"Scanning from task_id={scan_from}")
        log("=" * 50)

        while max_tasks is None or tasks_processed < max_tasks:
            # Find next available task
            task_id = self.find_next_task(scan_from, max_scan=100)

            if task_id is None:
                consecutive_no_work += 1
                if consecutive_no_work >= 3:
                    # No work available, advance scan window
                    scan_from += 100
                    log(f"No available tasks, advancing scan to {scan_from}")
                    consecutive_no_work = 0
                else:
                    log("No available tasks, waiting...")
                    await asyncio.sleep(5)
                continue

            consecutive_no_work = 0

            # Try to claim the task
            if not self.try_claim_task(task_id):
                log(f"[Task {task_id}] Claimed by another worker, skipping")
                continue

            # Check if already completed (claim returned True because task exists)
            if self.task_exists(task_id):
                # Already done, move on
                scan_from = max(scan_from, task_id + 1)
                continue

            # Generate the task
            current_docker_image = None
            try:
                success, result = await self.generate_task(task_id)

                if success and result is not None:
                    current_docker_image = result.pop("_docker_image", None)
                    upload_ok = self.save_task(task_id, result)
                    if upload_ok:
                        tasks_generated += 1
                        log(f"[Task {task_id}] ✓ Complete (total: {tasks_generated} generated, {tasks_failed} failed)")
                        # Only advance scan_from on successful upload
                        scan_from = max(scan_from, task_id + 1)
                    else:
                        tasks_failed += 1
                        log(f"[Task {task_id}] ✗ Upload failed, will retry (total: {tasks_generated} generated, {tasks_failed} failed)")
                elif not success:
                    tasks_failed += 1
                    log(f"[Task {task_id}] ✗ Generation failed, will retry (total: {tasks_generated} generated, {tasks_failed} failed)")

                tasks_processed += 1

                # Update metadata after each task
                self._update_metadata(task_id, success)

            finally:
                # Always release claim
                self._release_claim(task_id)
                # Clean up Docker resources after every task (success or failure)
                # Keep max 10 recent images to avoid disk space exhaustion
                cleanup_docker_resources(current_image=current_docker_image, max_images=10)

            # Progress logging
            if tasks_processed % batch_size == 0:
                log(f"Progress: {tasks_processed} processed, {tasks_generated} generated, {tasks_failed} failed")

        log("=" * 50)
        log(f"Completed: {tasks_processed} tasks processed, {tasks_generated} generated, {tasks_failed} failed")
        log("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Breaker Service - Distributed Task Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start single worker, generate 100 tasks
  python -m breaker.service --max-tasks 100

  # Start worker scanning from task 500
  python -m breaker.service --start-from 500

  # Run multiple workers in parallel (each finds unclaimed tasks)
  python -m breaker.service &
  python -m breaker.service &
  python -m breaker.service &
        """
    )
    parser.add_argument("--start-from", type=int, default=0,
                        help="Starting task_id to scan from (default: 0)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Tasks per progress log (default: 10)")
    parser.add_argument("--max-tasks", type=int,
                        help="Max tasks to generate (default: unlimited)")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE")
    parser.add_argument("--api-base", type=str, default="https://llm.chutes.ai/v1")
    parser.add_argument("--agent-type", type=str, default="miniswe",
                        choices=["miniswe", "ridge"])
    parser.add_argument("--dockerhub-username", type=str, default="jefzda")
    parser.add_argument("--run-scripts-dir", type=str, default="/app/run_scripts")
    parser.add_argument("--dockerfiles-dir", type=str, default="/app/dockerfiles")
    args = parser.parse_args()

    # Check required environment variables
    required_env = ["R2_ENDPOINT_URL", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
    missing = [e for e in required_env if not os.getenv(e)]
    if missing:
        print(f"Missing required environment variables: {missing}")
        sys.exit(1)

    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("Warning: CHUTES_API_KEY not set")

    service = BreakerService(
        r2_endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        r2_bucket=os.getenv("R2_BUCKET"),
        r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        agent_type=args.agent_type,
        dockerhub_username=args.dockerhub_username,
        run_scripts_dir=args.run_scripts_dir,
        dockerfiles_dir=args.dockerfiles_dir,
    )

    asyncio.run(service.run(
        start_from=args.start_from,
        batch_size=args.batch_size,
        max_tasks=args.max_tasks,
    ))


if __name__ == "__main__":
    main()
