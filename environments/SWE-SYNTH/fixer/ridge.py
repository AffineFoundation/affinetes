"""Ridge Fixer Agent - uses Docker sandbox with LLM proxy"""

import os
import sys
import subprocess
import tempfile
import time
from typing import Optional

import requests

from .base import BaseFixerAgent, FixerConfig, FixerResult
from . import config


class RidgeFixerAgent(BaseFixerAgent):
    """Fixer agent using Ridge's Docker sandbox with internal proxy"""

    def __init__(self, fixer_config: FixerConfig):
        super().__init__(fixer_config)
        self._temp_dir = None
        self._proxy_started = False
        self._proxy_container_name = None
        self._proxy_port = None

    def _get_ridges_module(self):
        """Import ridges_evaluate module"""
        ridge_project = config.get_ridge_project_path()
        if ridge_project not in sys.path:
            sys.path.insert(0, ridge_project)
        import ridges_evaluate
        return ridges_evaluate

    def _sanitize_git_history(self, repo_path: str) -> bool:
        """Remove git history to prevent cheating by looking at past commits.

        This creates an orphan commit with only the current working tree state,
        effectively removing all history that could reveal the fix.

        Returns:
            True if sanitization succeeded, False otherwise.
        """
        try:
            actual_repo = repo_path
            if os.path.exists(os.path.join(repo_path, "app")):
                actual_repo = os.path.join(repo_path, "app")

            # Save current state
            subprocess.run(["git", "add", "-A"], cwd=actual_repo, capture_output=True)

            # Create orphan branch (no parent commits)
            subprocess.run(
                ["git", "checkout", "--orphan", "sanitized_branch"],
                cwd=actual_repo, capture_output=True
            )

            # Commit current state as the only commit
            subprocess.run(
                ["git", "commit", "-m", "Initial state", "--allow-empty"],
                cwd=actual_repo, capture_output=True
            )

            # Delete old branches
            subprocess.run(["git", "branch", "-D", "main"], cwd=actual_repo, capture_output=True)
            subprocess.run(["git", "branch", "-D", "master"], cwd=actual_repo, capture_output=True)

            # Rename to main
            subprocess.run(["git", "branch", "-m", "main"], cwd=actual_repo, capture_output=True)

            # Clean up reflog and history traces
            import shutil
            logs_path = os.path.join(actual_repo, ".git", "logs")
            if os.path.exists(logs_path):
                shutil.rmtree(logs_path, ignore_errors=True)

            refs_orig_path = os.path.join(actual_repo, ".git", "refs", "original")
            if os.path.exists(refs_orig_path):
                shutil.rmtree(refs_orig_path, ignore_errors=True)

            subprocess.run(
                ["git", "reflog", "expire", "--expire=now", "--all"],
                cwd=actual_repo, capture_output=True
            )
            subprocess.run(
                ["git", "gc", "--prune=now"],
                cwd=actual_repo, capture_output=True
            )

            print("[RIDGE] Git history sanitized")
            return True
        except Exception as e:
            print(f"[RIDGE] Warning: Failed to sanitize git history: {e}")
            return False

    def _apply_patches_to_repo(
        self,
        repo_path: str,
        base_commit: Optional[str],
        gold_patch: Optional[str],
        bug_patch: Optional[str],
    ) -> bool:
        """Apply gold_patch and bug_patch to the repository.

        Returns:
            True if patches applied successfully, False otherwise.
        """
        try:
            actual_repo = repo_path
            if os.path.exists(os.path.join(repo_path, "app")):
                actual_repo = os.path.join(repo_path, "app")

            if base_commit:
                subprocess.run(
                    ["git", "reset", "--hard", base_commit],
                    cwd=actual_repo, capture_output=True, check=True
                )
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=actual_repo, capture_output=True, check=True
                )

            for name, patch in [("gold", gold_patch), ("bug", bug_patch)]:
                if patch:
                    patch_path = os.path.join(repo_path, f"{name}_patch.diff")
                    with open(patch_path, "w") as f:
                        f.write(patch)
                    # Apply patch and check result
                    result = subprocess.run(
                        ["git", "apply", "-v", patch_path],
                        cwd=actual_repo, capture_output=True, text=True
                    )
                    # Check for errors
                    if result.returncode != 0:
                        print(f"[RIDGE] Warning: {name}_patch may have failed to apply:")
                        print(f"[RIDGE]   stdout: {result.stdout[:300] if result.stdout else 'empty'}")
                        print(f"[RIDGE]   stderr: {result.stderr[:300] if result.stderr else 'empty'}")
                    else:
                        print(f"[RIDGE] {name}_patch applied successfully")

            # Sanitize git history to prevent cheating via git log/show
            self._sanitize_git_history(repo_path)

            # Verify code state after patches
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=actual_repo, capture_output=True, text=True
            )
            diff_result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=actual_repo, capture_output=True, text=True
            )
            print(f"[RIDGE] Post-patch git status: {status_result.stdout[:200] if status_result.stdout else 'clean'}")
            print(f"[RIDGE] Post-patch git diff stat: {diff_result.stdout[:200] if diff_result.stdout else 'no changes'}")

            return True
        except Exception as e:
            print(f"[RIDGE] Error applying patches: {e}")
            return False

    async def fix(
        self,
        problem_statement: str,
        docker_image: str,
        repo_path: Optional[str] = None,
        gold_patch: Optional[str] = None,
        bug_patch: Optional[str] = None,
        base_commit: Optional[str] = None,
    ) -> FixerResult:
        """Run Ridge agent to fix the bug"""
        ridges = None
        try:
            ridges = self._get_ridges_module()
            agent_path = self.config.get_ridge_agent_path()

            if not os.path.exists(agent_path):
                return FixerResult(
                    patch="", success=False,
                    error=f"Ridge agent not found: {agent_path}"
                )

            self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            temp_dir = self._temp_dir.name

            # Extract repo or use provided path
            if repo_path and os.path.exists(repo_path):
                local_repo_path = repo_path
            else:
                local_repo_path, extract_error = self._extract_repo_from_docker(docker_image, temp_dir)
                if not local_repo_path:
                    return FixerResult(
                        patch="", success=False,
                        error=f"Failed to extract repository from Docker image: {extract_error}"
                    )

            # Apply patches
            if gold_patch or bug_patch:
                self._apply_patches_to_repo(local_repo_path, base_commit, gold_patch, bug_patch)

            # Start proxy with unique port and container name for concurrency
            import random
            self._proxy_port = random.randint(9000, 9999)
            self._proxy_container_name = f"ridge-proxy-{os.urandom(4).hex()}"
            print(f"[RIDGE] Starting proxy container {self._proxy_container_name} on port {self._proxy_port}...")
            proxy_result = ridges.run_proxy_container(
                openai_api_key=self.config.api_key,
                openai_model=self.config.model,
                openai_base_url=self.config.api_base,
                temperature=self.config.temperature,
                seed=self.config.seed,
                port=self._proxy_port,
                container_name=self._proxy_container_name
            )

            if not proxy_result.get("success"):
                return FixerResult(
                    patch="", success=False,
                    error=f"Failed to start proxy: {proxy_result.get('error')}"
                )

            self._proxy_started = True

            # Run sandbox
            print("[RIDGE] Running agent sandbox...")
            result = ridges.run_ridges_sandbox(
                repo_path=local_repo_path,
                agent_path=agent_path,
                problem_statement=problem_statement,
                sandbox_proxy_url=f"http://host.docker.internal:{self._proxy_port}",
                timeout=self.config.timeout,
                actual_model=self.config.model,
            )

            # Fetch conversation and usage from proxy
            conversation = []
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            model_calls = 0
            try:
                proxy_url = f"http://localhost:{self._proxy_port}/api/usage"
                resp = requests.get(proxy_url, timeout=10)
                if resp.status_code == 200:
                    proxy_data = resp.json()
                    conversation = proxy_data.get("conversation", [])
                    usage = proxy_data.get("usage", usage)
                    model_calls = usage.get("total_requests", 0)
                    print(f"[RIDGE] Got {len(conversation)} conversation turns, {usage.get('total_tokens', 0)} tokens")
            except Exception as e:
                print(f"[RIDGE] Failed to fetch usage from proxy: {e}")

            if result.get("success"):
                return FixerResult(
                    patch=result.get("output", ""),
                    model_calls=model_calls,
                    model_cost=result.get("model_cost", 0.0),
                    total_tokens=usage.get("total_tokens", 0),
                    conversation=conversation,
                    success=True,
                )
            else:
                return FixerResult(
                    patch="",
                    model_calls=model_calls,
                    total_tokens=usage.get("total_tokens", 0),
                    conversation=conversation,
                    success=False,
                    error=result.get("error", "Unknown error")
                )

        except Exception as e:
            import traceback
            return FixerResult(patch="", success=False, error=traceback.format_exc())

        finally:
            self.cleanup()

    def _extract_repo_from_docker(self, docker_image: str, temp_dir: str) -> tuple[Optional[str], Optional[str]]:
        """Extract repository from SWE-bench Docker image

        Returns:
            (repo_path, None) on success, (None, error_message) on failure
        """
        container_name = f"ridge-extract-{os.urandom(8).hex()}"
        local_repo_path = os.path.join(temp_dir, "repo")

        try:
            # Pre-cleanup in case of leftover container
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

            create_result = subprocess.run(
                ["docker", "create", "--name", container_name, docker_image, "true"],
                capture_output=True, text=True
            )
            if create_result.returncode != 0:
                return None, f"docker create failed: {create_result.stderr.strip()}"

            cp_result = subprocess.run(
                ["docker", "cp", f"{container_name}:/app", local_repo_path],
                capture_output=True, text=True
            )

            if cp_result.returncode != 0:
                return None, f"docker cp failed: {cp_result.stderr.strip()}"

            return local_repo_path, None

        except Exception as e:
            return None, f"extract exception: {e}"

        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    def cleanup(self):
        """Clean up proxy container and temp directory"""
        if self._proxy_started and self._proxy_container_name:
            try:
                ridges = self._get_ridges_module()
                ridges.stop_proxy_container(self._proxy_container_name)
                print(f"[RIDGE] Proxy container {self._proxy_container_name} stopped")
            except Exception:
                pass
            self._proxy_started = False
            self._proxy_container_name = None
            self._proxy_port = None

        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except (PermissionError, OSError):
                pass
            self._temp_dir = None
