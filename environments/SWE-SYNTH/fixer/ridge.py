"""Ridge Fixer Agent - uses Docker sandbox with LLM proxy"""

import os
import sys
import subprocess
import tempfile
import time
from typing import Optional

from .base import BaseFixerAgent, FixerConfig, FixerResult
from . import config


class RidgeFixerAgent(BaseFixerAgent):
    """Fixer agent using Ridge's Docker sandbox with internal proxy"""

    def __init__(self, fixer_config: FixerConfig):
        super().__init__(fixer_config)
        self._temp_dir = None
        self._proxy_started = False

    def _get_ridges_module(self):
        """Import ridges_evaluate module"""
        ridge_project = config.get_ridge_project_path()
        if ridge_project not in sys.path:
            sys.path.insert(0, ridge_project)
        import ridges_evaluate
        return ridges_evaluate

    def _apply_patches_to_repo(
        self,
        repo_path: str,
        base_commit: Optional[str],
        gold_patch: Optional[str],
        bug_patch: Optional[str],
    ) -> bool:
        """Apply gold_patch and bug_patch to the repository"""
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
                    subprocess.run(
                        ["git", "apply", "-v", patch_path],
                        cwd=actual_repo, capture_output=True
                    )

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
                local_repo_path = self._extract_repo_from_docker(docker_image, temp_dir)
                if not local_repo_path:
                    return FixerResult(
                        patch="", success=False,
                        error="Failed to extract repository from Docker image"
                    )

            # Apply patches
            if gold_patch or bug_patch:
                self._apply_patches_to_repo(local_repo_path, base_commit, gold_patch, bug_patch)

            # Start proxy (use port 8001 to avoid conflict with affinetes server on 8000)
            proxy_port = 8001
            print(f"[RIDGE] Starting proxy container on port {proxy_port}...")
            proxy_result = ridges.run_proxy_container(
                openai_api_key=self.config.api_key,
                openai_model=self.config.model,
                openai_base_url=self.config.api_base,
                temperature=self.config.temperature,
                seed=self.config.seed,
                port=proxy_port,
                container_name="ridge-proxy"
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
                sandbox_proxy_url=f"http://host.docker.internal:{proxy_port}",
                timeout=self.config.timeout,
            )

            if result.get("success"):
                return FixerResult(
                    patch=result.get("output", ""),
                    model_calls=result.get("model_calls", 0),
                    model_cost=result.get("model_cost", 0.0),
                    total_tokens=result.get("total_tokens", 0),
                    conversation=result.get("conversation", []),
                    success=True,
                )
            else:
                return FixerResult(
                    patch="", success=False,
                    error=result.get("error", "Unknown error")
                )

        except Exception as e:
            import traceback
            return FixerResult(patch="", success=False, error=traceback.format_exc())

        finally:
            self.cleanup()

    def _extract_repo_from_docker(self, docker_image: str, temp_dir: str) -> Optional[str]:
        """Extract repository from SWE-bench Docker image"""
        try:
            container_name = f"ridge-extract-{int(time.time() * 1000)}"
            local_repo_path = os.path.join(temp_dir, "repo")

            subprocess.run(
                ["docker", "create", "--name", container_name, docker_image, "true"],
                capture_output=True, check=True
            )

            result = subprocess.run(
                ["docker", "cp", f"{container_name}:/app", local_repo_path],
                capture_output=True, text=True
            )

            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

            if result.returncode != 0:
                return None

            return local_repo_path

        except Exception:
            return None

    def cleanup(self):
        """Clean up proxy container and temp directory"""
        if self._proxy_started:
            try:
                ridges = self._get_ridges_module()
                ridges.stop_proxy_container("ridge-proxy")
                print("[RIDGE] Proxy container stopped")
            except Exception:
                pass
            self._proxy_started = False

        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except (PermissionError, OSError):
                pass
            self._temp_dir = None
