"""SWE-FRONTIER Environment

Evaluates coding agents on RFC-style benchmark tasks produced by the
InfiniteFrontierSWE pipeline. Each task is a Docker image published to
Docker Hub under `affinefoundation/swe_frontier_private:rfc-XXXX-...`. The
image bakes the entire task (BRIEF.md + verify/run.sh + verify/score.py +
verify/vectors.ndjson) under /task and uses verify_entry.sh as the
entrypoint, branching on IFS_STAGE.

Flow:
  evaluate(task_id) ->
    1. resolve task_id -> image_tag (currently a hardcoded mapping for testing)
    2. spawn solving container as root, hide /task/verify, copy afentctl into it,
       run afentctl to write a solution under /workspace
    3. tar /workspace out to a host temp dir
    4. run a fresh verify container with IFS_STAGE=solver (mounts workspace ro)
    5. run another fresh verify container with IFS_STAGE=score (no workspace)
    6. parse /out/score.json -> UnifiedScore -> return
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from agents import AfentAgent, AfentConfig

DOCKER_PULL_TIMEOUT = 300
VERIFY_TIMEOUT = 1800

# task_id -> dockerhub image tag.
# First version is hardcoded; once the catalog API on R2 is wired up this
# becomes a dynamic lookup.
TASK_ID_TO_IMAGE: Dict[int, str] = {
    1: "affinefoundation/swe_frontier_private:rfc-3550-python-0.1.0",
}


class FrontierActor:
    """SWE-FRONTIER evaluation actor."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self._setup_docker_auth()
        self._cleanup_stale_containers()

    # ===== Infrastructure =====================================================

    def _setup_docker_auth(self) -> None:
        username = os.getenv("DOCKER_HUB_USERNAME")
        token = os.getenv("DOCKER_HUB_TOKEN")
        if not username or not token:
            print("[SWE-FRONTIER] DOCKER_HUB_USERNAME/TOKEN not set, skipping login")
            return
        result = subprocess.run(
            ["docker", "login", "-u", username, "--password-stdin"],
            input=token, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"[SWE-FRONTIER] Docker Hub login succeeded for {username}")
        else:
            print(f"[SWE-FRONTIER] Warning: docker login failed: {result.stderr.strip()}")

    def _cleanup_stale_containers(self) -> None:
        for prefix in ("swe-frontier-afent-", "swe-frontier-verify-"):
            try:
                ps = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name={prefix}",
                     "--format", "{{.ID}}"],
                    capture_output=True, text=True, timeout=30,
                )
                cids = [c for c in ps.stdout.split() if c]
                if cids:
                    subprocess.run(
                        ["docker", "rm", "-f", *cids],
                        capture_output=True, timeout=60,
                    )
            except Exception as e:
                print(f"[SWE-FRONTIER] cleanup '{prefix}*' failed: {e}")

    def _resolve_image(self, task_id) -> str:
        # task_id may arrive as int or str (CLI typically passes str).
        if isinstance(task_id, str):
            try:
                task_id = int(task_id)
            except ValueError:
                pass
        if task_id in TASK_ID_TO_IMAGE:
            return TASK_ID_TO_IMAGE[task_id]
        # Allow passing a tag string directly to avoid editing the map for
        # ad-hoc testing.
        if isinstance(task_id, str) and ":" in task_id:
            return task_id
        raise ValueError(
            f"Unknown task_id={task_id!r}. Known ids: {sorted(TASK_ID_TO_IMAGE)}"
        )

    # ===== Verification =======================================================

    def _run_verify_stage(
        self,
        *,
        stage: str,
        image: str,
        workspace: Optional[Path],
        out_dir: Path,
    ) -> subprocess.CompletedProcess:
        """Run verify_entry.sh once for the given IFS_STAGE."""
        ctr = f"swe-frontier-verify-{stage}-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "--rm", "--init",
            "--name", ctr,
            "--user", "0:0",
            "--memory", "4g",
            # Verify must be sealed — the IFS contract for vendor'd images is
            # that everything required is baked in. This also prevents a
            # solver-planted oracle.sh from phoning home during grading.
            "--network", "none",
            "-e", f"IFS_STAGE={stage}",
            "-v", f"{out_dir}:/out",
        ]
        if workspace is not None:
            cmd += ["-v", f"{workspace}:/workspace:ro"]
        cmd.append(image)
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT,
        )

    def _verify(
        self, image: str, workspace: Path,
    ) -> tuple[float, Dict[str, Any]]:
        """Two-stage verify, returns (overall_score, parsed_score_payload)."""
        out_dir = Path(tempfile.mkdtemp(prefix="swe-frontier-out-"))
        try:
            print(f"[SWE-FRONTIER] Verify stage 1 (solver) image={image}")
            r1 = self._run_verify_stage(
                stage="solver", image=image, workspace=workspace, out_dir=out_dir,
            )
            if r1.returncode != 0:
                return 0.0, {
                    "error": "solver_stage_failed",
                    "exit_code": r1.returncode,
                    "stdout_tail": (r1.stdout + r1.stderr)[-1500:],
                }

            print(f"[SWE-FRONTIER] Verify stage 2 (score) image={image}")
            r2 = self._run_verify_stage(
                stage="score", image=image, workspace=None, out_dir=out_dir,
            )
            if r2.returncode != 0:
                return 0.0, {
                    "error": "score_stage_failed",
                    "exit_code": r2.returncode,
                    "stdout_tail": (r2.stdout + r2.stderr)[-1500:],
                }

            score_path = out_dir / "score.json"
            if not score_path.is_file():
                return 0.0, {"error": "no_score_json"}
            try:
                score_doc = json.loads(score_path.read_text())
            except Exception as e:
                return 0.0, {"error": f"bad_score_json: {e}"}

            overall_score = float(score_doc.get("overall_score", 0.0))
            return overall_score, {
                "overall_score": overall_score,
                "overall_pass": bool(score_doc.get("overall_pass", False)),
                "milestones": score_doc.get("milestones", {}),
                "raw": score_doc,
            }
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    # ===== One-shot Evaluation Interface =====================================

    async def evaluate(
        self,
        task_id,
        model: str = "affine/Kimi-K2.5",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        """Evaluate an afent agent on a SWE-FRONTIER task.

        Args:
            task_id: int id (lookup in TASK_ID_TO_IMAGE) or a full image tag.
            model: model name passed to afentctl.
            base_url: OpenAI-compatible endpoint.
            api_key: API key; falls back to env CHUTES_API_KEY.
            timeout: afent wall-time budget (seconds).
        """
        start = time.time()
        eval_api_key = api_key or self.api_key
        if not eval_api_key:
            raise ValueError(
                "api_key required (pass to evaluate() or set CHUTES_API_KEY env var)"
            )
        image_tag = self._resolve_image(task_id)
        print(f"[SWE-FRONTIER] Loaded task: id={task_id} image={image_tag}")

        agent = AfentAgent(AfentConfig(
            model=model, api_base=base_url, api_key=eval_api_key, timeout=timeout,
        ))
        agent_result = agent.solve(image_tag)

        if not agent_result.success or agent_result.workspace_dir is None:
            return {
                "task_name": "swe-frontier",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "extra": {
                    "task_id": task_id,
                    "image": image_tag,
                    "agent_type": "afent",
                    "model_calls": agent_result.model_calls,
                    "total_tokens": agent_result.total_tokens,
                    "conversation": agent_result.conversation,
                    "error": agent_result.error,
                },
            }

        workspace = agent_result.workspace_dir
        try:
            score, score_payload = self._verify(image_tag, workspace)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        return {
            "task_name": "swe-frontier",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "task_id": task_id,
                "image": image_tag,
                "agent_type": "afent",
                "model_calls": agent_result.model_calls,
                "total_tokens": agent_result.total_tokens,
                "conversation": agent_result.conversation,
                "score_payload": score_payload,
                "agent_warning": agent_result.error,
            },
        }


# Framework requires class named 'Actor'.
Actor = FrontierActor
