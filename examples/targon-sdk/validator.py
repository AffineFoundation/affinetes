import json
import time
import targon
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

AFFINE_DIR = Path(__file__).resolve().parent
REQUIREMENTS = AFFINE_DIR / "requirements.txt"

image = (
    targon.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install_from_requirements(str(REQUIREMENTS))
    .run_commands(
        [
            # Install Docker (for DIND) using official convenience script
            "curl -fsSL https://get.docker.com -o get-docker.sh",
            "sh get-docker.sh",
            # Install affinetes from GitHub
            "pip install git+https://github.com/affinefoundation/affinetes.git",
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = targon.App("affine", image=image)


@app.function(
    resource=targon.Compute.H200_MEDIUM, timeout=3600, min_replicas=1, max_replicas=1
)
@targon.concurrent(max_concurrency=1, target_concurrency=1)
def run(
    model_name: str,
    task_ids: list[int],
    *,
    image: str = "docker.io/affinefoundation/mth:pi",
    sglang_port: int = 30000,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """
    Run Math reasoning evaluations using affinetes environment

    Args:
        model_name: Model name for evaluation
        task_ids: List of task IDs to evaluate
        image: Affinetes environment image to use
        sglang_port: Port for SGLang service
        timeout: Timeout per task in seconds
    """
    import subprocess
    import asyncio
    import affinetes as af

    # Start Docker daemon (DIND)
    print("Starting Docker daemon...")
    docker_daemon = subprocess.Popen(
        ["dockerd", "--host=unix:///var/run/docker.sock"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Docker to be ready
    time.sleep(10)

    # Start SGLang service
    print(f"Starting SGLang service on port {sglang_port}...")
    sglang_process = subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_name,
            "--port",
            str(sglang_port),
            "--host",
            "0.0.0.0",
            "--chat-template",
            "llama-2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print("Waiting for LLM service to be ready...")
    time.sleep(60)

    try:
        # Construct base URL for SGLang service
        llm_base_url = f"http://localhost:{sglang_port}/v1"
        print(f"base URL: {llm_base_url}")

        # Load affinetes environment with host network mode
        # This allows the container to access sglang service on host network
        # Use custom port to avoid conflicts (default is 8000)
        print(f"Loading environment from affinetes...")

        async def run_evaluation():
            env = af.load_env(
                image=image,
                host_network=True,
            )

            async def evaluate_task(task_id: int):
                print(f"[Task {task_id}] Starting evaluation...")
                start = time.time()

                try:
                    result = await env.evaluate(
                        model=model_name,
                        base_url=llm_base_url,
                        task_id=task_id,
                        timeout=timeout,
                        _timeout=timeout + 60,
                    )

                    elapsed = time.time() - start
                    print(
                        f"[Task {task_id}] Completed in {elapsed:.2f}s - Score: {result.get('score', 0)}"
                    )

                    return {"task_id": task_id, "result": result, "elapsed": elapsed}
                except Exception as e:
                    print(f"[Task {task_id}] ERROR: {type(e).__name__}: {str(e)}")
                    return {
                        "task_id": task_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

            results = await asyncio.gather(
                *[evaluate_task(task_id) for task_id in task_ids]
            )
            return list(results)

        # Run evaluation
        results = asyncio.run(run_evaluation())

        # Calculate summary statistics
        successful_tasks = [r for r in results if "result" in r]
        total_score = sum(r["result"].get("score", 0) for r in successful_tasks)
        avg_score = total_score / len(successful_tasks) if successful_tasks else 0

        return {
            "model_name": model_name,
            "total_tasks": len(task_ids),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(task_ids) - len(successful_tasks),
            "average_score": avg_score,
            "total_score": total_score,
            "results": results,
        }

    finally:
        # Stop SGLang service
        print("Stopping SGLang service...")
        sglang_process.terminate()
        try:
            sglang_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sglang_process.kill()

        # Stop Docker daemon
        print("Stopping Docker daemon...")
        docker_daemon.terminate()
        docker_daemon.wait(timeout=10)


@app.local_entrypoint()
async def main(
    model_name: str,
    task_ids: str = "1,2",
    image: str = "docker.io/affinefoundation/mth:pi",
    sglang_port: int = 30000,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """
    Run the validator remotely
    
    Usage:
        targon run examples/targon-sdk/validator.py \
            --model-name "Qwen/Qwen2.5-7B-Instruct" \
            --task-ids "1,2,3,4,5,6,7,8,9,10" \
            --image "docker.io/affinefoundation/mth:pi"
    
    Args:
        model_name: Model name for evaluation
        task_ids: Comma-separated task IDs (e.g., "1,2,3,4,5,6,7,8,9,10")
        image: Affinetes environment image to use
        sglang_port: Port for SGLang service (default: 30000)
        timeout: Timeout per task in seconds
    """
    # Parse task IDs
    task_id_list = [int(x.strip()) for x in task_ids.split(",")]

    print(f"Starting evaluation:")
    print(f"  Model: {model_name}")
    print(f"  Image: {image}")
    print(f"  SGLang Port: {sglang_port}")
    print(f"  Task IDs: {task_id_list}")
    print(f"  Timeout: {timeout}s")
    print()

    result = await run.remote(
        model_name=model_name,
        task_ids=task_id_list,
        image=image,
        sglang_port=sglang_port,
        timeout=timeout,
    )

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"rollouts_{timestamp}.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print summary
    summary = {
        "model_name": result["model_name"],
        "total_tasks": result["total_tasks"],
        "successful_tasks": result["successful_tasks"],
        "failed_tasks": result["failed_tasks"],
        "average_score": result["average_score"],
        "total_score": result["total_score"],
        "output_path": str(output_path),
    }

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    return result
