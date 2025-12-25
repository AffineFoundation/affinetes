# /// script
# dependencies = [
#   "basilica-sdk>=0.10.0",
#   "affinetes",
# ]
# ///

"""
Basilica Evaluator - Deploy affinetes environment and run evaluation on Basilica

This script deploys the affinetes evaluation environment on Basilica GPU and
runs evaluations using a Basilica-deployed LLM service.

Both the LLM and the evaluation environment run on Basilica.

Usage:
    export BASILICA_API_TOKEN="your-token"

    # First deploy LLM (if not already running):
    python basilica_llm_server.py

    # Then run evaluation:
    python basilica_evaluator.py \
        --llm-url "https://your-llm.deployments.basilica.ai/v1" \
        --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
        --task-id-start 1 \
        --task-id-end 5
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import affinetes as af


async def evaluate_task(
    env,
    task_id: int,
    model_name: str,
    llm_url: str,
    timeout: int
) -> Dict[str, Any]:
    """Evaluate a single task."""
    try:
        print(f"[Task {task_id}] Starting evaluation...")
        start = time.time()

        result = await env.evaluate(
            model=model_name,
            base_url=llm_url,
            task_id=task_id,
            timeout=timeout,
            _timeout=timeout + 60,
            api_key="x",
        )

        elapsed = time.time() - start
        score = result.get("score", 0)
        print(f"[Task {task_id}] Completed in {elapsed:.2f}s - Score: {score}")

        return {
            "task_id": task_id,
            "result": result,
            "elapsed": elapsed
        }
    except Exception as e:
        print(f"[Task {task_id}] ERROR: {type(e).__name__}: {str(e)}")
        return {
            "task_id": task_id,
            "error": str(e),
            "error_type": type(e).__name__,
        }


async def run_evaluation(args: argparse.Namespace) -> tuple:
    """Deploy environment and run evaluation."""
    # Determine task IDs
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    else:
        task_ids = list(range(args.task_id_start, args.task_id_end + 1))

    print("=" * 60)
    print("BASILICA SDK - EVALUATION")
    print("=" * 60)
    print(f"LLM URL: {args.llm_url}")
    print(f"Model: {args.model_name}")
    print(f"Environment URL: {args.env_url}")
    print(f"Task IDs: {task_ids[:10]}{'...' if len(task_ids) > 10 else ''}")
    print(f"Total Tasks: {len(task_ids)}")
    print(f"Concurrency: {args.concurrent}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 60)
    print()

    # Connect to affinetes environment via URL
    print("Connecting to affinetes environment...")
    env = af.load_env(
        mode="url",
        base_url=args.env_url,
    )
    print("Environment connected successfully")
    print()

    # Run evaluations
    start_time = time.time()
    results = []

    try:
        if args.concurrent == 1:
            # Sequential evaluation
            for task_id in task_ids:
                result = await evaluate_task(
                    env=env,
                    task_id=task_id,
                    model_name=args.model_name,
                    llm_url=args.llm_url,
                    timeout=args.timeout,
                )
                results.append(result)
        else:
            # Concurrent evaluation in batches
            print(f"Running {len(task_ids)} evaluations with concurrency={args.concurrent}...")
            print()

            batches = [
                task_ids[i:i + args.concurrent]
                for i in range(0, len(task_ids), args.concurrent)
            ]

            for i, batch in enumerate(batches):
                print(f"Batch {i+1}/{len(batches)}: Processing tasks {batch}")
                tasks = [
                    evaluate_task(
                        env=env,
                        task_id=task_id,
                        model_name=args.model_name,
                        llm_url=args.llm_url,
                        timeout=args.timeout,
                    )
                    for task_id in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
    finally:
        # Cleanup environment connection
        await env.cleanup()

    total_time = time.time() - start_time
    return results, total_time, task_ids


def print_summary(results: List[Dict], total_time: float, task_ids: List[int]) -> Dict:
    """Print evaluation summary."""
    successful_tasks = [r for r in results if "result" in r]
    failed_tasks = [r for r in results if "error" in r]
    total_score = sum(r["result"].get("score", 0) for r in successful_tasks)
    avg_score = total_score / len(successful_tasks) if successful_tasks else 0

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Tasks: {len(task_ids)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    print(f"Average Score: {avg_score:.4f}")
    print(f"Total Score: {total_score:.2f}")
    print()

    print("TASK DETAILS:")
    print("-" * 60)
    for result in results:
        if "error" in result:
            print(f"Task {result['task_id']}: ERROR - {result['error_type']}: {result['error']}")
        else:
            score = result["result"].get("score", 0)
            elapsed = result["elapsed"]
            print(f"Task {result['task_id']}: SUCCESS - Score: {score:.4f}, Time: {elapsed:.2f}s")
    print("=" * 60)

    return {
        "total_time": total_time,
        "total_tasks": len(task_ids),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(failed_tasks),
        "average_score": avg_score,
        "total_score": total_score,
    }


def save_results(
    args: argparse.Namespace,
    results: List[Dict],
    summary: Dict,
    task_ids: List[int]
):
    """Save results to JSON file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"evaluation_results_{timestamp}.json")

    output_data = {
        "llm_url": args.llm_url,
        "model_name": args.model_name,
        "environment_url": args.env_url,
        "task_ids": task_ids,
        **summary,
        "results": results,
    }

    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print()
    print(f"Results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run affinetes evaluation with Basilica-deployed services"
    )
    parser.add_argument(
        "--llm-url",
        required=True,
        help="Basilica LLM deployment URL (e.g., https://xxx.deployments.basilica.ai/v1)"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name (e.g., Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--env-url",
        required=True,
        help="Basilica affinetes environment URL (e.g., https://xxx.deployments.basilica.ai)"
    )
    parser.add_argument(
        "--task-id-start",
        type=int,
        default=1,
        help="Start of task ID range (default: 1)"
    )
    parser.add_argument(
        "--task-id-end",
        type=int,
        default=5,
        help="End of task ID range (default: 5)"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        help="Comma-separated task IDs (e.g., 1,5,10,15,20)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per task in seconds (default: 1800)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent evaluations (default: 1)"
    )

    args = parser.parse_args()

    # Check API token
    api_token = os.environ.get("BASILICA_API_TOKEN")
    if not api_token:
        print("Error: BASILICA_API_TOKEN environment variable not set")
        print("Please set: export BASILICA_API_TOKEN='your-token'")
        sys.exit(1)

    results, total_time, task_ids = await run_evaluation(args)
    summary = print_summary(results, total_time, task_ids)
    save_results(args, results, summary, task_ids)


if __name__ == "__main__":
    asyncio.run(main())
