#!/usr/bin/env python3
"""Run affinetes environment on Basilica with parallel task execution.

Usage:
    export BASILICA_API_TOKEN="your-token"

    # Run a single task
    python run_on_basilica.py --task-id 1

    # Run tasks 1-10 with 4 concurrent
    python run_on_basilica.py --task-id-start 1 --task-id-end 10 --concurrent 4

    # Run specific tasks
    python run_on_basilica.py --task-ids "1,5,10,15,20" --concurrent 5
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

import affinetes as af_env


def log(msg: str, level: str = "info"):
    """Print a formatted log message."""
    ts = datetime.now().strftime("%H:%M:%S")

    if level == "info":
        prefix = f"\033[90m{ts}\033[0m \033[36m▸\033[0m"
    elif level == "success":
        prefix = f"\033[90m{ts}\033[0m \033[32m✓\033[0m"
    elif level == "error":
        prefix = f"\033[90m{ts}\033[0m \033[31m✗\033[0m"
    elif level == "warn":
        prefix = f"\033[90m{ts}\033[0m \033[33m⚠\033[0m"
    elif level == "start":
        prefix = f"\033[90m{ts}\033[0m \033[33m→\033[0m"
    else:
        prefix = f"\033[90m{ts}\033[0m  "

    print(f"{prefix} {msg}")


def log_header(title: str):
    """Print a section header."""
    print(f"\n\033[1m{'─' * 60}\033[0m")
    print(f"\033[1m{title}\033[0m")
    print(f"\033[1m{'─' * 60}\033[0m\n")


def log_config(key: str, value: str):
    """Print a config key-value pair."""
    print(f"  \033[36m{key:>12}\033[0m  {value}")


async def evaluate_task(
    env,
    task_id: int,
    model: str,
    base_url: str,
    timeout: int,
    api_key: str,
) -> Dict[str, Any]:
    """Evaluate a single task."""
    log(f"Task {task_id} started", "start")
    start = time.time()

    try:
        result = await env.evaluate(
            model=model,
            base_url=base_url,
            task_id=task_id,
            timeout=timeout,
            _timeout=timeout + 60,
            api_key=api_key,
        )

        elapsed = time.time() - start
        score = result.get("score", 0)
        log(f"Task {task_id} completed \033[90m→\033[0m score: \033[1m{score:.4f}\033[0m \033[90m({elapsed:.1f}s)\033[0m", "success")

        return {
            "task_id": task_id,
            "result": result,
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start
        error_msg = str(e)[:80]
        log(f"Task {task_id} failed: {error_msg} \033[90m({elapsed:.1f}s)\033[0m", "error")
        return {
            "task_id": task_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "elapsed": elapsed,
        }


async def run_evaluation(args: argparse.Namespace) -> tuple:
    """Deploy environment on Basilica and run parallel evaluations."""

    # Determine task IDs
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    elif args.task_id_start is not None and args.task_id_end is not None:
        task_ids = list(range(args.task_id_start, args.task_id_end + 1))
    else:
        task_ids = [args.task_id]

    # Build env_vars
    env_vars = {}
    if args.chutes_api_key:
        env_vars["CHUTES_API_KEY"] = args.chutes_api_key

    # Show config
    log_header("Configuration")
    log_config("Image", args.image)
    log_config("Model", args.model)
    log_config("Tasks", f"{len(task_ids)} tasks")
    log_config("Concurrency", str(args.concurrent))
    log_config("Timeout", f"{args.timeout}s")
    log_config("CPU", args.cpu)
    log_config("Memory", args.memory)

    # Load environment
    log_header("Execution")
    log("Loading environment on Basilica...")

    env = af_env.load_env(
        mode="basilica",
        image=args.image,
        cpu_limit=args.cpu,
        mem_limit=args.memory,
        env_vars=env_vars if env_vars else None,
        ttl_buffer=args.ttl_buffer,
    )

    log("Environment ready", "success")
    print()

    # Run evaluations
    start_time = time.time()
    results = []

    try:
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(args.concurrent)

        async def run_with_semaphore(task_id: int) -> Dict[str, Any]:
            async with semaphore:
                return await evaluate_task(
                    env=env,
                    task_id=task_id,
                    model=args.model,
                    base_url=args.base_url,
                    timeout=args.timeout,
                    api_key=args.chutes_api_key,
                )

        # Launch all tasks and collect results as they complete
        tasks = [run_with_semaphore(tid) for tid in task_ids]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)

    finally:
        print()
        log("Cleaning up environment...")
        await env.cleanup()
        log("Cleanup complete", "success")

    total_time = time.time() - start_time
    return results, total_time, task_ids


def print_summary(results: List[Dict], total_time: float, task_ids: List[int]) -> Dict:
    """Print evaluation summary."""
    successful_tasks = [r for r in results if "result" in r]
    failed_tasks = [r for r in results if "error" in r]
    total_score = sum(r["result"].get("score", 0) for r in successful_tasks)
    avg_score = total_score / len(successful_tasks) if successful_tasks else 0

    log_header("Summary")

    print(f"  \033[90mTotal Time\033[0m     {total_time:.2f}s")
    print(f"  \033[90mTasks\033[0m          {len(task_ids)} total, \033[32m{len(successful_tasks)} success\033[0m, \033[31m{len(failed_tasks)} failed\033[0m")
    print(f"  \033[90mAverage Score\033[0m  {avg_score:.4f}")
    print(f"  \033[90mTotal Score\033[0m    \033[1m{total_score:.2f}\033[0m")

    # Show errors if any
    if failed_tasks:
        print(f"\n  \033[31mErrors:\033[0m")
        for r in failed_tasks:
            print(f"    Task {r['task_id']}: {r['error_type']}: {r['error'][:60]}...")

    print()

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
    task_ids: List[int],
):
    """Save results to JSON file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"evaluation_results_{timestamp}.json")

    output_data = {
        "image": args.image,
        "model": args.model,
        "base_url": args.base_url,
        "task_ids": task_ids,
        **summary,
        "results": results,
    }

    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    log(f"Results saved to {output_path}", "info")


def main():
    parser = argparse.ArgumentParser(
        description="Run affinetes environment on Basilica with parallel tasks"
    )

    # Task selection (mutually exclusive options)
    task_group = parser.add_argument_group("task selection")
    task_group.add_argument(
        "--task-id",
        type=int,
        default=1,
        help="Single task ID to evaluate (default: 1)"
    )
    task_group.add_argument(
        "--task-id-start",
        type=int,
        help="Start of task ID range (inclusive)"
    )
    task_group.add_argument(
        "--task-id-end",
        type=int,
        help="End of task ID range (inclusive)"
    )
    task_group.add_argument(
        "--task-ids",
        type=str,
        help="Comma-separated task IDs (e.g., 1,5,10,15,20)"
    )

    # Concurrency
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1000,
        help="Number of concurrent evaluations (default: 1000)"
    )

    # Environment config
    parser.add_argument(
        "--image",
        default="epappas/mth:pi-fixed",
        help="Docker image to deploy (default: epappas/mth:pi-fixed)"
    )
    parser.add_argument(
        "--cpu",
        default="2000m",
        help="CPU limit (default: 2000m)"
    )
    parser.add_argument(
        "--memory",
        default="8Gi",
        help="Memory limit (default: 8Gi)"
    )
    parser.add_argument(
        "--ttl-buffer",
        type=int,
        default=60 * 10,
        help="Pod TTL buffer in seconds (default: 300)"
    )

    # Model config
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-32B",
        help="Model name for evaluation (default: Qwen/Qwen3-32B)"
    )
    parser.add_argument(
        "--base-url",
        default="https://llm.chutes.ai/v1",
        help="LLM API base URL (default: https://llm.chutes.ai/v1)"
    )

    # Evaluation config
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Evaluation timeout in seconds (default: 1800)"
    )
    parser.add_argument(
        "--chutes-api-key",
        default=os.environ.get("CHUTES_API_KEY"),
        help="Chutes API key (or set CHUTES_API_KEY env var)"
    )

    # Output
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Check for Basilica token
    if not os.environ.get("BASILICA_API_TOKEN"):
        log("BASILICA_API_TOKEN environment variable not set", "error")
        print("  Please set: export BASILICA_API_TOKEN='your-token'")
        sys.exit(1)

    # Run evaluation
    results, total_time, task_ids = asyncio.run(run_evaluation(args))

    # Print summary
    summary = print_summary(results, total_time, task_ids)

    # Save results if requested
    if args.save:
        save_results(args, results, summary, task_ids)


if __name__ == "__main__":
    main()