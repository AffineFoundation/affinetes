#!/usr/bin/env python3
"""Parallel ML evaluation on Basilica.

Each task gets its own isolated Kubernetes pod, created on demand and
destroyed after completion. Basilica handles pod scheduling -- all tasks
are launched concurrently, and the infrastructure scales automatically.

Prerequisites:
    pip install affinetes[basilica]
    export BASILICA_API_TOKEN="your-token"

Usage:
    # Single task
    python basilica_evaluate.py --image epappas/mth:pi-fixed --tasks 1

    # Range of tasks (all run in parallel)
    python basilica_evaluate.py --image epappas/mth:pi-fixed --tasks 1-10

    # Specific tasks
    python basilica_evaluate.py --image epappas/mth:pi-fixed --tasks 1,5,10,20

    # Custom model and resources
    python basilica_evaluate.py \\
        --image epappas/mth:pi-fixed \\
        --tasks 1-50 \\
        --model Qwen/Qwen3-32B \\
        --cpu 4000m --memory 16Gi \\
        --output results.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import affinetes


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    """Immutable evaluation configuration."""

    image: str
    task_ids: tuple[int, ...]
    model: str = "Qwen/Qwen3-32B"
    base_url: str = "https://llm.chutes.ai/v1"
    api_key: str = ""
    cpu: str = "2000m"
    memory: str = "8Gi"
    timeout: int = 1800
    ttl_buffer: int = 600
    output: str | None = None


@dataclass
class TaskResult:
    """Outcome of a single evaluation task."""

    task_id: int
    elapsed: float
    score: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

async def evaluate_task(
    env: affinetes.EnvironmentWrapper,
    task_id: int,
    config: EvalConfig,
) -> TaskResult:
    """Run a single evaluation task on an ephemeral Basilica pod."""
    start = time.monotonic()
    try:
        result = await env.evaluate(
            model=config.model,
            base_url=config.base_url,
            task_id=task_id,
            timeout=config.timeout,
            _timeout=config.timeout + 60,
            api_key=config.api_key,
        )
        return TaskResult(
            task_id=task_id,
            elapsed=time.monotonic() - start,
            score=result.get("score", 0.0),
            result=result,
        )
    except Exception as e:
        return TaskResult(
            task_id=task_id,
            elapsed=time.monotonic() - start,
            error=str(e),
            error_type=type(e).__name__,
        )


async def run(config: EvalConfig) -> list[TaskResult]:
    """Load environment, run all tasks in parallel, collect results."""
    env = affinetes.load_env(
        mode="basilica",
        image=config.image,
        cpu_limit=config.cpu,
        mem_limit=config.memory,
        env_vars={"CHUTES_API_KEY": config.api_key} if config.api_key else None,
        ttl_buffer=config.ttl_buffer,
    )

    results: list[TaskResult] = []
    try:
        coros = [evaluate_task(env, tid, config) for tid in config.task_ids]
        for future in asyncio.as_completed(coros):
            r = await future
            tag = f"score={r.score:.4f}" if r.ok else f"FAILED ({r.error_type})"
            print(f"  task {r.task_id:>5}  {r.elapsed:7.1f}s  {tag}")
            results.append(r)
    finally:
        await env.cleanup()

    return sorted(results, key=lambda r: r.task_id)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(results: list[TaskResult], wall_time: float) -> dict[str, Any]:
    """Print evaluation summary and return it as a dict."""
    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    total_score = sum(r.score for r in succeeded if r.score is not None)
    avg_score = total_score / len(succeeded) if succeeded else 0.0

    print(f"\n{'=' * 52}")
    print(f"  Tasks        {len(results)} total, {len(succeeded)} passed, {len(failed)} failed")
    print(f"  Avg Score    {avg_score:.4f}")
    print(f"  Total Score  {total_score:.2f}")
    print(f"  Wall Time    {wall_time:.1f}s")

    if failed:
        print(f"\n  Errors:")
        for r in failed:
            print(f"    task {r.task_id}: {r.error_type}: {str(r.error)[:60]}")

    print(f"{'=' * 52}\n")

    return {
        "total_tasks": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "average_score": avg_score,
        "total_score": total_score,
        "wall_time": wall_time,
    }


def save_results(
    config: EvalConfig,
    results: list[TaskResult],
    summary: dict[str, Any],
    path: str,
) -> None:
    """Persist results to a JSON file."""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": config.image,
        "model": config.model,
        "base_url": config.base_url,
        **summary,
        "results": [
            {
                "task_id": r.task_id,
                "elapsed": r.elapsed,
                **(
                    {"score": r.score, "result": r.result}
                    if r.ok
                    else {"error": r.error, "error_type": r.error_type}
                ),
            }
            for r in results
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_task_ids(value: str) -> tuple[int, ...]:
    """Parse task IDs from '42', '1-10', or '1,5,10,20'."""
    if "-" in value and "," not in value:
        lo, hi = value.split("-", 1)
        return tuple(range(int(lo), int(hi) + 1))
    if "," in value:
        return tuple(int(x.strip()) for x in value.split(","))
    return (int(value),)


def build_config(args: argparse.Namespace) -> EvalConfig:
    """Build an EvalConfig from parsed CLI arguments."""
    return EvalConfig(
        image=args.image,
        task_ids=parse_task_ids(args.tasks),
        model=args.model,
        base_url=args.base_url,
        api_key=os.environ.get("CHUTES_API_KEY", ""),
        cpu=args.cpu,
        memory=args.memory,
        timeout=args.timeout,
        ttl_buffer=args.ttl_buffer,
        output=args.output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel ML evaluation on Basilica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --image epappas/mth:pi-fixed --tasks 1\n"
            "  %(prog)s --image epappas/mth:pi-fixed --tasks 1-10\n"
            "  %(prog)s --image epappas/mth:pi-fixed --tasks 1,5,10 -o results.json"
        ),
    )

    parser.add_argument("--image", required=True, help="Docker image to deploy")
    parser.add_argument("--tasks", required=True, help="Task IDs: 42, 1-10, or 1,5,10")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="LLM model (default: Qwen/Qwen3-32B)")
    parser.add_argument("--base-url", default="https://llm.chutes.ai/v1", help="LLM API URL")
    parser.add_argument("--cpu", default="2000m", help="CPU per pod (default: 2000m)")
    parser.add_argument("--memory", default="8Gi", help="Memory per pod (default: 8Gi)")
    parser.add_argument("--timeout", type=int, default=1800, help="Task timeout in seconds (default: 1800)")
    parser.add_argument("--ttl-buffer", type=int, default=600, help="Pod TTL buffer in seconds (default: 600)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    if not os.environ.get("BASILICA_API_TOKEN"):
        print("Error: BASILICA_API_TOKEN environment variable not set")
        print("  export BASILICA_API_TOKEN='your-token'")
        sys.exit(1)

    config = build_config(args)

    print(f"\n{'=' * 52}")
    print(f"  Image      {config.image}")
    print(f"  Model      {config.model}")
    print(f"  Tasks      {len(config.task_ids)}")
    print(f"  Resources  {config.cpu} CPU, {config.memory} RAM")
    print(f"  Timeout    {config.timeout}s per task")
    print(f"{'=' * 52}\n")

    t0 = time.monotonic()
    results = asyncio.run(run(config))
    wall_time = time.monotonic() - t0

    summary = print_summary(results, wall_time)

    if config.output:
        save_results(config, results, summary, config.output)


if __name__ == "__main__":
    main()
