#!/usr/bin/env python3
"""
Autoppia IWA evaluation via affinetes.

Uses DOOD (Docker-out-of-Docker) to spawn demo website containers
as siblings by mounting the Docker socket.

Prerequisites:
    cd autoppia_affine && ./startup.sh build
    cd autoppia_affine && docker compose -f model/docker-compose.yml up -d

Usage:
    python examples/autoppia/evaluate.py
    python examples/autoppia/evaluate.py --base-url http://my-model:9000/act
    python examples/autoppia/evaluate.py --task-id autobooks-demo-task-1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

import affinetes as af  # noqa: E402

DEFAULT_IMAGE = "autoppia-affine-env:latest"
DEFAULT_MODEL_URL = "http://autoppia-affine-model:9000/act"
DOOD_VOLUMES = {
    "/var/run/docker.sock": {
        "bind": "/var/run/docker.sock",
        "mode": "rw",
    }
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autoppia IWA evaluation via affinetes",
    )
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help=f"Docker image (default: {DEFAULT_IMAGE})")
    p.add_argument("--model", default="test-model",
                   help="Model identifier (default: test-model)")
    p.add_argument("--base-url", default=DEFAULT_MODEL_URL,
                   help="Model /act endpoint URL")
    p.add_argument("--task-id", default=None,
                   help="Single task ID (default: all)")
    p.add_argument("--max-steps", type=int, default=30,
                   help="Max steps per task (default: 30)")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout in seconds (default: 600)")
    p.add_argument("--pull", action="store_true",
                   help="Pull image before run")
    p.add_argument("--no-force-recreate", action="store_false",
                   dest="force_recreate", default=True,
                   help="Reuse existing container")
    p.add_argument("--output-dir", default=None,
                   help="Result JSON directory")
    return p.parse_args(argv)


def load_autoppia_env(
    args: argparse.Namespace,
) -> Any:
    """Load the Autoppia environment with DOOD socket mount."""
    env_vars: dict[str, str] = {}
    chutes_key = os.getenv("CHUTES_API_KEY")
    if chutes_key:
        env_vars["CHUTES_API_KEY"] = chutes_key

    return af.load_env(
        image=args.image,
        mode="docker",
        env_type="http_based",
        env_vars=env_vars,
        pull=args.pull,
        force_recreate=args.force_recreate,
        cleanup=False,
        volumes=DOOD_VOLUMES,
        enable_logging=True,
        log_console=True,
    )


async def run_evaluation(
    env: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Call env.evaluate() with the CLI arguments."""
    kwargs: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "max_steps": args.max_steps,
    }
    if args.task_id is not None:
        kwargs["task_id"] = args.task_id
    return await env.evaluate(
        **kwargs, _timeout=args.timeout + 60,
    )


def print_result(result: dict[str, Any]) -> None:
    """Display evaluation summary."""
    score = result.get("total_score", 0)
    rate = result.get("success_rate", 0)
    count = result.get("evaluated", 0)

    print(f"\nTotal score:  {score:.2f}")
    print(f"Success rate: {rate:.1%}")
    print(f"Evaluated:    {count} task(s)")

    for d in result.get("details", []):
        tag = "PASS" if d.get("success") else "FAIL"
        tid = d.get("task_id", "?")
        sc = d.get("score", 0)
        steps = d.get("steps", 0)
        tp = d.get("tests_passed", 0)
        tt = d.get("total_tests", 0)
        print(f"  [{tag}] {tid}  score={sc:.2f}"
              f"  steps={steps}  tests={tp}/{tt}")

    if result.get("error"):
        print(f"\nError: {result['error']}")


def save_result(
    result: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write result JSON and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = output_dir / f"autoppia_{ts}.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"\nLoading environment: {args.image}")
    env = None
    try:
        env = load_autoppia_env(args)
        print(f"Environment ready: {env.name}")
        await env.list_methods()

        print(f"\nEvaluating  model={args.model}  "
              f"task={args.task_id or 'all'}  "
              f"max_steps={args.max_steps}")

        result = await run_evaluation(env, args)

        print("\n" + "=" * 50)
        print("EVALUATION RESULT")
        print("=" * 50)
        print_result(result)

        out_dir = Path(args.output_dir).resolve() \
            if args.output_dir else \
            Path(__file__).resolve().parent / "eval"
        path = save_result(result, out_dir)
        print(f"\nResults saved to: {path}")
        return 0

    except Exception as exc:
        print(f"\nEvaluation failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            try:
                await env.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
