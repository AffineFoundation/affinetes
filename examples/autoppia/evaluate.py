#!/usr/bin/env python3
"""
Autoppia IWA evaluation via affinetes.

The environment uses DOOD (Docker-out-of-Docker) to spawn demo website
containers as siblings, so the Docker socket is mounted automatically.

Prerequisites:
    1. Build the autoppia-affine-env image:
       cd autoppia_affine && ./startup.sh build
    2. Build and run the model container:
       cd autoppia_affine && docker compose -f model/docker-compose.yml up -d

Usage:
    python examples/autoppia/evaluate.py
    python examples/autoppia/evaluate.py --base-url http://my-model:9000/act
    python examples/autoppia/evaluate.py --task-id autobooks-demo-task-1
    python examples/autoppia/evaluate.py --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

import affinetes as af  # noqa: E402

DEFAULT_IMAGE = "autoppia-affine-env:latest"
DEFAULT_MODEL_URL = "http://autoppia-affine-model:9000/act"
DOCKER_SOCKET_VOLUME = {
    "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Autoppia IWA evaluation via affinetes",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help=f"Docker image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="test-model",
        help="Model identifier for /evaluate (default: test-model)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_MODEL_URL,
        help="Full URL of the model's /act endpoint",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Evaluate only this task ID (default: all tasks)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Max environment steps per task (default: 30)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--pull", action="store_true", help="Pull image from registry"
    )
    parser.add_argument(
        "--no-force-recreate",
        action="store_false",
        dest="force_recreate",
        default=True,
        help="Reuse existing container if running",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for result JSON",
    )
    parser.add_argument(
        "--require-chutes",
        action="store_true",
        help="Require CHUTES_API_KEY (default: optional)",
    )
    return parser.parse_args(argv)


def build_env_vars() -> dict[str, str]:
    """Build environment variables for the container (e.g. CHUTES_API_KEY)."""
    env_vars: dict[str, str] = {}
    chutes_key = os.getenv("CHUTES_API_KEY")
    if chutes_key:
        env_vars["CHUTES_API_KEY"] = chutes_key
    return env_vars


def validate_env(require_chutes: bool = False) -> None:
    """Validate env; exit if require_chutes and CHUTES_API_KEY missing."""
    if not require_chutes:
        return
    chutes_key = os.getenv("CHUTES_API_KEY")
    if not chutes_key:
        print(
            "Error: CHUTES_API_KEY environment variable not set.",
            file=sys.stderr,
        )
        print("Set it with: export CHUTES_API_KEY='your-key'", file=sys.stderr)
        sys.exit(1)


def load_autoppia_env(args: argparse.Namespace) -> Any:
    """Load Autoppia affinetes environment with DOOD socket mount."""
    env_vars = build_env_vars()
    return af.load_env(
        image=args.image,
        mode="docker",
        env_type="http_based",
        env_vars=env_vars,
        pull=args.pull,
        force_recreate=args.force_recreate,
        cleanup=False,
        volumes=DOCKER_SOCKET_VOLUME,
        enable_logging=True,
        log_console=True,
    )


def build_eval_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build keyword arguments for env.evaluate()."""
    kwargs: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "max_steps": args.max_steps,
    }
    if args.task_id is not None:
        kwargs["task_id"] = args.task_id
    return kwargs


async def run_evaluation(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Run evaluation on the environment and return the result dict."""
    eval_kwargs = build_eval_kwargs(args)
    return await env.evaluate(**eval_kwargs, _timeout=args.timeout + 60)


def print_result(result: dict[str, Any]) -> None:
    """Print evaluation summary and task details to stdout."""
    total_score = result.get("total_score", 0)
    success_rate = result.get("success_rate", 0)
    evaluated = result.get("evaluated", 0)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    print(f"Total score:  {total_score:.2f}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Evaluated:    {evaluated} task(s)")

    details = result.get("details", [])
    if details:
        print("\n--- Task Details ---")
        for detail in details:
            status = "PASS" if detail.get("success") else "FAIL"
            tid = detail.get("task_id", "N/A")
            sc = detail.get("score", 0)
            st = detail.get("steps", 0)
            tp = detail.get("tests_passed", 0)
            tt = detail.get("total_tests", 0)
            part = f"  [{status}] {tid}  score={sc:.2f}"
            part2 = f"  steps={st}  tests={tp}/{tt}"
            print(part + "  " + part2)

    if result.get("error"):
        print(f"\nError: {result['error']}")


def get_output_dir(args: argparse.Namespace) -> Path:
    """Resolve output directory for result JSON."""
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    return Path(__file__).resolve().parent / "eval"


def save_result(result: dict[str, Any], output_dir: Path) -> Path:
    """Write full result to a timestamped JSON file; return path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path = output_dir / f"autoppia_{ts}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return output_path


async def main(argv: list[str] | None = None) -> int:
    """Entry: parse args, load env, run evaluation, print and save results."""
    args = parse_args(argv)
    require = getattr(args, "require_chutes", False)
    validate_env(require_chutes=require)

    print("\n" + "=" * 60)
    print("Affinetes: Autoppia IWA Evaluation Example")
    print("=" * 60)

    env = None
    try:
        print(f"\nLoading environment from image: {args.image}")
        print("Mounting Docker socket for DOOD (sibling container pattern)")
        env = load_autoppia_env(args)
        print(f"Environment loaded: {env.name}")

        print("\nAvailable methods:")
        await env.list_methods()

        print("\nStarting evaluation...")
        print(f"  Model:     {args.model}")
        print(f"  Endpoint:  {args.base_url}")
        print(f"  Task ID:   {args.task_id or 'all'}")
        print(f"  Max steps: {args.max_steps}")
        print("-" * 60)

        result = await run_evaluation(env, args)
        print_result(result)

        output_dir = get_output_dir(args)
        output_path = save_result(result, output_dir)
        print(f"\nFull results saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"\nEvaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\nCleaning up...")
        if env is not None:
            try:
                await env.cleanup()
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
