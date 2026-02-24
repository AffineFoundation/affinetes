#!/usr/bin/env python3
"""
Autoppia IWA evaluation via affinetes.

The environment uses DOOD (Docker-out-of-Docker) to spawn demo website
containers as siblings, so the Docker socket is mounted automatically.

Prerequisites:
    1. Build the autoppia-affine-env image: cd autoppia_affine && ./startup.sh build
    2. Build and run the model container: cd autoppia_affine && docker compose -f model/docker-compose.yml up -d

Usage:
    python examples/autoppia/evaluate.py
    python examples/autoppia/evaluate.py --base-url http://my-model:9000/act
    python examples/autoppia/evaluate.py --task-id autobooks-demo-task-1
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import affinetes as af

DEFAULT_IMAGE = "autoppia-affine-env:latest"
DEFAULT_MODEL_URL = "http://autoppia-affine-model:9000/act"


async def main():
    parser = argparse.ArgumentParser(
        description="Autoppia IWA evaluation via affinetes",
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help=f"Docker image (default: {DEFAULT_IMAGE})")
    parser.add_argument("--model", type=str, default="test-model", help="Model identifier passed to /evaluate (default: test-model)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_MODEL_URL, help=f"Full URL of the model's /act endpoint (default: {DEFAULT_MODEL_URL})")
    parser.add_argument("--task-id", type=str, default=None, help="Evaluate only this task ID (default: all tasks)")
    parser.add_argument("--max-steps", type=int, default=30, help="Max environment steps per task (default: 30)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
    parser.add_argument("--pull", action="store_true", help="Pull image from registry")
    parser.add_argument("--no-force-recreate", action="store_false", dest="force_recreate", default=True, help="Reuse existing container if running")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Affinetes: Autoppia IWA Evaluation Example")
    print("=" * 60)

    try:
        # Step 1: Load environment
        print(f"\nLoading environment from image: {args.image}")
        print("Mounting Docker socket for DOOD (sibling container pattern)")

        env_vars = {}
        chutes_key = os.getenv("CHUTES_API_KEY")
        if chutes_key:
            env_vars["CHUTES_API_KEY"] = chutes_key

        env = af.load_env(
            image=args.image,
            mode="docker",
            env_type="http_based",
            env_vars=env_vars,
            pull=args.pull,
            force_recreate=args.force_recreate,
            cleanup=False,
            volumes={
                "/var/run/docker.sock": {
                    "bind": "/var/run/docker.sock",
                    "mode": "rw",
                }
            },
            enable_logging=True,
            log_console=True,
        )
        print(f"Environment loaded: {env.name}")

        # Step 2: List available methods
        print("\nAvailable methods:")
        await env.list_methods()

        # Step 3: Run evaluation
        print("\nStarting evaluation...")
        print(f"  Model:     {args.model}")
        print(f"  Endpoint:  {args.base_url}")
        print(f"  Task ID:   {args.task_id or 'all'}")
        print(f"  Max steps: {args.max_steps}")
        print("-" * 60)

        eval_kwargs = {
            "model": args.model,
            "base_url": args.base_url,
            "max_steps": args.max_steps,
        }
        if args.task_id is not None:
            eval_kwargs["task_id"] = args.task_id

        result = await env.evaluate(**eval_kwargs, _timeout=args.timeout + 60)

        # Step 4: Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULT")
        print("=" * 60)

        total_score = result.get("total_score", 0)
        success_rate = result.get("success_rate", 0)
        evaluated = result.get("evaluated", 0)

        print(f"Total score:  {total_score:.2f}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Evaluated:    {evaluated} task(s)")

        details = result.get("details", [])
        if details:
            print("\n--- Task Details ---")
            for detail in details:
                status = "PASS" if detail.get("success") else "FAIL"
                print(
                    f"  [{status}] {detail.get('task_id', 'N/A')}"
                    f"  score={detail.get('score', 0):.2f}"
                    f"  steps={detail.get('steps', 0)}"
                    f"  tests={detail.get('tests_passed', 0)}/{detail.get('total_tests', 0)}"
                )

        if result.get("error"):
            print(f"\nError: {result['error']}")

        # Save full result
        output_dir = Path(__file__).resolve().parent / "eval"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"autoppia_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to: {output_path}")

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("\nCleaning up...")
        try:
            await env.cleanup()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
