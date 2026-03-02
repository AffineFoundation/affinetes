#!/usr/bin/env python3
"""
QQR Travel Planning Evaluation Test

Usage:
    python test_qqr.py
"""

import asyncio
import json
import sys
import os

import affinetes as af
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
    api_key = os.getenv("CHUTES_API_KEY")
    amap_key = os.getenv("AMAP_MAPS_API_KEY")

    if not api_key:
        print("Error: CHUTES_API_KEY not set")
        sys.exit(1)

    if not amap_key:
        print("Warning: AMAP_MAPS_API_KEY not set, AMap tools may not work")

    print(f"API Keys loaded: CHUTES={api_key[:20]}..., AMAP={amap_key[:10] if amap_key else 'None'}...")

    # Build image
    print("\nBuilding image...")
    image_tag = af.build_image_from_env(
        env_path="environments/qqr",
        image_tag="qqr:latest",
        quiet=False
    )

    # Cache directory for MCP tools (shared between restarts)
    cache_dir = os.path.expanduser("~/.cache/qqr")
    os.makedirs(cache_dir, exist_ok=True)

    print("\nLoading environment...")
    env = af.load_env(
        image=image_tag,
        mode="docker",
        env_vars={
            "CHUTES_API_KEY": api_key,
            "AMAP_MAPS_API_KEY": amap_key or "",
        },
        volumes={
            cache_dir: {"bind": "/var/lib/qqr/cache", "mode": "rw"},
        },
        enable_logging=True,
        log_file="qqr_evaluation.log",
        log_console=True
    )
    print("Environment loaded\n")

    # Run evaluation
    print("Running evaluation with task_id=100...")
    result = await env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        task_id=100,
        timeout=300,
        temperature=0.7,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Print score summary
    if "score" in result:
        print(f"\nFinal Score: {result['score']:.2%}")
    if "extra" in result and "score_breakdown" in result["extra"]:
        breakdown = result["extra"]["score_breakdown"]
        if breakdown:
            print(f"  - Code Score: {breakdown.get('code_score', {}).get('subtotal', 0)}/60")
            print(f"  - LLM Score: {breakdown.get('llm_score', {}).get('subtotal', 0)}/40")
            if "hard_constraints" in breakdown:
                print(f"  - Hard Constraints: {breakdown['hard_constraints']}")
        else:
            print("  - Score breakdown is None (evaluation incomplete)")

    await env.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
