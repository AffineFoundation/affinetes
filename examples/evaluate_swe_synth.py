"""
SWE-SYNTH Environment Evaluation Example

Tests the breaker module integration with full affinetes environment.
"""

import affinetes as af_env
import os
import sys
import asyncio
from dotenv import load_dotenv
import json

load_dotenv(override=True)


async def main():
    print("\n" + "=" * 60)
    print("Affinetes: SWE-SYNTH Environment Evaluation Example")
    print("=" * 60)

    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("\n   CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        print("   Or create .env file with: CHUTES_API_KEY=your-key")
        sys.exit(1)

    print("\n1. Building SWE-SYNTH environment...")
    af_env.build_image_from_env(
        env_path="environments/SWE-SYNTH",
        image_tag="swe-synth:latest",
    )

    print("\n2. Loading SWE-SYNTH environment with DOOD support...")
    print("   Note: Docker socket will be mounted for container-in-container support")

    # Load environment with Docker socket mounted for DOOD
    env = af_env.load_env(
        image="swe-synth:latest",
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key},
        # Mount Docker socket for DOOD (Docker-out-of-Docker)
        volumes={
            "/var/run/docker.sock": {
                "bind": "/var/run/docker.sock",
                "mode": "rw"
            }
        },
    )

    try:
        print("\n3. Running evaluation on SWE-SYNTH task...")
        print("   - task_id=1: will generate a bug and have fixer attempt to fix it")
        print("   - This tests the full breaker module integration")

        result = await env.evaluate(
            task_id=1,
            model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
            base_url="https://llm.chutes.ai/v1",
            skip_cache=True,  # Force regenerate to test breaker
            breaker_max_iterations=15,
            breaker_cost_limit=3.0,
            max_iterations=15,
            cost_limit=5.0,
        )

        print("\n4. Evaluation Result:")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 60)

        # Check result
        if result.get("error"):
            print(f"\n   Evaluation error: {result['error']}")
        else:
            score = result.get("score", 0)
            print(f"\n   Score: {score}")
            if "bug_patch" in result:
                print(f"   Bug was injected successfully")
            if "problem_statement" in result:
                print(f"   Problem statement generated")

    except Exception as e:
        print(f"\n   Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n5. Cleaning up...")
        await env.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
