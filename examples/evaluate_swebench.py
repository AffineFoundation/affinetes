import affinetes as af_env
import os
import sys
import asyncio
from dotenv import load_dotenv
import json

load_dotenv(override=True)


async def main():
    print("\n" + "=" * 60)
    print("Affinetes: SWE-bench Pro Environment Evaluation Example")
    print("=" * 60)

    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("\n   ❌ CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        print("   Or create .env file with: CHUTES_API_KEY=your-key")
        sys.exit(1)

    print("\n1. Loading SWE-bench Pro environment with DOOD support...")
    print("   Note: Docker socket will be mounted for container-in-container support")
    
    af_env.build_image_from_env(
        env_path="environments/SWE-bench_Pro-os",
        image_tag="swebench_pro:latest",
    )
    # Load environment with Docker socket mounted for DOOD
    env = af_env.load_env(
        image="swebench_pro:latest",
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key},
        # pull=False,
        # force_recreate=True,
        # Mount Docker socket for DOOD (Docker-out-of-Docker)
        volumes={
            "/var/run/docker.sock": {
                "bind": "/var/run/docker.sock",
                "mode": "rw"
            }
        },
    )

    try:
        print("\n2. Running evaluation on SWE-bench Verified task...")
        result = await env.evaluate(
            task_id=0,
            model="deepseek-ai/DeepSeek-V3.2",
            base_url="https://llm.chutes.ai/v1",
        )
        
        print("\n3. Evaluation Result:")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 60)

    except Exception as e:
        print(f"\n   ❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())