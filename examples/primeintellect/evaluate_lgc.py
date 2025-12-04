import affinetes as af_env
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
    print("\n" + "=" * 60)
    print("Affinetes: Logic Environment Evaluation Example")
    print("=" * 60)

    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("\n   ❌ CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        print("   Or create .env file with: CHUTES_API_KEY=your-key")
        sys.exit(1)

    print("\n1. Building Docker image for logic environment...")
    # Build the image from the lgc environment directory
    af_env.build_image_from_env(
        env_path="environments/primeintellect/lgc",
        image_tag="logic:latest",
    )
    print("   ✓ Image built successfully")
    
    print("\n2. Loading logic environment from image 'logic:latest'...")
    env = af_env.load_env(
        image="logic:latest",
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key},
        pull=False,
        cleanup=False,
    )
    print("   ✓ Environment loaded (container started with HTTP server)")

    try:
        print("\n3. Available methods in environment:")
        await env.list_methods(print_info=True)

        print("\n4. Running logic evaluation in container (async)...")
        result = await env.evaluate(
            model="deepseek-ai/DeepSeek-V3",
            base_url="https://llm.chutes.ai/v1",
            task_id=100,  # Deterministic task selection
            temperature=0.7,
            timeout=600
        )
        
        if 'error' in result:
            print(f"\n   Error occurred:")
            print(f"     Error type: {result.get('error_type', 'unknown')}")
            print(f"     Error: {result['error'][:200]}...")
            return

        print(f"\nResults:")
        print(f"   Task: {result['task_name']}")
        print(f"   Score: {result['score']}")
        print(f"   Success: {result['success']}")
        print(f"   Time taken: {result['time_taken']:.2f}s")
        
        if 'extra' in result:
            print(f"\n   Extra info:")
            print(f"     Seed: {result['extra'].get('seed')}")
            print(f"     Dataset index: {result['extra'].get('dataset_index')}")
            print(f"     Task type: {result['extra'].get('task_type')}")
            
            if 'conversation' in result['extra']:
                conv = result['extra']['conversation']
                print(f"\n   Conversation:")
                print(f"     Question: {conv[0]['content'][:200]}...")
                print(f"     Response: {conv[1]['content'][:200]}...")
                
    except Exception as e:
        print(f"\n   ❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n5. Cleaning up...")
        # Cleanup is handled automatically by affinetes framework


if __name__ == "__main__":
    asyncio.run(main())