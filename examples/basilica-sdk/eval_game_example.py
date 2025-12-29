#!/usr/bin/env python3
import asyncio
import os
import sys
from dotenv import load_dotenv
import json
import affinetes as af_env

load_dotenv()

async def main():
    basilica_token = os.getenv("BASILICA_API_TOKEN")
    chutes_key = os.getenv("CHUTES_API_KEY")
    
    if not basilica_token:
        print("\n❌ BASILICA_API_TOKEN not set")
        print("   Export it: export BASILICA_API_TOKEN='your-token'")
        sys.exit(1)
    
    if not chutes_key:
        print("\n❌ CHUTES_API_KEY not set")
        print("   Export it: export CHUTES_API_KEY='your-key'")
        sys.exit(1)

    env = af_env.load_env(
        mode="basilica",
        image="affinefoundation/game:openspiel",
        cpu_limit="4000m",
        mem_limit="16Gi",
        env_vars={
            "CHUTES_API_KEY": chutes_key,
        },
        ttl_buffer=300,
    )
    
    result = await env.evaluate(
        model="Qwen/Qwen3-32B",
        base_url="https://llm.chutes.ai/v1",
        task_id=123,
        timeout=1800
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())