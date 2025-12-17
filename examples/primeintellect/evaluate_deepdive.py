#!/usr/bin/env python3

import asyncio
import json
import sys
import os
import affinetes as af
from dotenv import load_dotenv

load_dotenv(override=True)

async def main():
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("Error: CHUTES_API_KEY not set")
        sys.exit(1)
    
    # Check for required API keys for DeepDive
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        print("Warning: SERPER_API_KEY not set - search functionality may be limited")

    image_tag = af.build_image_from_env(
        env_path="environments/primeintellect/deepdive",
        image_tag="deepdive:latest"
    )
    
    env_vars = {"CHUTES_API_KEY": api_key}
    if serper_api_key:
        env_vars["SERPER_API_KEY"] = serper_api_key
    
    env = af.load_env(
        image=image_tag,
        mode="docker",
        env_vars=env_vars,
        cleanup=False,
        force_recreate=True,
    )

    result = await env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        task_id=0,
        judge_model="deepseek-ai/DeepSeek-V3.2-Speciale",
        judge_base_url="https://llm.chutes.ai/v1",
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())