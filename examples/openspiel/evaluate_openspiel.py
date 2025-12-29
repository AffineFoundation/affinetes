#!/usr/bin/env python3
"""Evaluate OpenSpiel environment"""

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

    image_tag = af.build_image_from_env(
        env_path="environments/openspiel",
        image_tag="openspiel:latest"
    )
    
    env = af.load_env(
        image=image_tag,
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key},
        cleanup=False,
        force_recreate=True,
    )

    test_cases = [
        {"task_id": 388240510, "seed": 1231233},
        # {"task_id": 100000001, "seed": 134234},
    ]

    # Run tests in parallel
    tasks = [
        env.evaluate(
            model="Qwen/Qwen3-32B",
            base_url="https://llm.chutes.ai/v1",
            task_id=case["task_id"],
            seed=case["seed"],
            temperature=0.0,
        )
        for case in test_cases
    ]
    
    results = await asyncio.gather(*tasks)
    
    output = []
    for case, result in zip(test_cases, results):
        print(f"Task: {case}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        output.append({
            "task_id": case["task_id"],
            "seed": case["seed"],
            "result": result
        })
        
    print("All tests completed!")
    with open("openspiel_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
if __name__ == "__main__":
    asyncio.run(main())