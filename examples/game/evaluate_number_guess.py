#!/usr/bin/env python3
"""
Evaluate Number Guessing interactive environment

Usage:
    python evaluate_number_guess.py
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
    if not api_key:
        print("Error: CHUTES_API_KEY not set")
        sys.exit(1)

    image_tag = af.build_image_from_env(
        env_path="environments/game/number_guess",
        image_tag="game:guess",
        quiet=False
    )
    
    print("Loading environment...")
    # Method 1: Auto-enable logging on load
    # Pass enable_logging=True to load_env() to automatically start log streaming
    env = af.load_env(
        image=image_tag,
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key},
        enable_logging=True,
        log_file="number_guess_evaluation.log",
        log_console=True
    )
    print("Environment loaded\n")
    
    # Method 2: Manual logging control (alternative approach)
    # Load environment without auto-logging, then manually start/stop logging:
    # env = af.load_env(
    #     image=image_tag,
    #     mode="docker",
    #     env_vars={"CHUTES_API_KEY": api_key}
    # )
    # env.start_logging(
    #     file="number_guess_evaluation.log",  # Optional: log file path
    #     console=True,                         # Optional: print to console (default: True)
    #     tail="all",                           # Optional: "all" or number of lines (default: "all")
    #     timestamps=True                       # Optional: include timestamps (default: True)
    # )
    # # ... do your work ...
    # env.stop_logging()  # Stop logging when done (or use context manager)
    
    result = await env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        task_id=100
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    

if __name__ == "__main__":
    asyncio.run(main())