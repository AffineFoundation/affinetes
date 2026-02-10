#!/usr/bin/env python3
"""
Test script for running the number_guess game in Docker using affinetes.

Usage:
    # Build the image first (if not already built):
    afs build environments/game/number_guess --tag number-guess:latest

    # Run this test:
    python guess_number_openEnv.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path (works from any location)
sys.path.insert(0, str(Path(__file__).parent))

from affinetes import load_env


async def main():
    """Main test function."""
    
    print("=" * 60)
    print("Number Guess Game - Docker Test")
    print("=" * 60)
    
    # =========================================================
    # Step 1: Load environment in Docker mode
    # =========================================================
    print("\n[1] Loading environment in Docker...")
    
    env = load_env(
        image="number-guess:latest",    # Docker image name
        mode="docker",                   # Run in Docker container
        container_name="number-guess-test",
        force_recreate=True,             # Recreate if exists
        cleanup=False                    # Keep container after test
    )
    
    print(f"    Container: {env.name}")
    print("    Status: Running")
    
    # =========================================================
    # Step 2: List available methods
    # =========================================================
    print("\n[2] Listing available methods...")
    
    methods = await env.list_methods()
    actor_methods = [m for m in methods if m.get('source') == 'Actor']
    
    for m in actor_methods:
        print(f"    - {m['name']}")
    
    # =========================================================
    # Step 3: Reset - Start a new game
    # =========================================================
    print("\n[3] Starting new game (reset)...")
    
    result = await env.reset(task_id=42, seed=12345)
    episode_id = result['episode_id']
    
    print(f"    Episode ID: {episode_id}")
    print(f"    Task ID: {result['info']['task_id']}")
    print(f"    Attempts: {result['info']['attempts_left']}")
    
    # =========================================================
    # Step 4: Play game using binary search
    # =========================================================
    print("\n[4] Playing game with binary search...")
    
    low, high = 1, 1000
    
    for i in range(1, 11):
        # Binary search: guess the middle value
        guess = (low + high) // 2
        
        # Submit guess to environment
        result = await env.step(action=str(guess), episode_id=episode_id)
        
        obs = result['observation']
        done = result['done']
        reward = result['reward']
        
        # Parse hint and update search range
        if "higher" in obs.lower():
            low = guess + 1
            hint = "higher"
        elif "lower" in obs.lower():
            high = guess - 1
            hint = "lower"
        else:
            hint = "correct!" if reward > 0 else "game over"
        
        print(f"    Attempt {i}: {guess} -> {hint}")
        
        if done:
            print(f"\n    Result: {'WIN' if reward > 0 else 'LOSE'} (reward={reward})")
            break
    
    # =========================================================
    # Step 5: Test state endpoint
    # =========================================================
    print("\n[5] Testing state endpoint...")
    
    state = await env.state(episode_id=episode_id)
    print(f"    Done: {state['done']}")
    print(f"    Attempts left: {state['info']['attempts_left']}")
    
    # =========================================================
    # Step 6: Test stop endpoint
    # =========================================================
    print("\n[6] Testing stop endpoint...")
    
    stop_result = await env.stop(episode_id=episode_id)
    print(f"    Result: {stop_result}")
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print(f"\nContainer '{env.name}' is still running.")
    print("To stop it: docker stop number-guess-test && docker rm number-guess-test")


if __name__ == "__main__":
    asyncio.run(main())
