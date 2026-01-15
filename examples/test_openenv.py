#!/usr/bin/env python3
"""
Test script for OpenEnv protocol endpoints.

This script tests the /reset, /step, and /state endpoints that enable
training frameworks to interact with Affinetes environments.

Usage:
    # Test trace environment locally (requires running the env)
    python examples/test_openenv.py --env trace --url http://localhost:8000

    # Test with a deployed environment
    python examples/test_openenv.py --env trace --url http://your-env-url:8000

    # Test OpenSpiel environment
    python examples/test_openenv.py --env openspiel --url http://localhost:8000
"""

import argparse
import httpx
import json
import sys
from typing import Optional


def test_openenv_protocol(base_url: str, env_type: str = "trace", task_id: int = 1, seed: int = 42):
    """
    Test the OpenEnv protocol endpoints.
    
    Args:
        base_url: Base URL of the environment server (e.g., http://localhost:8000)
        env_type: Environment type ("trace" or "openspiel")
        task_id: Task ID for deterministic testing
        seed: Random seed for reproducibility
    """
    client = httpx.Client(timeout=60.0)
    base_url = base_url.rstrip("/")
    
    print(f"\n{'='*60}")
    print(f"Testing OpenEnv Protocol: {env_type} environment")
    print(f"Base URL: {base_url}")
    print(f"Task ID: {task_id}, Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Test 1: Reset
    print("1. Testing /reset endpoint...")
    reset_payload = {
        "task_id": task_id,
        "seed": seed,
    }
    if env_type == "openspiel":
        reset_payload["kwargs"] = {"opponent": "random"}
    
    try:
        response = client.post(f"{base_url}/reset", json=reset_payload)
        response.raise_for_status()
        reset_result = response.json()
        print(f"   Status: SUCCESS")
        print(f"   Observation length: {len(reset_result.get('observation', ''))}")
        print(f"   Done: {reset_result.get('done')}")
        print(f"   Info keys: {list(reset_result.get('info', {}).keys())}")
        
        # Print first 200 chars of observation
        obs = reset_result.get("observation", "")
        if obs:
            print(f"   Observation preview: {obs[:200]}...")
    except Exception as e:
        print(f"   Status: FAILED - {e}")
        return False
    
    print()
    
    # Test 2: State (should return same observation)
    print("2. Testing /state endpoint...")
    try:
        response = client.get(f"{base_url}/state")
        response.raise_for_status()
        state_result = response.json()
        print(f"   Status: SUCCESS")
        print(f"   Observation length: {len(state_result.get('observation', ''))}")
        print(f"   Done: {state_result.get('done')}")
        
        # Verify observation matches reset
        if state_result.get("observation") == reset_result.get("observation"):
            print(f"   Observation matches reset: YES")
        else:
            print(f"   Observation matches reset: NO (may be expected if state changed)")
    except Exception as e:
        print(f"   Status: FAILED - {e}")
        return False
    
    print()
    
    # Test 3: Step (with a dummy action)
    print("3. Testing /step endpoint...")
    
    # Generate a test action based on environment type
    if env_type == "trace":
        # For trace, any text is valid (it's the model's prediction)
        test_action = "This is a test prediction output"
    else:
        # For OpenSpiel, we need a valid action
        # Get legal actions from info
        legal_actions = reset_result.get("info", {}).get("legal_actions", [])
        if legal_actions:
            test_action = str(legal_actions[0])
        else:
            test_action = "0"
    
    step_payload = {"action": test_action}
    
    try:
        response = client.post(f"{base_url}/step", json=step_payload)
        response.raise_for_status()
        step_result = response.json()
        print(f"   Status: SUCCESS")
        print(f"   Action: {test_action[:50]}...")
        print(f"   Reward: {step_result.get('reward')}")
        print(f"   Done: {step_result.get('done')}")
        
        if env_type == "trace":
            # Trace is single-step, so it should be done
            if step_result.get("done"):
                print(f"   Single-step completed correctly: YES")
            else:
                print(f"   Single-step completed correctly: NO (unexpected)")
        else:
            # OpenSpiel may continue
            if step_result.get("done"):
                print(f"   Game completed in one move (short game)")
            else:
                print(f"   Game continues, observation length: {len(step_result.get('observation', ''))}")
    except Exception as e:
        print(f"   Status: FAILED - {e}")
        return False
    
    print()
    
    # Test 4: Full episode loop (for OpenSpiel)
    if env_type == "openspiel" and not step_result.get("done"):
        print("4. Testing full episode loop...")
        max_steps = 100
        step_count = 1  # Already did one step
        
        while not step_result.get("done") and step_count < max_steps:
            legal_actions = step_result.get("info", {}).get("legal_actions", [])
            if not legal_actions:
                print(f"   No legal actions available, breaking")
                break
            
            # Take first legal action
            test_action = str(legal_actions[0])
            step_payload = {"action": test_action}
            
            try:
                response = client.post(f"{base_url}/step", json=step_payload)
                response.raise_for_status()
                step_result = response.json()
                step_count += 1
            except Exception as e:
                print(f"   Step {step_count} failed: {e}")
                return False
        
        print(f"   Episode completed in {step_count} steps")
        print(f"   Final reward: {step_result.get('reward')}")
        print(f"   Score: {step_result.get('info', {}).get('score')}")
    
    print()
    print(f"{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}\n")
    
    return True


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    client = httpx.Client(timeout=10.0)
    base_url = base_url.rstrip("/")
    
    try:
        response = client.get(f"{base_url}/health")
        response.raise_for_status()
        result = response.json()
        return result.get("status") == "ok"
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenEnv protocol endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the environment server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--env",
        choices=["trace", "openspiel"],
        default="trace",
        help="Environment type to test (default: trace)"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=1,
        help="Task ID for testing (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check before testing"
    )
    
    args = parser.parse_args()
    
    # Check health first
    if not args.skip_health:
        print(f"Checking server health at {args.url}...")
        if not test_health(args.url):
            print("Server health check failed. Is the server running?")
            print(f"Try: curl {args.url}/health")
            sys.exit(1)
        print("Server is healthy.\n")
    
    # Run tests
    success = test_openenv_protocol(
        base_url=args.url,
        env_type=args.env,
        task_id=args.task_id,
        seed=args.seed
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
