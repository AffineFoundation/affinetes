#!/usr/bin/env python3
"""
Local test script for OpenEnv protocol methods.

This script tests the reset(), step(), and state() methods directly on the
Actor classes without requiring a running HTTP server.

Usage:
    # Activate virtual environment first
    source .venv/bin/activate

    # Test trace environment (from repo root)
    cd environments/trace && python ../../examples/test_openenv_local.py --env trace

    # Test OpenSpiel environment
    cd environments/openspiel && python ../../examples/test_openenv_local.py --env openspiel
"""

import argparse
import asyncio
import sys
import os


def test_trace_env():
    """Test trace environment OpenEnv methods."""
    print("\n" + "="*60)
    print("Testing Trace Environment OpenEnv Protocol")
    print("="*60 + "\n")
    
    # Import trace environment
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../environments/trace")
    from env import Actor
    
    actor = Actor()
    
    # Test 1: Reset
    print("1. Testing reset()...")
    result = asyncio.run(actor.reset(task_id=1, seed=42))
    print(f"   Done: {result['done']}")
    print(f"   Observation length: {len(result['observation'])}")
    print(f"   Info: {list(result['info'].keys())}")
    print(f"   Observation preview: {result['observation'][:150]}...")
    assert not result['done'], "Episode should not be done after reset"
    assert result['observation'], "Should have observation after reset"
    print("   PASSED\n")
    
    # Test 2: State
    print("2. Testing state()...")
    state_result = actor.state()
    print(f"   Done: {state_result['done']}")
    print(f"   Observation matches reset: {state_result['observation'] == result['observation']}")
    assert state_result['observation'] == result['observation'], "State should match reset"
    print("   PASSED\n")
    
    # Test 3: Step with wrong answer
    print("3. Testing step() with wrong answer...")
    step_result = asyncio.run(actor.step(action="wrong answer"))
    print(f"   Done: {step_result['done']}")
    print(f"   Reward: {step_result['reward']}")
    print(f"   Score: {step_result['info'].get('score')}")
    assert step_result['done'], "Trace should be done after single step"
    assert step_result['reward'] == 0.0, "Wrong answer should have 0 reward"
    print("   PASSED\n")
    
    # Test 4: Reset and step with (probably) wrong answer
    print("4. Testing reset() then step() cycle...")
    result = asyncio.run(actor.reset(task_id=2, seed=123))
    step_result = asyncio.run(actor.step(action="some prediction"))
    print(f"   Reset done: {result['done']}")
    print(f"   Step done: {step_result['done']}")
    print(f"   Reward: {step_result['reward']}")
    print("   PASSED\n")
    
    print("="*60)
    print("All Trace environment tests passed!")
    print("="*60 + "\n")
    return True


def test_openspiel_env():
    """Test OpenSpiel environment OpenEnv methods."""
    print("\n" + "="*60)
    print("Testing OpenSpiel Environment OpenEnv Protocol")
    print("="*60 + "\n")
    
    # Import openspiel environment
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../environments/openspiel")
    from env import Actor
    
    actor = Actor()
    
    # Test 1: Reset (use liars_dice which is quick)
    # task_id 100000000 = game index 1 (liars_dice) with config 0
    print("1. Testing reset() with liars_dice...")
    result = actor.reset(task_id=100000000, seed=42, opponent="random")
    print(f"   Game: {result['info'].get('game_name')}")
    print(f"   Done: {result['done']}")
    print(f"   LLM Player ID: {result['info'].get('llm_player_id')}")
    print(f"   Legal actions: {result['info'].get('legal_actions', [])[:5]}...")
    print(f"   Observation length: {len(result['observation'])}")
    print(f"   Observation preview: {result['observation'][:200]}...")
    assert not result['done'], "Episode should not be done after reset"
    assert result['observation'], "Should have observation after reset"
    assert result['info'].get('legal_actions'), "Should have legal actions"
    print("   PASSED\n")
    
    # Test 2: State
    print("2. Testing state()...")
    state_result = actor.state()
    print(f"   Done: {state_result['done']}")
    print(f"   Has observation: {bool(state_result['observation'])}")
    print("   PASSED\n")
    
    # Test 3: Step with first legal action
    print("3. Testing step() with first legal action...")
    legal_actions = result['info'].get('legal_actions', [])
    if legal_actions:
        action = str(legal_actions[0])
        step_result = actor.step(action=action)
        print(f"   Action taken: {action}")
        print(f"   Done: {step_result['done']}")
        print(f"   Reward: {step_result['reward']}")
        if not step_result['done']:
            print(f"   New legal actions: {step_result['info'].get('legal_actions', [])[:5]}...")
    else:
        print("   No legal actions available (game may have ended)")
    print("   PASSED\n")
    
    # Test 4: Play until game ends
    print("4. Testing full episode loop...")
    result = actor.reset(task_id=100000000, seed=123, opponent="random")
    step_count = 0
    max_steps = 50
    
    while not result['done'] and step_count < max_steps:
        legal_actions = result['info'].get('legal_actions', [])
        if not legal_actions:
            print(f"   No legal actions at step {step_count}")
            break
        
        action = str(legal_actions[0])  # Always take first action
        result = actor.step(action=action)
        step_count += 1
    
    print(f"   Episode completed in {step_count} steps")
    print(f"   Final done: {result['done']}")
    print(f"   Final reward: {result['reward']}")
    print(f"   Score: {result['info'].get('score')}")
    print(f"   Returns: {result['info'].get('returns')}")
    print("   PASSED\n")
    
    # Test 5: Test with a different game (goofspiel - game index 0)
    print("5. Testing with goofspiel (task_id=0)...")
    result = actor.reset(task_id=0, seed=42, opponent="random")
    print(f"   Game: {result['info'].get('game_name')}")
    print(f"   Done: {result['done']}")
    print(f"   Legal actions count: {len(result['info'].get('legal_actions', []))}")
    
    step_count = 0
    max_steps = 100
    while not result['done'] and step_count < max_steps:
        legal_actions = result['info'].get('legal_actions', [])
        if not legal_actions:
            break
        action = str(legal_actions[0])
        result = actor.step(action=action)
        step_count += 1
    
    print(f"   Episode completed in {step_count} steps")
    print(f"   Score: {result['info'].get('score')}")
    print("   PASSED\n")
    
    print("="*60)
    print("All OpenSpiel environment tests passed!")
    print("="*60 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Local test for OpenEnv protocol methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--env",
        choices=["trace", "openspiel", "all"],
        default="all",
        help="Environment to test (default: all)"
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.env in ["trace", "all"]:
        try:
            success = test_trace_env() and success
        except Exception as e:
            print(f"Trace test failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    if args.env in ["openspiel", "all"]:
        try:
            success = test_openspiel_env() and success
        except Exception as e:
            print(f"OpenSpiel test failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
