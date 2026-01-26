#!/usr/bin/env python3
"""
Logic V2 Environment OpenEnv Example

Demonstrates using the OpenEnv reset/step/stop interface for the Logic V2 environment.
This is a single-turn evaluation task for logic puzzles (e.g., Dyck language).

Usage:
    python examples/lgc-v2/openenv_lgc_v2.py
    python examples/lgc-v2/openenv_lgc_v2.py --task-id 42

Task ID encoding:
    - 0-99,999,999: dyck_language tasks
    - Task ID encodes both task type and seed
"""

import argparse
import asyncio

import affinetes as af


def simple_solver(observation: str) -> str:
    """
    A simple heuristic solver for logic tasks.

    In practice, you would use an LLM to solve the puzzle.
    This example just returns a placeholder to demonstrate the interface.
    """
    # For demo purposes, return a simple answer
    # A real implementation would parse and solve the logic puzzle
    return ")"


async def main():
    parser = argparse.ArgumentParser(description="Logic V2 Environment OpenEnv Example")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Task ID (encodes task type and seed)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (unused - seed encoded in task-id)")
    args = parser.parse_args()

    print("Building Logic V2 environment image...")
    image_tag = af.build_image_from_env(
        env_path="environments/primeintellect/lgc-v2",
        image_tag="lgc-v2:openenv",
        quiet=False,
    )

    print("Loading environment...")
    env = af.load_env(
        image=image_tag,
        mode="docker",
    )

    print(f"\n{'='*80}")
    print("Starting Logic V2 evaluation")
    if args.task_id is not None:
        print(f"Task ID: {args.task_id}")
    print(f"{'='*80}\n")

    # Reset the environment to get a challenge
    sess = await env.openenv().reset(
        task_id=args.task_id,
        seed=args.seed,
    )

    obs = sess.observation
    print("RESET (Challenge)")
    print("-" * 80)
    # Show the challenge prompt
    print(obs[:2000] if len(obs) > 2000 else obs)
    print("-" * 80)

    # Generate an answer
    # In a real scenario, you would call an LLM here
    answer = simple_solver(obs)
    print(f"\nAnswer: {answer!r}")

    # Submit the answer
    step_resp = await sess.step(answer)

    obs = step_resp["observation"]
    reward = float(step_resp.get("reward", 0.0))
    done = step_resp.get("done", False)
    info = step_resp.get("info", {})

    print("\nSTEP (Evaluation Result)")
    print("-" * 80)
    print(f"Reward (Score): {reward}")
    print(f"Done: {done}")
    print(f"\nResult:\n{obs}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Score: {reward}")
    print(f"Task ID: {info.get('task_id', 'N/A')}")
    print(f"Task Type: {info.get('task_type', 'N/A')}")
    print(f"Seed: {info.get('seed', 'N/A')}")

    # Clean up
    await sess.stop()
    await env.cleanup()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
