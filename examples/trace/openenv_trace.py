#!/usr/bin/env python3
"""
Trace Environment OpenEnv Example

Demonstrates using the OpenEnv reset/step/stop interface for the Trace environment.
This is a single-turn evaluation task where you predict the stdout output of code.

Usage:
    python examples/trace/openenv_trace.py
    python examples/trace/openenv_trace.py --task-id 42 --seed 12345
"""

import argparse
import asyncio

import affinetes as af


def simple_prediction(observation: str) -> str:
    """
    A simple heuristic to predict stdout output.

    In practice, you would use an LLM to analyze the code and predict the output.
    This example just returns a placeholder to demonstrate the interface.
    """
    # For demo purposes, return a simple prediction
    # A real implementation would parse the code and predict output
    return "1\n2\n3\n"


async def main():
    parser = argparse.ArgumentParser(description="Trace Environment OpenEnv Example")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Task ID (index into dataset)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print("Building Trace environment image...")
    image_tag = af.build_image_from_env(
        env_path="environments/trace",
        image_tag="trace:openenv",
        quiet=False,
    )

    print("Loading environment...")
    env = af.load_env(
        image=image_tag,
        mode="docker",
    )

    print(f"\n{'='*80}")
    print("Starting Trace evaluation")
    if args.task_id is not None:
        print(f"Task ID: {args.task_id}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")

    # Reset the environment to get a challenge
    sess = await env.openenv().reset(
        task_id=args.task_id,
        seed=args.seed,
    )

    obs = sess.observation
    print("RESET (Challenge)")
    print("-" * 80)
    # Show the challenge prompt (truncate if too long)
    print(obs[:3000] if len(obs) > 3000 else obs)
    print("-" * 80)

    # Generate a prediction
    # In a real scenario, you would call an LLM here
    prediction = simple_prediction(obs)
    print(f"\nPrediction: {prediction!r}")

    # Submit the prediction
    step_resp = await sess.step(prediction)

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
    print(f"Dataset Index: {info.get('dataset_index', 'N/A')}")

    # Clean up
    await sess.stop()
    await env.cleanup()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
