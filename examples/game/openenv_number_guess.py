#!/usr/bin/env python3

import argparse
import asyncio
import json
from typing import Optional, Tuple

import affinetes as af


def _pick_guess(observation: str, low: int, high: int, last_guess: Optional[int]) -> Tuple[int, int, int]:
    """
    Simple heuristic policy:
    - Start with mid
    - If observation says "higher than X" -> low = X + 1
    - If observation says "lower than X"  -> high = X - 1
    """
    if last_guess is not None:
        if "higher than" in observation:
            low = max(low, last_guess + 1)
        elif "lower than" in observation:
            high = min(high, last_guess - 1)

    if low > high:
        # Fallback
        low, high = 1, 1000

    guess = (low + high) // 2
    return guess, low, high


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=100, help="Deterministic task instance id")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed (will be echoed back in info)")
    args = parser.parse_args()

    image_tag = af.build_image_from_env(
        env_path="environments/game/number_guess",
        image_tag="game:number-guess-openenv",
        quiet=False,
    )

    env = af.load_env(
        image=image_tag,
        mode="docker",
    )

    sess = await env.openenv().reset(task_id=args.task_id, seed=args.seed)
    obs = sess.observation
    print("RESET\n" + "=" * 80)
    print(obs)

    # A (minimal) info: env no longer returns min/max/max_attempts in info.public.
    # Use the known environment range for this example.
    low, high = 1, 1000
    last_guess = None
    total_reward = 0.0

    for _ in range(20):
        guess, low, high = _pick_guess(obs, low, high, last_guess)
        last_guess = guess

        step_resp = await sess.step(str(guess))

        obs = step_resp["observation"]
        total_reward += float(step_resp.get("reward", 0.0))

        print("\nSTEP\n" + "-" * 80)
        print(f"action={guess} reward={step_resp['reward']} done={step_resp['done']} truncated={step_resp.get('truncated', False)}")
        print(obs)

        if step_resp["done"] or step_resp.get("truncated", False):
            print("\nFINAL\n" + "=" * 80)
            print(json.dumps(step_resp, indent=2, ensure_ascii=False))
            break

    print(f"\nTotal reward: {total_reward}")

    # Prefer explicit stop over relying on __del__ (more deterministic).
    await sess.stop()
    await env.cleanup()


if __name__ == "__main__":
    asyncio.run(main())


