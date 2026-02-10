#!/usr/bin/env python3
"""
OpenSpiel Training Interface Example

Demonstrates using the OpenEnv reset/step/stop interface for training.
This example uses a simple random policy to interact with the game.

Usage:
    python examples/openspiel/openenv_openspiel.py --task-id 100000000 --seed 12345
    python examples/openspiel/openenv_openspiel.py --game liars_dice --opponent random
"""

import argparse
import asyncio
import json
import random
from typing import Optional, List

import affinetes as af


def parse_legal_actions(observation: str) -> List[int]:
    """Extract legal action IDs from observation text."""
    actions = []
    in_legal_actions = False

    for line in observation.split("\n"):
        if "Legal Actions:" in line:
            in_legal_actions = True
            continue
        if in_legal_actions:
            if line.strip().startswith("Your choice"):
                break
            # Parse lines like "  0 -> raise" or "  5 -> fold"
            parts = line.strip().split(" -> ")
            if parts and parts[0].isdigit():
                actions.append(int(parts[0]))

    return actions


def select_random_action(legal_actions: List[int]) -> str:
    """Select a random action from legal actions."""
    if not legal_actions:
        return "0"  # Fallback
    return str(random.choice(legal_actions))


async def main():
    parser = argparse.ArgumentParser(description="OpenSpiel Training Interface Example")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Task ID (encodes game type and config)")
    parser.add_argument("--game", type=str, default="liars_dice",
                        help="Game name (only used if task-id not specified)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "mcts"],
                        help="Opponent type")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    args = parser.parse_args()

    # Map game names to task_id ranges
    GAME_TASK_IDS = {
        "goofspiel": 0,
        "liars_dice": 100000000,
        "leduc_poker": 200000000,
        "gin_rummy": 300000000,
        "othello": 400000000,
        "backgammon": 500000000,
        "hex": 600000000,
        "clobber": 700000000,
        "hearts": 800000000,
        "euchre": 900000000,
    }

    # Determine task_id
    task_id = args.task_id
    if task_id is None:
        task_id = GAME_TASK_IDS.get(args.game, 100000000)  # Default to liars_dice

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    print(f"Building OpenSpiel environment image...")
    image_tag = af.build_image_from_env(
        env_path="environments/openspiel",
        image_tag="openspiel:training",
    )

    print(f"Loading environment...")
    env = af.load_env(
        image=image_tag,
        mode="docker",
    )

    print(f"\n{'='*80}")
    print(f"Starting game with task_id={task_id}, seed={seed}")
    print(f"Opponent type: {args.opponent}")
    print(f"{'='*80}\n")

    # Reset the environment
    sess = await env.openenv().reset(
        task_id=task_id,
        seed=seed,
        kwargs={"opponent": args.opponent}
    )

    obs = sess.observation
    print("RESET (Initial State)")
    print("-" * 80)
    print(obs[:2000] if len(obs) > 2000 else obs)  # Truncate if too long

    total_reward = 0.0
    step_count = 0

    # Game loop
    for step_num in range(args.max_steps):
        # Parse legal actions from observation
        legal_actions = parse_legal_actions(obs)

        if not legal_actions:
            print("\nNo legal actions found - game may have ended")
            break

        # Select action (random policy for demo)
        action = select_random_action(legal_actions)

        # Execute step
        step_resp = await sess.step(action)

        obs = step_resp["observation"]
        reward = float(step_resp.get("reward", 0.0))
        done = step_resp.get("done", False)
        truncated = step_resp.get("truncated", False)
        info = step_resp.get("info", {})

        total_reward += reward
        step_count += 1

        print(f"\nSTEP {step_num + 1}")
        print("-" * 40)
        print(f"Action: {action}")
        print(f"Reward: {reward:.4f}")
        print(f"Done: {done}, Truncated: {truncated}")
        print(f"Step count: {info.get('step_count', 'N/A')}")

        # Show truncated observation
        obs_display = obs[:1000] if len(obs) > 1000 else obs
        print(f"Observation:\n{obs_display}")

        if done or truncated:
            print("\n" + "=" * 80)
            print("GAME FINISHED")
            print("=" * 80)
            print(f"Final Info: {json.dumps(info, indent=2, ensure_ascii=False)}")
            break

    # Summary
    print("\n" + "=" * 80)
    print("EPISODE SUMMARY")
    print("=" * 80)
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Game name: {sess.last.get('info', {}).get('game_name', 'unknown')}")

    # Clean up
    await sess.stop()
    await env.cleanup()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
