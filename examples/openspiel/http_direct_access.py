#!/usr/bin/env python3
"""
Example: Direct HTTP access to OpenSpiel environment (without SDK)

This example demonstrates how to:
1. Use affinetes SDK to start a container with host network mode
2. Access the container's HTTP endpoints directly using requests/httpx
3. Use the new RESTful-style endpoints (POST /{method_name})

Usage:
    python examples/openspiel/http_direct_access.py
    python examples/openspiel/http_direct_access.py --game leduc_poker --opponent mcts
"""

import argparse
import asyncio
import json
import random
from typing import List

import httpx
import affinetes as af


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
            parts = line.strip().split(" -> ")
            if parts and parts[0].isdigit():
                actions.append(int(parts[0]))

    return actions


def select_random_action(legal_actions: List[int]) -> str:
    """Select a random action from legal actions."""
    if not legal_actions:
        return "0"
    return str(random.choice(legal_actions))


async def main():
    parser = argparse.ArgumentParser(description="OpenSpiel HTTP Direct Access Example")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Task ID (encodes game type and config)")
    parser.add_argument("--game", type=str, default="liars_dice",
                        help="Game name (only used if task-id not specified)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--opponent", type=str, default="mcts",
                        choices=["random", "mcts"],
                        help="Opponent type")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--port", type=int, default=9000,
                        help="Host port for the container")
    args = parser.parse_args()

    # Determine task_id
    task_id = args.task_id
    if task_id is None:
        task_id = GAME_TASK_IDS.get(args.game, 100000000)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    # Build the image (nocache=True to pick up env.py changes)
    print("Building OpenSpiel environment image...")
    image_tag = af.build_image_from_env(
        env_path="environments/openspiel",
        image_tag="openspiel:http",
        nocache=True,
        quiet=False,
    )

    # Start container with host network mode
    print("Starting container with host network mode...")
    env = af.load_env(
        image=image_tag,
        mode="docker",
        host_network=True,
        host_port=args.port,
        cleanup=False,
    )

    base_url = f"http://localhost:{args.port}"

    print("=" * 80)
    print("Container started with host network mode")
    print(f"HTTP API available at: {base_url}")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Health check
        print("\n[1] Health Check")
        resp = await client.get(f"{base_url}/health")
        print(f"    GET /health -> {resp.json()}")

        # 2. List available methods
        print("\n[2] List Methods")
        resp = await client.get(f"{base_url}/methods")
        methods_info = resp.json()
        print(f"    GET /methods -> {json.dumps(methods_info, indent=4)}")


        # 3. Reset using RESTful-style endpoint
        print(f"\n[3] Reset Game (task_id={task_id}, seed={seed}, opponent={args.opponent})")
        resp = await client.post(
            f"{base_url}/reset",
            json={
                "task_id": task_id,
                "seed": seed,
                "opponent": args.opponent
            }
        )
        result = resp.json()
        reset_data = result["result"]
        episode_id = reset_data["episode_id"]
        obs = reset_data["observation"]
        info = reset_data.get("info", {})

        print(f"    POST /reset -> episode_id={episode_id}")
        print(f"    Game: {info.get('game_name', 'unknown')}")
        print(f"    Player ID: {info.get('llm_player_id', 'unknown')}")
        print(f"    Opponent: {info.get('opponent_type', 'unknown')}")
        print(f"\n    Initial Observation (truncated):")
        print("    " + obs[:500].replace("\n", "\n    ") + "...")

        # 4. Game loop using direct HTTP calls
        print(f"\n[4] Playing game with random policy...")
        total_reward = 0.0
        step_count = 0

        for step_num in range(args.max_steps):
            # Parse legal actions
            legal_actions = parse_legal_actions(obs)

            if not legal_actions:
                print(f"    No legal actions found - game may have ended")
                break

            # Select random action
            action = select_random_action(legal_actions)

            # Call step via RESTful endpoint
            resp = await client.post(
                f"{base_url}/step",
                json={
                    "action": action,
                    "episode_id": episode_id
                }
            )
            result = resp.json()

            if result["status"] != "success":
                print(f"    Step error: {result.get('error')}")
                break

            step_data = result["result"]
            obs = step_data["observation"]
            reward = float(step_data.get("reward", 0.0))
            done = step_data.get("done", False)
            truncated = step_data.get("truncated", False)

            total_reward += reward
            step_count += 1

            # Show progress
            status = "WIN" if reward > 0.5 else "LOSS" if reward < 0.5 else "DRAW" if done else "..."
            print(f"    Step {step_num + 1}: action={action}, reward={reward:.3f}, done={done} [{status}]")

            if done or truncated:
                break

        # 5. Get final state
        print(f"\n[5] Final State")
        resp = await client.post(
            f"{base_url}/state",
            json={"episode_id": episode_id}
        )
        state_result = resp.json()
        if state_result["status"] == "success":
            final_info = state_result["result"].get("info", {})
            print(f"    Total steps: {final_info.get('step_count', step_count)}")
            print(f"    Cumulative reward: {final_info.get('cumulative_reward', total_reward):.4f}")

        # 6. Stop episode
        print(f"\n[6] Stop Episode")
        resp = await client.post(
            f"{base_url}/stop",
            json={"episode_id": episode_id}
        )
        print(f"    POST /stop -> {resp.json()}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"\nContainer is still running. You can access it directly:")
    print(f"  curl {base_url}/health")
    print(f"  curl {base_url}/methods")
    print(f"  curl -X POST {base_url}/reset -H 'Content-Type: application/json' \\")
    print(f"       -d '{{\"task_id\": {task_id}, \"seed\": 42}}'")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
