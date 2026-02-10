#!/usr/bin/env python3
"""
Example: Direct HTTP access to affinetes environment (without SDK)

This example demonstrates how to:
1. Use affinetes SDK to start a container with host network mode
2. Access the container's HTTP endpoints directly using requests/httpx
3. Use the new RESTful-style endpoints (POST /{method_name})

Usage:
    python http_direct_access.py
"""

import asyncio
import json
import httpx
import affinetes as af


async def main():
    # Build the image
    image_tag = af.build_image_from_env(
        env_path="environments/game/number_guess",
        image_tag="game:number-guess-http",
        quiet=False,
    )

    # Start container with host network mode
    # - host_network=True: Container uses host's network stack (port 8000 directly accessible)
    # - cleanup=False: Container keeps running after program exits
    # - force_recreate=False: Reuse existing container if available
    env = af.load_env(
        image=image_tag,
        mode="docker",
        host_network=True,
        host_port=8000,
        cleanup=False,
        force_recreate=False,
    )

    print("=" * 80)
    print("Container started with host network mode")
    print("You can now access the HTTP API directly at http://localhost:8000")
    print("=" * 80)

    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health check
        print("\n[1] Health Check")
        resp = await client.get(f"{base_url}/health")
        print(f"    GET /health -> {resp.json()}")

        # 2. List available methods
        print("\n[2] List Methods")
        resp = await client.get(f"{base_url}/methods")
        methods_info = resp.json()
        print(f"    GET /methods -> {json.dumps(methods_info, indent=4)}")

        # 3. Reset game via RESTful endpoint
        print("\n[3] Reset via /reset")
        resp = await client.post(
            f"{base_url}/reset",
            json={"task_id": 42, "seed": 123}
        )
        result = resp.json()
        episode_id = result["result"]["episode_id"]
        print(f"    POST /reset -> episode_id={episode_id}")
        print(f"    observation: {result['result']['observation'][:100]}...")

        # 4. Step via RESTful endpoint
        print("\n[4] Step via /step")
        resp = await client.post(
            f"{base_url}/step",
            json={
                "action": "500",  # Guess 500
                "episode_id": episode_id
            }
        )
        result = resp.json()
        print(f"    POST /step -> reward={result['result']['reward']}, done={result['result']['done']}")
        print(f"    observation: {result['result']['observation'][:100]}...")

        # 5. Play a few more steps
        print("\n[5] Playing game with binary search...")
        low, high = 1, 1000
        last_guess = 500

        for i in range(10):
            obs = result["result"]["observation"]

            # Parse hint from observation
            if "higher than" in obs:
                low = max(low, last_guess + 1)
            elif "lower than" in obs:
                high = min(high, last_guess - 1)

            if result["result"]["done"]:
                print(f"    Game finished!")
                break

            guess = (low + high) // 2
            last_guess = guess

            resp = await client.post(
                f"{base_url}/step",
                json={"action": str(guess), "episode_id": episode_id}
            )
            result = resp.json()
            print(f"    Guess {guess}: reward={result['result']['reward']}, done={result['result']['done']}")

        # 6. Stop the episode
        print("\n[6] Stop via /stop")
        resp = await client.post(
            f"{base_url}/stop",
            json={"episode_id": episode_id}
        )
        print(f"    POST /stop -> {resp.json()}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("Container is still running. You can access it directly:")
    print(f"  curl {base_url}/health")
    print(f"  curl {base_url}/methods")
    print(f"  curl -X POST {base_url}/reset -H 'Content-Type: application/json' -d '{{\"task_id\": 100}}'")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
