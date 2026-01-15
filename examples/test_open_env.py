#!/usr/bin/env python3
"""
Test OpenEnv protocol with the Trace environment using Docker.

This script:
1. Builds and deploys the trace environment using affinetes
2. Tests the /reset, /step, /state OpenEnv endpoints
3. Uses a real LLM from Chutes to generate responses

Usage:
    source .venv/bin/activate
    export CHUTES_API_KEY="your-key"
    
    # Build and test (first time)
    python test_open_env.py
    
    # Skip build if image already exists
    python test_open_env.py --skip-build
    
    # Use a specific image
    python test_open_env.py --image trace:latest
"""

import argparse
import asyncio
import httpx
import os
import sys
import time

# Add affinetes to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import affinetes as af


def log(msg: str, level: str = "info"):
    """Print a formatted log message."""
    ts = time.strftime("%H:%M:%S")
    
    if level == "info":
        prefix = f"\033[90m{ts}\033[0m \033[36m▸\033[0m"
    elif level == "success":
        prefix = f"\033[90m{ts}\033[0m \033[32m✓\033[0m"
    elif level == "error":
        prefix = f"\033[90m{ts}\033[0m \033[31m✗\033[0m"
    elif level == "warn":
        prefix = f"\033[90m{ts}\033[0m \033[33m⚠\033[0m"
    else:
        prefix = f"\033[90m{ts}\033[0m  "
    
    print(f"{prefix} {msg}")


def log_header(title: str):
    """Print a section header."""
    print(f"\n\033[1m{'─' * 60}\033[0m")
    print(f"\033[1m{title}\033[0m")
    print(f"\033[1m{'─' * 60}\033[0m\n")


async def call_llm(prompt: str, model: str, base_url: str, api_key: str) -> str:
    """Call LLM API to generate a response."""
    import openai
    
    client = openai.AsyncOpenAI(
        base_url=base_url.rstrip('/'),
        api_key=api_key,
        timeout=httpx.Timeout(120.0),
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    )
    
    return response.choices[0].message.content.strip()


async def test_openenv_with_llm(
    base_url: str,
    model: str = "Qwen/Qwen3-32B",
    llm_base_url: str = "https://llm.chutes.ai/v1",
    api_key: str = None,
    task_id: int = 1,
    seed: int = 42,
):
    """
    Test OpenEnv protocol with a real LLM.
    
    Args:
        base_url: Environment server URL (e.g., http://localhost:8000)
        model: LLM model to use
        llm_base_url: LLM API base URL
        api_key: API key for LLM
        task_id: Task ID for the trace environment
        seed: Random seed for reproducibility
    """
    client = httpx.AsyncClient(timeout=60.0)
    base_url = base_url.rstrip("/")
    
    log_header("Testing OpenEnv Protocol with LLM")
    log(f"Environment URL: {base_url}")
    log(f"Model: {model}")
    log(f"Task ID: {task_id}, Seed: {seed}")
    print()
    
    # Step 1: Test health endpoint
    log("Testing /health endpoint...")
    try:
        response = await client.get(f"{base_url}/health")
        response.raise_for_status()
        log("Health check passed", "success")
    except Exception as e:
        log(f"Health check failed: {e}", "error")
        return False
    
    # Step 2: Reset environment
    log("Testing /reset endpoint...")
    try:
        response = await client.post(
            f"{base_url}/reset",
            json={"task_id": task_id, "seed": seed}
        )
        response.raise_for_status()
        reset_result = response.json()
        
        observation = reset_result.get("observation", "")
        done = reset_result.get("done", False)
        info = reset_result.get("info", {})
        
        log(f"Reset successful", "success")
        log(f"  Done: {done}")
        log(f"  Observation length: {len(observation)} chars")
        log(f"  Dataset index: {info.get('dataset_index')}")
        
        if done:
            log("Episode already done after reset (unexpected)", "error")
            return False
            
    except Exception as e:
        log(f"Reset failed: {e}", "error")
        return False
    
    # Step 3: Test state endpoint
    log("Testing /state endpoint...")
    try:
        response = await client.get(f"{base_url}/state")
        response.raise_for_status()
        state_result = response.json()
        
        if state_result.get("observation") == observation:
            log("State matches reset observation", "success")
        else:
            log("State differs from reset (may be okay)", "warn")
            
    except Exception as e:
        log(f"State check failed: {e}", "error")
        return False
    
    # Step 4: Call LLM to generate a response
    log_header("Calling LLM for Response")
    log(f"Sending prompt to {model}...")
    log(f"Prompt preview: {observation[:200]}...")
    print()
    
    try:
        start_time = time.time()
        llm_response = await call_llm(
            prompt=observation,
            model=model,
            base_url=llm_base_url,
            api_key=api_key
        )
        elapsed = time.time() - start_time
        
        log(f"LLM response received in {elapsed:.1f}s", "success")
        log(f"Response length: {len(llm_response)} chars")
        log(f"Response preview: {llm_response[:200]}...")
        
    except Exception as e:
        log(f"LLM call failed: {e}", "error")
        # Use a dummy response to continue testing
        llm_response = "Test response - LLM call failed"
    
    # Step 5: Submit response via /step
    log_header("Testing /step Endpoint")
    log("Submitting LLM response to environment...")
    
    try:
        response = await client.post(
            f"{base_url}/step",
            json={"action": llm_response}
        )
        response.raise_for_status()
        step_result = response.json()
        
        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        info = step_result.get("info", {})
        score = info.get("score", reward)
        success = info.get("success", score > 0)
        
        log(f"Step completed", "success")
        log(f"  Done: {done}")
        log(f"  Reward: {reward}")
        log(f"  Score: {score}")
        log(f"  Success: {success}")
        
        if not done:
            log("Trace environment should be done after one step", "warn")
            
    except Exception as e:
        log(f"Step failed: {e}", "error")
        return False
    
    # Step 6: Verify state after completion
    log("Verifying final state...")
    try:
        response = await client.get(f"{base_url}/state")
        response.raise_for_status()
        final_state = response.json()
        
        if final_state.get("done"):
            log("Final state shows done=True", "success")
        else:
            log("Final state shows done=False (unexpected)", "warn")
            
    except Exception as e:
        log(f"Final state check failed: {e}", "error")
    
    # Summary
    log_header("Test Summary")
    log(f"Task ID: {task_id}")
    log(f"Seed: {seed}")
    log(f"Model: {model}")
    log(f"Score: {score}")
    log(f"Success: {success}")
    
    if success:
        log("LLM correctly predicted the trace output!", "success")
    else:
        log("LLM prediction did not match expected output", "info")
        # Show ground truth if available
        ground_truth = info.get("ground_truth", "")
        if ground_truth:
            log(f"Expected output preview: {ground_truth[:200]}...")
    
    await client.aclose()
    return True


async def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Wait for server to become ready."""
    start = time.time()
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() - start < timeout:
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def deploy_container(image_tag: str, api_key: str, port: int) -> str:
    """Deploy container manually using Docker and return the URL."""
    import subprocess
    
    container_name = "trace-openenv-test"
    
    # Stop and remove existing container
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)
    
    # Start new container with port mapping
    result = subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "-e", "AFFINETES_PORT=8000",
        "-e", f"CHUTES_API_KEY={api_key}",
        "-p", f"{port}:8000",
        image_tag
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")
    
    return f"http://localhost:{port}", container_name


async def cleanup_container(container_name: str):
    """Stop and remove container."""
    import subprocess
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)


async def main(args):
    """Main function to run the OpenEnv test."""
    
    # Get API key from environment
    api_key = os.environ.get("CHUTES_API_KEY")
    if not api_key:
        log("CHUTES_API_KEY not set", "error")
        print("  Please set: export CHUTES_API_KEY='your-key'")
        sys.exit(1)
    
    image_tag = args.image
    container_name = None
    
    # Build the image if needed
    if not args.skip_build and not args.image:
        log_header("Building Trace Environment Docker Image")
        log("This may take a few minutes on first run...")
        
        try:
            image_tag = af.build_image_from_env(
                env_path="environments/trace",
                image_tag="trace:openenv-test",
                quiet=False,  # Show build output
            )
            log(f"Image built: {image_tag}", "success")
        except Exception as e:
            log(f"Failed to build image: {e}", "error")
            sys.exit(1)
    elif args.skip_build:
        image_tag = "trace:openenv-test"
        log(f"Skipping build, using existing image: {image_tag}")
    
    log_header("Deploying Trace Environment")
    
    try:
        # Deploy container manually (works better on Mac with port mapping)
        log(f"Starting container from image: {image_tag}")
        env_url, container_name = await deploy_container(image_tag, api_key, args.port)
        log(f"Container started: {container_name}", "success")
        
        # Wait for server to be ready
        log(f"Waiting for server at {env_url}...")
        
        if await wait_for_server(env_url, timeout=60):
            log("Server is ready", "success")
        else:
            log("Server failed to start within timeout", "error")
            sys.exit(1)
        
        # Run the OpenEnv tests
        success = await test_openenv_with_llm(
            base_url=env_url,
            model=args.model,
            llm_base_url="https://llm.chutes.ai/v1",
            api_key=api_key,
            task_id=1,
            seed=42,
        )
        
        # Run a second test with a different task
        log_header("Running Second Test (Different Task)")
        success2 = await test_openenv_with_llm(
            base_url=env_url,
            model=args.model,
            llm_base_url="https://llm.chutes.ai/v1",
            api_key=api_key,
            task_id=5,
            seed=123,
        )
        
        log_header("Final Results")
        log(f"Test 1 (task_id=1): {'PASSED' if success else 'FAILED'}", 
            "success" if success else "error")
        log(f"Test 2 (task_id=5): {'PASSED' if success2 else 'FAILED'}", 
            "success" if success2 else "error")
        
    except Exception as e:
        log(f"Test failed with exception: {e}", "error")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if container_name:
            log("Cleaning up container...")
            try:
                await cleanup_container(container_name)
                log("Container cleaned up", "success")
            except Exception as e:
                log(f"Cleanup error: {e}", "warn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test OpenEnv protocol with Trace environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Docker image build (use existing image)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Docker image to use (implies --skip-build)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the environment server (default: 8765)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="LLM model to use for testing (default: Qwen/Qwen3-32B)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))
