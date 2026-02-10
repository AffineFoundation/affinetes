import os
import json
import subprocess
import tempfile
import shutil
import time


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally"""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())


def run_ridges_sandbox(repo_path, agent_path, problem_statement, sandbox_proxy_url="http://74.82.63.163:9001", timeout=1500, actual_model=None):
    """
    Run the ridges sandbox using Docker out of Docker pattern.

    Uses docker cp instead of bind mounts to support running from within containers.
    """
    # Get absolute paths
    repo_path = os.path.abspath(repo_path)
    agent_path = os.path.abspath(agent_path)
    sandbox_dir = os.path.dirname(os.path.abspath(__file__))
    dockerfile_dir = os.path.join(sandbox_dir, "sandbox")

    # Validate inputs
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent path does not exist: {agent_path}")

    # Create a temporary directory for local files
    temp_dir_obj = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    temp_dir = temp_dir_obj.name
    container_name = f"ridges-sandbox-{os.urandom(8).hex()}"

    try:
        print(f"[RIDGES] Created temporary directory: {temp_dir}")

        # Prepare local files
        sandbox_temp = os.path.join(temp_dir, "sandbox_context")
        os.makedirs(sandbox_temp, exist_ok=True)

        # Copy Dockerfile and packages_py.txt for image build
        dockerfile_src = os.path.join(dockerfile_dir, "Dockerfile")
        dockerfile_dst = os.path.join(sandbox_temp, "Dockerfile")
        shutil.copy(dockerfile_src, dockerfile_dst)

        packages_py_src = os.path.join(dockerfile_dir, "packages_py.txt")
        packages_py_dst = os.path.join(sandbox_temp, "packages_py.txt")
        if os.path.exists(packages_py_src):
            shutil.copy(packages_py_src, packages_py_dst)
        else:
            with open(packages_py_dst, "w") as f:
                f.write("")

        # Prepare files to copy into container
        agent_local = os.path.join(sandbox_temp, "agent.py")
        shutil.copy(agent_path, agent_local)

        agent_runner_src = os.path.join(dockerfile_dir, "AGENT_RUNNER.py")
        agent_runner_local = os.path.join(sandbox_temp, "AGENT_RUNNER.py")
        shutil.copy(agent_runner_src, agent_runner_local)

        # Create input.json
        input_data = {"problem_statement": problem_statement}
        input_json_local = os.path.join(sandbox_temp, "input.json")
        with open(input_json_local, "w") as f:
            json.dump(input_data, f, indent=2)

        # Copy repository to temp (for docker cp)
        repo_local = os.path.join(sandbox_temp, "repo")
        if os.path.isdir(repo_path):
            shutil.copytree(repo_path, repo_local, symlinks=False, ignore=shutil.ignore_patterns('.git'))
        else:
            os.makedirs(repo_local, exist_ok=True)
            shutil.copy(repo_path, repo_local)

        print(f"[RIDGES] Prepared local files")

        # Build Docker image (skip if already exists)
        image_name = "ridges-sandbox:latest"
        if image_exists(image_name):
            print(f"[RIDGES] Docker image {image_name} already exists, skipping build")
        else:
            print(f"[RIDGES] Building Docker image: {image_name}")
            build_cmd = ["docker", "build", "-t", image_name, sandbox_temp]
            try:
                subprocess.run(build_cmd, cwd=sandbox_temp, capture_output=True, text=True, check=True)
                print(f"[RIDGES] Docker image built successfully")
            except subprocess.CalledProcessError as e:
                print(f"[RIDGES] Docker build failed: {e.stderr}")
                raise

        # Step 1: Start container with sleep (like minisweagent)
        print(f"[RIDGES] Starting container: {container_name}")
        run_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--add-host=host.docker.internal:host-gateway",
            "-e", f"SANDBOX_PROXY_URL={sandbox_proxy_url}",
            "-e", f"TIMEOUT={timeout}",
            "-e", f"ACTUAL_MODEL={actual_model or 'unknown'}",
            image_name,
            "sleep", str(timeout + 60)
        ]
        result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
        print(f"[RIDGES] Container started: {result.stdout.strip()[:12]}")

        # Step 2: Copy files into container using docker cp
        print(f"[RIDGES] Copying files into container...")

        # Create /sandbox directory
        subprocess.run(["docker", "exec", container_name, "mkdir", "-p", "/sandbox"], check=True)

        # Copy agent.py
        subprocess.run(["docker", "cp", agent_local, f"{container_name}:/sandbox/agent.py"], check=True)

        # Copy AGENT_RUNNER.py
        subprocess.run(["docker", "cp", agent_runner_local, f"{container_name}:/sandbox/AGENT_RUNNER.py"], check=True)

        # Copy input.json
        subprocess.run(["docker", "cp", input_json_local, f"{container_name}:/sandbox/input.json"], check=True)

        # Copy repo directory
        subprocess.run(["docker", "cp", repo_local, f"{container_name}:/sandbox/repo"], check=True)

        print(f"[RIDGES] Files copied successfully")

        # Step 3: Execute AGENT_RUNNER.py using docker exec
        print(f"[RIDGES] Running AGENT_RUNNER.py...")
        try:
            exec_result = subprocess.run(
                ["docker", "exec", container_name, "python", "/sandbox/AGENT_RUNNER.py"],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            print(f"[RIDGES] Execution finished (exit code: {exec_result.returncode})")
            if exec_result.stdout:
                print(f"[RIDGES] stdout:\n{exec_result.stdout}")
            if exec_result.stderr:
                print(f"[RIDGES] stderr:\n{exec_result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"[RIDGES] Execution timed out after {timeout} seconds")
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            return {"success": False, "error": f"Execution timed out after {timeout} seconds"}

        # Step 4: Copy output.json from container
        output_json_local = os.path.join(temp_dir, "output.json")
        copy_result = subprocess.run(
            ["docker", "cp", f"{container_name}:/sandbox/output.json", output_json_local],
            capture_output=True, text=True
        )

        if copy_result.returncode != 0:
            print(f"[RIDGES] Failed to copy output.json from container")
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            return {"success": False, "error": "output.json was not created by the agent"}

        # Read output.json
        with open(output_json_local, "r") as f:
            output_data = json.load(f)

        print(f"output_data: {json.dumps(output_data, indent=4)}")
        print(f"[RIDGES] Read output.json")
        return output_data

    finally:
        # Cleanup container
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        # Cleanup temp directory
        try:
            temp_dir_obj.cleanup()
        except (PermissionError, OSError) as e:
            print(f"[RIDGES] Warning: Could not fully cleanup temp directory: {e}")


def run_proxy_container(openai_api_key, openai_model, openai_base_url="https://api.openai.com/v1", temperature=0.0, seed=None, port=8000, container_name="proxy-api"):
    """
    Build and run the proxy container.
    
    Args:
        openai_api_key: OpenAI API key
        openai_model: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        openai_base_url: OpenAI base URL (default: https://api.openai.com/v1)
        temperature: Model temperature (default: 0.0)
        seed: Random seed for reproducibility (optional)
        port: Port to expose the API on (default: 8000)
        container_name: Name for the Docker container (default: proxy-api)
    
    Returns:
        dict: Status information with 'success', 'container_name', 'port', and 'url' keys
    """
    proxy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy")
    
    # Validate proxy directory exists
    if not os.path.exists(proxy_dir):
        return {
            "success": False,
            "error": f"Proxy directory not found: {proxy_dir}"
        }
    
    # Check if container is already running and stop it (to ensure fresh config)
    check_cmd = ["docker", "ps", "-q", "-f", f"name={container_name}"]
    check_result = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if check_result.stdout.strip():
        print(f"[PROXY] Container '{container_name}' is already running, stopping it to apply new configuration...")
        subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Remove any stopped container with the same name
    subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Build Docker image (skip if already exists)
    image_name = "proxy-api:latest"
    if image_exists(image_name):
        print(f"[PROXY] Docker image {image_name} already exists, skipping build")
    else:
        print(f"[PROXY] Building Docker image: {image_name}")
        build_cmd = ["docker", "build", "-t", image_name, proxy_dir]

        try:
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"[PROXY] Docker image built successfully")
        except subprocess.CalledProcessError as e:
            print(f"[PROXY] Docker build failed:")
            print(e.stdout)
            print(e.stderr)
            return {
                "success": False,
                "error": "Failed to build Docker image",
                "details": e.stderr
            }
    
    # Run Docker container
    print(f"[PROXY] Starting Docker container: {container_name}")
    run_cmd = [
        "docker", "run",
        "-d",  # Detached mode
        "--name", container_name,
        "-p", f"{port}:8000",
        "-e", f"OPENAI_API_KEY={openai_api_key}",
        "-e", f"OPENAI_BASE_URL={openai_base_url}",
        "-e", f"OPENAI_MODEL={openai_model}",
        "-e", f"OPENAI_TEMPERATURE={temperature}",
    ]
    
    # Add seed if provided
    if seed is not None:
        run_cmd.extend(["-e", f"OPENAI_SEED={seed}"])
    
    run_cmd.append(image_name)
    
    try:
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        container_id = result.stdout.strip()
        print(f"[PROXY] Container started: {container_id[:12]}")
        
        # Wait a moment for container to initialize
        time.sleep(2)
        
        # Check if container is still running
        check_cmd = ["docker", "ps", "-q", "-f", f"name={container_name}"]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if not check_result.stdout.strip():
            # Container stopped, get logs
            logs_cmd = ["docker", "logs", container_name]
            logs_result = subprocess.run(logs_cmd, capture_output=True, text=True)
            print(f"[PROXY] Container failed to start. Logs:")
            print(logs_result.stdout)
            print(logs_result.stderr)
            return {
                "success": False,
                "error": "Container started but stopped immediately",
                "logs": logs_result.stdout + logs_result.stderr
            }
        
        print(f"[PROXY] Proxy API is running at http://localhost:{port}/api/inference")
        return {
            "success": True,
            "container_name": container_name,
            "container_id": container_id[:12],
            "port": port,
            "url": f"http://localhost:{port}",
            "endpoint": f"http://localhost:{port}/api/inference",
            "status": "started"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"[PROXY] Failed to start container:")
        print(e.stderr)
        return {
            "success": False,
            "error": "Failed to start Docker container",
            "details": e.stderr
        }

def stop_proxy_container(container_name="proxy-api"):
    """
    Stop and remove the proxy container.
    
    Args:
        container_name: Name of the Docker container to stop (default: proxy-api)
    
    Returns:
        dict: Status information
    """
    print(f"[PROXY] Stopping container: {container_name}")
    
    # Stop and remove container
    result = subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"[PROXY] Container stopped and removed")
        return {
            "success": True,
            "message": "Container stopped and removed"
        }
    else:
        if "No such container" in result.stderr:
            print(f"[PROXY] Container not found")
            return {
                "success": True,
                "message": "Container was not running"
            }
        else:
            print(f"[PROXY] Failed to stop container:")
            print(result.stderr)
            return {
                "success": False,
                "error": "Failed to stop container",
                "details": result.stderr
            }

if __name__ == "__main__":
    # Example usage
    result = run_ridges_sandbox(
        repo_path="./agents",
        agent_path="./agents/agent01.py",
        problem_statement="Fix the bug in the code",
        sandbox_proxy_url="http://74.82.63.163:9001",  # Optional
        timeout=1500
    )
    print(json.dumps(result, indent=2))

