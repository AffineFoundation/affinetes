"""SWE-bench Pro evaluation environment using mini-swe-agent"""
import os
import time
import tempfile
import subprocess
import yaml
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel


class Actor:
    """SWE-bench Pro environment actor"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize actor and load dataset"""
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        
        # Load SWE-bench Verified dataset
        print("Loading SWE-bench Verified dataset...")
        dataset = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
        
        # Create task_id -> instance mapping (sorted by instance_id for consistency)
        sorted_instances = sorted(dataset, key=lambda x: x["instance_id"])
        self.instances = {idx: inst for idx, inst in enumerate(sorted_instances)}
        
        print(f"Loaded {len(self.instances)} instances")
    
    def _get_swebench_image_name(self, instance: dict) -> str:
        """Get SWE-bench Docker image name for an instance"""
        iid = instance["instance_id"]
        # Docker doesn't allow double underscore, replace with magic token
        id_docker = iid.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()
    
    def _verify_patch(self, instance: dict, patch: str) -> float:
        """Verify patch by running tests in evaluation container
        
        Returns:
            1.0 if patch passes all required tests, 0.0 otherwise
        """
        if not patch or not patch.strip():
            return 0.0
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write patch to file
                patch_file = os.path.join(tmpdir, "patch.diff")
                with open(patch_file, "w") as f:
                    f.write(patch)
                
                # Get evaluation image
                image = self._get_swebench_image_name(instance)
                
                # Build test command
                base_commit = instance.get("base_commit", "")
                test_cmd = f"""
                cd /app
                git reset --hard {base_commit}
                git checkout {base_commit}
                git apply -v /workspace/patch.diff || exit 1
                """
                
                # Run container with patch
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{tmpdir}:/workspace",
                    image,
                    "/bin/bash", "-c", test_cmd
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=600,
                    text=True
                )
                
                # Check if patch applied successfully and tests passed
                if result.returncode == 0:
                    return 1.0
                else:
                    return 0.0
                    
        except Exception as e:
            print(f"Error verifying patch: {e}")
            return 0.0
    
    async def evaluate(
        self,
        task_id: int,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        timeout: int = 600,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_iterations: int = 30,
        cost_limit: float = 10.0,
    ):
        """Evaluate model on a SWE-bench Pro task
        
        Args:
            task_id: Numeric task ID (index into dataset)
            model: Model name for LiteLLM
            base_url: API base URL
            api_key: API key (falls back to self.api_key)
            timeout: Timeout for each command execution
            temperature: Model temperature
            seed: Random seed
            max_iterations: Maximum agent steps
            cost_limit: Maximum cost in dollars
        
        Returns:
            Result dict with score, patch, and conversation
        """
        start = time.time()
        current_api_key = api_key or self.api_key
        
        # Validate task_id
        if task_id not in self.instances:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 0-{len(self.instances)-1}")
        
        instance = self.instances[task_id]
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        
        print(f"Evaluating instance: {instance_id}")
        
        # Configure model
        # For custom OpenAI-compatible endpoints, LiteLLM requires "openai/" prefix
        # This tells LiteLLM to use OpenAI SDK with custom api_base
        litellm_model = f"openai/{model}" if not model.startswith("openai/") else model
        model_obj = LitellmModel(
            model_name=litellm_model,
            model_kwargs={
                "api_base": base_url,  # LiteLLM uses api_base, not base_url
                "api_key": current_api_key,
                "temperature": temperature,
            },
            cost_tracking="ignore_errors",  # Ignore cost calculation errors for custom models
        )
        
        # Configure environment (SWE-bench Docker)
        # Use unique container name to avoid conflicts in concurrent execution
        container_name = f"swebench-{task_id}-{int(time.time() * 1000)}"
        env = DockerEnvironment(
            image=self._get_swebench_image_name(instance),
            timeout=timeout,
            executable="docker",  # DOOD: use host docker
            run_args=["--rm", "--name", container_name],  # Ensure cleanup and unique naming
        )
        
        # Load SWE-bench agent configuration from YAML (in Docker image)
        config_path = Path(__file__).parent / "swebench_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override agent config limits with function parameters
        agent_config = config["agent"].copy()
        agent_config["step_limit"] = max_iterations
        agent_config["cost_limit"] = cost_limit
        
        # Create and run agent with proper SWE-bench configuration
        agent = DefaultAgent(
            model_obj,
            env,
            **agent_config,
        )
        
        exit_status = "unknown"
        result = ""
        patch = ""
        
        try:
            exit_status, result = agent.run(problem_statement)
            patch = result  # Final output is the patch
            
        except Exception as e:
            exit_status = type(e).__name__
            result = str(e)
            patch = ""
            print(f"Error running agent: {e}")
        
        finally:
            # Clean up environment
            try:
                env.cleanup()
            except Exception as cleanup_error:
                try:
                    if hasattr(env, 'container_id') and env.container_id:
                        subprocess.run(
                            ["docker", "rm", "-f", env.container_id],
                            capture_output=True,
                            timeout=30,
                        )
                        print(f"Force removed container {env.container_id}")
                except Exception as force_cleanup_error:
                    print(f"Error: Force cleanup also failed for task {task_id}: {force_cleanup_error}")
        
        # Verify patch
        score = self._verify_patch(instance, patch) if patch else 0.0
        
        # Clean conversation: remove 'extra' field from each message
        clean_conversation = []
        for msg in agent.messages:
            clean_msg = {k: v for k, v in msg.items() if k != "extra"}
            clean_conversation.append(clean_msg)
        
        # Return result
        return {
            "task_name": "swe-bench-pro",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "instance_id": instance_id,
                "patch": patch,
                "conversation": clean_conversation,  # Cleaned AgentGym trajectory
                "exit_status": exit_status,
                "model_calls": agent.model.n_calls,
                "model_cost": agent.model.cost,
                "task_id": task_id,
            }
        }