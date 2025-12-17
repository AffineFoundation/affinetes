"""
SWE-bench Pro Environment Wrapper

This wrapper implements the complete evaluation flow:
1. Patch Generation: Uses mini-swe-agent to generate patches via LLM in Docker
2. Patch Verification: Uses exact logic from swe_bench_pro_eval.py to verify patches

The evaluate() function handles both stages and returns standardized results.
"""

import os
import sys
import time
import asyncio
import tempfile
import subprocess
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset

# Add mini-swe-agent to path
MINI_SWE_AGENT_PATH = "/app/mini-swe-agent/src"
if MINI_SWE_AGENT_PATH not in sys.path:
    sys.path.insert(0, MINI_SWE_AGENT_PATH)

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def get_dockerhub_image_uri(uid: str, dockerhub_username: str, repo_name: str) -> str:
    """Generate Docker Hub image URI matching SWE-bench_Pro-os naming scheme"""
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")
    
    # Special case handling
    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = 'element-web'
    elif 'element-hq' in repo_name.lower() and 'element-web' in repo_name.lower():
        repo_name_only = 'element'
        if hsh.endswith('-vnan'):
            hsh = hsh[:-5]
    elif hsh.endswith('-vnan'):
        hsh = hsh[:-5]
    
    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    
    return f"{dockerhub_username}/sweap-images:{tag}"


def load_base_docker(instance_id: str, dockerfiles_dir: str = "/app/dockerfiles") -> str:
    """Load base Dockerfile content"""
    dockerfile_path = f"{dockerfiles_dir}/base_dockerfile/{instance_id}/Dockerfile"
    with open(dockerfile_path) as fp:
        return fp.read()


def load_instance_docker(instance_id: str, dockerfiles_dir: str = "/app/dockerfiles") -> str:
    """Load instance-specific Dockerfile content"""
    dockerfile_path = f"{dockerfiles_dir}/instance_dockerfile/{instance_id}/Dockerfile"
    with open(dockerfile_path) as fp:
        return fp.read()


def extract_env_commands(base_dockerfile: str, instance_dockerfile: str) -> str:
    """Extract ENV commands from Dockerfiles and convert to export statements"""
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)
    
    return "\n".join(env_cmds)


def create_entryscript(instance: dict, dockerfiles_dir: str = "/app/dockerfiles") -> str:
    """Create entryscript for patch verification (from swe_bench_pro_eval.py)"""
    instance_id = instance["instance_id"]
    base_commit = instance.get("base_commit", "")
    
    before_repo_set_cmd = instance.get("before_repo_set_cmd", "").strip()
    if before_repo_set_cmd:
        before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1]
    
    selected_test_files = instance.get("selected_test_files_to_run", "[]")
    if isinstance(selected_test_files, str):
        try:
            selected_test_files = eval(selected_test_files)
        except:
            selected_test_files = []
    test_files_str = ",".join(selected_test_files) if selected_test_files else ""
    
    try:
        base_dockerfile = load_base_docker(instance_id, dockerfiles_dir)
        instance_dockerfile = load_instance_docker(instance_id, dockerfiles_dir)
        env_cmds = extract_env_commands(base_dockerfile, instance_dockerfile)
    except Exception as e:
        print(f"Warning: Could not load Dockerfiles for {instance_id}: {e}")
        env_cmds = ""
    
    entryscript = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {test_files_str} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    
    return entryscript


class Actor:
    """
    SWE-bench Pro evaluation actor.
    
    Implements complete flow:
    1. Generate patch using mini-swe-agent (DefaultAgent + DockerEnvironment)
    2. Verify patch using exact swe_bench_pro_eval.py logic
    """
    
    def __init__(self, api_key: Optional[str] = None, dockerhub_username: str = "jefzda"):
        """
        Initialize actor with SWE-bench Pro dataset
        
        Args:
            api_key: API key for LLM (optional, can also use environment variables)
            dockerhub_username: Docker Hub username (default: jefzda)
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.dockerhub_username = dockerhub_username
        
        # Load SWE-bench Pro dataset
        print("Loading SWE-bench Pro dataset...")
        dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        
        # Create task_id -> instance mapping
        sorted_instances = sorted(dataset, key=lambda x: x["instance_id"])
        self.instances = {idx: inst for idx, inst in enumerate(sorted_instances)}
        
        print(f"Loaded {len(self.instances)} SWE-bench Pro instances")
        
        # Paths
        self.run_scripts_dir = "/app/run_scripts"
        self.dockerfiles_dir = "/app/dockerfiles"
    
    def _load_instance_script(self, instance_id: str, script_name: str) -> Optional[str]:
        """Load instance-specific script (run_script.sh or parser.py)"""
        script_path = Path(self.run_scripts_dir) / instance_id / script_name
        if not script_path.exists():
            return None
        with open(script_path, 'r') as f:
            return f.read()
    
    async def _generate_patch(
        self,
        instance: dict,
        model: str,
        base_url: str,
        api_key: str,
        timeout: int,
        temperature: float,
        max_iterations: int,
        cost_limit: float,
        seed: Optional[int]
    ) -> tuple[Optional[str], Optional[Dict[str, Any]], list]:
        """
        Generate patch using mini-swe-agent.
        
        Returns:
            Tuple of (patch, metadata, conversation)
        """
        try:
            instance_id = instance["instance_id"]
            task_id = instance["task_id"]
            repo = instance.get("repo", "")
            problem_statement = instance.get("problem_statement", "")
            
            # Get Docker image
            image = get_dockerhub_image_uri(instance_id, self.dockerhub_username, repo)
            print(f"Generating patch using image: {image}")
            
            # Initialize model (format for LiteLLM)
            litellm_model_name = f"openai/{model}" if not model.startswith("openai/") else model
            model_kwargs = {
                "api_base": base_url,
                "api_key": api_key,
                "temperature": temperature,
            }
            if seed is not None:
                model_kwargs["seed"] = seed
            
            model_obj = LitellmModel(
                model_name=litellm_model_name,
                model_kwargs=model_kwargs,
                cost_tracking="ignore_errors",
            )
            
            # Initialize Docker environment
            container_name = f"swebench-pro-{task_id}-{int(time.time() * 1000)}"
            env = DockerEnvironment(
                image=image,
                cwd="/app",
                timeout=timeout,
                executable="docker",
                run_args=["--rm", "--entrypoint", "", "--name", container_name],
                container_timeout=str(timeout),
            )
            
            # Load agent configuration
            config_path = Path(__file__).parent / "swebench_config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                agent_config = config.get("agent", {}).copy()
            else:
                # Default config if file doesn't exist
                agent_config = {}
            
            agent_config["step_limit"] = max_iterations
            agent_config["cost_limit"] = cost_limit
            
            # Create and run agent
            agent = DefaultAgent(model_obj, env, **agent_config)
            
            error = ""
            patch = ""
            
            try:
                loop = asyncio.get_event_loop()
                _, result = await loop.run_in_executor(
                    None,
                    agent.run,
                    problem_statement
                )
                patch = result
                
            except Exception as e:
                import traceback
                error = traceback.format_exc()
                patch = ""
                print(f"Error running agent: {type(e).__name__}: {str(e)}")
            
            finally:
                try:
                    env.cleanup()
                    print(f"Stopped container: {container_name}")
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")
            
            # Extract usage statistics from conversation
            total_completion_tokens = 0
            total_prompt_tokens = 0
            total_tokens = 0
            clean_conversation = []
            
            for msg in agent.messages:
                if isinstance(msg, dict) and "extra" in msg:
                    msg_extra = msg.get("extra", {})
                    if isinstance(msg_extra, dict):
                        msg_usage = None
                        if "response" in msg_extra and isinstance(msg_extra["response"], dict):
                            msg_usage = msg_extra["response"].get("usage")
                        elif "usage" in msg_extra:
                            msg_usage = msg_extra["usage"]
                        
                        if msg_usage and isinstance(msg_usage, dict):
                            total_completion_tokens += msg_usage.get("completion_tokens", 0)
                            total_prompt_tokens += msg_usage.get("prompt_tokens", 0)
                            total_tokens += msg_usage.get("total_tokens", 0)
                    
                    clean_msg = {k: v for k, v in msg.items() if k != "extra"}
                    clean_conversation.append(clean_msg)
                else:
                    clean_conversation.append(msg)
            
            usage = {
                "completion_tokens": total_completion_tokens,
                "prompt_tokens": total_prompt_tokens,
                "total_tokens": total_tokens
            } if total_tokens > 0 else None
            
            metadata = {
                "model_calls": agent.model.n_calls,
                "model_cost": agent.model.cost,
                "usage": usage,
            }
            
            if error:
                metadata["error"] = error
            
            return patch, metadata, clean_conversation
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error generating patch: {e}")
            print(error_trace)
            return None, {
                "error": error_trace,
                "model_calls": 0,
                "model_cost": 0,
                "usage": None
            }, []
    
    def _verify_patch(self, instance: dict, patch: str) -> tuple[float, Optional[Dict[str, Any]]]:
        """
        Verify patch using Docker container.
        Implements exact logic from swe_bench_pro_eval.py eval_with_docker()
        
        Returns:
            Tuple of (score, test_stats)
        """
        if not patch or not patch.strip():
            return 0.0, {"error": "no patch"}

        try:
            instance_id = instance["instance_id"]
            
            # Get test requirements
            fail_to_pass = instance.get("FAIL_TO_PASS", instance.get("fail_to_pass", "[]"))
            pass_to_pass = instance.get("PASS_TO_PASS", instance.get("pass_to_pass", "[]"))
            
            # Parse test lists
            if isinstance(fail_to_pass, str):
                try:
                    fail_to_pass = eval(fail_to_pass)
                except:
                    fail_to_pass = []
            if isinstance(pass_to_pass, str):
                try:
                    pass_to_pass = eval(pass_to_pass)
                except:
                    pass_to_pass = []
            
            f2p = set(fail_to_pass)
            p2p = set(pass_to_pass)
            required_tests = f2p | p2p
            
            if not required_tests:
                return 0.0, {"error": f"Warning: No required tests for {instance_id}"}
            
            # Load instance-specific scripts
            run_script = self._load_instance_script(instance_id, "run_script.sh")
            parser_script = self._load_instance_script(instance_id, "parser.py")
            
            if not run_script or not parser_script:
                return 0.0, {"error": f"✗ Missing scripts for {instance_id}"}
            
            # Create entryscript with embedded files (for DooD compatibility)
            entryscript = create_entryscript(instance, self.dockerfiles_dir)
            
            # Prepare shell script with embedded files
            # Use base64 encoding to safely embed files in shell script
            import base64
            
            patch_b64 = base64.b64encode(patch.encode('utf-8')).decode('ascii')
            run_script_b64 = base64.b64encode(run_script.encode('utf-8')).decode('ascii')
            parser_script_b64 = base64.b64encode(parser_script.encode('utf-8')).decode('ascii')
            entryscript_b64 = base64.b64encode(entryscript.encode('utf-8')).decode('ascii')
            
            full_script = f"""#!/bin/bash
# Create workspace directory
mkdir -p /workspace

# Decode and write files
echo "{patch_b64}" | base64 -d > /workspace/patch.diff
echo "{run_script_b64}" | base64 -d > /workspace/run_script.sh
echo "{parser_script_b64}" | base64 -d > /workspace/parser.py
echo "{entryscript_b64}" | base64 -d > /workspace/entryscript.sh

# Make scripts executable
chmod +x /workspace/run_script.sh
chmod +x /workspace/entryscript.sh

# Run entryscript
bash /workspace/entryscript.sh
EXIT_CODE=$?

# Always try to output the JSON if it exists, regardless of exit code
if [ -f /workspace/output.json ]; then
    echo "===SWEBENCH_OUTPUT_BEGIN==="
    cat /workspace/output.json
    echo "===SWEBENCH_OUTPUT_END==="
else
    echo "Error: output.json not found. Entryscript exited with code $EXIT_CODE" >&2
fi
"""
            
            # Get Docker image
            repo = instance.get("repo", "")
            image = get_dockerhub_image_uri(instance_id, self.dockerhub_username, repo)
            print(f"Verifying patch using image: {image}")
            
            # Pull image if needed
            try:
                subprocess.run(
                    ["docker", "pull", image],
                    check=False,
                    capture_output=True,
                    timeout=300
                )
            except Exception as e:
                print(f"Warning: Could not pull image {image}: {e}")
                return 0.0, {"error": str(e)}
            
            # Run Docker container with script passed via stdin
            cmd = [
                "docker", "run", "--rm", "-i",
                "--entrypoint", "/bin/bash",
                image
            ]
            
            result = subprocess.run(
                cmd,
                input=full_script,
                capture_output=True,
                timeout=1800,
                text=True
            )
            
            # Parse output from stdout
            stdout = result.stdout
            
            # Extract JSON between markers
            begin_marker = "===SWEBENCH_OUTPUT_BEGIN==="
            end_marker = "===SWEBENCH_OUTPUT_END==="
            
            if begin_marker not in stdout or end_marker not in stdout:
                print(f"✗ No output markers found for {instance_id}")
                print(f"stdout: {stdout[:500]}")
                print(f"stderr: {result.stderr[:500]}")
                return 0.0, {"error": result.stderr}
            
            json_start = stdout.index(begin_marker) + len(begin_marker)
            json_end = stdout.index(end_marker)
            json_str = stdout[json_start:json_end].strip()
            
            try:
                output = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"✗ Failed to parse JSON for {instance_id}: {e}")
                print(f"JSON string: {json_str[:500]}")
                return 0.0, {"error": str(e)}
            
            # Verify tests (exact logic from swe_bench_pro_eval.py lines 530-533)
            passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}
            test_result = (f2p | p2p) <= passed_tests
            
            # Calculate statistics
            required_passed = len(required_tests & passed_tests)
            total_required = len(required_tests)
            test_stats = {
                "test_result": f"{required_passed}/{total_required}",
                "required_tests": total_required,
                "passed_required_tests": required_passed,
                "fail_to_pass_tests": len(f2p),
                "pass_to_pass_tests": len(p2p),
            }
            
            if test_result:
                print(f"✓ All required tests passed for {instance_id} ({required_passed}/{total_required})")
                return 1.0, test_stats
            else:
                missing = required_tests - passed_tests
                print(f"✗ Missing tests for {instance_id}: {missing} ({required_passed}/{total_required})")
                test_stats["missing_tests"] = missing
                return 0.0, test_stats
                
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while verifying patch for {instance.get('instance_id', 'unknown')}"
            print(error_msg)
            return 0.0, {"error": error_msg}
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error verifying patch: {e}")
            print(error_msg)
            return 0.0, {"error": error_msg}
    
    async def evaluate(
        self,
        task_id: int,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        timeout: int = 1800,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_iterations: int = 30,
        cost_limit: float = 10.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete evaluation: generate patch + verify it.
        
        Args:
            task_id: Task ID (index into dataset)
            model: Model name for LiteLLM
            base_url: API base URL
            api_key: Override API key. If not provided, uses instance api_key
            timeout: Timeout for each command execution
            temperature: Model temperature
            seed: Random seed for reproducibility
            max_iterations: Maximum agent steps
            cost_limit: Maximum cost in dollars
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Result dict with score and metadata
        """
        start = time.time()
        current_api_key = api_key or self.api_key
        
        # Validate task_id
        if task_id not in self.instances:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 0-{len(self.instances)-1}")
        
        instance = self.instances[task_id]
        instance["task_id"] = task_id
        instance_id = instance["instance_id"]
        
        print(f"Evaluating SWE-bench Pro instance: {instance_id}")
        
        if not current_api_key:
            raise ValueError("API key required for patch generation")
        
        # Step 1: Generate patch
        print(f"Step 1: Generating patch for {instance_id}...")
        patch, generation_metadata, conversation = await self._generate_patch(
            instance,
            model,
            base_url,
            current_api_key,
            timeout,
            temperature,
            max_iterations,
            cost_limit,
            seed
        )
        
        # Handle patch generation failure
        if not patch:
            result = {
                "task_name": "swe-bench-pro",
                "score": 0.0,
                "success": False,
                "time_taken": time.time() - start,
                "extra": {
                    "instance_id": instance_id,
                    "task_id": task_id,
                    "patch": "",
                    "conversation": conversation,
                    "model_calls": generation_metadata.get("model_calls", 0),
                    "model_cost": generation_metadata.get("model_cost", 0),
                    "usage": generation_metadata.get("usage"),
                }
            }
            # Add error information if present
            if generation_metadata.get("error"):
                result["extra"]["error"] = generation_metadata["error"]
            return result
        
        print(f"Generated patch ({len(patch)} chars)")
        
        # Step 2: Verify patch
        print(f"Step 2: Verifying patch for {instance_id}...")
        score, test_stats = self._verify_patch(instance, patch)
        
        # Build result
        result_dict = {
            "task_name": "swe-bench-pro",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "instance_id": instance_id,
                "task_id": task_id,
                "patch": patch,
                "conversation": conversation,
                "model_calls": generation_metadata.get("model_calls", 0),
                "model_cost": generation_metadata.get("model_cost", 0),
                "usage": generation_metadata.get("usage"),
            }
        }
        
        # Add test statistics if available
        if test_stats:
            result_dict["extra"].update(test_stats)
        
        return result_dict
    
    async def local_evaluate(
        self,
        task_id: int,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        timeout: int = 1800,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_iterations: int = 30,
        cost_limit: float = 10.0,
        max_tokens: int = 2048,
        input_cost_per_million: float = 0.01,
        output_cost_per_million: float = 0.03,
    ):
        """Evaluate model on a SWE-bench Pro task using local transformers model
        
        Args:
            task_id: Numeric task ID (index into dataset)
            model: Loaded AutoModelForCausalLM instance
            tokenizer: Loaded AutoTokenizer instance
            timeout: Timeout for each command execution
            temperature: Model temperature
            seed: Random seed
            max_iterations: Maximum agent steps
            cost_limit: Maximum cost in dollars
            max_tokens: Maximum tokens to generate per response
            input_cost_per_million: Cost per million input tokens (default: $0.01)
            output_cost_per_million: Cost per million output tokens (default: $0.03)
        
        Returns:
            Result dict with score, patch, and conversation
        """
        start = time.time()
        
        # Validate task_id
        if task_id not in self.instances:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 0-{len(self.instances)-1}")
        
        instance = self.instances[task_id]
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        
        print(f"Evaluating instance: {instance_id}")
        
        # Create a custom model wrapper that matches LitellmModel interface
        class TransformersModel:
            """Local transformers model wrapper matching mini-swe-agent Model interface"""
            
            def __init__(self, model, tokenizer, temperature=0.0, seed=None, max_tokens=2048,
                        input_cost_per_million=0.01, output_cost_per_million=0.03):
                self.model = model
                self.tokenizer = tokenizer
                self.temperature = temperature
                self.seed = seed
                self.max_tokens = max_tokens
                self.cost = 0.0
                self.n_calls = 0
                self.input_cost_per_million = input_cost_per_million
                self.output_cost_per_million = output_cost_per_million
                
                # Ensure pad token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
                """Query the local model. Must match LitellmModel.query() signature.
                
                Returns dict with 'content' and 'extra' keys matching LitellmModel format.
                """
                self.n_calls += 1
                
                # Extract only role and content from messages (ignore timestamp, extra, etc.)
                clean_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
                
                # Convert messages to model input using chat template
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    prompt = self.tokenizer.apply_chat_template(
                        clean_messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    # Fallback: simple concatenation for models without chat template
                    prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in clean_messages])
                    prompt += "\n\nASSISTANT:"
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length or 4096
                ).to(self.model.device)
                
                # Set seed for reproducibility
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.seed)
                
                # Prepare generation kwargs
                gen_kwargs = {
                    "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                # Add temperature/sampling settings
                if self.temperature > 0:
                    gen_kwargs.update({
                        "temperature": self.temperature,
                        "do_sample": True,
                        "top_p": kwargs.get("top_p", 0.95),
                    })
                else:
                    gen_kwargs["do_sample"] = False
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode only the new tokens (exclude the prompt)
                prompt_length = inputs['input_ids'].shape[1]
                response_ids = outputs[0][prompt_length:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Calculate token counts
                completion_tokens = len(response_ids)
                prompt_tokens = prompt_length
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate cost based on token usage
                input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_million
                output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_million
                call_cost = input_cost + output_cost
                self.cost += call_cost
                
                # Return in the format expected by DefaultAgent
                # Must have 'content' key and 'extra' dict with usage info
                return {
                    "content": response_text,
                    "extra": {
                        "response": {
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": response_text
                                }
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                        }
                    }
                }
            
            def get_template_vars(self) -> dict:
                """Return template variables for rendering prompts.
                Must match LitellmModel.get_template_vars() signature.
                """
                return {
                    "n_model_calls": self.n_calls,
                    "model_cost": self.cost,
                    "model_name": self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else "local",
                }
        
        # Create model wrapper
        model_obj = TransformersModel(
            model, 
            tokenizer, 
            temperature=temperature, 
            seed=seed,
            max_tokens=max_tokens,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million
        )
        
        # Configure environment (SWE-bench Docker)
        container_name = f"swebench-{task_id}-{int(time.time() * 1000)}"
        env = DockerEnvironment(
            image=self._get_swebench_image_name(instance),
            timeout=timeout,
            executable="docker",
            run_args=["--rm", "--name", container_name],
            container_timeout=str(timeout),
        )
        
        # Load SWE-bench agent configuration
        config_path = Path(__file__).parent / "swebench_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override agent config limits
        agent_config = config["agent"].copy()
        agent_config["step_limit"] = max_iterations
        agent_config["cost_limit"] = cost_limit
        
        # Create and run agent
        agent = DefaultAgent(model_obj, env, **agent_config)
        
        exit_status = "unknown"
        result = ""
        patch = ""
        error = None
        
        try:
            # Run agent.run() in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            exit_status, result = await loop.run_in_executor(
                None,
                agent.run,
                problem_statement
            )
            patch = result
            
        except Exception as e:
            import traceback
            exit_status = type(e).__name__
            result = str(e)
            patch = ""
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"Error running agent: {e}")
        
        finally:
            # Clean up environment
            try:
                env.cleanup()
            except Exception:
                try:
                    if hasattr(env, 'container_id') and env.container_id:
                        subprocess.run(
                            ["docker", "rm", "-f", env.container_id],
                            capture_output=True,
                            timeout=30,
                        )
                        print(f"Force removed container {env.container_id}")
                except Exception as force_cleanup_error:
                    print(f"Error: Force cleanup failed for task {task_id}: {force_cleanup_error}")
        
        # Verify patch
        score = self._verify_patch(instance, patch) if patch else 0.0
        
        # Extract usage information and clean conversation
        total_completion_tokens = 0
        total_prompt_tokens = 0
        total_tokens = 0
        clean_conversation = []
        
        for msg in agent.messages:
            if isinstance(msg, dict) and "extra" in msg:
                msg_extra = msg.get("extra", {})
                if isinstance(msg_extra, dict):
                    msg_usage = None
                    if "response" in msg_extra and isinstance(msg_extra["response"], dict):
                        msg_usage = msg_extra["response"].get("usage")
                    elif "usage" in msg_extra:
                        msg_usage = msg_extra["usage"]
                    
                    if msg_usage and isinstance(msg_usage, dict):
                        total_completion_tokens += msg_usage.get("completion_tokens", 0)
                        total_prompt_tokens += msg_usage.get("prompt_tokens", 0)
                        total_tokens += msg_usage.get("total_tokens", 0)
                
                clean_msg = {k: v for k, v in msg.items() if k != "extra"}
                clean_conversation.append(clean_msg)
            else:
                clean_conversation.append(msg)
        
        usage = {
            "completion_tokens": total_completion_tokens,
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_tokens
        } if total_tokens > 0 else None
        
        # Return result
        result_dict = {
            "task_name": "swe-bench-pro",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "instance_id": instance_id,
                "patch": patch,
                "conversation": clean_conversation,
                "model_calls": agent.model.n_calls,
                "model_cost": agent.model.cost,
                "task_id": task_id,
                "usage": usage,
            }
        }
        
        if error:
            result_dict["extra"]["error"] = error
            result_dict["extra"]["error_type"] = exit_status
        
        return result_dict