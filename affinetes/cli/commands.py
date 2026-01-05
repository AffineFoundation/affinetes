"""CLI command implementations"""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ..api import load_env, build_image_from_env, get_environment
from ..utils.logger import logger
from .templates import (
    ACTOR_ENV_PY,
    BASIC_ENV_PY,
    FUNCTION_DOCKERFILE,
    FASTAPI_ENV_PY,
    FASTAPI_DOCKERFILE
)


async def run_environment(
    image: Optional[str],
    env_dir: Optional[str],
    tag: Optional[str],
    name: Optional[str],
    env_vars: Dict[str, str],
    pull: bool,
    mem_limit: Optional[str],
    no_cache: bool
) -> None:
    """Start an environment container"""
    
    try:
        # Build image from directory if env_dir is provided
        if env_dir:
            if not tag:
                # Auto-generate tag from directory name
                dir_name = env_dir.rstrip('/').split('/')[-1]
                tag = f"{dir_name}:latest"
            
            logger.info(f"Building image '{tag}' from '{env_dir}'")
            
            # Build image
            image = build_image_from_env(
                env_path=env_dir,
                image_tag=tag,
                nocache=no_cache,
                quiet=False
            )
            
            logger.info(f"Image '{image}' built successfully")
        
        # Validate image parameter
        if not image:
            logger.error("Either image or env_dir must be specified")
            return
        
        if "CHUTES_API_KEY" not in env_vars and os.environ.get("CHUTES_API_KEY"):
            env_vars["CHUTES_API_KEY"] = os.environ.get("CHUTES_API_KEY")

        # Load environment using SDK
        env = load_env(
            image=image,
            container_name=name,
            env_vars=env_vars,
            cleanup=False,
            force_recreate=True,
            pull=pull,
            mem_limit=mem_limit
        )
        
        logger.info(f"✓ Environment started: {env.name}")
        
        # Show available methods immediately
        await env.list_methods(print_info=True)
        
        print(f"\nUsage:")
        print(f"  afs call {env.name} <method> --arg key=value")
    
    except Exception as e:
        logger.error(f"Failed to start environment: {e}")
        raise


async def call_method(
    name: str,
    method: str,
    args: Dict[str, Any],
    timeout: Optional[int] = 300
) -> None:
    """Call a method on running environment"""
    
    try:
        logger.info(f"Calling {method}({args}) on {name}...")
        
        # Try to get from registry first
        env = get_environment(name)
        
        if not env or not env.is_ready():
            # Not in registry, try to connect to existing container
            logger.debug(f"Environment '{name}' not in registry, connecting to container...")
            try:
                env = load_env(
                    container_name=name,
                    cleanup=False,
                    connect_only=True
                )
                logger.debug(f"Successfully connected to container '{name}'")
            except Exception as e:
                logger.error(
                    f"Failed to connect to container '{name}': {e}\n"
                    f"Please ensure the container is running with: docker ps"
                )
                return
        
        # Call method using SDK's dynamic dispatch
        method_func = getattr(env, method)
        result = await method_func(_timeout=timeout, **args)
        
        logger.info("✓ Method completed successfully")
        
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)
    
    except asyncio.TimeoutError:
        logger.error(f"Method call timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"Failed to call method: {e}")
        raise


async def build_and_push(
    env_dir: str,
    tag: str,
    push: bool,
    registry: Optional[str],
    no_cache: bool,
    quiet: bool,
    build_args: Optional[Dict[str, str]] = None
) -> None:
    """Build environment image and optionally push to registry"""
    
    try:
        env_path = Path(env_dir).resolve()
        
        # Validate environment directory
        if not env_path.exists():
            logger.error(f"Environment directory not found: {env_dir}")
            return
        
        if not (env_path / "env.py").exists():
            logger.error(f"Missing env.py in {env_dir}")
            return
        
        if not (env_path / "Dockerfile").exists():
            logger.error(f"Missing Dockerfile in {env_dir}")
            return
        
        logger.info(f"Building image '{tag}' from '{env_dir}'")
        
        # Build image
        final_tag = build_image_from_env(
            env_path=str(env_path),
            image_tag=tag,
            nocache=no_cache,
            quiet=quiet,
            buildargs=build_args,
            push=push,
            registry=registry
        )
        
        if push:
            logger.info(f"✓ Image built and pushed successfully: {final_tag}")
            logger.info(f"\nTo use this image:")
            logger.info(f"  afs run {final_tag}")
        else:
            logger.info(f"✓ Image built successfully: {final_tag}")
            logger.info(f"\nTo push to registry:")
            logger.info(f"  afs build {env_dir} --tag {tag} --push --registry <registry-url>")
            logger.info(f"\nTo run locally:")
            logger.info(f"  afs run {tag}")
    
    except Exception as e:
        logger.error(f"Failed to build image: {e}")
        raise


def init_environment(
    name: str,
    env_type: str,
    template: str
) -> None:
    """Initialize a new environment directory with template files"""
    
    try:
        env_path = Path(name)
        
        # Check if directory already exists
        if env_path.exists():
            logger.error(f"Directory '{name}' already exists")
            return
        
        # Create directory
        env_path.mkdir(parents=True)
        logger.info(f"Created directory: {name}/")
        
        # Generate files based on template
        if template == 'basic' or (template == 'actor' and env_type == 'function'):
            _create_function_based_env(env_path, use_actor=(template == 'actor'))
        elif template == 'fastapi' or env_type == 'http':
            _create_http_based_env(env_path)
        else:
            _create_function_based_env(env_path, use_actor=False)
        
        logger.info(f"✓ Environment '{name}' initialized successfully")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Build image:")
        logger.info(f"     afs build {name} --tag {name}:v1")
        logger.info(f"")
        logger.info(f"  2. Run environment:")
        logger.info(f"     afs run {name}:v1 --name {name}")
        logger.info(f"")
        logger.info(f"  3. Call methods:")
        logger.info(f"     afs call {name} add --arg a=10.5 --arg b=20.3")
        logger.info(f"     afs call {name} multiply --arg a=3.5 --arg b=4.2")
    
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise


def _create_function_based_env(env_path: Path, use_actor: bool = False) -> None:
    """Create function-based environment files"""
    
    # env.py
    env_py_content = ACTOR_ENV_PY if use_actor else BASIC_ENV_PY
    (env_path / "env.py").write_text(env_py_content)
    logger.info(f"  Created: env.py (function-based)")
    
    # Dockerfile
    (env_path / "Dockerfile").write_text(FUNCTION_DOCKERFILE)
    logger.info(f"  Created: Dockerfile")


def _create_http_based_env(env_path: Path) -> None:
    """Create HTTP-based environment files with FastAPI"""

    # env.py
    (env_path / "env.py").write_text(FASTAPI_ENV_PY)
    logger.info(f"  Created: env.py (HTTP-based)")

    # Dockerfile
    (env_path / "Dockerfile").write_text(FASTAPI_DOCKERFILE)
    logger.info(f"  Created: Dockerfile")


def _generate_seed(env_name: str, task_id: int) -> int:
    """Generate deterministic seed from env_name and task_id

    Uses the same algorithm as affine system:
    SHA256(env_name:task_id) % 2^32
    """
    seed_string = f"{env_name}:{task_id}"
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder='big') % (2**32)


async def test_environment(
    env_dir: str,
    num_tests: int,
    task_id_start: int,
    task_id_end: Optional[int],
    output_dir: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    timeout: int
) -> None:
    """Validate environment seed consistency and generate test rollouts"""

    try:
        env_path = Path(env_dir).resolve()

        # Validate environment directory
        if not env_path.exists():
            logger.error(f"Environment directory not found: {env_dir}")
            return

        if not (env_path / "env.py").exists():
            logger.error(f"Missing env.py in {env_dir}")
            return

        # Build and run environment in container
        logger.info("Building environment image...")

        # Generate image tag
        dir_name = env_path.name
        image_tag = f"affinetes-validate-{dir_name}:latest"

        try:
            from ..api import build_image_from_env, load_env

            # Build image
            build_image_from_env(
                env_path=str(env_path),
                image_tag=image_tag,
                nocache=False,
                quiet=True
            )

            logger.info(f"✓ Image built: {image_tag}")

        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            return

        # Prepare environment variables
        env_vars = {}
        if api_key:
            env_vars["CHUTES_API_KEY"] = api_key
        elif os.getenv("CHUTES_API_TOKEN"):
            env_vars["CHUTES_API_KEY"] = os.getenv("CHUTES_API_TOKEN")
        elif os.getenv("CHUTES_API_KEY"):
            env_vars["CHUTES_API_KEY"] = os.getenv("CHUTES_API_KEY")

        if os.getenv("MINER_SLUG"):
            env_vars["MINER_SLUG"] = os.getenv("MINER_SLUG")

        # Start environment
        logger.info("Starting environment container...")
        try:
            env = load_env(
                image=image_tag,
                container_name=f"affinetes-validate-{dir_name}",
                env_vars=env_vars,
                cleanup=False,
                force_recreate=True
            )

            logger.info(f"✓ Environment started: {env.name}")

        except Exception as e:
            logger.error(f"Failed to start environment: {e}")
            return

        logger.info("="*80)
        logger.info("Environment Validation Suite")
        logger.info("="*80)

        results = []

        # Determine task_id range
        if task_id_end is None:
            task_id_end = task_id_start + num_tests - 1

        # Calculate actual number of tests based on range
        actual_num_tests = task_id_end - task_id_start + 1

        if actual_num_tests != num_tests:
            logger.info(f"Note: task_id range [{task_id_start}, {task_id_end}] = {actual_num_tests} tests (overriding --num-tests={num_tests})")
            num_tests = actual_num_tests

        # Test: Generate tests with seed consistency validation
        logger.info(f"\nRunning {num_tests} tests with task_id range [{task_id_start}, {task_id_end}]")
        logger.info("Each test runs twice with different seeds to validate that problem generation only depends on task_id")
        logger.info("-"*80)

        try:
            # Determine base_url
            use_base_url = base_url
            if not use_base_url:
                miner_slug = os.getenv("MINER_SLUG")
                if miner_slug:
                    use_base_url = f"https://{miner_slug}.chutes.ai/v1"
                else:
                    use_base_url = "https://llm.chutes.ai/v1"

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            rollouts = []
            success_count = 0
            seed_consistency_failures = 0
            all_prompts = {}  # Track all prompts to verify diversity

            # Get environment name from path
            env_name = env_path.name

            for i, task_id in enumerate(range(task_id_start, task_id_end + 1)):

                try:
                    # Generate two different seeds for same task_id
                    seed1 = _generate_seed(env_name, task_id)
                    seed2 = _generate_seed(env_name, task_id + 100000)  # Different seed for same task_id

                    # First evaluation with seed1
                    result1 = await env.evaluate(
                        task_id=task_id,
                        seed=seed1,
                        base_url=use_base_url,
                        temperature=temperature,
                        timeout=timeout,
                        _timeout=timeout + 10
                    )

                    # Second evaluation with same task_id but different seed
                    result2 = await env.evaluate(
                        task_id=task_id,
                        seed=seed2,
                        base_url=use_base_url,
                        temperature=temperature,
                        timeout=timeout,
                        _timeout=timeout + 10
                    )

                    # Check seed consistency (same task_id should generate same question regardless of seed)
                    conv1 = result1.get("extra", {}).get("conversation", [])
                    conv2 = result2.get("extra", {}).get("conversation", [])

                    prompt1 = conv1[0].get("content", "") if len(conv1) > 0 else ""
                    prompt2 = conv2[0].get("content", "") if len(conv2) > 0 else ""

                    seed_consistent = prompt1 == prompt2

                    if not seed_consistent:
                        seed_consistency_failures += 1
                        logger.warning(f"Test {task_id}: Same task_id generated different prompts with different seeds! (seed1={seed1}, seed2={seed2})")

                    # Track prompt for diversity check
                    all_prompts[task_id] = prompt1

                    # Use first result for rollout, but add seed consistency info
                    result = result1
                    result["seed_consistent"] = seed_consistent
                    result["extra"]["seed1"] = seed1
                    result["extra"]["seed2"] = seed2
                    result["extra"]["second_run"] = {
                        "score": result2.get("score"),
                        "success": result2.get("success")
                    }

                    if result.get("success"):
                        success_count += 1

                    rollouts.append(result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i + 1}/{num_tests} tests completed")

                except Exception as e:
                    logger.warning(f"Error on test {i + 1}: {e}")
                    rollouts.append({
                        "task_id": task_id,
                        "error": str(e),
                        "success": False,
                        "seed_consistent": False
                    })
                    seed_consistency_failures += 1

            # Check seed diversity (different seeds should generate different questions)
            unique_prompts = len(set(all_prompts.values()))
            total_prompts = len(all_prompts)
            seed_diversity_rate = unique_prompts / total_prompts if total_prompts > 0 else 0

            # Save individual rollouts with task_id in filename
            for rollout in rollouts:
                # Get task_id from top level or from extra
                task_id = rollout.get("task_id") or rollout.get("extra", {}).get("task_id", 0)
                output_file = output_path / f"test_task{task_id:05d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(rollout, f, indent=2, ensure_ascii=False)

            # Save summary
            summary = {
                "total_tests": num_tests,
                "task_id_range": {
                    "start": task_id_start,
                    "end": task_id_end
                },
                "success_count": success_count,
                "success_rate": success_count / num_tests if num_tests > 0 else 0,
                "seed_consistency_failures": seed_consistency_failures,
                "seed_consistency_rate": (num_tests - seed_consistency_failures) / num_tests if num_tests > 0 else 0,
                "seed_diversity": {
                    "unique_prompts": unique_prompts,
                    "total_prompts": total_prompts,
                    "diversity_rate": seed_diversity_rate
                },
                "rollouts": rollouts
            }

            summary_file = output_path / "summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"\n✓ Completed {num_tests} tests")
            logger.info(f"Output directory: {output_dir}/")
            logger.info(f"Success rate: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")
            logger.info(f"Task-ID consistency: {num_tests - seed_consistency_failures}/{num_tests} ({(num_tests - seed_consistency_failures)/num_tests*100:.1f}%)")
            logger.info(f"Task-ID diversity: {unique_prompts}/{total_prompts} unique questions ({seed_diversity_rate*100:.1f}%)")

            if seed_consistency_failures > 0:
                logger.warning(f"\n⚠️  {seed_consistency_failures} task_ids generated different problems (not solely depending on task_id)!")

            if seed_diversity_rate < 1.0:
                duplicate_count = total_prompts - unique_prompts
                logger.warning(f"\n⚠️  {duplicate_count} duplicate questions detected (different task_ids generated same question)!")

            # Pass validation if both consistency and diversity are perfect
            validation_passed = (seed_consistency_failures == 0 and seed_diversity_rate == 1.0)
            results.append(validation_passed)

        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results.append(False)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("Validation Summary")
        logger.info("="*80)

        if all(results):
            logger.info("✓ All validations passed!")
        else:
            logger.error("✗ Validation failed - problem generation does not solely depend on task_id")

    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        raise