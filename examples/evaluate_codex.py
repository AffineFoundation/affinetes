#!/usr/bin/env python3
"""
Example script for evaluating models on the Codex Django Bug-Fix environment.

This environment challenges models to fix bugs in Django that were
introduced by OpenAI Codex CLI.

Usage:
    python examples/evaluate_codex.py --model deepseek-ai/DeepSeek-V3

Environment Variables:
    CHUTES_API_KEY: API key for Chutes LLM service
"""

import argparse
import asyncio
import json
import os
import sys

# Add the affinetes package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_evaluation(
    model: str,
    base_url: str,
    num_tasks: int,
    temperature: float,
    seed: int = None,
):
    """Run evaluation on the Codex Django bug-fix environment"""
    
    # Import here to ensure path is set up
    # Note: In practice, you would run this inside the Docker container
    # For local testing, we import the Actor directly
    try:
        from environments.codex.env import Actor
    except ImportError:
        print("Error: Cannot import Actor. Run this inside the Codex Docker container.")
        print("Build and run with:")
        print("  docker build -t codex-env environments/codex/")
        print("  docker run -e CHUTES_API_KEY=$CHUTES_API_KEY codex-env python /app/examples/evaluate_codex.py")
        return
    
    api_key = os.environ.get("CHUTES_API_KEY")
    if not api_key:
        print("Error: CHUTES_API_KEY environment variable is required")
        sys.exit(1)
    
    actor = Actor(api_key=api_key)
    
    results = []
    total_score = 0.0
    
    print(f"\n{'='*60}")
    print(f"Codex Django Bug-Fix Evaluation")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Tasks: {num_tasks}")
    print(f"{'='*60}\n")
    
    for i in range(num_tasks):
        task_id = i if seed is None else seed + i
        
        print(f"Task {i+1}/{num_tasks} (task_id={task_id})...")
        
        result = await actor.evaluate(
            model=model,
            base_url=base_url,
            temperature=temperature,
            seed=seed,
            task_id=task_id,
        )
        
        results.append(result)
        total_score += result["score"]
        
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  {status} - {result.get('extra', {}).get('file_path', 'unknown')}")
        print(f"  Time: {result['time_taken']:.2f}s")
        
        if result.get("error"):
            print(f"  Error: {result['error'][:100]}...")
        
        print()
    
    # Summary
    avg_score = total_score / num_tasks if num_tasks > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tasks: {num_tasks}")
    print(f"Passed: {int(total_score)}")
    print(f"Failed: {num_tasks - int(total_score)}")
    print(f"Average Score: {avg_score:.2%}")
    print(f"{'='*60}\n")
    
    # Save results to file
    output_file = f"codex_results_{model.replace('/', '_')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": model,
            "base_url": base_url,
            "num_tasks": num_tasks,
            "temperature": temperature,
            "average_score": avg_score,
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on Codex Django Bug-Fix environment"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V3",
        help="Model to evaluate (default: deepseek-ai/DeepSeek-V3)"
    )
    parser.add_argument(
        "--base-url",
        default="https://llm.chutes.ai/v1",
        help="LLM API base URL (default: https://llm.chutes.ai/v1)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="Number of tasks to run (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(
        model=args.model,
        base_url=args.base_url,
        num_tasks=args.num_tasks,
        temperature=args.temperature,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()

