"""Code task generator and evaluator using code execution"""

import json
import logging
import os
import re
import sys
from typing import Callable

import httpx
from datasets import load_dataset
from openai import AsyncOpenAI

sys.path.insert(0, '/app')
from models import Challenge

# We set higher timeouts than default to avoid judge timeout during eval
HTTPX_TIMEOUT = httpx.Timeout(1200)
HTTPX_LIMITS = httpx.Limits(
    max_connections=8192,
    max_keepalive_connections=8192,
)

logger = logging.getLogger("i3_code")
handler = logging.StreamHandler(sys.stderr)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
logger.addHandler(handler)
logger.setLevel(os.environ.get("I3_CODE_LOG_LEVEL", "INFO"))

INSTRUCTION_PROMPT = """Write a Python function to solve the following problem. 
Put your complete solution in a markdown code block (```python ... ```).
Make sure your function handles all test cases correctly."""


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Try to find ```python blocks first
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Fall back to any ``` blocks
    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no code blocks, return the text as-is
    return text.strip()


class CodeTask:
    """Code task generator and evaluator using INTELLECT-3-RL dataset"""
    
    def __init__(
        self,
        dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
        dataset_subset: str = "code",
        dataset_split: str = "train",
        dataset_shuffle: bool = False,
        difficulty_key: str = "avg@8_qwen3_4b_instruct_2507",
        min_avg_reward: float = 0.0,
        max_avg_reward: float = 1.0,
    ):
        """
        Initialize CodeTask with dataset configuration
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_subset: Dataset subset to use
            dataset_split: Dataset split (train/test/validation)
            dataset_shuffle: Whether to shuffle the dataset
            difficulty_key: Key for filtering by difficulty
            min_avg_reward: Minimum average reward filter
            max_avg_reward: Maximum average reward filter
        """
        logger.info(f"Loading dataset: {dataset_name}/{dataset_subset} split={dataset_split}")
        
        # Load and filter dataset
        def process_example(x):
            info = json.loads(x["info"])
            # Keep tests as JSON string to avoid PyArrow type issues
            return {
                "question": INSTRUCTION_PROMPT + "\n\n" + x["question"],
                "tests": info["tests"],  # Keep as string
                "source": info.get("source", "")
            }
        
        self.dataset = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            .filter(lambda x: min_avg_reward <= x.get(difficulty_key, 0) <= max_avg_reward)
            .map(process_example)
        )
        
        if dataset_shuffle:
            self.dataset = self.dataset.shuffle(seed=42)
        
        logger.info(f"Dataset loaded: {len(self.dataset)} examples")
    
    async def generate(self, task_id: int = None) -> Challenge:
        """
        Generate a code task challenge
        
        Args:
            task_id: Optional task ID for deterministic selection.
                     If provided, used as index into dataset.
                     If not provided, random sample is selected.
        """
        if task_id is not None:
            # Use task_id as index for deterministic selection
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            # Random selection
            import random
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        return Challenge(
            env="code",
            prompt=sample["question"],
            extra={
                "tests": sample["tests"],
                "source": sample.get("source", ""),
                "task_id": task_id,
                "dataset_index": idx
            }
        )
    
    async def evaluate(self, response: str, challenge: Challenge) -> float:
        """
        Evaluate code response by running test cases
        
        Args:
            response: Model response containing code
            challenge: Original challenge with test cases
        
        Returns:
            Score between 0.0 and 1.0 (percentage of tests passed)
        """
        tests_str = challenge.extra.get("tests", "")
        if not tests_str:
            logger.warning("No tests provided")
            return 0.0
        
        # Extract code from response
        code = extract_code_from_markdown(response)
        if not code:
            logger.warning("No code found in response")
            return 0.0
        
        logger.debug(f"Extracted code:\n{code}")
        
        # Parse tests
        try:
            tests = json.loads(tests_str)
        except Exception as e:
            logger.error(f"Failed to parse tests: {e}")
            return 0.0
        
        # Extract test data
        fn_name = tests.get("fn_name", "")
        inputs = tests.get("inputs", [])
        outputs = tests.get("outputs", [])
        
        if not inputs or not outputs:
            logger.warning("No test inputs/outputs found")
            return 0.0
        
        if len(inputs) != len(outputs):
            logger.error(f"Mismatch: {len(inputs)} inputs vs {len(outputs)} outputs")
            return 0.0
        
        # Run tests
        passed = 0
        total = len(inputs)
        
        for i in range(total):
            try:
                # Create a namespace for code execution
                namespace = {}
                
                # Execute the code
                exec(code, namespace)
                
                # Get the function by name or find first callable
                func = namespace.get(fn_name)
                if func is None:
                    for name, obj in namespace.items():
                        if callable(obj) and not name.startswith('_'):
                            func = obj
                            break
                
                if func is None:
                    logger.warning(f"Test {i}: No function found in code")
                    continue
                
                # Parse input (it's a JSON string)
                test_input = json.loads(inputs[i])
                expected_output = json.loads(outputs[i])
                
                # Run the function
                if isinstance(test_input, list):
                    result = func(*test_input)
                else:
                    result = func(test_input)
                
                # Convert result to string for comparison (as outputs are strings)
                result_str = str(result)
                expected_str = str(expected_output)
                
                # Compare result with expected output
                # Try direct comparison first, then string comparison
                if result == expected_output or result_str == expected_str:
                    passed += 1
                    logger.debug(f"Test {i}: PASSED")
                else:
                    logger.debug(f"Test {i}: FAILED - expected {expected_output} ({expected_str}), got {result} ({result_str})")
                
            except Exception as e:
                logger.debug(f"Test {i}: ERROR - {e}")
                continue
        
        score = passed / total if total > 0 else 0.0
        logger.info(f"Evaluation complete: {passed}/{total} tests passed, score={score}")
        
        return score