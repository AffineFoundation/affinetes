import itertools
import random
import re
import uuid

from base.data import Data
from .config import derive_params_from_seed, format_question


class GameOf24Generator:
    """Generate Game of 24 challenges with seed-based determinism"""

    def __init__(self):
        pass

    def generate(self, seed: int = None, **kwargs):
        """
        Generate a single Game of 24 challenge based on seed

        All parameters are derived from seed, so config kwargs are ignored

        Args:
            seed: Random seed for deterministic generation
            **kwargs: Config parameters (ignored, for compatibility)

        Returns:
            Data object containing question, answer, and metadata
        """
        if seed is None:
            seed = random.randint(0, 99999999)

        # Derive all parameters from seed
        params = derive_params_from_seed(seed)

        # Create RNG for number generation
        rng = random.Random(seed)

        # Generate numbers using derived parameters
        numbers = [
            rng.randint(params['min_val'], params['max_val'])
            for _ in range(params['num_count'])
        ]
        numbers = sorted(numbers)

        # Find all solutions (for metadata, not filtering)
        solutions = self._find_all_solutions(
            numbers,
            params['operators'],
            params['target']
        )

        # Format question using config
        question, lang = format_question(
            numbers,
            params['operators'],
            params['target'],
            params['prompt_idx']
        )

        # Build metadata
        metadata = {
            "seed": seed,
            "trace_id": str(uuid.uuid4()),
            "numbers": numbers,
            "solutions_count": len(solutions),
            "operators": params['operators'],
            "target": params['target'],
            "num_of_numbers": params['num_count'],
            "language": lang,
            "is_solvable": len(solutions) > 0,
        }

        return Data(
            question=question,
            answer="",  # No reference answer needed
            metadata=metadata
        )

    def _find_all_solutions(self, numbers, operators, target_result):
        """Find all possible solutions for given numbers"""
        solutions = set()

        for nums in itertools.permutations(numbers):
            for ops in itertools.product(operators, repeat=len(nums) - 1):
                cur_nums = list(nums)
                cur_ops = list(ops)

                # Evaluate left to right
                while cur_ops:
                    op = cur_ops.pop(0)
                    cur_num1 = cur_nums.pop(0)
                    cur_num2 = cur_nums.pop(0)

                    try:
                        result = eval(f"{cur_num1} {op} {cur_num2}")
                        cur_nums.insert(0, result)
                    except (ZeroDivisionError, ValueError):
                        break

                # Check if result matches target
                if cur_nums and abs(cur_nums[0] - target_result) < 1e-10:
                    solutions.add(f"nums: {nums}, ops: {ops}")

        return list(solutions)

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract answer from model response

        Looks for ```python code block and extracts the expression
        Returns the extracted expression or None string
        """
        regex_pattern = "```python.*?```"
        matches = re.findall(regex_pattern, test_solution, re.DOTALL)

        if matches:
            answer = matches[-1].replace("```python", "").replace("```", "").strip()
            return answer

        return ""
