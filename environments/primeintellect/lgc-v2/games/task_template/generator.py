"""
Task Template - Generator

TODO: Replace with your task name and description
"""

import random
from base.data import Data


class TaskTemplateGenerator:
    """
    TODO: Replace 'TaskTemplate' with your task name

    Example: SimpleMathGenerator, SudokuGenerator, etc.
    """

    def __init__(self):
        """Initialize your generator"""
        # TODO: Add initialization code
        pass

    def generate(
        self,
        seed: int,
        # TODO: Add your custom parameters
        # difficulty: str = "easy",
        # max_value: int = 100,
    ):
        """
        Generate a single task

        Args:
            seed: Seed for reproducibility
            # TODO: Document your parameters

        Returns:
            Data: Game data object
        """
        # Create deterministic RNG
        rng = random.Random(seed)

        # Generate question
        question, answer, metadata = self._generate_one(
            rng,
            # TODO: Pass your parameters
            # difficulty=difficulty,
            # max_value=max_value,
        )

        # Create and return Data object
        return Data(
            question=question,
            answer=answer,
            metadata={
                "seed": seed,
                **metadata
            }
        )

    def _generate_one(
        self,
        rng: random.Random,
        # TODO: Add your parameters
    ):
        """
        Generate a single question

        Args:
            rng: Random number generator for deterministic generation
            # TODO: Document your parameters

        Returns:
            tuple: (question, answer, metadata)
        """
        # TODO: Implement your generation logic

        # Example structure:
        # 1. Generate random values using rng
        # value1 = rng.randint(1, 100)
        # value2 = rng.choice(['option1', 'option2'])

        # 2. Create question text
        # question = f"Your question text with {value1}"

        # 3. Calculate correct answer
        # answer = str(value1 * 2)

        # 4. Prepare metadata
        # metadata = {
        #     "value1": value1,
        #     "value2": value2,
        #     "difficulty": difficulty
        # }

        # 5. Return
        # return question, answer, metadata

        raise NotImplementedError("TODO: Implement _generate_one()")

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract answer from model response (optional but recommended)

        Args:
            test_solution: Raw model output

        Returns:
            str: Cleaned answer
        """
        # TODO: Implement answer extraction logic

        # Example: Simple cleaning
        return test_solution.strip()

        # Example: Extract numbers
        # import re
        # numbers = re.findall(r'-?\d+', test_solution)
        # return numbers[0] if numbers else ""

        # Example: Parse JSON
        # import json
        # try:
        #     data = json.loads(test_solution)
        #     return str(data.get('answer', ''))
        # except:
        #     return ""
