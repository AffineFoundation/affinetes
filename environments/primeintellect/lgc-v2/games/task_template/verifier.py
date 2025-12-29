"""
Task Template - Verifier

TODO: Replace with your task name and description
"""

from base.data import Data
from base.verifier import Verifier


class TaskTemplateVerifier(Verifier):
    """
    TODO: Replace 'TaskTemplate' with your task name

    Example: SimpleMathVerifier, SudokuVerifier, etc.
    """

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct

        Args:
            data: Game data containing correct answer
            test_solution: Model's response

        Returns:
            bool: True if correct, False otherwise
        """
        # TODO: Implement verification logic

        # Get correct answer
        correct_answer = data.answer

        # Extract user answer (using generator's extract_answer)
        from .generator import TaskTemplateGenerator

        generator = TaskTemplateGenerator()
        user_answer = generator.extract_answer(test_solution)

        # Compare
        # TODO: Implement comparison logic

        # Example: Exact match
        return user_answer == correct_answer

        # Example: Case-insensitive match
        # return user_answer.lower() == correct_answer.lower()

        # Example: Numerical comparison
        # try:
        #     return float(user_answer) == float(correct_answer)
        # except:
        #     return False

        # Example: Parse and compare structures
        # import json
        # try:
        #     user_data = json.loads(user_answer)
        #     correct_data = json.loads(correct_answer)
        #     return user_data == correct_data
        # except:
        #     return False
