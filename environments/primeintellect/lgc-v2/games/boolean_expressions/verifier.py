"""
Boolean Expressions Task - Verifier

Verifies model responses for boolean expression evaluation problems.
"""

import re
from base.data import Data
from base.verifier import Verifier


class BooleanExpressionsVerifier(Verifier):
    """Verify boolean expression evaluation answers."""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct.

        Args:
            data: Game data containing correct answer
            test_solution: Model's response

        Returns:
            bool: True if correct, False otherwise
        """
        try:
            # Extract answer from response
            from .generator import BooleanExpressionsGenerator

            generator = BooleanExpressionsGenerator()
            test_answer = generator.extract_answer(test_solution)

            if test_answer is None:
                return False

            # Extract letters from both answers
            test_letters = re.findall(r"[a-zA-Z]", test_answer)
            ground_truth_letters = re.findall(r"[a-zA-Z]", data.answer)

            # Normalize to lowercase
            test_set = set(letter.lower() for letter in test_letters)
            ground_truth_set = set(letter.lower() for letter in ground_truth_letters)

            # Compare sets
            return test_set == ground_truth_set

        except Exception:
            return False
