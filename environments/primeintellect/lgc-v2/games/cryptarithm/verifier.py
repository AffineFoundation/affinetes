"""
Cryptarithm - Verifier

Verify cryptarithmetic puzzle solutions
"""

from base.data import Data
from base.verifier import Verifier


class CryptarithmVerifier(Verifier):
    """Verify cryptarithmetic puzzle solutions"""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct

        Args:
            data: Game data containing correct answer
            test_solution: Model's response

        Returns:
            bool: True if correct, False otherwise
        """
        try:
            # Extract answer from response
            from .generator import CryptarithmGenerator

            generator = CryptarithmGenerator()
            test_answer = generator.extract_answer(test_solution)

            if not test_answer:
                return False

            # Get correct answer
            correct_answer = data.answer

            # Normalize both answers (remove spaces)
            test_answer = test_answer.strip().replace(" ", "")
            correct_answer = correct_answer.strip().replace(" ", "")

            # Compare
            return test_answer == correct_answer

        except Exception:
            return False
