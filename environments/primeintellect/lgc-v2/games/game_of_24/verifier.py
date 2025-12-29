import re

from base.data import Data
from base.verifier import Verifier


class GameOf24Verifier(Verifier):
    """Verify Game of 24 solutions"""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct

        Args:
            data: Game data containing numbers and target result
            test_solution: Model's response

        Returns:
            bool: True if correct, False otherwise
        """
        try:
            # Extract answer from response
            from .generator import GameOf24Generator

            generator = GameOf24Generator()
            test_answer = generator.extract_answer(test_solution)

            if not test_answer:
                return False

            # Get metadata
            numbers = data.metadata["numbers"]
            operators = data.metadata["operators"]
            target_result = data.metadata["target"]
            is_solvable = data.metadata.get("is_solvable", True)

            # Check if model claims no solution
            if test_answer.strip() == "None":
                # Model says no solution - verify this is correct
                return not is_solvable

            # Model provided an expression - verify it
            # Extract numbers from answer
            input_numbers = [str(num) for num in numbers]
            answer_numbers_str = re.sub("[^0-9]", " ", test_answer)
            answer_numbers = [num for num in answer_numbers_str.split() if num]

            # Extract operators and check for unknown characters
            answer_wo_numbers_str = re.sub("[0-9\\s]", "", test_answer)
            unknown_chars = []
            for c in answer_wo_numbers_str:
                if c in operators:
                    continue
                if c in ["(", ")"]:
                    continue
                unknown_chars.append(c)

            if len(unknown_chars) > 0:
                return False

            # Verify all numbers are from input and used correct times
            for num in answer_numbers:
                if num not in input_numbers:
                    return False
                if answer_numbers.count(num) > input_numbers.count(num):
                    return False

            # Verify all numbers are used
            if len(answer_numbers) != len(input_numbers):
                return False

            # Evaluate expression
            evaluated = eval(test_answer)
            return abs(float(evaluated) - float(target_result)) < 1e-10

        except Exception:
            return False
