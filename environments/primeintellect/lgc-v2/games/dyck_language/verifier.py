"""
Dyck Language Verifier

Verifies bracket matching completion answers.
"""

from base.data import Data
from base.verifier import Verifier


class DyckLanguageVerifier(Verifier):
    """Verify Dyck language completion answers"""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct and minimal

        Args:
            data: Game data containing question sequence
            test_solution: Model's response

        Returns:
            bool: True if correct and minimal, False otherwise
        """
        # Extract user answer
        from .generator import DyckLanguageGenerator

        generator = DyckLanguageGenerator()
        user_answer = generator.extract_answer(test_solution)

        # Get question sequence from metadata
        question_sequence = data.metadata.get("question_sequence", "")

        if not question_sequence:
            return False

        # Combine question + user answer
        full_sequence = question_sequence + user_answer

        # 1. Check if full sequence is valid Dyck language
        if not self._is_valid_dyck(full_sequence):
            return False

        # 2. Check if user answer is minimal (only necessary closing brackets)
        required_closing = self._calculate_required_closing(question_sequence)

        return user_answer == required_closing

    def _is_valid_dyck(self, sequence: str) -> bool:
        """Check if a sequence is valid Dyck language"""
        stack = []
        bracket_pairs = {
            ")": "(", "]": "[", "}": "{", ">": "<",
            "⟩": "⟨", "⟧": "⟦", "⦄": "⦃", "⦆": "⦅"
        }

        for char in sequence:
            if char in "([{<⟨⟦⦃⦅":
                stack.append(char)
            elif char in ")]}>⟩⟧⦄⦆":
                if not stack or stack[-1] != bracket_pairs[char]:
                    return False
                stack.pop()

        return len(stack) == 0

    def _calculate_required_closing(self, prefix: str) -> str:
        """Calculate minimal closing brackets needed for prefix"""
        stack = []
        bracket_pairs = {
            ")": "(", "]": "[", "}": "{", ">": "<",
            "⟩": "⟨", "⟧": "⟦", "⦄": "⦃", "⦆": "⦅"
        }

        # Build stack from prefix
        for char in prefix:
            if char in "([{<⟨⟦⦃⦅":
                stack.append(char)
            elif char in ")]}>⟩⟧⦄⦆":
                if stack and stack[-1] == bracket_pairs[char]:
                    stack.pop()

        # Generate required closing brackets
        required_closing = []
        while stack:
            char = stack.pop()
            for close, open_br in bracket_pairs.items():
                if open_br == char:
                    required_closing.append(close)
                    break

        return "".join(required_closing)
