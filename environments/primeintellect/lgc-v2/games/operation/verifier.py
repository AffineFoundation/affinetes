"""
Operation Verifier

Verifies operation task answers
"""

import re
from base.verifier import Verifier
from base.data import Data


class OperationVerifier(Verifier):
    """Verifier for operation tasks"""

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        Verify if the test answer matches the ground truth

        Args:
            data: Data object containing the challenge
            test_answer: Model's answer string

        Returns:
            bool: True if correct, False otherwise
        """
        try:
            # Extract answer from model response
            extracted_answer = self.extract_answer(test_answer)
            if extracted_answer is None:
                return False

            # Parse both answers
            ground_truth = self._parse_number(data.answer)
            parsed_answer = self._parse_number(extracted_answer)

            if ground_truth is None or parsed_answer is None:
                return False

            # Compare with tolerance for floating point
            return self._are_equal(parsed_answer, ground_truth)

        except Exception:
            return False

    def extract_answer(self, answer_str: str) -> str:
        """
        Extract answer from \\boxed{} notation

        Args:
            answer_str: Answer string potentially containing \\boxed{}

        Returns:
            str: Extracted answer or None
        """
        if not answer_str:
            return None

        # Find last occurrence of \boxed{
        last_box_index = answer_str.rfind("\\boxed{")
        if last_box_index == -1:
            return None

        # Extract content between braces
        start_index = last_box_index + len("\\boxed{")
        bracket_stack = 1
        end_index = start_index

        while end_index < len(answer_str) and bracket_stack > 0:
            if answer_str[end_index] == "{":
                bracket_stack += 1
            elif answer_str[end_index] == "}":
                bracket_stack -= 1
            end_index += 1

        if bracket_stack != 0:
            return None

        latex_content = answer_str[start_index : end_index - 1].strip()
        return latex_content

    def _parse_number(self, text: str):
        """
        Parse number from text (supports integers, floats, fractions)

        Args:
            text: Text containing number

        Returns:
            float: Parsed number or None
        """
        if not text:
            return None

        text = text.strip()

        # Remove LaTeX formatting
        text = text.replace("\\", "")
        text = text.replace(",", "")  # Remove thousand separators

        # Try to parse as integer or float
        try:
            # Handle scientific notation
            if "e" in text.lower():
                return float(text)

            # Handle fractions like "1/2"
            if "/" in text:
                parts = text.split("/")
                if len(parts) == 2:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator != 0:
                        return numerator / denominator

            # Handle negative numbers
            if text.startswith("-"):
                return float(text)

            # Try simple float conversion
            return float(text)

        except (ValueError, ZeroDivisionError):
            pass

        # Try to extract first number from text
        number_pattern = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"
        match = re.search(number_pattern, text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None

    def _are_equal(self, a: float, b: float, tolerance: float = 1e-6) -> bool:
        """
        Check if two numbers are equal within tolerance

        Args:
            a: First number
            b: Second number
            tolerance: Absolute tolerance

        Returns:
            bool: True if equal within tolerance
        """
        # Handle exact integer matches
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if int(a) == a and int(b) == b:
                return int(a) == int(b)

            # Floating point comparison
            return abs(a - b) < tolerance

        return False
