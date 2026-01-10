"""
Cryptarithm - Verifier

Verify cryptarithmetic puzzle solutions
"""

import re
from typing import Dict, List, Optional

from base.data import Data
from base.verifier import Verifier


class CryptarithmVerifier(Verifier):
    """Verify cryptarithmetic puzzle solutions by validating the equation"""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct by checking:
        1. Extract numeric equation from model output
        2. Verify the equation is mathematically correct
        3. Verify the digit mapping satisfies all constraints

        Args:
            data: Game data containing question and metadata
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

            # Parse the numeric equation
            parsed = self._parse_equation(test_answer)
            if not parsed:
                return False

            numbers, operators = parsed

            # Verify equation is mathematically correct
            if not self._verify_equation(numbers, operators, generator):
                return False

            # Extract letter words and operators from original question
            letter_info = self._extract_letter_equation(data.question)
            if not letter_info:
                return False

            letter_words, letter_operators = letter_info

            # Verify structure matches (same number of operators and operands)
            if len(numbers) != len(letter_words) or operators != letter_operators:
                return False

            # Build and verify digit mapping
            digit_map = self._build_digit_mapping(letter_words, numbers)
            if not digit_map:
                return False

            # Verify mapping constraints (no leading zeros, unique digits)
            if not self._verify_mapping_constraints(letter_words, numbers, digit_map):
                return False

            return True

        except Exception:
            return False

    def _parse_equation(self, equation: str) -> Optional[tuple[List[int], List[str]]]:
        """Parse numeric equation into numbers and operators"""
        # Normalize spaces
        equation = equation.strip().replace(" ", "")

        # Match pattern: number (op number)* = result
        pattern = r"^(\d+)([\+\-\*]\d+)*=(\d+)$"
        if not re.match(pattern, equation):
            return None

        # Split by equals sign
        parts = equation.split("=")
        if len(parts) != 2:
            return None

        left_side = parts[0]
        result = int(parts[1])

        # Extract numbers and operators from left side
        numbers = []
        operators = []

        # Parse left side
        current_num = ""
        for char in left_side:
            if char.isdigit():
                current_num += char
            elif char in ["+", "-", "*"]:
                if current_num:
                    numbers.append(int(current_num))
                    current_num = ""
                operators.append(char)
            else:
                return None

        # Add last number
        if current_num:
            numbers.append(int(current_num))

        # Add result
        numbers.append(result)

        return numbers, operators

    def _verify_equation(
        self, numbers: List[int], operators: List[str], generator
    ) -> bool:
        """
        Verify equation is mathematically correct

        Args:
            numbers: All numbers including result (e.g., [46, 87, 133])
            operators: Operators (e.g., ['+'])
        """
        # numbers should contain operands + result
        # So len(numbers) should be len(operators) + 2
        if len(numbers) < 2:
            return False

        try:
            # Split into operands and result
            operands = numbers[:-1]
            expected_result = numbers[-1]

            # operands should have len(operators) + 1 elements
            if len(operands) != len(operators) + 1:
                return False

            # Calculate using generator's method (handles operator precedence)
            calculated_result = generator._calculate_equation(operands, operators)
            return calculated_result == expected_result
        except Exception:
            return False

    def _extract_letter_equation(self, question: str) -> Optional[tuple[List[str], List[str]]]:
        """Extract letter words and operators from question"""
        # Find equation pattern in question
        # Match uppercase letter words separated by operators
        pattern = r"([A-Z]+)\s*([\+\-\*])\s*([A-Z]+(?:\s*[\+\-\*]\s*[A-Z]+)*)\s*=\s*([A-Z]+)"

        match = re.search(pattern, question)
        if not match:
            return None

        # Parse match
        first_word = match.group(1)
        first_op = match.group(2)
        middle_part = match.group(3)
        result_word = match.group(4)

        # Build letter words list
        letter_words = [first_word]
        operators = [first_op]

        # Parse middle part for additional operands and operators
        parts = re.split(r"\s*([\+\-\*])\s*", middle_part)
        for i, part in enumerate(parts):
            if part.strip():
                if part in ["+", "-", "*"]:
                    operators.append(part)
                else:
                    letter_words.append(part.strip())

        # Add result word
        letter_words.append(result_word)

        return letter_words, operators

    def _build_digit_mapping(
        self, letter_words: List[str], numbers: List[int]
    ) -> Optional[Dict[str, int]]:
        """Build letter to digit mapping from letter words and numbers"""
        if len(letter_words) != len(numbers):
            return None

        digit_map = {}

        for word, num in zip(letter_words, numbers):
            num_str = str(num)

            if len(word) != len(num_str):
                return None

            for letter, digit_char in zip(word, num_str):
                digit = int(digit_char)

                if letter in digit_map:
                    # Check consistency
                    if digit_map[letter] != digit:
                        return None
                else:
                    digit_map[letter] = digit

        return digit_map

    def _verify_mapping_constraints(
        self, letter_words: List[str], numbers: List[int], digit_map: Dict[str, int]
    ) -> bool:
        """Verify mapping satisfies all constraints"""
        # Check no leading zeros
        for word, num in zip(letter_words, numbers):
            if len(word) > 1:
                first_letter = word[0]
                if digit_map[first_letter] == 0:
                    return False

        # Check unique digits (each letter maps to different digit)
        used_digits = set()
        for letter, digit in digit_map.items():
            if digit in used_digits:
                return False
            used_digits.add(digit)

        return True
