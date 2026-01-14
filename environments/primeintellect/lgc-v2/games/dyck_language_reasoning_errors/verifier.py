"""
Dyck Language Reasoning Errors Verifier

Verifies that the model correctly identifies error positions in reasoning steps.
"""

import re
from typing import Set
from base.verifier import Verifier
from base.data import Data


class DyckLanguageReasoningErrorsVerifier(Verifier):
    """Verifier for Dyck language reasoning error identification tasks"""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the model's answer matches the expected error indices

        Args:
            data: Game data containing the correct answer
            test_solution: Model's response

        Returns:
            bool: True if the answer is correct
        """
        # Get expected error indices
        expected_answer = data.answer.strip()
        expected_indices = self._parse_indices(expected_answer)

        # Parse model's answer
        model_answer = self._extract_answer(test_solution)
        model_indices = self._parse_indices(model_answer)

        # Compare sets
        return expected_indices == model_indices

    def _parse_indices(self, answer: str) -> Set[int]:
        """Parse comma-separated indices into a set of integers"""
        if not answer or answer.strip() == "":
            return set()

        # Extract all numbers
        numbers = re.findall(r'\d+', answer)
        return set(int(n) for n in numbers)

    def _extract_answer(self, test_solution: str) -> str:
        """Extract error indices from model response"""
        if not test_solution:
            return ""

        solution = test_solution.strip()

        # Check for empty/no error indicators
        no_error_indicators = [
            "", "none", "no errors", "no error", "empty", '""', "''",
            "no mistakes", "correct", "all correct", "no issues"
        ]
        if solution.lower() in no_error_indicators:
            return ""

        # If the answer is just comma-separated numbers
        if re.match(r'^[0-9,\s，]+$', solution):
            return solution.replace(" ", "").replace("，", ",")

        # Look for answer patterns
        # Pattern 1: "Answer: X,Y,Z" or "answer: X,Y,Z"
        answer_pattern = r'[Aa]nswer[:\s]+([0-9,，\s]+)'
        match = re.search(answer_pattern, solution)
        if match:
            return match.group(1).replace(" ", "").replace("，", ",")

        # Pattern 2: "errors in thoughts X, Y, Z" or similar
        error_pattern = r'(?:errors?|mistakes?|wrong|incorrect).*?([0-9]+(?:[,，\s]+[0-9]+)*)'
        match = re.search(error_pattern, solution, re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "").replace("，", ",")

        # Pattern 3: Find comma-separated number list
        num_list_pattern = r'([0-9]+(?:[,，\s]+[0-9]+)+)'
        match = re.search(num_list_pattern, solution)
        if match:
            answer = match.group(1).strip()
            return re.sub(r'[,，\s]+', ',', answer)

        # Pattern 4: Single number at the end or standalone
        # Look for numbers that appear to be answers
        lines = solution.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[0-9,，\s]+$', line):
                return line.replace(" ", "").replace("，", ",")

        # Pattern 5: Extract all thought numbers mentioned
        thought_pattern = r'[Tt]hought\s+(\d+)'
        thought_matches = re.findall(thought_pattern, solution)
        if thought_matches:
            return ','.join(thought_matches)

        # Pattern 6: Just find any numbers in the response
        all_nums = re.findall(r'(?<!\d)([1-9]\d*)(?!\d)', solution)
        if all_nums:
            # Filter out numbers that are too large (probably not thought indices)
            valid_nums = [n for n in all_nums if int(n) <= 50]
            if valid_nums:
                unique_nums = sorted(set(map(int, valid_nums)))
                return ','.join(map(str, unique_nums))

        return ""
