"""
Cryptarithm - Generator

Generate cryptarithmetic puzzles where letters represent unique digits.
Example: SEND + MORE = MONEY
"""

import operator
import random
import re
import string
import uuid
from typing import Dict, List, Tuple, Optional

from base.data import Data


class CryptarithmGenerator:
    """Generate cryptarithmetic puzzles with seed-based determinism"""

    def __init__(self):
        self.operators_map = {"+": operator.add, "-": operator.sub, "*": operator.mul}
        self.operator_symbols = {1: ["+", "-"], 2: ["+", "-"], 3: ["+", "-", "*"]}

    def generate(self, seed: int = None, **kwargs):
        """
        Generate a single cryptarithm challenge based on seed

        All parameters are derived from seed for reproducibility

        Args:
            seed: Random seed for deterministic generation
            **kwargs: Config parameters (ignored, for compatibility)

        Returns:
            Data object containing question, answer, and metadata
        """
        if seed is None:
            seed = random.randint(0, 99999999)

        rng = random.Random(seed)

        # Derive parameters from seed
        num_letter = rng.randint(4, 8)  # 4-8 unique letters
        operator_num = rng.choice([1, 2, 3])  # 1-3 operators
        operator_level = rng.choice([1, 2, 3])  # difficulty level

        # Generate valid puzzle with retries
        max_attempts = 100
        for _ in range(max_attempts):
            result = self._generate_valid_puzzle(rng, num_letter, operator_num, operator_level)
            if result:
                numbers, operators_list, letter_words, digit_map = result
                break
        else:
            # Fallback: generate simpler puzzle
            numbers, operators_list, letter_words, digit_map = self._generate_simple_puzzle(rng)

        # Generate question
        is_chinese = rng.choice([True, False])
        question = self._generate_prompt(letter_words, operators_list, is_chinese)

        # Construct answer
        answer = self._construct_answer(numbers, operators_list)

        # Build metadata
        metadata = {
            "seed": seed,
            "trace_id": str(uuid.uuid4()),
            "numbers": numbers,
            "letter_words": letter_words,
            "operators": operators_list,
            "digit_map": digit_map,
            "num_letter": num_letter,
            "operator_num": operator_num,
            "operator_level": operator_level,
            "language": "chinese" if is_chinese else "english",
        }

        return Data(question=question, answer=answer, metadata=metadata)

    def _generate_valid_puzzle(
        self, rng: random.Random, num_letter: int, operator_num: int, operator_level: int
    ) -> Optional[Tuple[List[int], List[str], List[str], Dict[str, int]]]:
        """Generate a valid cryptarithm puzzle"""
        # Sample digits to use
        digits = rng.sample(range(10), min(num_letter, 10))

        # Generate valid equation
        equation_data = self._generate_valid_equation(rng, digits, operator_num, operator_level)
        if not equation_data:
            return None

        numbers, operators_list, used_digits_list = equation_data

        # Verify we have correct number of unique digits
        all_digits_str = []
        for num in numbers:
            all_digits_str.extend(list(str(num)))

        unique_digits = set(all_digits_str)
        if len(unique_digits) != num_letter:
            return None

        # Verify unique solution
        if not self._verify_unique_solution(numbers, operators_list):
            return None

        # Convert to letters
        letter_words, digit_map = self._convert_to_letters(rng, numbers)

        return numbers, operators_list, letter_words, digit_map

    def _generate_valid_equation(
        self, rng: random.Random, digits: List[int], operator_num: int, operator_level: int
    ) -> Optional[Tuple[List[int], List[str], List[int]]]:
        """Generate valid equation with given constraints"""
        max_attempts = 50

        for _ in range(max_attempts):
            # Choose operators
            available_operators = self.operator_symbols[operator_level]
            operators_list = [rng.choice(available_operators) for _ in range(operator_num)]

            # Generate numbers (operands)
            numbers = []
            used_digits_list = []

            for i in range(operator_num + 1):
                # Determine number length (3-6 digits mostly, 1-2 occasionally)
                if rng.random() < 0.1:
                    num_length = rng.randint(1, 2)
                else:
                    num_length = rng.randint(3, 6)

                # Build number digit by digit
                number_digits = []

                # First digit cannot be 0
                non_zero_digits = [d for d in digits if d != 0]
                if not non_zero_digits:
                    continue

                first_digit = rng.choice(non_zero_digits)
                number_digits.append(first_digit)
                used_digits_list.append(first_digit)

                # Remaining digits
                for _ in range(num_length - 1):
                    digit = rng.choice(digits)
                    number_digits.append(digit)
                    used_digits_list.append(digit)

                number = int("".join(map(str, number_digits)))
                numbers.append(number)

            # Calculate result
            try:
                result = self._calculate_equation(numbers, operators_list)
            except Exception:
                continue

            if result <= 0:
                continue

            # Check if result digits are in allowed set
            result_digits = [int(d) for d in str(result)]
            if not all(d in digits for d in result_digits):
                continue

            used_digits_list.extend(result_digits)
            numbers.append(result)

            # Require at least 4 unique digits used
            if len(set(used_digits_list)) < min(len(digits), 4):
                continue

            return numbers, operators_list, used_digits_list

        return None

    def _generate_simple_puzzle(self, rng: random.Random) -> Tuple[List[int], List[str], List[str], Dict[str, int]]:
        """Generate a simple fallback puzzle (single addition)"""
        # Simple two-number addition with 4-5 unique digits
        digits = rng.sample(range(10), 5)

        # Build two 2-3 digit numbers
        num1_digits = rng.sample([d for d in digits if d != 0], 2)
        num2_digits = rng.sample([d for d in digits if d != 0], 2)

        num1 = num1_digits[0] * 10 + num1_digits[1]
        num2 = num2_digits[0] * 10 + num2_digits[1]
        result = num1 + num2

        numbers = [num1, num2, result]
        operators_list = ["+"]

        letter_words, digit_map = self._convert_to_letters(rng, numbers)

        return numbers, operators_list, letter_words, digit_map

    def _convert_to_letters(
        self, rng: random.Random, numbers: List[int]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Convert numbers to letter words"""
        # Collect all unique digits
        all_digits = set()
        for num in numbers:
            for digit in str(num):
                all_digits.add(int(digit))

        # Map to random uppercase letters
        alphabet = rng.sample(string.ascii_uppercase, len(all_digits))
        digit_to_letter = {digit: letter for digit, letter in zip(all_digits, alphabet)}

        # Convert each number to letter word
        letter_words = []
        for num in numbers:
            word = "".join(digit_to_letter[int(d)] for d in str(num))
            letter_words.append(word)

        # Create letter-to-digit map for metadata
        letter_to_digit = {letter: digit for digit, letter in digit_to_letter.items()}

        return letter_words, letter_to_digit

    def _generate_prompt(self, letter_words: List[str], operators_list: List[str], is_chinese: bool) -> str:
        """Generate question prompt"""
        if len(operators_list) == 1:
            # Single operator case
            word1, word2, word3 = letter_words[0], letter_words[1], letter_words[2]
            len1, len2, len3 = len(word1), len(word2), len(word3)
            op = operators_list[0]

            if is_chinese:
                prompts = [
                    f"已知：{word1} {op} {word2} = {word3}，每个字母代表一个数字（{word1} 是 {len1} 位数、{word2} 是 {len2} 位数、{word3} 是 {len3} 位数），字母与字母之间代表的数字不重复。若要让等式成立，求得题中的数字等式。",
                    f"有一个等式：{word1} {op} {word2} = {word3}，其中每个字母代表一个不同的数字（{word1} 是 {len1} 位数、{word2} 是 {len2} 位数、{word3} 是 {len3} 位数）。请找出每个字母代表的数字，使等式成立。",
                    f"给定一个字母等式：{word1} {op} {word2} = {word3}，每个字母表示一个数字，不同字母表示不同数字。其中{word1}是{len1}位数，{word2}是{len2}位数，{word3}是{len3}位数。请计算出满足条件的数字等式。",
                ]
                prompt = random.choice(prompts)
                prompt += " 请在回答的最后一行使用以下格式：答案是 $YOUR_ANSWER。$YOUR_ANSWER 应该是替换为数字后的等式。"
            else:
                prompts = [
                    f"Given: {word1} {op} {word2} = {word3}, where each letter represents a unique digit ({word1} is a {len1}-digit number, {word2} is a {len2}-digit number, and {word3} is a {len3}-digit number). Find the numeric equation that makes the equality valid.",
                    f"In the cryptarithm: {word1} {op} {word2} = {word3}, each letter stands for a different digit ({word1} is {len1} digits, {word2} is {len2} digits, and {word3} is {len3} digits). Determine what each letter represents to make the equation true.",
                    f"Solve the following alphametic puzzle: {word1} {op} {word2} = {word3}, where each letter represents a unique digit. Note that {word1} is a {len1}-digit number, {word2} is a {len2}-digit number, and {word3} is a {len3}-digit number.",
                ]
                prompt = random.choice(prompts)
                prompt += " Please end your response in the last line with the following format: The answer is $YOUR_ANSWER. $YOUR_ANSWER should be the equation with letters replaced by digits."
        else:
            # Multiple operators case
            equation = letter_words[0]
            for i in range(len(operators_list)):
                equation += f" {operators_list[i]} {letter_words[i + 1]}"
            equation += f" = {letter_words[-1]}"

            words_description = ""
            for word in letter_words:
                words_description += f"{word}是{len(word)}位数、"
            words_description = words_description[:-1]

            if is_chinese:
                prompts = [
                    f"已知字母等式：{equation}（其中{words_description}），每个字母代表一个数字，不同字母代表不同数字。请找出让等式成立的数字替换方案。",
                    f"解决密码算术谜题：{equation}（其中{words_description}），其中每个字母表示0-9之间的一个数字，不同字母代表不同数字。请计算出使等式成立的数字等式。",
                ]
                prompt = random.choice(prompts)
                prompt += " 请在回答的最后一行使用以下格式：答案是 $YOUR_ANSWER。$YOUR_ANSWER 应该是替换为数字后的等式。"
            else:
                prompts = [
                    f"Solve this cryptarithm: {equation}, where each letter represents a unique digit. Find the digit substitution that makes the equation true.",
                    f"In this alphametic puzzle: {equation}, each letter represents a distinct digit from 0-9. Determine the digits that make the equation valid.",
                ]
                prompt = random.choice(prompts)
                prompt += " Please end your response in the last line with the following format: The answer is $YOUR_ANSWER. $YOUR_ANSWER should be the equation with letters replaced by digits."

        return prompt

    def _verify_unique_solution(self, numbers: List[int], operators_list: List[str]) -> bool:
        """Verify puzzle has exactly one solution"""
        # Verify equation is correct
        result = self._calculate_equation(numbers[:-1], operators_list)
        if result != numbers[-1]:
            return False

        # Collect all unique digit characters
        all_digits_str = []
        for num in numbers:
            all_digits_str.extend(list(str(num)))

        unique_letters = set(all_digits_str)
        letter_list = list(unique_letters)
        letter_count = len(letter_list)

        # Find first positions (cannot be 0)
        first_positions = {}
        for num in numbers:
            num_str = str(num)
            if len(num_str) > 1:
                first_char = num_str[0]
                if first_char not in first_positions:
                    first_positions[first_char] = True

        # Helper to evaluate number from digit mapping
        def evaluate_number(num_str, digit_map):
            result = 0
            for c in num_str:
                result = result * 10 + digit_map[c]
            return result

        # Check if mapping satisfies equation
        def is_equation_valid(digit_map):
            equation_numbers = []
            for num in numbers:
                num_str = str(num)
                equation_numbers.append(evaluate_number(num_str, digit_map))

            result = self._calculate_equation(equation_numbers[:-1], operators_list)
            return result == equation_numbers[-1]

        # Backtracking to count valid solutions
        valid_mappings = []

        def backtrack(index, used_digits, digit_map):
            # Stop if we found more than one solution
            if len(valid_mappings) > 1:
                return

            if index == letter_count:
                if is_equation_valid(digit_map):
                    valid_mappings.append(digit_map.copy())
                return

            current_letter = letter_list[index]

            # First position letters cannot be 0
            start_digit = 1 if current_letter in first_positions else 0

            for digit in range(start_digit, 10):
                if digit not in used_digits:
                    digit_map[current_letter] = digit
                    used_digits.add(digit)

                    backtrack(index + 1, used_digits, digit_map)

                    used_digits.remove(digit)
                    digit_map.pop(current_letter)

        backtrack(0, set(), {})

        return len(valid_mappings) == 1

    def _calculate_equation(self, operands: List[int], operators_list: List[str]) -> int:
        """Calculate equation result with operator precedence"""
        if len(operands) != len(operators_list) + 1:
            raise ValueError("Operand count must be operator count + 1")

        # Handle operator precedence (multiplication first)
        ops = operators_list.copy()
        nums = operands.copy()

        # Process multiplication first
        i = 0
        while i < len(ops):
            if ops[i] == "*":
                nums[i] = nums[i] * nums[i + 1]
                nums.pop(i + 1)
                ops.pop(i)
            else:
                i += 1

        # Process addition and subtraction left to right
        result = nums[0]
        for i in range(len(ops)):
            if ops[i] == "+":
                result += nums[i + 1]
            elif ops[i] == "-":
                result -= nums[i + 1]

        return result

    def _construct_answer(self, numbers: List[int], operators_list: List[str]) -> str:
        """Construct answer string"""
        equation = str(numbers[0])
        for i, op in enumerate(operators_list):
            equation += f" {op} {numbers[i + 1]}"
        equation += f" = {numbers[-1]}"
        return equation

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract answer from model response

        Looks for numeric equations in various formats
        """
        if not test_solution:
            return ""

        # Normalize answer markers
        test_solution = test_solution.replace("THE ANSWER IS", "The answer is")
        test_solution = test_solution.replace("答案是：", "答案是:")
        test_solution = test_solution.replace("答案：", "答案:")

        # Try direct equation patterns first
        equation_patterns = [
            r"(\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+)",
            r"(\d+\s*(?:\+|\-|\*)\s*\d+\s*=\s*-?\d+)",
        ]

        for pattern in equation_patterns:
            matches = re.findall(pattern, test_solution)
            if matches:
                return matches[-1].strip()

        # Try Chinese answer patterns
        cn_patterns = [
            r"答案是[：:]\s*([0-9\s\+\-\*=]+)",
            r"答案[：:]\s*([0-9\s\+\-\*=]+)",
            r"数字等式是[：:]\s*([0-9\s\+\-\*=]+)",
            r"等式为[：:]\s*([0-9\s\+\-\*=]+)",
        ]

        # Try English answer patterns
        en_patterns = [
            r"[Tt]he answer is[：:=]\s*([0-9\s\+\-\*=]+)",
            r"[Aa]nswer[：:=]\s*([0-9\s\+\-\*=]+)",
            r"[Tt]he equation is[：:=]\s*([0-9\s\+\-\*=]+)",
        ]

        for pattern in cn_patterns + en_patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                answer = matches[-1].strip()
                answer = answer.replace("$", "").replace("。", "").replace(".", "")

                # Verify it's a valid equation
                if re.match(r"\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+", answer):
                    return answer

        # Try last line or any equation
        lines = test_solution.strip().split("\n")
        for line in reversed(lines):
            equation_match = re.search(r"\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+", line)
            if equation_match:
                return equation_match.group(0)

        return ""
