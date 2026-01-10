"""
Dyck Language Generator

Generates bracket matching completion tasks based on Dyck language.
"""

import random
import uuid
from base.data import Data


class DyckLanguageGenerator:
    """Generate Dyck language bracket matching tasks"""

    def __init__(self):
        self.brackets = [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
            ("<", ">"),
            ("⟨", "⟩"),
            ("⟦", "⟧"),
            ("⦃", "⦄"),
            ("⦅", "⦆")
        ]

    def generate(
        self,
        seed: int,
        n_types: int = 6,
        total_length: int = 0,
        to_fill_length: int = 0,
        nesting_depth: int = 0,
        max_attempts: int = 1000
    ):
        """
        Generate a single Dyck language task

        Args:
            seed: Seed for reproducibility
            n_types: Number of bracket types (1-8)
            total_length: Total sequence length (0 for random)
            to_fill_length: Length to fill (0 for random)
            nesting_depth: Minimum nesting depth
            max_attempts: Max attempts to generate valid sequence

        Returns:
            Data: Game data object
        """
        # Validate parameters
        if n_types < 1 or n_types > 8:
            raise ValueError("n_types must be between 1 and 8")

        self.used_brackets = self.brackets[:n_types]
        rng = random.Random(seed)

        # Determine lengths
        current_total_length = total_length
        if current_total_length <= 0:
            # Random length between 40-60 (always even)
            current_total_length = rng.randint(20, 30) * 2
        elif current_total_length % 2 != 0:
            current_total_length -= 1

        current_fill_length = to_fill_length
        if current_fill_length <= 0:
            current_fill_length = rng.randint(
                max(1, int(current_total_length * 0.2)),
                min(int(current_total_length * 0.5), current_total_length // 2),
            )

        cut_point = current_total_length - current_fill_length

        # Generate sequence
        sequence = self._generate_valid_sequence(
            current_total_length, cut_point, nesting_depth, seed, max_attempts
        )

        question_sequence = sequence[:cut_point]
        closing_sequence = sequence[cut_point:]  # Only the closing brackets

        # Format question
        question = self._format_question(question_sequence)

        # Create Data object
        return Data(
            question=question,
            answer=closing_sequence,  # Store only closing brackets, not full sequence
            metadata={
                "seed": seed,
                "trace_id": str(uuid.uuid4()),
                "full_sequence": sequence,
                "question_sequence": question_sequence,
                "closing_sequence": closing_sequence,  # Add this for clarity
                "n_types": n_types,
                "total_length": current_total_length,
                "fill_length": current_fill_length,
                "nesting_depth": nesting_depth,
            }
        )

    def _generate_valid_sequence(self, total_length, cut_point, nesting_depth, seed, max_attempts):
        """Generate valid Dyck sequence where cut_point divides prefix and closing-only suffix"""
        rng = random.Random(seed) if seed is not None else random.Random()

        for attempt in range(max_attempts):
            try:
                if seed is not None:
                    rng.seed(seed + attempt)

                if total_length % 2 != 0:
                    total_length -= 1

                # Generate prefix (up to cut_point) with some unmatched opening brackets
                prefix = []
                prefix_stack = []
                current_depth = 0
                max_depth = 0

                # Add required nesting depth at the beginning
                if nesting_depth > 0:
                    for _ in range(nesting_depth):
                        bracket = rng.choice(self.used_brackets)
                        prefix.append(bracket[0])
                        prefix_stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)

                # Continue building prefix up to cut_point
                while len(prefix) < cut_point:
                    # Randomly decide whether to add opening or closing bracket
                    # But ensure we don't close all brackets before reaching cut_point
                    remaining_chars = cut_point - len(prefix)

                    if len(prefix_stack) == 0:
                        # Must add opening bracket
                        bracket = rng.choice(self.used_brackets)
                        prefix.append(bracket[0])
                        prefix_stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
                    elif remaining_chars > len(prefix_stack):
                        # Can add either opening or closing
                        if rng.random() < 0.6:  # Prefer opening brackets
                            bracket = rng.choice(self.used_brackets)
                            prefix.append(bracket[0])
                            prefix_stack.append(bracket)
                            current_depth += 1
                            max_depth = max(max_depth, current_depth)
                        else:
                            bracket = prefix_stack.pop()
                            prefix.append(bracket[1])
                            current_depth -= 1
                    else:
                        # Must ensure some brackets remain unclosed
                        bracket = rng.choice(self.used_brackets)
                        prefix.append(bracket[0])
                        prefix_stack.append(bracket)
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)

                # Generate suffix: only closing brackets from prefix_stack
                suffix = []
                while prefix_stack:
                    bracket = prefix_stack.pop()
                    suffix.append(bracket[1])

                result = "".join(prefix + suffix)

                # Validate
                if len(result) != total_length:
                    raise ValueError(f"Invalid length: {len(result)} != {total_length}")
                if nesting_depth > 0 and max_depth < nesting_depth:
                    raise ValueError("Insufficient nesting depth")
                if len(prefix) != cut_point:
                    raise ValueError(f"Invalid cut point: {len(prefix)} != {cut_point}")

                return result

            except ValueError:
                continue

        raise ValueError("Failed to generate valid sequence")

    def _format_question(self, sequence):
        """Format the question prompt"""
        return f"""Complete the following Dyck language sequence by adding the minimal necessary closing brackets.

Sequence: {sequence}

Rules:
- Add only the closing brackets needed to match all unmatched opening brackets
- Do not add any extra bracket pairs beyond what is required

Provide only the complete valid sequence."""

    def extract_answer(self, test_solution: str) -> str:
        """Extract bracket sequence from model response - O(n) algorithm"""
        if not test_solution:
            return ""

        # Clean text
        text = "".join(test_solution.split())
        text = text.replace("\\n", "").replace("\\t", "").replace("\\r", "")

        # Remove quotes
        if len(text) >= 2:
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]

        bracket_chars = set("()[]{}<>⟨⟩⟦⟧⦃⦄⦅⦆")
        opening_chars = set("([{<⟨⟦⦃⦅")
        bracket_pairs = {
            ")": "(", "]": "[", "}": "{", ">": "<",
            "⟩": "⟨", "⟧": "⟦", "⦄": "⦃", "⦆": "⦅"
        }

        def find_longest_valid_in_segment(s):
            """Find longest valid Dyck substring in a pure bracket string - O(n)"""
            if not s:
                return ""

            n = len(s)
            stack = []  # stores indices of unmatched brackets

            for i, char in enumerate(s):
                if char in opening_chars:
                    stack.append(i)
                else:
                    # closing bracket
                    if stack and s[stack[-1]] == bracket_pairs.get(char):
                        stack.pop()
                    else:
                        stack.append(i)  # unmatched closing bracket

            if not stack:
                return s  # entire string is valid

            # Find longest valid substring between unmatched positions
            unmatched = [-1] + stack + [n]

            max_len = 0
            best_start = 0

            for i in range(len(unmatched) - 1):
                length = unmatched[i + 1] - unmatched[i] - 1
                if length > max_len:
                    max_len = length
                    best_start = unmatched[i] + 1

            if max_len == 0:
                return ""

            return s[best_start:best_start + max_len]

        # Extract contiguous bracket segments and find longest valid sequence
        best_sequence = ""
        current_segment = []

        for char in text:
            if char in bracket_chars:
                current_segment.append(char)
            else:
                if current_segment:
                    segment = "".join(current_segment)
                    valid = find_longest_valid_in_segment(segment)
                    if len(valid) > len(best_sequence):
                        best_sequence = valid
                    current_segment = []

        # Don't forget the last segment
        if current_segment:
            segment = "".join(current_segment)
            valid = find_longest_valid_in_segment(segment)
            if len(valid) > len(best_sequence):
                best_sequence = valid

        return best_sequence
