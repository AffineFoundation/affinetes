"""
Dyck Language 2 Generator

Generates reverse bracket completion tasks - given closing brackets, complete opening brackets.
Supports multiple prompt variants for diverse question formats.
"""

import random
import uuid
from base.data import Data


class DyckLanguage2Generator:
    """Generate reverse Dyck language bracket completion tasks"""

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

        # Multiple prompt templates for diversity
        # All templates require answer in backticks
        self.prompt_templates = [
            # Template 0: Direct instruction
            {
                "template": """Complete the following Dyck language sequence by adding the minimal necessary opening brackets at the beginning.

Suffix: {suffix}

Rules:
- Add only the opening brackets needed to match all unmatched closing brackets
- The opening brackets should be prepended to form a valid sequence
- Do not add any extra bracket pairs beyond what is required

Please provide your final answer (the complete valid sequence) in backticks like `answer`.""",
                "name": "direct"
            },
            # Template 1: Reverse perspective
            {
                "template": """You are given the ending portion of a bracket sequence. Find the beginning.

Ending: {suffix}

Your task:
- Determine what opening brackets must come before this ending
- The result must be a valid, balanced bracket sequence
- Use the minimum number of opening brackets necessary

Please provide your final answer (the complete sequence) in backticks like `answer`.""",
                "name": "reverse_perspective"
            },
            # Template 2: Puzzle format
            {
                "template": """Bracket Puzzle: What's missing at the start?

Given suffix: {suffix}

Solve:
1. Each closing bracket needs a matching opening bracket
2. Add the minimal opening brackets to the left
3. The final sequence must be properly nested

Please provide your final answer (the full balanced sequence) in backticks like `answer`.""",
                "name": "puzzle"
            },
            # Template 3: Technical format
            {
                "template": """[DYCK LANGUAGE TASK]

INPUT: {suffix}
DIRECTION: reverse (prepend opening brackets)

CONSTRAINTS:
- Minimize added brackets
- Result must be valid Dyck word
- Preserve input as suffix

Please provide your final answer in backticks like `answer`.""",
                "name": "technical"
            },
            # Template 4: Story format
            {
                "template": """A sequence of brackets was cut, and only the end remains:

"{suffix}"

Restore the missing beginning so that:
- Every closing bracket has its opening partner
- No extra brackets are added
- The sequence is properly nested

Please provide your final answer (the restored complete sequence) in backticks like `answer`.""",
                "name": "story"
            },
            # Template 5: Mathematical format
            {
                "template": """Let S = "{suffix}" be a suffix of a Dyck word.

Find the minimal prefix P such that P + S forms a valid Dyck word.

Please provide your final answer (P + S, the complete Dyck word) in backticks like `answer`.""",
                "name": "mathematical"
            },
            # Template 6: Q&A format
            {
                "template": """Q: What opening brackets should precede "{suffix}" to form a valid bracket sequence?

Requirements:
- Use exactly the brackets needed (no more, no less)
- Maintain proper nesting order

Please provide your final answer (the entire resulting sequence) in backticks like `answer`.""",
                "name": "qa"
            },
            # Template 7: Fill-in format
            {
                "template": """Fill in the blank:

_____ + {suffix} = valid Dyck sequence

Rules:
- The blank contains only opening brackets
- Use the minimum required

Please provide your final answer (the complete filled sequence) in backticks like `answer`.""",
                "name": "fill_in"
            },
        ]

    def generate(
        self,
        seed: int,
        n_types: int = 0,
        total_length: int = 0,
        to_fill_length: int = 0,
        nesting_depth: int = 0,
        prompt_variant: int = -1,
        max_attempts: int = 1000
    ):
        """
        Generate a single reverse Dyck language task

        Args:
            seed: Seed for reproducibility
            n_types: Number of bracket types (1-8), 0 for random
            total_length: Total sequence length (6-60), 0 for random
            to_fill_length: Length to fill with opening brackets (0 for random)
            nesting_depth: Minimum nesting depth
            prompt_variant: Which prompt template to use (-1 for random)
            max_attempts: Max attempts to generate valid sequence

        Returns:
            Data: Game data object
        """
        rng = random.Random(seed)

        # Determine n_types from seed if not specified (3-8)
        current_n_types = n_types
        if current_n_types <= 0:
            current_n_types = rng.randint(3, 8)
        elif current_n_types < 3:
            current_n_types = 3
        elif current_n_types > 8:
            current_n_types = 8

        self.used_brackets = self.brackets[:current_n_types]

        # Determine length from seed if not specified (6-60, must be even)
        current_total_length = total_length
        if current_total_length <= 0:
            current_total_length = rng.randint(3, 30) * 2  # 6-60
        else:
            if current_total_length < 6:
                current_total_length = 6
            if current_total_length % 2 != 0:
                current_total_length -= 1

        current_fill_length = to_fill_length
        if current_fill_length <= 0:
            current_fill_length = rng.randint(
                max(1, int(current_total_length * 0.2)),
                min(int(current_total_length * 0.5), current_total_length // 2),
            )

        # For reverse: fill_length is opening brackets, suffix is the rest
        prefix_length = current_fill_length
        suffix_start = prefix_length

        # Generate sequence
        sequence = self._generate_valid_sequence(
            current_total_length, prefix_length, nesting_depth, seed, max_attempts
        )

        opening_prefix = sequence[:prefix_length]
        suffix_sequence = sequence[prefix_length:]

        # Select prompt variant
        if prompt_variant < 0 or prompt_variant >= len(self.prompt_templates):
            prompt_variant = rng.randint(0, len(self.prompt_templates) - 1)

        question = self._format_question(suffix_sequence, prompt_variant)

        return Data(
            question=question,
            answer=sequence,  # Full sequence is the answer
            metadata={
                "seed": seed,
                "trace_id": str(uuid.uuid4()),
                "full_sequence": sequence,
                "opening_prefix": opening_prefix,
                "suffix_sequence": suffix_sequence,
                "n_types": current_n_types,
                "total_length": current_total_length,
                "fill_length": current_fill_length,
                "nesting_depth": nesting_depth,
                "prompt_variant": prompt_variant,
                "prompt_name": self.prompt_templates[prompt_variant]["name"],
                "task_type": "reverse"
            }
        )

    def _generate_valid_sequence(self, total_length, prefix_length, nesting_depth, seed, max_attempts):
        """Generate valid Dyck sequence where prefix_length opening brackets come first"""
        rng = random.Random(seed) if seed is not None else random.Random()

        for attempt in range(max_attempts):
            try:
                if seed is not None:
                    rng.seed(seed + attempt)

                if total_length % 2 != 0:
                    total_length -= 1

                # Build a sequence where first prefix_length chars are opening brackets
                # that get closed in the suffix

                prefix = []
                prefix_stack = []

                # Start with required opening brackets that will be closed later
                for _ in range(prefix_length):
                    bracket = rng.choice(self.used_brackets)
                    prefix.append(bracket[0])
                    prefix_stack.append(bracket)

                # Check nesting depth requirement
                if nesting_depth > 0 and len(prefix_stack) < nesting_depth:
                    continue

                # Build suffix: mix of matched pairs and closing the prefix_stack
                suffix = []
                suffix_length = total_length - prefix_length

                # We need to close all prefix_stack brackets in suffix
                # But we can also add matched pairs in between

                temp_stack = prefix_stack.copy()
                remaining = suffix_length

                while remaining > 0:
                    if len(temp_stack) == remaining:
                        # Must close all remaining
                        while temp_stack:
                            bracket = temp_stack.pop()
                            suffix.append(bracket[1])
                            remaining -= 1
                    elif len(temp_stack) < remaining and remaining - len(temp_stack) >= 2:
                        # Can add a matched pair or close one
                        choice = rng.random()
                        if choice < 0.4 and remaining - len(temp_stack) >= 2:
                            # Add a complete matched pair
                            bracket = rng.choice(self.used_brackets)
                            suffix.append(bracket[0])
                            suffix.append(bracket[1])
                            remaining -= 2
                        elif temp_stack:
                            # Close one from stack
                            bracket = temp_stack.pop()
                            suffix.append(bracket[1])
                            remaining -= 1
                        else:
                            # Add matched pair
                            bracket = rng.choice(self.used_brackets)
                            suffix.append(bracket[0])
                            suffix.append(bracket[1])
                            remaining -= 2
                    elif temp_stack:
                        bracket = temp_stack.pop()
                        suffix.append(bracket[1])
                        remaining -= 1
                    else:
                        break

                result = "".join(prefix + suffix)

                if len(result) != total_length:
                    continue

                # Validate it's a proper Dyck sequence
                if not self._is_valid_dyck(result):
                    continue

                # Validate prefix is all opening brackets
                opening_chars = set(b[0] for b in self.used_brackets)
                if not all(c in opening_chars for c in result[:prefix_length]):
                    continue

                return result

            except (ValueError, IndexError):
                continue

        raise ValueError("Failed to generate valid sequence")

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

    def _format_question(self, suffix: str, variant: int) -> str:
        """Format the question using selected prompt template"""
        template_info = self.prompt_templates[variant]
        return template_info["template"].format(suffix=suffix)

    def extract_answer(self, test_solution: str) -> str:
        """Extract bracket sequence from backticks in model response"""
        if not test_solution:
            return ""

        bracket_chars = set("()[]{}<>⟨⟩⟦⟧⦃⦄⦅⦆")

        # Find all backtick-enclosed content, return the last valid one
        results = []
        i = 0
        while i < len(test_solution):
            if test_solution[i] == '`':
                # Find closing backtick
                j = i + 1
                while j < len(test_solution) and test_solution[j] != '`':
                    j += 1
                if j < len(test_solution):
                    content = test_solution[i+1:j].strip()
                    # Only accept if content is pure brackets
                    if content and all(c in bracket_chars for c in content):
                        results.append(content)
                    i = j + 1
                else:
                    break
            else:
                i += 1

        return results[-1] if results else ""
