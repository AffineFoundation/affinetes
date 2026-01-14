"""
Dyck Language Reasoning Errors Generator

Generates tasks where the model must identify errors in bracket matching reasoning steps.
"""

import random
import uuid
from typing import List, Tuple
from base.data import Data


class DyckLanguageReasoningErrorsGenerator:
    """Generate Dyck language reasoning error identification tasks"""

    def __init__(self):
        self.brackets = [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
            ("<", ">"),
        ]
        self.open_brackets = [b[0] for b in self.brackets]
        self.close_brackets = [b[1] for b in self.brackets]
        self.bracket_pairs = {b[1]: b[0] for b in self.brackets}
        self.reverse_pairs = {b[0]: b[1] for b in self.brackets}

    def generate(
        self,
        seed: int,
        n_types: int = 0,
        total_length: int = 0,
        n_errors: int = 0,
    ) -> Data:
        """
        Generate a single Dyck language reasoning error task

        Args:
            seed: Seed for reproducibility
            n_types: Number of bracket types (1-4), 0 for random
            total_length: Total sequence length, 0 for random (10-30)
            n_errors: Number of errors to introduce, 0 for random (1-5)

        Returns:
            Data: Game data object
        """
        rng = random.Random(seed)

        # Determine parameters from seed if not specified
        if n_types <= 0:
            n_types = rng.randint(2, 4)
        n_types = min(n_types, 4)

        if total_length <= 0:
            total_length = rng.randint(5, 15) * 2  # Even number between 10-30

        if n_errors <= 0:
            max_errors = min(5, total_length // 4)
            n_errors = rng.randint(2, max(2, max_errors))  # At least 2 errors

        # Select bracket types
        selected_brackets = self.brackets[:n_types]

        # Generate valid Dyck sequence
        dyck_sequence = self._generate_valid_dyck_sequence(rng, selected_brackets, total_length)

        # Generate thoughts with errors
        thoughts, error_indices = self._generate_thoughts_with_errors(
            rng, dyck_sequence, n_errors, selected_brackets
        )

        # Format question
        question = self._format_question(dyck_sequence, thoughts, rng)

        # Format answer
        answer = ",".join(map(str, sorted(error_indices))) if error_indices else ""

        return Data(
            question=question,
            answer=answer,
            metadata={
                "seed": seed,
                "trace_id": str(uuid.uuid4()),
                "dyck_sequence": dyck_sequence,
                "thoughts": thoughts,
                "error_indices": error_indices,
                "n_types": n_types,
                "total_length": len(dyck_sequence),
                "n_errors": len(error_indices),
            }
        )

    def _generate_valid_dyck_sequence(
        self, rng: random.Random, selected_brackets: List[Tuple[str, str]], total_length: int
    ) -> str:
        """Generate a valid Dyck language sequence"""
        if total_length % 2 != 0:
            total_length += 1

        sequence = []
        stack = []
        remaining = total_length

        while remaining > 0:
            if not stack or (rng.random() < 0.5 and remaining > len(stack) * 2):
                bracket_type = rng.choice(selected_brackets)
                sequence.append(bracket_type[0])
                stack.append(bracket_type[1])
            else:
                sequence.append(stack.pop())
            remaining -= 1

        while stack:
            sequence.append(stack.pop())

        return "".join(sequence)

    def _generate_thoughts_with_errors(
        self,
        rng: random.Random,
        dyck_sequence: str,
        n_errors: int,
        selected_brackets: List[Tuple[str, str]]
    ) -> Tuple[List[str], List[int]]:
        """
        Generate reasoning steps with intentional errors

        Each step shows the current character and the resulting stack state.
        Errors are introduced by showing incorrect stack states.
        """
        thoughts = []
        error_indices = []

        # Initial thoughts
        thoughts.append("We should process the input step by step and track the stack state.")
        thoughts.append("Stack: empty")

        # Maintain correct stack
        correct_stack = []

        # Choose error positions (indices in the sequence)
        seq_len = len(dyck_sequence)
        max_error_positions = min(n_errors, seq_len)
        error_positions = set(rng.sample(range(seq_len), max_error_positions)) if max_error_positions > 0 else set()

        open_brackets_set = set(b[0] for b in selected_brackets)
        all_brackets = [b[0] for b in selected_brackets] + [b[1] for b in selected_brackets]

        for i, char in enumerate(dyck_sequence):
            # Update correct stack
            if char in open_brackets_set:
                correct_stack.append(char)
            else:
                if correct_stack:
                    correct_stack.pop()

            # Decide whether to introduce error at this position
            if i in error_positions:
                # Generate wrong stack state
                wrong_stack = self._generate_wrong_stack(rng, correct_stack, selected_brackets)
                if wrong_stack != correct_stack:
                    stack_str = self._format_stack(wrong_stack)
                    thoughts.append(f"{char} ; Stack: {stack_str}")
                    error_indices.append(len(thoughts))  # 1-indexed (will add "Thought N:" later)
                else:
                    # Failed to generate different stack, use correct
                    stack_str = self._format_stack(correct_stack)
                    thoughts.append(f"{char} ; Stack: {stack_str}")
            else:
                stack_str = self._format_stack(correct_stack)
                thoughts.append(f"{char} ; Stack: {stack_str}")

        # Final thought
        if correct_stack:
            thoughts.append("Now we have reached the end. The final stack is not empty.")
        else:
            thoughts.append("Now we have reached the end. The final stack is empty.")

        # Ensure at least one error if n_errors > 0
        if n_errors > 0 and len(error_indices) == 0:
            # Force an error in a middle thought
            if len(thoughts) > 4:
                error_idx = rng.randint(2, len(thoughts) - 2)
                thoughts[error_idx] = self._corrupt_thought(rng, thoughts[error_idx], selected_brackets)
                error_indices.append(error_idx + 1)

        # Add "Thought N:" prefix to all thoughts
        numbered_thoughts = [f"Thought {i+1}: {thought}" for i, thought in enumerate(thoughts)]

        return numbered_thoughts, error_indices

    def _generate_wrong_stack(
        self,
        rng: random.Random,
        correct_stack: List[str],
        selected_brackets: List[Tuple[str, str]]
    ) -> List[str]:
        """Generate an incorrect stack state"""
        open_brackets = [b[0] for b in selected_brackets]
        all_brackets = open_brackets + [b[1] for b in selected_brackets]

        error_type = rng.choice(["wrong_pop", "no_pop", "wrong_push", "stack_corruption"])
        modified_stack = correct_stack.copy()

        if error_type == "wrong_pop" and modified_stack:
            modified_stack.pop()
            if modified_stack:
                modified_stack[-1] = rng.choice(open_brackets)

        elif error_type == "no_pop":
            if modified_stack:
                modified_stack[-1] = rng.choice(open_brackets)
            else:
                modified_stack.append(rng.choice(open_brackets))

        elif error_type == "wrong_push":
            modified_stack.append(rng.choice(all_brackets))

        elif error_type == "stack_corruption" and modified_stack:
            idx = rng.randint(0, len(modified_stack) - 1)
            # Choose a different bracket
            current = modified_stack[idx]
            choices = [b for b in open_brackets if b != current]
            if choices:
                modified_stack[idx] = rng.choice(choices)
            else:
                modified_stack.append(rng.choice(open_brackets))

        # If still same as correct, force a change
        if modified_stack == correct_stack:
            if modified_stack:
                modified_stack.append(rng.choice(open_brackets))
            else:
                modified_stack = [rng.choice(open_brackets)]

        return modified_stack

    def _corrupt_thought(
        self,
        rng: random.Random,
        thought: str,
        selected_brackets: List[Tuple[str, str]]
    ) -> str:
        """Corrupt a thought by modifying its stack state"""
        parts = thought.split(" ; Stack: ")
        if len(parts) != 2:
            return thought

        char, stack_str = parts
        open_brackets = [b[0] for b in selected_brackets]
        all_brackets = open_brackets + [b[1] for b in selected_brackets]

        if stack_str == "empty":
            new_stack = rng.choice(open_brackets)
        else:
            # Modify the stack string
            if rng.random() < 0.5 and len(stack_str) > 0:
                # Remove a character
                pos = rng.randint(0, len(stack_str) - 1)
                new_stack = stack_str[:pos] + stack_str[pos+1:]
                if not new_stack:
                    new_stack = rng.choice(open_brackets)
            else:
                # Add a character
                pos = rng.randint(0, len(stack_str))
                new_stack = stack_str[:pos] + rng.choice(all_brackets) + stack_str[pos:]

        return f"{char} ; Stack: {new_stack}"

    def _format_stack(self, stack: List[str]) -> str:
        """Format stack for display"""
        if not stack:
            return "empty"
        return "".join(stack)

    def _format_question(self, dyck_sequence: str, thoughts: List[str], rng: random.Random) -> str:
        """Format the question prompt"""
        thoughts_text = "\n".join(thoughts)

        # Use different prompt templates
        template_id = rng.randint(0, 1)

        if template_id == 0:
            return f"""You are an expert in Dyck language, where you must complete bracket sequences (e.g., [], {{}}, <>). You need to analyze whether the bracket matching steps are correct according to Dyck language rules.

Given an initial Dyck language sequence and steps for deriving the closing bracket sequence (given in the form of a thought process), your task is to identify the positions with incorrect reasoning. There may be multiple errors.

Possible errors include: forgetting to close a bracket, using the wrong closing bracket, or incorrectly copying the stack state.

Task: Check the sequence and verify the bracket matching process.
Input: {dyck_sequence}
{thoughts_text}
Question: Are there any reasoning errors in this process? If no errors, output an empty string ""; if there are errors, output the step numbers where errors occur.

Note: If there are multiple errors, output in the format: 1,3,9"""

        else:
            return f"""As a Dyck language expert, you need to verify the correctness of bracket matching reasoning steps.

A Dyck sequence must have all brackets properly matched and nested. The reasoning process tracks a stack: opening brackets are pushed, and closing brackets pop the matching opening bracket.

Sequence: {dyck_sequence}
{thoughts_text}
Task: Identify which thought steps contain errors in the stack state.

If no errors exist, output "". If errors exist, output the thought numbers separated by commas (e.g., 3,5,8)."""

    def extract_answer(self, test_solution: str) -> str:
        """Extract error indices from model response"""
        import re

        if not test_solution:
            return ""

        solution = test_solution.strip()

        # Check for empty/no error indicators
        if solution == "" or solution.lower() in ["", "none", "no errors", "no error", "empty"]:
            return ""

        # If the answer is just comma-separated numbers
        if re.match(r'^[0-9,\s]+$', solution):
            return solution.replace(" ", "").replace("，", ",")

        # Find comma-separated number list
        num_list_pattern = r'([0-9]+(?:[,，\s]+[0-9]+)+)'
        num_list_match = re.search(num_list_pattern, solution)
        if num_list_match:
            answer = num_list_match.group(1).strip()
            answer = re.sub(r'[,，\s]+', ',', answer)
            return answer

        # Single number
        single_num_pattern = r'(?<!\d)([0-9]+)(?!\d)'
        single_num_match = re.search(single_num_pattern, solution)
        if single_num_match:
            return single_num_match.group(1)

        # Extract all numbers and deduplicate
        all_nums = re.findall(r'(?<!\d)([1-9]\d*)(?!\d)', solution)
        if all_nums:
            unique_nums = sorted(set(map(int, all_nums)))
            return ','.join(map(str, unique_nums))

        return ""
