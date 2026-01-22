"""ARC-GEN task generator and evaluator."""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from models import Challenge

from agent.arc_agi_generator import ARC2Generator

_PROMPT_HEADER = """You are given ARC-style tasks.
Each grid is a matrix of integers 0-9 representing colors.
Infer the transformation from each input to its output.
Apply the same transformation to the test input.
Return ONLY the output grid as JSON (a list of lists of integers)."""


class ArcGenTask:
    """ARC-GEN task generator and evaluator."""

    def __init__(self, num_train: int = 3, num_test: int = 1) -> None:
        self.num_train = num_train
        self.num_test = num_test
        self.generator = ARC2Generator()

    @staticmethod
    def _format_grid(grid: List[List[int]]) -> str:
        return json.dumps(grid, separators=(",", ":"))

    def _build_prompt(self, train_examples: List[Dict[str, Any]], test_input: List[List[int]]) -> str:
        lines = [_PROMPT_HEADER, "", "Training examples:"]
        for idx, example in enumerate(train_examples, 1):
            lines.append(f"Example {idx}:")
            lines.append(f"input: {self._format_grid(example['input'])}")
            lines.append(f"output: {self._format_grid(example['output'])}")
        lines.append("")
        lines.append("Test input:")
        lines.append(self._format_grid(test_input))
        lines.append("")
        lines.append("Output grid (JSON only):")
        return "\n".join(lines)

    async def generate(
        self,
        task_id: Optional[int] = None,
        num_train: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> Challenge:
        
        if num_train is None:
            num_train = self.num_train
        if num_test is None:
            num_test = self.num_test
        if num_test < 1:
            raise ValueError("num_test must be >= 1")

        generation_result = self.generator.generate_problem_set(task_id = task_id)
        train_examples = generation_result["train_examples"]
        test_input = generation_result["test_input"]
        test_output = generation_result["test_output"]

        prompt = self._build_prompt(train_examples, test_input)

        return Challenge(
            env="affine:arc-gen",
            prompt=prompt,
            extra={
                "task_id": task_id,
                "expected_output": test_output,
                "test_input" : test_input,
                "train_examples" : train_examples
            },
        )

    @staticmethod
    def _strip_response(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?answer>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_candidate(text: str) -> str:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if blocks:
            return blocks[-1].strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text.strip()

    @staticmethod
    def _is_valid_grid(grid: Any) -> bool:
        if not isinstance(grid, list) or not grid:
            return False
        row_len = None
        for row in grid:
            if not isinstance(row, list) or not row:
                return False
            if row_len is None:
                row_len = len(row)
            if len(row) != row_len:
                return False
            for val in row:
                if isinstance(val, bool) or not isinstance(val, int):
                    return False
                if val < 0 or val > 9:
                    return False
        return True

    def _parse_grid(self, text: str) -> Optional[List[List[int]]]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return None
        if not self._is_valid_grid(parsed):
            return None
        return parsed

    def extract_grid(self, response: str) -> Optional[List[List[int]]]:
        if not response:
            return None
        cleaned = self._strip_response(response)
        candidate = self._extract_candidate(cleaned)
        return self._parse_grid(candidate)

    @staticmethod
    def _cell_accuracy(predicted: List[List[int]], expected: List[List[int]]) -> Optional[float]:
        if not predicted or not expected:
            return None
        if len(predicted) != len(expected):
            return None
        if len(predicted[0]) != len(expected[0]):
            return None
        total = 0
        correct = 0
        for r in range(len(expected)):
            if len(predicted[r]) != len(expected[r]):
                return None
            for c in range(len(expected[r])):
                total += 1
                if predicted[r][c] == expected[r][c]:
                    correct += 1
        if total == 0:
            return None
        return correct / total

    async def evaluate(self, response: str, challenge: Challenge) -> Tuple[float, Optional[float], Optional[List[List[int]]]]:
        expected = challenge.extra.get("expected_output")
        if expected is None:
            return 0.0, None, None

        predicted = self.extract_grid(response or "")
        if predicted is None:
            return 0.0, None, None

        if predicted == expected:
            return 1.0, 1.0, predicted

        cell_accuracy = self._cell_accuracy(predicted, expected)
        return 0.0, cell_accuracy, predicted
