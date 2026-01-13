"""
Sudoku - Verifier

Verify sudoku solutions by checking:
1. The grid is valid (rows, columns, boxes all contain 1-9)
2. The solution respects the original puzzle (given cells match)
"""

import re
from base.data import Data
from base.verifier import Verifier


class SudokuVerifier(Verifier):
    """Verify Sudoku solutions"""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is a valid completed sudoku

        Args:
            data: Game data containing the original puzzle
            test_solution: Model's response

        Returns:
            bool: True if valid solution, False otherwise
        """
        # Extract answer from response
        from .generator import SudokuGenerator
        generator = SudokuGenerator()
        answer_str = generator.extract_answer(test_solution)

        # Parse the answer grid
        answer_grid = self._parse_grid(answer_str)
        if answer_grid is None:
            return False

        # Parse the original puzzle
        puzzle_str = data.metadata.get("puzzle", "")
        puzzle_grid = self._parse_grid(puzzle_str)
        if puzzle_grid is None:
            return False

        # Check 1: Answer respects the original puzzle (given cells match)
        if not self._respects_puzzle(answer_grid, puzzle_grid):
            return False

        # Check 2: Answer is a valid completed sudoku
        if not self._is_valid_sudoku(answer_grid):
            return False

        return True

    def _parse_grid(self, grid_str: str) -> list:
        """
        Parse grid string into 9x9 list of integers

        Args:
            grid_str: String representation of grid

        Returns:
            9x9 list of integers, or None if invalid
        """
        if not grid_str:
            return None

        lines = grid_str.strip().split('\n')
        grid = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract digits (1-9) and dots/zeros
            row = []
            for char in line:
                if char.isdigit():
                    row.append(int(char))
                elif char == '.':
                    row.append(0)
                # Skip other characters (spaces, separators, etc.)

            if len(row) == 9:
                grid.append(row)

        if len(grid) != 9:
            return None

        return grid

    def _respects_puzzle(self, answer: list, puzzle: list) -> bool:
        """
        Check if answer respects the given cells in puzzle

        Args:
            answer: 9x9 answer grid
            puzzle: 9x9 puzzle grid (0 for empty cells)

        Returns:
            bool: True if all given cells match
        """
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] != 0:
                    if answer[i][j] != puzzle[i][j]:
                        return False
        return True

    def _is_valid_sudoku(self, grid: list) -> bool:
        """
        Check if grid is a valid completed sudoku

        Args:
            grid: 9x9 grid

        Returns:
            bool: True if valid
        """
        # Check all cells are filled with 1-9
        for i in range(9):
            for j in range(9):
                if grid[i][j] < 1 or grid[i][j] > 9:
                    return False

        # Check rows
        for i in range(9):
            if not self._is_valid_group(grid[i]):
                return False

        # Check columns
        for j in range(9):
            col = [grid[i][j] for i in range(9)]
            if not self._is_valid_group(col):
                return False

        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = []
                for i in range(3):
                    for j in range(3):
                        box.append(grid[box_row * 3 + i][box_col * 3 + j])
                if not self._is_valid_group(box):
                    return False

        return True

    def _is_valid_group(self, group: list) -> bool:
        """
        Check if a group (row/column/box) contains exactly 1-9

        Args:
            group: List of 9 integers

        Returns:
            bool: True if contains exactly 1-9
        """
        return sorted(group) == list(range(1, 10))
