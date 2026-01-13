"""
Sudoku - Generator

Generate sudoku puzzles with seed-based determinism.
Difficulty is controlled by the number of given cells.
"""

import random
import re
import uuid
from base.data import Data


class SudokuGenerator:
    """Generate Sudoku puzzles with seed-based determinism"""

    def __init__(self):
        pass

    def generate(self, seed: int = None, **kwargs):
        """
        Generate a single Sudoku puzzle based on seed

        All parameters are derived from seed for full determinism.

        Args:
            seed: Random seed for deterministic generation
            **kwargs: Config parameters (ignored, for compatibility)

        Returns:
            Data object containing question, answer, and metadata
        """
        if seed is None:
            seed = random.randint(0, 99999999)

        rng = random.Random(seed)

        # Derive difficulty from seed
        params = self._derive_params_from_seed(seed)

        # Generate complete sudoku
        solution = self._generate_complete_sudoku(rng)

        # Create puzzle by removing cells
        puzzle = self._create_puzzle(solution, params['given_count'], rng)

        # Format question
        question, lang = self._format_question(puzzle, params['prompt_idx'], rng)

        # Build metadata
        metadata = {
            "seed": seed,
            "trace_id": str(uuid.uuid4()),
            "given_count": params['given_count'],
            "difficulty": params['difficulty'],
            "language": lang,
            "solution": self._grid_to_string(solution),
            "puzzle": self._grid_to_string(puzzle),
        }

        return Data(
            question=question,
            answer=self._grid_to_string(solution),
            metadata=metadata
        )

    def _derive_params_from_seed(self, seed: int) -> dict:
        """Derive all parameters from seed"""
        rng = random.Random(seed)

        # Difficulty levels: easy(36-45), medium(28-35), hard(22-27)
        difficulty_roll = rng.random()
        if difficulty_roll < 0.3:
            difficulty = "easy"
            given_count = rng.randint(36, 45)
        elif difficulty_roll < 0.7:
            difficulty = "medium"
            given_count = rng.randint(28, 35)
        else:
            difficulty = "hard"
            given_count = rng.randint(22, 27)

        prompt_idx = rng.randint(0, 5)

        return {
            "difficulty": difficulty,
            "given_count": given_count,
            "prompt_idx": prompt_idx,
        }

    def _generate_complete_sudoku(self, rng: random.Random) -> list:
        """Generate a complete valid sudoku grid"""
        grid = [[0] * 9 for _ in range(9)]

        # Fill diagonal 3x3 boxes first (they are independent)
        for box in range(3):
            self._fill_box(grid, box * 3, box * 3, rng)

        # Fill the rest using backtracking
        self._solve_sudoku(grid, rng)

        return grid

    def _fill_box(self, grid: list, row: int, col: int, rng: random.Random):
        """Fill a 3x3 box with random valid numbers"""
        nums = list(range(1, 10))
        rng.shuffle(nums)
        idx = 0
        for i in range(3):
            for j in range(3):
                grid[row + i][col + j] = nums[idx]
                idx += 1

    def _solve_sudoku(self, grid: list, rng: random.Random = None) -> bool:
        """Solve sudoku using backtracking with optional randomization"""
        empty = self._find_empty(grid)
        if not empty:
            return True

        row, col = empty
        nums = list(range(1, 10))
        if rng:
            rng.shuffle(nums)

        for num in nums:
            if self._is_valid_placement(grid, row, col, num):
                grid[row][col] = num
                if self._solve_sudoku(grid, rng):
                    return True
                grid[row][col] = 0

        return False

    def _find_empty(self, grid: list):
        """Find first empty cell"""
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def _is_valid_placement(self, grid: list, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check row
        if num in grid[row]:
            return False

        # Check column
        if num in [grid[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if grid[i][j] == num:
                    return False

        return True

    def _create_puzzle(self, solution: list, given_count: int, rng: random.Random) -> list:
        """Create puzzle by removing cells from solution"""
        puzzle = [row[:] for row in solution]
        cells = [(i, j) for i in range(9) for j in range(9)]
        rng.shuffle(cells)

        cells_to_remove = 81 - given_count
        for i in range(cells_to_remove):
            row, col = cells[i]
            puzzle[row][col] = 0

        return puzzle

    def _grid_to_string(self, grid: list) -> str:
        """Convert grid to string representation"""
        rows = []
        for row in grid:
            rows.append("".join(str(x) if x != 0 else "." for x in row))
        return "\n".join(rows)

    def _format_question(self, puzzle: list, prompt_idx: int, rng: random.Random) -> tuple:
        """Format the puzzle as a question prompt"""
        # Determine language
        lang = "en" if rng.random() < 0.5 else "zh"

        puzzle_str = self._grid_to_string(puzzle)

        prompts_en = [
            f"""Solve the following Sudoku puzzle. Fill in the empty cells (marked with '.') with digits 1-9.

Rules:
- Each row must contain digits 1-9 with no repetition
- Each column must contain digits 1-9 with no repetition
- Each 3x3 box must contain digits 1-9 with no repetition

Puzzle:
```
{puzzle_str}
```

Provide your answer as the complete 9x9 grid (9 lines, 9 digits each) wrapped in triple backticks.""",

            f"""Complete this Sudoku puzzle. Replace each '.' with a digit from 1 to 9.

```
{puzzle_str}
```

Rules: Each row, column, and 3x3 box must contain all digits 1-9 exactly once.

Output the solved grid in the same format, wrapped in triple backticks.""",

            f"""Sudoku Challenge:

```
{puzzle_str}
```

Fill in the blanks ('.') following standard Sudoku rules. Output the complete solution grid wrapped in ``` marks.""",
        ]

        prompts_zh = [
            f"""请解决以下数独谜题。用数字1-9填充空格（用'.'标记）。

规则：
- 每行必须包含数字1-9，不能重复
- 每列必须包含数字1-9，不能重复
- 每个3x3宫格必须包含数字1-9，不能重复

谜题：
```
{puzzle_str}
```

请将完整的9x9网格（9行，每行9个数字）用三个反引号包裹后输出。""",

            f"""完成这道数独题目。将每个'.'替换为1到9之间的数字。

```
{puzzle_str}
```

规则：每行、每列、每个3x3宫格都必须恰好包含数字1-9各一次。

以相同格式输出解答，用三个反引号包裹。""",

            f"""数独挑战：

```
{puzzle_str}
```

按照标准数独规则填充空格（'.'）。请输出完整的解答网格，用```包裹。""",
        ]

        prompts = prompts_zh if lang == "zh" else prompts_en
        prompt = prompts[prompt_idx % len(prompts)]

        return prompt, lang

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract the sudoku grid from model response

        Looks for ```...``` block containing the grid, or 9 lines of 9 digits
        """
        # First try: find 9 lines of 9 digits anywhere in the response
        # This handles most common output formats
        lines = test_solution.strip().split('\n')
        grid_lines = []
        for line in lines:
            # Remove backticks and other common formatting
            clean_line = line.replace('`', '').strip()
            # Extract digits from line
            digits = re.findall(r'[1-9]', clean_line)
            if len(digits) == 9:
                grid_lines.append(''.join(digits))

        if len(grid_lines) >= 9:
            return '\n'.join(grid_lines[:9])

        # Fallback: try to find code block
        pattern = r"```([\s\S]*?)```"
        matches = re.findall(pattern, test_solution)

        if matches:
            # Get the last code block
            grid_str = matches[-1].strip()
            # Remove language identifier if present
            if grid_str and '\n' in grid_str:
                first_line = grid_str.split('\n')[0]
                if not re.search(r'[1-9]', first_line):
                    grid_str = '\n'.join(grid_str.split('\n')[1:])
            return grid_str

        return test_solution.strip()
