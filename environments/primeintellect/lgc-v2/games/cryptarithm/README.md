# Cryptarithm Task

Cryptarithmetic puzzles (also known as alphametic puzzles or verbal arithmetic) where letters represent unique digits.

## Task Description

Given an arithmetic equation where letters replace digits, find the digit assignment that makes the equation valid.

**Classic Example**: SEND + MORE = MONEY

**Constraints**:
- Each letter represents a unique digit (0-9)
- Different letters must represent different digits
- Leading digits cannot be 0 (e.g., in SEND, S ≠ 0)
- The equation must be mathematically correct

## Task ID Allocation

- **Task Type ID**: 3
- **ID Range**: 300,000,000 - 399,999,999
- **Capacity**: 100 million unique puzzles

### Examples

```
task_id = 300,000,000 → cryptarithm with seed 0
task_id = 300,000,500 → cryptarithm with seed 500
task_id = 300,050,000 → cryptarithm with seed 50,000
```

## Generation Parameters

All parameters are **derived from seed** for full reproducibility:

- **num_letter** (4-8): Number of unique letters/digits
- **operator_num** (1-3): Number of operators in equation
- **operator_level** (1-3):
  - Level 1: `+`, `-` only
  - Level 2: `+`, `-` only
  - Level 3: `+`, `-`, `*`

## Example Puzzles

### Simple Addition (1 operator)
```
Question: AB + CD = EFG
where each letter is a unique digit
A is 2 digits, B is 2 digits, C is 3 digits

Answer: 23 + 45 = 68
```

### Multiple Operations
```
Question: ABC + DEF - GH = IJK
where each letter is a unique digit

Answer: 234 + 567 - 89 = 712
```

## Multilingual Support

The task generates prompts in both Chinese and English (randomly selected):

### Chinese Example
```
已知：ABC + DEF = GHI，每个字母代表一个数字（ABC 是 3 位数、DEF 是 3 位数、GHI 是 3 位数），
字母与字母之间代表的数字不重复。若要让等式成立，求得题中的数字等式。
请在回答的最后一行使用以下格式：答案是 $YOUR_ANSWER。$YOUR_ANSWER 应该是替换为数字后的等式。
```

### English Example
```
Given: ABC + DEF = GHI, where each letter represents a unique digit (ABC is a 3-digit number,
DEF is a 3-digit number, and GHI is a 3-digit number). Find the numeric equation that makes
the equality valid.
Please end your response in the last line with the following format: The answer is $YOUR_ANSWER.
$YOUR_ANSWER should be the equation with letters replaced by digits.
```

## Answer Format

Models should respond with the numeric equation:

```
The answer is 123 + 456 = 579
```

or

```
答案是 123 + 456 = 579
```

## Verification

The verifier:
1. Extracts the numeric equation from the response
2. Normalizes spaces
3. Compares with the correct answer (exact match)

## Unique Solution Guarantee

The generator uses **backtracking** to verify that each puzzle has **exactly one solution**:
- Tests all possible digit assignments
- Respects leading-digit constraints (cannot be 0)
- Ensures equation validity
- Rejects puzzles with multiple solutions or no solution

## Implementation Details

### Generator (`generator.py`)
- Seed-based deterministic generation
- Automatic parameter derivation from seed
- Fallback to simple puzzles if generation fails
- Unique solution verification via backtracking

### Verifier (`verifier.py`)
- Flexible answer extraction (supports multiple formats)
- Handles both Chinese and English responses
- Normalizes whitespace for comparison

## Usage Example

```python
from env import Actor

actor = Actor(api_key="your-api-key")

# Generate cryptarithm puzzle with seed 1000
result = await actor.evaluate(task_id=300_001_000)

print(f"Task: {result['extra']['metadata']['letter_words']}")
print(f"Score: {result['score']}")
```

## Performance Notes

- Generation may take multiple attempts to find valid unique-solution puzzles
- Backtracking verification ensures high puzzle quality
- Fallback mechanism guarantees puzzle availability
- Average generation time: <100ms per puzzle
