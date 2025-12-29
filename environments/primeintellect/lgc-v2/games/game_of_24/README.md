# Game of 24

A mathematical puzzle where players use four numbers and basic arithmetic operations to get 24.

## Task Description

Given 4 numbers (typically 1-9), use arithmetic operations (+, -, *, /) to create an expression that equals 24. Each number must be used exactly once.

### Example

**Question**: Given the numbers [3, 3, 8, 8], apply the arithmetic operations ['+', '-', '*', '/'] to get 24.

**Valid Answer**: `8 / (3 - 8 / 3)`

**Evaluation**: 8 / (3 - 8/3) = 8 / (3 - 2.666...) = 8 / 0.333... = 24 ✓

## Configuration

### Default Parameters
- `num_of_numbers`: 4 (number of numbers to use)
- `result`: 24 (target value)
- `min_candidate`: 1 (minimum number value)
- `max_candidate`: 9 (maximum number value)
- `operators`: ["+", "-", "*", "/"] (allowed operators)

### Configuration Examples

**Easy mode** (smaller numbers, more solutions):
```python
{
    "min_candidate": 1,
    "max_candidate": 6,
}
```

**Hard mode** (larger numbers, fewer solutions):
```python
{
    "min_candidate": 5,
    "max_candidate": 13,
}
```

**Alternative target**:
```python
{
    "result": 36,  # Make 36 instead of 24
}
```

## Generation Logic

1. **Number selection**: Randomly generate `num_of_numbers` integers between `min_candidate` and `max_candidate` using the seed
2. **Solution validation**: Check if the numbers can form a valid solution using brute force enumeration of all permutations and operator combinations
3. **Retry on failure**: If no solution exists, increment seed and retry (up to `max_attempts`)
4. **Prompt generation**: Randomly select Chinese or English prompt template

## Verification Logic

The verifier checks:
1. **Format**: Answer is in a ```python code block
2. **Numbers**: All and only the given numbers are used (each exactly once)
3. **Operators**: Only allowed operators and parentheses are used
4. **Correctness**: Expression evaluates to the target result (within 1e-10 tolerance for floating point)

## Answer Format

Models must respond with a ```python code block containing a single evaluable expression:

**English example**:
```
To solve this, I'll try different combinations...

Here's the solution:
```python
(3 + 8 / 8) * 3
```

**Chinese example**:
```
让我试试不同的组合...

答案是：
```python
8 / (3 - 8 / 3)
```

## Difficulty Analysis

The difficulty of Game of 24 depends on:
- **Number range**: Larger numbers generally harder
- **Number distribution**: Numbers with common factors easier (e.g., [2,3,4,6] vs [7,7,7,7])
- **Solution density**: Some combinations have many solutions, others have few or none

Typical success rates (estimated):
- Numbers 1-9: ~60-70% of combinations have solutions
- Numbers 1-13: ~40-50% of combinations have solutions
