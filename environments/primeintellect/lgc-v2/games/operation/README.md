# Operation Task

Symbol operation task: define custom operators with conditional rules and evaluate expressions.

## Task Description

This task tests the model's ability to:
- Understand custom operator definitions
- Apply conditional logic based on operand properties
- Follow operator precedence rules
- Evaluate complex arithmetic expressions

## Example Challenge

```
Define △，the rules are as follows:
when x and y are both even, x △ y = x * y + 2;
otherwise, x △ y = x + y * 2.
The precedence of operations: ** > * = / = % > + = - = △.
Parentheses have the highest priority, and the expression inside the parentheses is calculated first.
Please calculate the value of the expression: 3 △ 4 + 2
Please fill your final answer in \boxed{}.
```

**Solution:**
- 3 and 4 are not both even, so use default: 3 △ 4 = 3 + 4 * 2 = 3 + 8 = 11
- Final: 11 + 2 = 13
- Answer: `\boxed{13}`

## Features

### Seed-based Generation
- All parameters derived from seed for full reproducibility
- Same seed always generates same challenge

### Customizable Complexity
- **Number of symbols**: 1-3 custom operators
- **Conditional definitions**: 30%-60% chance per operator
- **Expression length**: 4-6 operands
- **Operator precedence**: Random precedence levels (1-5)

### Multilingual Support
- Chinese and English prompts
- Language randomly selected based on seed

## Difficulty Levels

The task difficulty varies based on:
1. **Number of custom operators** (1-3)
2. **Presence of conditions** (conditional vs unconditional)
3. **Expression complexity** (number of operations)
4. **Nested operations** (custom operators calling other custom operators)

## Task ID Range

- **Range**: 200,000,000 - 299,999,999
- **Capacity**: 100 million unique challenges

## Examples

### Example 1: Simple (seed=200000000)
```
Define ○, the rules are as follows:
on the real number field, x ○ y = x + y.
The precedence of operations: ** > * = / = % > + = - = ○.
Please calculate the value of the expression: 3 ○ 5
```
Answer: 8

### Example 2: Conditional (seed=200000500)
```
Define △，规则如下：
当x是偶数时，x △ y = x * y；
其他情况下，x △ y = x + y。
运算优先级：** > * = / = % > + = - > △。
请计算表达式的值: 4 △ 3 + 2
```
Answer: 14 (4△3=12, 12+2=14)

### Example 3: Multiple Operators (seed=200001000)
```
Define △ and ○，the rules are as follows:
when x is greater than y, x △ y = x - y;
otherwise, x △ y = x + y.
on the real number field, x ○ y = x * y + 1.
The precedence of operations: ** > * = / = % = △ > + = - > ○.
Please calculate the value of the expression: 5 △ 3 ○ 2
```
Answer: 5 ((5△3)○2 = 2○2 = 2*2+1 = 5)

## Condition Types

The task supports 15 different condition types:
- Parity conditions: even/odd checks on x, y, or both
- Comparison: x > y, x < y, x = y
- Divisibility: multiples of 3 or 5
- Combined conditions: sum parity checks

## Answer Format

Models must return the numerical answer in `\boxed{}` notation:
```
\boxed{42}
```

The verifier extracts the number from the last `\boxed{}` occurrence and compares with tolerance of 1e-6 for floating-point results.

## Implementation Details

### Generator (`generator.py`)
- Seed-based deterministic generation
- Fallback mechanism for edge cases
- Validates expressions using SymPy
- Ensures answer is reasonable (|answer| < 100,000)

### Verifier (`verifier.py`)
- Extracts answer from `\boxed{}` notation
- Handles integers, floats, and fractions
- Tolerance-based comparison for floating-point
- Robust to formatting variations

## Performance Notes

- Generation success rate: ~95% (fallback for edge cases)
- Average generation time: <100ms per challenge
- Memory efficient: expressions simplified before evaluation
