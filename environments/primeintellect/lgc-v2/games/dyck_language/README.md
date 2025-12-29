# Dyck Language Task

## Overview

The Dyck language task tests a model's ability to complete bracket sequences with proper nesting and matching. Given a partial bracket sequence, the model must complete it with the correct closing brackets.

## Task Description

**Input**: A partial sequence of opening and closing brackets from multiple types: `()`, `[]`, `{}`, `<>`

**Output**: The complete valid Dyck sequence

**Example**:
```
Input:  "([{<"
Output: "([{<>}])"
```

## Difficulty Parameters

- **n_types** (1-4): Number of bracket types used
  - 1: Only `()`
  - 2: `()` and `[]`
  - 3: `()`, `[]`, and `{}`
  - 4: All types including `<>`

- **total_length** (0 or positive): Total sequence length
  - 0: Random length (4n to 8n, where n = n_types)
  - Positive: Fixed length (must be even)

- **to_fill_length** (0 or positive): Number of brackets to complete
  - 0: Random (20%-50% of total)
  - Positive: Exact number to fill

- **nesting_depth** (0 or positive): Minimum nesting depth
  - 0: No constraint
  - Positive: Ensures at least this depth

## Validation Rules

1. **Balanced**: Every opening bracket has a matching closing bracket
2. **Proper Nesting**: Brackets are properly nested (no crossing)
3. **Unique Completion**: Only one valid way to complete the sequence

## Examples

### Easy (n_types=2, length=8)
```
Question: "([("
Answer:   "([()]))"
```

### Medium (n_types=3, length=16)
```
Question: "([{[{("
Answer:   "([{[{()}]}])"
```

### Hard (n_types=4, length=24, depth=4)
```
Question: "([{<[{<("
Answer:   "([{<[{<()>}]>}])"
```

## Scoring

- **Correct**: 1.0 - Exact match of complete valid sequence
- **Incorrect**: 0.0 - Any other output

## Implementation Details

### Generator

**File**: `generator.py`

**Key Methods**:
- `generate()`: Main generation method
- `_generate_valid_sequence()`: Creates valid Dyck sequences
- `_check_unique_completion()`: Validates unique completion property
- `extract_answer()`: Extracts bracket sequence from model output

**Determinism**: Uses seed-based random number generation for reproducibility

### Verifier

**File**: `verifier.py`

**Key Methods**:
- `verify()`: Compares extracted answer with correct sequence

**Validation**: Extracts longest valid bracket sequence from model output and compares with ground truth

## Configuration

Default configuration:
```python
{
    "n_types": 3,
    "total_length": 0,      # Random
    "to_fill_length": 0,    # Random
    "nesting_depth": 0      # No constraint
}
```

Custom configuration example:
```python
{
    "n_types": 4,           # All bracket types
    "total_length": 32,     # Fixed length
    "to_fill_length": 16,   # Fill half
    "nesting_depth": 5      # Deep nesting
}
```

## Task ID Range

**0 - 99,999,999** (100 million unique tasks)

## Usage

```python
from env import Actor

actor = Actor(api_key="your-key")

# Use default configuration
result = await actor.evaluate(task_id=1000)

# Use custom configuration
actor = Actor(
    api_key="your-key",
    task_configs={
        "dyck_language": {
            "n_types": 4,
            "total_length": 40,
            "nesting_depth": 6
        }
    }
)
result = await actor.evaluate(task_id=1000)
```

## References

- [Dyck Language (Wikipedia)](https://en.wikipedia.org/wiki/Dyck_language)
- Based on formal language theory and context-free grammars
