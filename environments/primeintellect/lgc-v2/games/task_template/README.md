# Task Template

## Overview

TODO: Provide a brief overview of your task.

Example: This task tests the model's ability to solve arithmetic problems.

## Task Description

**Input**: TODO: Describe what the model receives as input

**Output**: TODO: Describe what the model should output

**Example**:
```
Input:  TODO: Example input
Output: TODO: Example output
```

## Difficulty Parameters

TODO: Document your configuration parameters

- **parameter1**: Description
  - Value1: Effect
  - Value2: Effect

- **parameter2**: Description
  - Range: Min-Max
  - Default: Value

## Validation Rules

TODO: List the rules for a valid answer

1. Rule 1
2. Rule 2
3. Rule 3

## Examples

### Easy
```
Question: TODO
Answer:   TODO
```

### Medium
```
Question: TODO
Answer:   TODO
```

### Hard
```
Question: TODO
Answer:   TODO
```

## Scoring

- **Correct**: 1.0 - TODO: Define what counts as correct
- **Incorrect**: 0.0 - TODO: Define what counts as incorrect

## Implementation Details

### Generator

**File**: `generator.py`

**Key Methods**:
- `generate()`: Main generation method
- `_generate_one()`: Generate single instance
- `extract_answer()`: Extract answer from model output

**Determinism**: Uses seed-based random number generation

### Verifier

**File**: `verifier.py`

**Key Methods**:
- `verify()`: Verify answer correctness

## Configuration

Default configuration:
```python
{
    "parameter1": default_value1,
    "parameter2": default_value2,
}
```

Custom configuration example:
```python
{
    "parameter1": custom_value1,
    "parameter2": custom_value2,
}
```

## Task ID Range

**TODO** - **TODO** (100 million unique tasks)

Example: **100,000,000 - 199,999,999** for task_type_id=1

## Usage

```python
from env import Actor

actor = Actor(api_key="your-key")

# Use default configuration (replace with your task_id range)
result = await actor.evaluate(task_id=100_000_000)

# Use custom configuration
actor = Actor(
    api_key="your-key",
    task_configs={
        "your_task_name": {
            "parameter1": value1,
            "parameter2": value2
        }
    }
)
result = await actor.evaluate(task_id=100_000_000)
```

## References

TODO: Add relevant references, papers, or resources

- [Reference 1](URL)
- [Reference 2](URL)
