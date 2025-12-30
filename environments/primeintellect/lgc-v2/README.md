# LGC-V2 Environment

Seed-based logic reasoning task generation environment with fully reproducible evaluation.

## Quick Start

```python
from env import Actor

actor = Actor(api_key="your-api-key")

# task_id automatically identifies task type and generates deterministic challenge
result = await actor.evaluate(task_id=1000)

print(f"Score: {result['score']}")
print(f"Task Type: {result['extra']['task_type']}")  # dyck_language
```

## Core Features

### Intelligent Task ID Encoding

```
task_id = task_type_id × 100,000,000 + seed

Examples:
  1000        → dyck_language (type_id=0), seed=1000
  100,000,500 → future_task (type_id=1), seed=500
```

- ✅ **Automatic task type detection** - No manual specification needed
- ✅ **Fully reproducible** - Same task_id produces same result
- ✅ **Unlimited scalability** - 100 million unique challenges per task type

## Directory Structure

```
lgc-v2/
├── games/              # All task implementations
│   ├── verifiers.py   # Verifier registry
│   ├── dyck_language/ # Example task
│   │   ├── generator.py
│   │   ├── verifier.py
│   │   └── README.md
│   └── task_template/ # New task template
├── env.py              # Actor API
├── logic_task_v2.py   # Task loader
├── models.py           # Data models
└── docs/               # Detailed documentation
```

## Adding New Tasks

### 1. Copy Template
```bash
cp -r games/task_template games/your_task
```

### 2. Implement Code
- `games/your_task/generator.py` - Implement `generate()` method
- `games/your_task/verifier.py` - Implement `verify()` method
- `games/your_task/README.md` - Write documentation

### 3. Register Task

**logic_task_v2.py**:
```python
SUPPORTED_TASKS = {
    "your_task": {
        "task_type_id": 1,
        "module": "games.your_task.generator",
        "class": "YourTaskGenerator",
        "default_config": {...}
    }
}
```

**games/verifiers.py**:
```python
from games.your_task.verifier import YourTaskVerifier

verifier_classes = {
    "your_task": YourTaskVerifier,
}
```

## Usage Examples

### Batch Evaluation
```python
for task_id in range(1000):
    result = await actor.evaluate(task_id=task_id)
    print(f"Task {task_id}: {'✓' if result['success'] else '✗'}")
```

### Custom Configuration
```python
actor = Actor(
    api_key="your-key",
    task_configs={
        "dyck_language": {
            "n_types": 4,
            "total_length": 40,
            "nesting_depth": 5
        }
    }
)
```

### Encode/Decode Task ID
```python
from logic_task_v2 import LogicTaskV2

# Encode
task_id = LogicTaskV2.encode_task_id("dyck_language", 500)  # → 500

# Decode
task_type, seed = LogicTaskV2.decode_task_id(500)  # → ("dyck_language", 500)
```

## Task ID Allocation

| Task Type | ID Range | Capacity |
|-----------|----------|----------|
| dyck_language | 0-99,999,999 | 100M |
| game_of_24 | 100,000,000-199,999,999 | 100M |
| operation | 200,000,000-299,999,999 | 100M |
| cryptarithm | 300,000,000-399,999,999 | 100M |
| Reserved 1 | 400,000,000-499,999,999 | 100M |

## Documentation
- `games/dyck_language/README.md` - Dyck language task documentation
- `games/game_of_24/README.md` - Game of 24 task documentation
- `games/operation/README.md` - Operation task documentation
- `games/cryptarithm/README.md` - Cryptarithm task documentation

## TODO: Task Integration Roadmap

Based on analysis of 33 task types from `lgc/i3_logic`, the following tasks are candidates for integration:

### High Priority (Ready for Integration)

1. ✅ **game_of_24** (24 Game) - COMPLETED
   - Given 4-6 numbers, use arithmetic operations to reach target (24/36/48/60/72/100)
   - Seed-based generation with multilingual prompts (Chinese/English)
   - Supports unsolvable problems (model returns None)
   - Model success rate: ~10%

2. ✅ **operation** (Symbol Operations) - COMPLETED
   - Define custom operators (e.g., a△b = 2a+b), then evaluate expressions
   - Seed-based generation with conditional branches
   - Supports 1-3 custom symbols with operator precedence
   - Multilingual prompts (Chinese/English)

3. ✅ **cryptarithm** (Cryptarithmetic Puzzles) - COMPLETED
   - SEND + MORE = MONEY (letters represent digits)
   - Pure logic reasoning with unique solution verification
   - Seed-based generation with multilingual prompts (Chinese/English)
   - Backtracking ensures exactly one unique solution

4. **boolean_expressions** (Boolean Logic)
   - Evaluate nested logical statements with true/false facts
   - Pure logic, simple verification
   - Config: nesting depth (2-5), option count (3-6)

### Medium Priority (Requires Adaptation)

5. **web_of_lies** (Truth-Teller Logic)
   - Deduce who tells truth/lies from given statements
   - Complex reasoning, unique solutions
   - Challenge: complex generation logic

6. **object_counting** (Counting Problems)
   - Multi-step counting scenarios with narrative context
   - Practical math reasoning
   - Challenge: story template handling

7. **time_sequence** (Schedule Reasoning)
   - Optimize schedules given rules and constraints
   - Real-world application scenarios
   - Challenge: complex verification logic

### Puzzle Games (For Specialized Training)

8. **sudoku**, **futoshiki**, **number_wall**, **minesweeper**
   - Classic grid-based logic puzzles
   - Well-defined rules with unique solutions

### Integration Order

Recommended sequence based on implementation complexity and value:
1. ✅ dyck_language (completed)
2. ✅ game_of_24 (completed)
3. ✅ operation (completed)
4. ⏳ cryptarithm
5. ⏳ boolean_expressions
