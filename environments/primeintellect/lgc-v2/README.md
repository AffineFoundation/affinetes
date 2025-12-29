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
| Reserved 1 | 100,000,000-199,999,999 | 100M |
| Reserved 2 | 200,000,000-299,999,999 | 100M |

## Documentation
- `games/dyck_language/README.md` - Example task documentation
