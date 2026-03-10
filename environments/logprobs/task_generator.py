"""Task ID registry and prompt generator.

Task ID allocation (100M-per-type scheme):

  task_type_id = task_id // 100_000_000
  seed         = task_id %  100_000_000

  ┌─────────────────────┬───────────────────────────────────────────────┐
  │ task_type_id        │ task                                          │
  ├─────────────────────┼───────────────────────────────────────────────┤
  │ 0  (0–99,999,999)   │ common_sense_combo  (fact Q&A combos)         │
  │ 1  (100M–199,999,999│ qqr_travel  (QQR travel planning prompts)     │
  │ 2  (200M–299,999,999│ long_context  (long reading comprehension)     │
  │ 3  (300M–399,999,999│ openspiel_game  (OpenSpiel game prompts)       │
  │ 4+ (400M+)          │ reserved – raises NotImplementedError         │
  └─────────────────────┴───────────────────────────────────────────────┘
"""

import sys
from typing import Tuple

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

TASK_ID_RANGE = 100_000_000

# task_type_id → name
TASK_NAMES = {
    0: "common_sense_combo",
    1: "qqr_travel",
    2: "long_context",
    3: "openspiel_game",
}

# Lazy-loaded instances
_task_instances: dict = {}


def decode_task_id(task_id: int) -> Tuple[str, int]:
    """Return (task_type_name, seed) for the given task_id."""
    type_id = task_id // TASK_ID_RANGE
    seed = task_id % TASK_ID_RANGE

    if type_id not in TASK_NAMES:
        raise NotImplementedError(
            f"task_type_id={type_id} is reserved. "
            f"Supported ids: {sorted(TASK_NAMES)}"
        )
    return TASK_NAMES[type_id], seed


def encode_task_id(task_type: str, seed: int) -> int:
    """Inverse of decode_task_id."""
    for type_id, name in TASK_NAMES.items():
        if name == task_type:
            return type_id * TASK_ID_RANGE + seed
    raise ValueError(f"Unknown task type: {task_type!r}")


def _get_task(task_type: str):
    """Return a cached task instance for the given type."""
    if task_type not in _task_instances:
        if task_type == "common_sense_combo":
            from tasks.common_sense_combo import CommonSenseComboTask
            _task_instances[task_type] = CommonSenseComboTask()

        elif task_type == "qqr_travel":
            from tasks.qqr_travel import QQRTravelTask
            _task_instances[task_type] = QQRTravelTask()

        elif task_type == "long_context":
            from tasks.long_context import LongContextTask
            _task_instances[task_type] = LongContextTask()

        elif task_type == "openspiel_game":
            from tasks.openspiel_game import OpenSpielGameTask
            _task_instances[task_type] = OpenSpielGameTask()

        else:
            raise NotImplementedError(f"No implementation for task type: {task_type!r}")

    return _task_instances[task_type]


async def generate_prompt(task_id: int) -> Tuple[str, str, int, dict]:
    """Generate a prompt for the given task_id.

    Returns:
        (prompt, task_type, seed, metadata)
    """
    task_type, seed = decode_task_id(task_id)
    task = _get_task(task_type)
    prompt, metadata = task.generate(seed=seed)
    return prompt, task_type, seed, metadata
