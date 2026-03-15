"""Agent registry for SWE-INFINITE.

Provides CodexAgent and MiniSWEAgent, with automatic selection based on task.
"""

from .codex import CodexAgent, CodexConfig, CodexResult
from .miniswe import MiniSWEAgent, MiniSWEConfig, MiniSWEResult

SUPPORTED_AGENTS = ("codex", "miniswe")

# Languages where codex CLI works well (native tool-use, single-turn)
_CODEX_PREFERRED_LANGUAGES = frozenset()  # empty for now; miniswe is default


def select_agent(task: dict, override: str = "") -> str:
    """Pick the best agent for a task.

    Priority: explicit override > task metadata > task_id parity (alternating).
    Even task_id → codex, odd task_id → miniswe.
    """
    if override and override in SUPPORTED_AGENTS:
        return override

    # Task-level override (set in R2 JSON if needed)
    task_agent = task.get("agent")
    if task_agent and task_agent in SUPPORTED_AGENTS:
        return task_agent

    # Alternate by numeric task_id parity: even → codex, odd → miniswe
    task_id_num = task.get("_task_id", 0)
    return "codex" if task_id_num % 2 == 0 else "miniswe"


__all__ = [
    "CodexAgent", "CodexConfig", "CodexResult",
    "MiniSWEAgent", "MiniSWEConfig", "MiniSWEResult",
    "SUPPORTED_AGENTS", "select_agent",
]
