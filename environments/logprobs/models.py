"""Data models for logprobs environment."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Challenge:
    """A generated task challenge."""
    env: str
    prompt: str
    extra: Dict[str, Any]
