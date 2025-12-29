"""Data models for Logic V2 environment"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Challenge:
    """Challenge data structure"""
    env: str
    prompt: str
    extra: Dict[str, Any]
