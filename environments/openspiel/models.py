"""Data models for OpenSpiel environment"""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class Challenge(BaseModel):
    """Represents a game challenge"""
    prompt: str
    game_name: str
    task_id: int
    seed: int
    extra: Optional[Dict[str, Any]] = None