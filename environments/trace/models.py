"""Data models for trace task challenges and evaluations"""

import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Challenge(BaseModel):
    """Challenge specification for evaluation"""
    
    env: str
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[float] = Field(default_factory=lambda: time.time())

