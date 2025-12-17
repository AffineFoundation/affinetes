"""Data models for DeepSWE environment"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Challenge:
    """Challenge specification for DeepSWE tasks"""
    
    env: str
    prompt: str
    extra: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}