"""Data models for ARC-GEN challenges"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Challenge:
    """Challenge specification for evaluation"""

    env: str
    prompt: str
    extra: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = field(default_factory=lambda: time.time())
