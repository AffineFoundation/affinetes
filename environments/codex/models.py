"""Data models for Codex Django bug-fix challenges"""

import time
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class Challenge(BaseModel):
    """Challenge specification for Codex Django bug-fix evaluation"""
    
    env: str
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[float] = Field(default_factory=lambda: time.time())


class BugSpec(BaseModel):
    """Specification for a bug introduced by Codex"""
    
    file_path: str
    original_content: str
    buggy_content: str
    bug_description: str
    affected_tests: List[str] = Field(default_factory=list)

