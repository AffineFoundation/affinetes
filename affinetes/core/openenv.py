"""OpenEnv shared protocol models.

This module is intended to be usable in two contexts:
1) Host-side Python (normal affinetes package).
2) Environment containers (via minimal package injection under /app/affinetes).
"""

from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    """OpenEnv reset request"""

    task_id: Optional[int] = None
    seed: Optional[int] = None
    # Additional environment-specific parameters can be passed
    kwargs: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    """OpenEnv step request"""

    action: str
    # Optional: bind step to a specific episode (training-friendly)
    episode_id: Optional[str] = None
    # Additional step-level parameters
    kwargs: Optional[Dict[str, Any]] = None


class OpenEnvResponse(BaseModel):
    """OpenEnv response for reset/step/state"""

    # Episode identifier for multi-step interaction
    episode_id: Optional[str] = None
    observation: str
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


