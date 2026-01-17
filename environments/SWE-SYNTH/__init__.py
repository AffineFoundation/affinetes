"""
SWE-SYNTH: Synthetic SWE-bench evaluation environment

Dynamic bug generation + fixing evaluation with caching for reproducibility.
"""

from .env import SynthActor, Actor
from .breaker import BUG_TYPES, generate_bug
from .cache import TwoLevelCache, LocalCache, R2Cache, CacheLockError

__all__ = [
    "SynthActor",
    "Actor",
    "BUG_TYPES",
    "generate_bug",
    "TwoLevelCache",
    "LocalCache",
    "R2Cache",
    "CacheLockError",
]
