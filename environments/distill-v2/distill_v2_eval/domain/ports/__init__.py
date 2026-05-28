"""Protocol-based ports used by the eval-only slice.

Only the ports the evaluator actually depends on are exposed —
generator-side ports (sandbox, teacher, task source, metadata store,
...) are intentionally absent from this environment.
"""

from distill_v2_eval.domain.ports.forward_engine import ForwardEngine
from distill_v2_eval.domain.ports.scoring import (
    RolloutScoringStrategy,
    ScoringStrategy,
)

__all__ = [
    "ForwardEngine",
    "RolloutScoringStrategy",
    "ScoringStrategy",
]
