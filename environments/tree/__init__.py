"""Tree Deduction Environment.

Deduce a tree's structure from partial observations.

Parameters:
    n: Number of nodes
    reveal_fraction: Fraction of edges revealed (0 = hard, 1 = trivial)

Usage:
    from tree import Actor

    actor = Actor()
    result = await actor.evaluate_deduction(
        n=8,
        reveal_fraction=0.5,
        task_id=42
    )
"""

from .env import Actor
from ._hidden_tree import HiddenTree
from ._deduction import (
    generate_problem,
    generate_observations,
    evaluate_answer,
    compute_ambiguity,
    DeductionProblem,
    TreeObservations,
    Observation,
    TreeDeductionTask,
)

__all__ = [
    "Actor",
    "HiddenTree",
    "TreeDeductionTask",
    "generate_problem",
    "generate_observations",
    "evaluate_answer",
    "compute_ambiguity",
    "DeductionProblem",
    "TreeObservations",
    "Observation",
]
