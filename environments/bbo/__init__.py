"""Principled Black-Box Optimization Environment.

Everything flows from a single seed via maximum-entropy priors.
No presets, no hand-crafting.

f(x) = γ·(quadratic + softmax_planes) + (1-γ)·spectral + ε‖x‖²
"""

from .env import Actor, Session, Query
from .generator import generate, Problem, Hyperparameters, PRIORS
from .function_ast import (
    RandomFunction,
    FunctionProperties,
    random_psd,
    random_orthogonal,
    matern_frequencies,
)

__all__ = [
    "Actor", "Session", "Query",
    "generate", "Problem", "Hyperparameters", "PRIORS",
    "RandomFunction", "FunctionProperties",
    "random_psd", "random_orthogonal", "matern_frequencies",
]
