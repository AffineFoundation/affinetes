"""Principled black-box optimization environment.

Everything flows from a single seed via maximum-entropy priors.
Scoring uses information-theoretic regret bounds.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import time
import numpy as np

from .generator import generate, Problem


@dataclass
class Query:
    x: np.ndarray
    value: float
    gradient: Optional[np.ndarray] = None
    time: float = field(default_factory=time.time)


@dataclass
class Session:
    problem: Problem
    allow_gradient: bool = True
    queries: List[Query] = field(default_factory=list)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def budget(self) -> int:
        return self.problem.hp.query_budget

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.n_queries)

    def _validate(self, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        d = self.problem.hp.dimension
        if x.shape != (d,):
            raise ValueError(f"Expected {d}D point, got {x.shape}")
        b = self.problem.bounds
        if np.any(x < b[:, 0] - 1e-10) or np.any(x > b[:, 1] + 1e-10):
            raise ValueError(f"Point outside [{b[0,0]:.1f}, {b[0,1]:.1f}]^{d}")
        return x

    def query(self, x: Sequence[float], with_grad: bool = False) -> Dict[str, Any]:
        if self.n_queries >= self.budget:
            raise ValueError(f"Budget exhausted ({self.budget} queries)")
        if with_grad and not self.allow_gradient:
            raise ValueError("Gradient queries not allowed")

        x = self._validate(x)
        f = self.problem.function
        val = f(x)
        grad = f.gradient(x) if with_grad else None
        self.queries.append(Query(x.copy(), val, grad.copy() if grad is not None else None))

        result = {"value": val, "queries_used": self.n_queries, "queries_remaining": self.remaining}
        if grad is not None:
            result["gradient"] = grad.tolist()
        return result

    def score(self, x: Sequence[float]) -> Dict[str, Any]:
        x = self._validate(x)
        f_sub = self.problem.function(x)
        f_opt, f_init = self.problem.f_opt, self.problem.f_init

        regret = f_sub - f_opt
        gap = f_init - f_opt
        normalized = regret / gap if gap > 1e-10 else (0.0 if regret < 1e-10 else float('inf'))

        bound = self.problem.function.properties().theoretical_regret_bound(self.n_queries)
        ratio = min(regret / bound, 100.0) if bound > 1e-10 else (0.0 if regret < 1e-10 else 100.0)

        return {
            "score": float(np.exp(-ratio)),
            "success": bool(ratio < 2.0),
            "regret": float(regret),
            "normalized_regret": float(normalized),
            "regret_ratio": float(ratio),
            "theoretical_lower_bound": float(bound),
            "f_submitted": f_sub,
            "f_optimal": f_opt,
            "f_initial": f_init,
            "queries_used": self.n_queries,
            "query_budget": self.budget,
        }


class Actor:
    """Black-box optimization via seed-generated problems.

    No presets. Everything from the seed through max-entropy priors.
    """

    def __init__(self, seed: Optional[int] = None):
        seed = seed if seed is not None else np.random.default_rng().integers(2**31)
        self.session = Session(generate(seed))

    async def reset(self, seed: Optional[int] = None, allow_gradient: bool = True) -> Dict[str, Any]:
        seed = seed if seed is not None else np.random.default_rng().integers(2**31)
        self.session = Session(generate(seed), allow_gradient)
        return await self.spec()

    async def spec(self) -> Dict[str, Any]:
        s = self.session
        return {
            **s.problem.spec(),
            "task": "Minimize f: ℝⁿ → ℝ",
            "queries_used": s.n_queries,
            "queries_remaining": s.remaining,
            "allow_gradient": s.allow_gradient,
        }

    async def query(self, x: Sequence[float]) -> Dict[str, Any]:
        return self.session.query(x, with_grad=False)

    async def query_gradient(self, x: Sequence[float]) -> Dict[str, Any]:
        return self.session.query(x, with_grad=True)

    async def submit(self, x: Sequence[float]) -> Dict[str, Any]:
        t0 = time.time()
        result = self.session.score(x)
        p = self.session.problem
        hp = p.hp
        result.update(
            task_name="black_box_optimization",
            time_taken=time.time() - t0,
            optimal_x=p.x_opt.tolist(),
            optimal_value=p.f_opt,
            submitted_x=list(x),
            hyperparameters=dict(
                dimension=hp.dimension, smoothness=hp.smoothness,
                convexity_weight=hp.convexity, condition_number=hp.condition,
                length_scale=hp.length_scale, n_spectral=hp.n_spectral, n_planes=hp.n_planes,
            ),
        )
        return result

    async def query_history(self) -> List[Dict[str, Any]]:
        return [
            {"x": q.x.tolist(), "value": q.value,
             "gradient": q.gradient.tolist() if q.gradient is not None else None, "time": q.time}
            for q in self.session.queries
        ]

    async def best_so_far(self) -> Dict[str, Any]:
        qs = self.session.queries
        if not qs:
            return {"x": None, "value": None, "index": None}
        i = min(range(len(qs)), key=lambda j: qs[j].value)
        return {"x": qs[i].x.tolist(), "value": qs[i].value, "index": i}

    async def evaluate(self, x: Optional[Sequence[float]] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if x is None:
            return {**(await self.reset(seed)), "message": "query(x) → f(x), submit(x) → score"}
        return await self.submit(x)

    async def info(self) -> Dict[str, Any]:
        return {
            "design": ["Seed determines everything", "Max-entropy priors", "Info-theoretic scoring"],
            "priors": {
                "dimension": "Geometric(0.1)", "smoothness": "Gamma(2,2)",
                "length_scale": "LogNormal(0,1)", "convexity": "Uniform[0,1]",
                "condition": "LogUniform[1,22000]", "n_spectral": "Poisson(50)", "n_planes": "Poisson(10)",
            },
            "scoring": "exp(-regret/lower_bound), success if ratio < 2",
            "function": "γ·(quadratic + softmax_planes) + (1-γ)·spectral + ε‖x‖²",
        }
