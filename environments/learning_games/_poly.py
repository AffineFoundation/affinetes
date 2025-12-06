from __future__ import annotations

import ast
import numbers
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from ._utils import BudgetMixin


POLY_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Tuple,
}


def compile_poly_expr(expr: str) -> Callable[[float], float]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    for node in ast.walk(tree):
        if type(node) not in POLY_ALLOWED_NODES:
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id != "x":
            raise ValueError("Only variable 'x' is allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, numbers.Number):
            raise ValueError("Only numeric constants allowed")

    code = compile(tree, filename="<user-poly>", mode="eval")

    def fn(x: float) -> float:
        return eval(code, {"__builtins__": {}}, {"x": x})

    return fn


def eval_poly(coeffs: Sequence[float], x: float) -> float:
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


@dataclass
class HiddenPoly:
    coeffs: List[int]

    def __post_init__(self):
        if not self.coeffs:
            raise ValueError("Coefficient list cannot be empty")
        if self.coeffs[-1] == 0:
            raise ValueError("Leading coefficient must be non-zero")

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def evaluate(self, x: float) -> float:
        return eval_poly(self.coeffs, x)


@dataclass
class Session(BudgetMixin):
    poly: HiddenPoly
    query_budget: int
    queries_used: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def query(self, x: float) -> float:
        self._count()
        return self.poly.evaluate(x)

    def score_submission(self, submission) -> dict:
        coeffs = None
        fn = None

        if isinstance(submission, (list, tuple)):
            if not all(isinstance(c, numbers.Number) for c in submission):
                raise ValueError("Coefficient list must be numeric")
            coeffs = list(submission)
        elif isinstance(submission, str):
            try:
                literal = ast.literal_eval(submission)
                if isinstance(literal, (list, tuple)) and all(isinstance(c, numbers.Number) for c in literal):
                    coeffs = list(literal)
            except Exception:
                pass
            if coeffs is None:
                fn = compile_poly_expr(submission)
        else:
            raise ValueError("Submission must be a list/tuple of coeffs or an expression string")

        if coeffs is not None:
            if coeffs and coeffs[-1] == 0:
                raise ValueError("Leading coefficient must be non-zero")
            if coeffs == self.poly.coeffs:
                return {"accuracy": 1.0, "score": 1.0}
            fn = lambda x: eval_poly(coeffs, x)

        test_points = list(range(-50, 51))
        correct = 0
        for x in test_points:
            try:
                if fn(x) == self.poly.evaluate(x):
                    correct += 1
            except Exception:
                continue
        accuracy = correct / len(test_points)
        return {"accuracy": accuracy, "score": accuracy}


def default_poly() -> HiddenPoly:
    return HiddenPoly([7, -2, 3])


def random_poly(max_degree: int, coeff_range: range, rng: random.Random) -> HiddenPoly:
    degree = rng.randint(1, max_degree)
    coeffs: List[int] = []
    for i in range(degree + 1):
        if i == degree:
            c = 0
            while c == 0:
                c = rng.choice(coeff_range)
        else:
            c = rng.choice(coeff_range)
        coeffs.append(int(c))
    return HiddenPoly(coeffs)


def preset(difficulty: str, seed: int | None) -> Session:
    rng = random.Random(seed)
    level = difficulty.lower()
    if level == "easy":
        return Session(poly=default_poly(), query_budget=8, rng=rng)
    if level == "medium":
        return Session(poly=random_poly(3, range(-8, 9), rng), query_budget=10, rng=rng)
    if level == "hard":
        return Session(poly=random_poly(4, range(-10, 11), rng), query_budget=12, rng=rng)
    raise ValueError(f"Unknown difficulty '{difficulty}'. Choose easy, medium, or hard.")

