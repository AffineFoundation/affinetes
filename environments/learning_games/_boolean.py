from __future__ import annotations

import ast
import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

from ._utils import BudgetMixin


ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Tuple,
}


def _validate_expr_tree(expr: str, variables: Sequence[str]) -> ast.AST:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST_NODES:
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in variables:
            raise ValueError(f"Unknown variable '{node.id}'")
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            raise ValueError("Only boolean constants True/False are allowed")
    return tree


def compile_expr(expr: str, variables: Sequence[str]) -> Callable[..., bool]:
    tree = _validate_expr_tree(expr, variables)
    code = compile(tree, filename="<user-expr>", mode="eval")

    def fn(**kwargs):
        if set(kwargs.keys()) != set(variables):
            missing = set(variables) - set(kwargs)
            extra = set(kwargs) - set(variables)
            raise ValueError(f"Expected variables {variables}. Missing {missing}, extra {extra}")
        return bool(eval(code, {"__builtins__": {}}, kwargs))

    return fn


def random_expr(vars_: Sequence[str], rng: random.Random, depth: int) -> str:
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(vars_)
    if rng.random() < 0.25:
        inner = random_expr(vars_, rng, depth - 1)
        return f"(not {inner})"
    left = random_expr(vars_, rng, depth - 1)
    right = random_expr(vars_, rng, depth - 1)
    op = "and" if rng.random() < 0.5 else "or"
    return f"({left} {op} {right})"


@dataclass
class HiddenFormula:
    variables: Sequence[str]
    expr: str
    fn: Callable[..., bool]

    def evaluate(self, assignments: Dict[str, bool]) -> bool:
        return self.fn(**assignments)


@dataclass
class Session(BudgetMixin):
    formula: HiddenFormula
    query_budget: int
    queries_used: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def query(self, assignments: Dict[str, bool]) -> bool:
        self._count()
        missing = set(self.formula.variables) - set(assignments)
        extra = set(assignments) - set(self.formula.variables)
        if missing or extra:
            raise ValueError(f"Expected variables {self.formula.variables}. Missing {missing}, extra {extra}")
        return self.formula.evaluate(assignments)

    def score_submission(self, expr: str) -> Dict[str, float]:
        user_fn = compile_expr(expr, self.formula.variables)
        combos = list(itertools.product([False, True], repeat=len(self.formula.variables)))
        correct = 0
        for combo in combos:
            assignment = dict(zip(self.formula.variables, combo))
            if user_fn(**assignment) == self.formula.evaluate(assignment):
                correct += 1
        accuracy = correct / len(combos)
        return {"accuracy": accuracy, "score": accuracy}


def easy_formula() -> HiddenFormula:
    variables = ["x", "y", "z"]
    expr = "(x and y) or (not z and x)"
    fn = compile_expr(expr, variables)
    return HiddenFormula(variables, expr, fn)


def random_formula(variables: Sequence[str], depth: int, seed: int | None) -> HiddenFormula:
    rng = random.Random(seed)
    expr = random_expr(variables, rng, depth)
    fn = compile_expr(expr, variables)
    return HiddenFormula(list(variables), expr, fn)


def preset(difficulty: str, seed: int | None) -> Session:
    level = difficulty.lower()
    if level == "easy":
        return Session(formula=easy_formula(), query_budget=6, rng=random.Random(seed))
    if level == "medium":
        vars_ = ["a", "b", "c", "d"]
        form = random_formula(vars_, depth=3, seed=seed)
        return Session(formula=form, query_budget=8, rng=random.Random(seed))
    if level == "hard":
        vars_ = ["a", "b", "c", "d", "e"]
        form = random_formula(vars_, depth=4, seed=seed)
        return Session(formula=form, query_budget=10, rng=random.Random(seed))
    raise ValueError(f"Unknown difficulty '{difficulty}'. Choose easy, medium, or hard.")

