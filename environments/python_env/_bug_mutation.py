"""AST mutation utilities for the bug-finding task."""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import random
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from _program import DEFAULT_ALLOWED_OPS, build_program


@dataclass(frozen=True)
class Mutation:
    line: int
    col: int
    original: str
    mutated: str
    operator: str


@dataclass(frozen=True)
class Challenge:
    original_code: str
    mutated_code: str
    expected_output: str
    actual_output: str
    mutation: Mutation
    seed: int


def _safe_exec(code: str) -> str | None:
    """Execute code safely, returning stdout or None on error."""
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<string>", "exec"), {})
        return buf.getvalue()
    except Exception:
        return None


def mutate_binop(node: ast.BinOp, rng: random.Random) -> Tuple[ast.BinOp, str] | None:
    swaps = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}
    if type(node.op) not in swaps:
        return None
    new = copy.deepcopy(node)
    new.op = swaps[type(node.op)]()
    return new, "AOR"


def mutate_augassign(node: ast.AugAssign, rng: random.Random) -> Tuple[ast.AugAssign, str] | None:
    swaps = {ast.Add: ast.Sub, ast.Sub: ast.Add}
    if type(node.op) not in swaps:
        return None
    new = copy.deepcopy(node)
    new.op = swaps[type(node.op)]()
    return new, "AOR"


def mutate_constant(node: ast.Constant, rng: random.Random) -> Tuple[ast.Constant, str] | None:
    if not isinstance(node.value, int):
        return None
    new = copy.deepcopy(node)
    delta = rng.choice([-1, 1]) if node.value > 0 else 1
    new.value = node.value + delta
    return new, "CR"


def mutate_call(node: ast.Call, rng: random.Random) -> Tuple[ast.Call, str] | None:
    if not isinstance(node.func, ast.Name):
        return None
    swaps = {"min": "max", "max": "min"}
    if node.func.id not in swaps:
        return None
    new = copy.deepcopy(node)
    new.func.id = swaps[node.func.id]
    return new, "MR"


def mutate_method(node: ast.Expr, rng: random.Random) -> Tuple[ast.Pass, str] | None:
    if not isinstance(node.value, ast.Call):
        return None
    if not isinstance(node.value.func, ast.Attribute):
        return None
    if node.value.func.attr not in ("append", "extend", "insert", "pop", "reverse", "sort"):
        return None
    return ast.Pass(), "SD"


MUTATORS = [
    (ast.BinOp, mutate_binop),
    (ast.AugAssign, mutate_augassign),
    (ast.Constant, mutate_constant),
    (ast.Call, mutate_call),
    (ast.Expr, mutate_method),
]


class Mutator(ast.NodeTransformer):
    """Single-shot AST mutator."""

    def __init__(self, target_line: int, target_col: int, new_node: ast.AST):
        self.target_line = target_line
        self.target_col = target_col
        self.new_node = new_node
        self.applied = False

    def generic_visit(self, node):
        if (
            not self.applied
            and hasattr(node, "lineno")
            and node.lineno == self.target_line
            and getattr(node, "col_offset", 0) == self.target_col
        ):
            self.applied = True
            return self.new_node
        return super().generic_visit(node)


def find_mutation_candidates(tree: ast.AST, rng: random.Random) -> List[Tuple[ast.AST, ast.AST, str]]:
    candidates = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        for node_type, mutator in MUTATORS:
            if isinstance(node, node_type):
                result = mutator(node, rng)
                if result:
                    new_node, op = result
                    candidates.append((node, new_node, op))
    return candidates


def apply_mutation(code: str, rng: random.Random) -> Tuple[str, Mutation] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    candidates = find_mutation_candidates(tree, rng)
    if not candidates:
        return None

    rng.shuffle(candidates)

    for original_node, mutated_node, operator in candidates:
        line = original_node.lineno
        col = getattr(original_node, "col_offset", 0)

        new_tree = copy.deepcopy(tree)
        mutator = Mutator(line, col, copy.deepcopy(mutated_node))
        mutated_tree = mutator.visit(new_tree)
        ast.fix_missing_locations(mutated_tree)

        if not mutator.applied:
            continue

        try:
            new_code = ast.unparse(mutated_tree)
        except Exception:
            continue

        if new_code == code:
            continue

        new_output = _safe_exec(new_code)
        if new_output is None:
            continue

        orig_output = _safe_exec(code)
        if orig_output is None or new_output == orig_output:
            continue

        try:
            orig_repr = ast.unparse(original_node)
            mut_repr = ast.unparse(mutated_node)
        except Exception:
            continue

        mutation = Mutation(line=line, col=col, original=orig_repr, mutated=mut_repr, operator=operator)
        return new_code, mutation

    return None


def generate_challenge(
    seed: int,
    allowed_ops: Sequence[str] = DEFAULT_ALLOWED_OPS,
    op_count: int = 8,
    max_digits: int = 4,
) -> Challenge | None:
    rng = random.Random(seed)

    for attempt in range(10):
        code_seed = seed + attempt * 10000
        lines, _ = build_program(seed=code_seed, op_count=op_count, allowed_ops=allowed_ops, max_digits=max_digits)
        code = "\n".join(lines)

        expected = _safe_exec(code)
        if expected is None:
            continue

        result = apply_mutation(code, rng)
        if result is None:
            continue

        mutated_code, mutation = result
        actual = _safe_exec(mutated_code)

        if actual is None or actual == expected:
            continue

        return Challenge(
            original_code=code,
            mutated_code=mutated_code,
            expected_output=expected,
            actual_output=actual,
            mutation=mutation,
            seed=code_seed,
        )

    return None


def normalize(code: str) -> str:
    """Normalize code via AST for comparison."""
    try:
        return ast.unparse(ast.parse(code, mode="eval"))
    except Exception:
        try:
            return ast.unparse(ast.parse(code, mode="exec"))
        except Exception:
            return " ".join(code.split())


def parse_response(response: str) -> tuple[int | None, str | None]:
    line_match = re.search(r"<LINE>\s*(\d+)\s*</LINE>", response)
    fix_match = re.search(r"<FIX>\s*(.*?)\s*</FIX>", response, re.DOTALL)
    line = int(line_match.group(1)) if line_match else None
    fix = fix_match.group(1).strip() if fix_match else None
    return line, fix


def evaluate_response(response: str, challenge: Challenge) -> bool:
    provided_line, provided_fix = parse_response(response)
    if provided_line is None or provided_fix is None:
        return False
    if provided_line != challenge.mutation.line:
        return False
    return normalize(provided_fix) == normalize(challenge.mutation.original)


def format_prompt(challenge: Challenge) -> str:
    numbered = "\n".join(f"{i + 1:3}| {line}" for i, line in enumerate(challenge.mutated_code.split("\n")))
    return (
        "Find the bug in this Python code.\n\n"
        "## Code\n"
        f"```\n{numbered}\n```\n\n"
        "## Current Output (wrong)\n"
        f"```\n{challenge.actual_output}```\n\n"
        "## Expected Output\n"
        f"```\n{challenge.expected_output}```\n\n"
        "## Instructions\n"
        "Identify the bug and provide the fix in this exact format:\n"
        "<LINE>line_number</LINE>\n"
        "<FIX>corrected_expression</FIX>\n\n"
        "Example: If line 5 has `x - 1` but should be `x + 1`:\n"
        "<LINE>5</LINE>\n"
        "<FIX>x + 1</FIX>"
    )
