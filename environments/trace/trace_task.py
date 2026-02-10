"""Trace task generator and evaluator using code execution and print injection"""

import ast
import asyncio
import gc
import json
import logging
import os
import random
import re
import signal
import sys
import tempfile
import subprocess
from typing import List, Generator, Tuple, Optional, Any, Dict
from datasets import load_dataset

sys.path.insert(0, '/app')
from models import Challenge

logger = logging.getLogger("trace_task")
handler = logging.StreamHandler(sys.stderr)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
logger.addHandler(handler)
logger.setLevel(os.environ.get("TRACE_LOG_LEVEL", "INFO"))

# Default timeout per execution (seconds)
DEFAULT_EXECUTION_TIMEOUT = 10

class PrintInjector(ast.NodeTransformer):
    def __init__(self, seed: int, max_injections: int = 6):
        self.rng = random.Random(seed)
        self.max_injections = max_injections
        self.injections_done = 0
        self.scopes: List[List[str]] = [[]]  # Stack of scopes

    @property
    def live_vars(self) -> List[str]:
        """Variables visible in the current scope"""
        all_vars = []
        for scope in self.scopes:
            for v in scope:
                if v not in all_vars:
                    all_vars.append(v)
        return all_vars

    def _add_live_var(self, name: str):
        if name not in self.scopes[-1]:
            self.scopes[-1].append(name)

    def _make_print(self) -> ast.Expr | None:
        """Create a print statement with deterministic output.

        For each variable, generates:
            repr(var) if isinstance(var, (int, float, str, bool, type(None))) else type(var).__name__

        This preserves actual values for basic types (high information, anti-memorization)
        while avoiding memory addresses for complex objects (functions, methods, etc.).
        """
        vars_available = self.live_vars
        if not vars_available:
            return None

        # Pick a few variables to print
        k = self.rng.randint(1, min(3, len(vars_available)))
        vars_to_print = self.rng.sample(vars_available, k)

        tag = f"__DBG_{self.injections_done}__"
        print_args = [ast.Constant(tag)]

        for var_name in vars_to_print:
            # Generate: repr(var) if isinstance(var, (int, float, str, bool, type(None))) else type(var).__name__
            var_node = ast.Name(id=var_name, ctx=ast.Load())

            # isinstance(var, (int, float, str, bool, type(None)))
            isinstance_check = ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[
                    var_node,
                    ast.Tuple(elts=[
                        ast.Name(id="int", ctx=ast.Load()),
                        ast.Name(id="float", ctx=ast.Load()),
                        ast.Name(id="str", ctx=ast.Load()),
                        ast.Name(id="bool", ctx=ast.Load()),
                        ast.Call(
                            func=ast.Name(id="type", ctx=ast.Load()),
                            args=[ast.Constant(None)],
                            keywords=[]
                        )
                    ], ctx=ast.Load())
                ],
                keywords=[]
            )

            # repr(var)
            repr_call = ast.Call(
                func=ast.Name(id="repr", ctx=ast.Load()),
                args=[ast.Name(id=var_name, ctx=ast.Load())],
                keywords=[]
            )

            # type(var).__name__
            type_name = ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="type", ctx=ast.Load()),
                    args=[ast.Name(id=var_name, ctx=ast.Load())],
                    keywords=[]
                ),
                attr="__name__",
                ctx=ast.Load()
            )

            # Ternary: repr(var) if isinstance(...) else type(var).__name__
            ternary = ast.IfExp(test=isinstance_check, body=repr_call, orelse=type_name)
            print_args.append(ternary)

        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=print_args,
                keywords=[]
            )
        )

    def _extract_names(self, target: ast.AST) -> Generator[str, None, None]:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from self._extract_names(elt)

    def _update_live_vars(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                for name in self._extract_names(t):
                    self._add_live_var(name)
        elif isinstance(stmt, ast.AugAssign):
            for name in self._extract_names(stmt.target):
                self._add_live_var(name)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            for name in self._extract_names(stmt.target):
                self._add_live_var(name)

    def _maybe_inject(self, body: List[ast.stmt]) -> List[ast.stmt]:
        new_body = []
        for stmt in body:
            # First, update live vars from this statement
            self._update_live_vars(stmt)
            
            # Visit the statement itself (to handle nested bodies)
            new_body.append(self.visit(stmt))

            if self.injections_done >= self.max_injections:
                continue

            # inject only after "safe" statements
            if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                if self.rng.random() < 0.6:
                    p = self._make_print()
                    if p:
                        new_body.append(p)
                        self.injections_done += 1

            elif isinstance(stmt, ast.If):
                if self.rng.random() < 0.4:
                    p = self._make_print()
                    if p:
                        new_body.append(p)
                        self.injections_done += 1

        return new_body

    def visit_Module(self, node: ast.Module):
        node.body = self._maybe_inject(node.body)
        return node

    def visit_For(self, node: ast.For):
        node.body = self._maybe_inject(node.body)
        node.orelse = self._maybe_inject(node.orelse)
        return node

    def visit_If(self, node: ast.If):
        node.body = self._maybe_inject(node.body)
        node.orelse = self._maybe_inject(node.orelse)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.scopes.append([]) # Push new local scope
        for arg in node.args.args:
            self._add_live_var(arg.arg)

        node.body = self._maybe_inject(node.body)
        self.scopes.pop() # Pop local scope
        return node
    
    def visit_Assign(self, node: ast.Assign):
        return node

    def visit_AugAssign(self, node: ast.AugAssign):
        return node


def clean_source(source: str) -> str:
    """Strip markdown code blocks if present"""
    source = source.strip()
    if source.startswith("```"):
        # Find the end of the first line (e.g., ```python)
        first_newline = source.find("\n")
        if first_newline != -1:
            source = source[first_newline:].strip()
        if source.endswith("```"):
            source = source[:-3].strip()
    return source


def inject_non_overfittable_prints(
    source: str,
    seed: int,
    max_injections: int = 6,
) -> str:
    """
    Inject execution-dependent print statements into a Python program.
    """
    source = clean_source(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"# Syntax Error in source: {e}\n{source}"

    injector = PrintInjector(seed=seed, max_injections=max_injections)
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def run_code_sync(code: str, input_data: str = "", timeout: int = 10) -> Tuple[str, Optional[str]]:
    """Execute code and capture stdout, optionally with stdin input"""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr if result.returncode != 0 else None
    except subprocess.TimeoutExpired:
        return "", "Timeout"
    except Exception as e:
        return "", str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def clean_llm_prediction(prediction: str) -> str:
    """Remove reasoning/thinking tags and markdown code blocks from LLM response"""
    # Remove thinking/reasoning tags
    prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
    prediction = re.sub(r"<thinking>.*?</thinking>", "", prediction, flags=re.DOTALL)
    
    # Handle remaining </> or </thinking> tags for thinking models
    if "</think>" in prediction:
        prediction = prediction.split("</think>")[-1].strip()
    if "</thinking>" in prediction:
        prediction = prediction.split("</thinking>")[-1].strip()

    # Remove markdown code blocks (handle both ```python ... ``` and ``` ... ```)
    code_block_match = re.search(r"```(?:[a-zA-Z]*)\n?(.*?)\n?```", prediction, flags=re.DOTALL)
    if code_block_match:
        prediction = code_block_match.group(1)
    
    return prediction.strip()


def compare_outputs(expected: str, actual: str) -> bool:
    """
    Compare two strings for equality, but with leniency:
    1. Normalize line endings
    2. Ignore all internal whitespace differences
    3. Case-insensitive comparison
    """
    def normalize(s):
        # Remove all whitespace and convert to lowercase
        return "".join(s.split()).lower()
    
    return normalize(expected) == normalize(actual)


class TraceTask:
    """Trace task generator and evaluator"""
    
    def __init__(
        self,
        dataset_name: str = "satpalsr/rl-python",
        dataset_split: str = "train",
        dataset_shuffle: bool = False,
    ):
        logger.info(f"Loading dataset: {dataset_name} split={dataset_split}")
        HF_TOKEN = os.environ.get("HF_TOKEN")
        self.dataset = load_dataset(dataset_name, split=dataset_split, token=HF_TOKEN)
        
        if dataset_shuffle:
            self.dataset = self.dataset.shuffle(seed=42)
        
        logger.info(f"Dataset loaded: {len(self.dataset)} examples")
    
    async def generate(self, task_id: int = None) -> Challenge:
        """
        Generate a trace task challenge
        """
        if task_id is not None:
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        source = sample.get("program", "")
        inputs = sample.get("inputs", "")
        
        # Use task_id as seed for deterministic print injection
        seed = task_id if task_id is not None else random.randint(0, 1000000)
        transformed = inject_non_overfittable_prints(source, seed, max_injections=6)
        
        # Get ground truth
        stdout, stderr = run_code_sync(transformed, input_data=inputs)
        if stderr:
            # If transformation causes error, try again with a different seed or different task
            # For simplicity, we just log it and return what we have (evaluation will likely fail)
            logger.warning(f"Transformation caused error for task {idx}: {stderr}")
        
        ground_truth = stdout.strip()
        
        prompt = f"""Predict the exact and complete standard output (stdout) of the following Python program, including every single print statement.

The program contains several injected debug print statements starting with '__DBG_'. You must include these in your prediction exactly as they would appear in the output, along with any other output the program produces.

Program:
```python
{transformed}
```

Input (stdin):
```
{inputs}
```

Provide the full stdout content. Do not provide any explanations or commentary outside of the predicted output."""

        return Challenge(
            env="trace",
            prompt=prompt,
            extra={
                "ground_truth": ground_truth,
                "transformed_code": transformed,
                "inputs": inputs,
                "seed": seed,
                "dataset_index": idx,
                "task_id": task_id
            }
        )
    
    async def evaluate(self, response: str, challenge: Challenge) -> Tuple[float, str]:
        """
        Evaluate trace response
        """
        ground_truth = challenge.extra.get("ground_truth", "")
        cleaned_prediction = clean_llm_prediction(response)
        
        score = 1.0 if compare_outputs(ground_truth, cleaned_prediction) else 0.0
        
        return score, "1/1" if score > 0 else "0/1"

