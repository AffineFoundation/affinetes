"""
Bug Type Definitions

Defines the types of bugs that can be injected and their descriptions.
"""

from typing import List, Dict, Any


# All supported bug types
BUG_TYPES = [
    "off-by-one",           # Boundary errors (< vs <=, > vs >=)
    "logic-inversion",      # Logic errors (and vs or, == vs !=)
    "wrong-variable",       # Using wrong variable
    "missing-check",        # Missing null/boundary check
    "type-error",           # Type conversion errors
    "wrong-operator",       # Arithmetic operator errors (+, -, *, /)
    "missing-return",       # Missing return statement
    "exception-swallow",    # Swallowing exceptions incorrectly
    "wrong-constant",       # Wrong constant/magic number
    "string-format",        # String formatting errors
    "order-dependency",     # Wrong order of operations
    "early-exit",           # Premature return/break/continue
    "wrong-default",        # Wrong default value
]


BUG_TYPE_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "off-by-one": {
        "description": "Boundary errors like < vs <=, wrong loop bounds, index ±1 errors",
        "examples": [
            "for i in range(n) → for i in range(n-1)",
            "if x <= 0 → if x < 0",
            "array[i+1] → array[i]",
        ]
    },
    "logic-inversion": {
        "description": "Logic errors like and/or swap, condition negation, comparison flip",
        "examples": [
            "if a and b → if a or b",
            "if x == y → if x != y",
            "if not flag → if flag",
        ]
    },
    "wrong-variable": {
        "description": "Using incorrect but similar variable names",
        "examples": [
            "return result → return results",
            "self.value → self.values",
            "user_id → user_ids",
        ]
    },
    "missing-check": {
        "description": "Removing important null/boundary/type checks",
        "examples": [
            "if x is not None: use(x) → use(x)",
            "if len(arr) > 0: → (remove check)",
            "if isinstance(x, str): → (remove check)",
        ]
    },
    "type-error": {
        "description": "Type conversion or coercion errors",
        "examples": [
            "int(x) → str(x)",
            "float(x) → int(x)",
            "str(x) → x",
        ]
    },
    "wrong-operator": {
        "description": "Arithmetic or bitwise operator errors",
        "examples": [
            "a + b → a - b",
            "a * b → a / b",
            "a | b → a & b",
        ]
    },
    "missing-return": {
        "description": "Forgetting return statements",
        "examples": [
            "return result → result",
            "return True → (nothing)",
        ]
    },
    "exception-swallow": {
        "description": "Incorrectly handling or swallowing exceptions",
        "examples": [
            "except ValueError: raise → except ValueError: pass",
            "except Exception as e: log(e) → except: pass",
        ]
    },
    "wrong-constant": {
        "description": "Using wrong constant or magic number",
        "examples": [
            "timeout = 1000 → timeout = 100",
            "MAX_RETRIES = 3 → MAX_RETRIES = 0",
            "BUFFER_SIZE = 4096 → BUFFER_SIZE = 40",
        ]
    },
    "string-format": {
        "description": "String formatting or interpolation errors",
        "examples": [
            'f"{name}" → f"{names}"',
            '"{} {}".format(a, b) → "{} {}".format(b, a)',
            'f"Error: {msg}" → f"Error: {err}"',
        ]
    },
    "order-dependency": {
        "description": "Wrong order of operations or statements",
        "examples": [
            "validate(); process() → process(); validate()",
            "lock(); access() → access(); lock()",
            "init(); use() → use(); init()",
        ]
    },
    "early-exit": {
        "description": "Premature return, break, or continue",
        "examples": [
            "if error: log(); return → if error: return",
            "for item in items: process(item) → for item in items: break",
            "while condition: work(); check() → while condition: return",
        ]
    },
    "wrong-default": {
        "description": "Using wrong default value for parameters or variables",
        "examples": [
            "def f(x=None): → def f(x=0):",
            "count = 0 → count = 1",
            "enabled = True → enabled = False",
        ]
    },
}


def get_bug_type_info(bug_type: str) -> Dict[str, Any]:
    """Get description and examples for a bug type"""
    return BUG_TYPE_DESCRIPTIONS.get(bug_type, {
        "description": bug_type,
        "examples": []
    })


def format_bug_types_for_prompt(bug_types: List[str]) -> str:
    """Format bug types with descriptions for use in prompts"""
    lines = []
    for bug_type in bug_types:
        info = get_bug_type_info(bug_type)
        desc = info.get("description", bug_type)
        lines.append(f"- {bug_type}: {desc}")
        for ex in info.get("examples", [])[:2]:
            lines.append(f"  Example: {ex}")
    return "\n".join(lines)


def validate_bug_types(bug_types: List[str]) -> List[str]:
    """Validate and return only known bug types"""
    return [bt for bt in bug_types if bt in BUG_TYPES]
