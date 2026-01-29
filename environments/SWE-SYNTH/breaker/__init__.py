"""
Breaker Module

A modular bug injection system for generating synthetic software engineering tasks.

Usage:
    from breaker import BreakerInput, run_breaker

    input = BreakerInput(
        docker_image="swe-bench/django:latest",
        base_commit="abc123",
        repo="django/django",
        instance_id="django__django-12345",
        gold_patch="...",
        test_cases=["test_auth_login", "test_auth_logout"],
        test_runner_script="...",
        test_parser_script="...",
        test_files="tests/auth/",
        bug_types=["off-by-one", "logic-inversion"],
        seed=42,
        model="Qwen/Qwen3-Coder-480B",
        api_base="https://xxx.chutes.ai/v1",
        api_key="xxx",
    )

    output = await run_breaker(input)
    print(output.problem_statement)
    print(output.bug_patch)
"""

from .types import (
    BreakerInput,
    BreakerOutput,
    InjectionResult,
    AgentResult,
    BreakerException,
)
from .bug_types import (
    BUG_TYPES,
    BUG_TYPE_DESCRIPTIONS,
    get_bug_type_info,
    format_bug_types_for_prompt,
    validate_bug_types,
)
from .orchestrator import (
    BreakerOrchestrator,
    run_breaker,
)
from .injector import (
    BugInjector,
    create_injector,
)
from .summarizer import (
    ProblemSummarizer,
    create_summarizer,
)
from .agents import (
    BaseCodeAgent,
    AgentConfig,
    AgentResult,
    MiniSweAgent,
    RidgeCodeAgent,
)

__all__ = [
    # Types
    "BreakerInput",
    "BreakerOutput",
    "InjectionResult",
    "AgentResult",
    "BreakerException",
    # Bug types
    "BUG_TYPES",
    "BUG_TYPE_DESCRIPTIONS",
    "get_bug_type_info",
    "format_bug_types_for_prompt",
    "validate_bug_types",
    # Orchestrator
    "BreakerOrchestrator",
    "run_breaker",
    # Components
    "BugInjector",
    "create_injector",
    "ProblemSummarizer",
    "create_summarizer",
    # Agents
    "BaseCodeAgent",
    "AgentConfig",
    "MiniSweAgent",
    "RidgeCodeAgent",
]
