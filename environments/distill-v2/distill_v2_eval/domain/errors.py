"""Domain error hierarchy.

Errors raised in adapters wrap the underlying provider error (with .__cause__)
and are caught/translated by the application layer. The hierarchy is intentionally
shallow: distinguish only what callers act on differently.
"""

from __future__ import annotations


class AfrError(Exception):
    """Base for all application-defined errors."""


class TeacherError(AfrError):
    """Anything that goes wrong while calling a teacher provider."""


class TeacherRateLimited(TeacherError):
    pass


class TeacherTimeout(TeacherError):
    pass


class TeacherBadResponse(TeacherError):
    """Provider returned something we couldn't parse or that violated contract."""


class SandboxError(AfrError):
    """Anything that goes wrong inside the SWE-rebench docker sandbox."""


class SandboxTimeout(SandboxError):
    pass


class NormalizationError(AfrError):
    """Raised when a teacher trajectory can't be normalized.

    Producers wrap it: the rollout is still persisted with status='parse_failed'
    plus a structured error payload so we can later improve the normalizer.
    """


class TaskSourceError(AfrError):
    """Failed to load tasks from the configured dataset / version."""


class PairMiningError(AfrError):
    pass


class EvalError(AfrError):
    pass


class StorageError(AfrError):
    pass


class ContractError(AfrError):
    """Adapter implementation violated its port contract. Bug, not runtime issue."""
