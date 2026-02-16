"""Input validation utilities for affinetes.

Provides reusable validation functions to ensure correctness
of user-supplied parameters before they reach the backend layer.
"""

import re
from typing import Any, Dict, List, Optional

from .exceptions import ValidationError


def validate_image_tag(image: str) -> str:
    """Validate and normalize a Docker image tag.

    Args:
        image: Docker image reference (e.g. "myimage:latest", "registry/repo:v1").

    Returns:
        The validated image tag string.

    Raises:
        ValidationError: If the image tag is empty or has invalid characters.
    """
    if not image or not isinstance(image, str):
        raise ValidationError("Image tag must be a non-empty string")

    image = image.strip()
    if not image:
        raise ValidationError("Image tag must be a non-empty string")

    # Basic Docker image tag pattern (simplified)
    # Allow: letters, digits, -, _, ., /, :, @
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._\-/:@]*$'
    if not re.match(pattern, image):
        raise ValidationError(
            f"Invalid image tag '{image}'. "
            "Must start with alphanumeric and contain only "
            "letters, digits, '.', '-', '_', '/', ':', '@'."
        )

    return image


def validate_replicas(replicas: int) -> int:
    """Validate the replica count.

    Args:
        replicas: Number of environment replicas.

    Returns:
        Validated replica count.

    Raises:
        ValidationError: If replicas is not a positive integer.
    """
    if not isinstance(replicas, int) or replicas < 1:
        raise ValidationError(
            f"replicas must be a positive integer, got {replicas!r}"
        )
    return replicas


def validate_env_vars(env_vars: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Validate environment variables dictionary.

    Args:
        env_vars: Dictionary of environment variable key-value pairs.

    Returns:
        The validated env_vars or None.

    Raises:
        ValidationError: If keys or values are not strings.
    """
    if env_vars is None:
        return None

    if not isinstance(env_vars, dict):
        raise ValidationError(
            f"env_vars must be a dict, got {type(env_vars).__name__}"
        )

    for key, value in env_vars.items():
        if not isinstance(key, str) or not key:
            raise ValidationError(
                f"env_vars keys must be non-empty strings, got {key!r}"
            )
        if not isinstance(value, str):
            raise ValidationError(
                f"env_vars values must be strings, got {type(value).__name__} "
                f"for key '{key}'"
            )

    return env_vars


def validate_mem_limit(mem_limit: Optional[str]) -> Optional[str]:
    """Validate a memory limit string.

    Accepts Docker-style memory limits like "512m", "1g", "2Gi", "256Mi".

    Args:
        mem_limit: Memory limit string or None.

    Returns:
        The validated memory limit or None.

    Raises:
        ValidationError: If the format is invalid.
    """
    if mem_limit is None:
        return None

    if not isinstance(mem_limit, str):
        raise ValidationError(
            f"mem_limit must be a string, got {type(mem_limit).__name__}"
        )

    pattern = r'^\d+(\.\d+)?\s*[bBkKmMgGtT][iI]?[bB]?$'
    if not re.match(pattern, mem_limit.strip()):
        raise ValidationError(
            f"Invalid mem_limit '{mem_limit}'. "
            "Expected format like '512m', '1g', '2Gi', '256Mi'."
        )

    return mem_limit.strip()


def validate_load_balance_strategy(strategy: str) -> str:
    """Validate load balancing strategy.

    Args:
        strategy: Load balance strategy name.

    Returns:
        Validated strategy string.

    Raises:
        ValidationError: If strategy is not recognized.
    """
    valid_strategies = {"random", "round_robin"}
    if strategy not in valid_strategies:
        raise ValidationError(
            f"Invalid load_balance strategy '{strategy}'. "
            f"Must be one of: {', '.join(sorted(valid_strategies))}"
        )
    return strategy


def validate_hosts(
    hosts: Optional[List[str]], replicas: int
) -> Optional[List[str]]:
    """Validate hosts list against replica count.

    Args:
        hosts: List of host addresses or None for local deployment.
        replicas: Number of replicas to deploy.

    Returns:
        Validated hosts list or None.

    Raises:
        ValidationError: If hosts list is invalid or insufficient.
    """
    if hosts is None:
        return None

    if not isinstance(hosts, list):
        raise ValidationError(
            f"hosts must be a list, got {type(hosts).__name__}"
        )

    if len(hosts) < replicas:
        raise ValidationError(
            f"Not enough hosts ({len(hosts)}) for replicas ({replicas}). "
            f"Either provide enough hosts or set hosts=None for local deployment."
        )

    for i, host in enumerate(hosts):
        if not isinstance(host, str) or not host:
            raise ValidationError(
                f"hosts[{i}] must be a non-empty string, got {host!r}"
            )

    return hosts
