"""Global configuration with environment variable overrides"""

import os
import re


def image_reference_name(image: str, *, kubernetes: bool = False) -> str:
    """Derive a bounded runtime name from a tag or immutable digest reference."""

    if not isinstance(image, str) or not image:
        raise ValueError("image must be a non-empty string")
    leaf = image.rsplit("/", 1)[-1]
    if kubernetes:
        value = re.sub(r"[^a-z0-9-]+", "-", leaf.lower())
        value = value.strip("-")
        return (value or "environment")[:63].rstrip("-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", leaf)
    value = value.strip("-._")
    return (value or "environment")[:128].rstrip("-._")


class Config:
    """Global configuration with sensible defaults"""
    
    # Container configuration
    CONTAINER_STARTUP_TIMEOUT: int = 30  # seconds
    CONTAINER_NAME_PREFIX: str = "affinetes"
    
    # Image configuration
    IMAGE_BUILD_TIMEOUT: int = 600  # seconds
    DEFAULT_IMAGE_PREFIX: str = "affinetes"
    DEFAULT_REGISTRY: str | None = None
    
    # Logging
    LOG_LEVEL: str = os.getenv("AFFINETES_LOG_LEVEL", "INFO")
    
    # Environment file path (inside container)
    ENV_MODULE_PATH: str = "/app/env.py"
    
    @classmethod
    def get_log_level(cls) -> str:
        """Get log level from env or default"""
        return os.getenv("AFFINETES_LOG_LEVEL", cls.LOG_LEVEL)
