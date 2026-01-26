"""Ridge evaluation module for SWE-SYNTH"""

from .ridges_evaluate import (
    run_proxy_container,
    run_ridges_sandbox,
    stop_proxy_container,
    image_exists,
)

__all__ = [
    "run_proxy_container",
    "run_ridges_sandbox",
    "stop_proxy_container",
    "image_exists",
]
