"""Fixer Agent Configuration - Ridge project paths"""

import os


def _get_default_ridge_path() -> str:
    """Get default Ridge path relative to this file"""
    # fixer/config.py -> .. -> SWE-SYNTH -> ridges
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ridges"))


def get_ridge_project_path() -> str:
    """Get Ridge project path (env var > default)"""
    return os.getenv("RIDGE_PROJECT_PATH", _get_default_ridge_path())


def get_ridge_agent_path() -> str:
    """Get Ridge agent path (env var > default)"""
    default = os.path.join(get_ridge_project_path(), "agents/agent01.py")
    return os.getenv("RIDGE_AGENT_PATH", default)
