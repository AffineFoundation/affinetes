"""Fixer Agent Configuration - Ridge project paths"""

import os

RIDGE_PROJECT_PATH = "/home/ubuntu/ridges/affine-ridges-env"


def get_ridge_project_path() -> str:
    """Get Ridge project path (env var > default)"""
    return os.getenv("RIDGE_PROJECT_PATH", RIDGE_PROJECT_PATH)


def get_ridge_agent_path() -> str:
    """Get Ridge agent path (env var > default)"""
    default = os.path.join(get_ridge_project_path(), "agents/agent01.py")
    return os.getenv("RIDGE_AGENT_PATH", default)
