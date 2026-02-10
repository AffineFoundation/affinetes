"""Stub Fixer Agents - reserved for future implementations"""

from typing import Optional
from .base import BaseFixerAgent, FixerConfig, FixerResult


class SWEAgentFixerAgent(BaseFixerAgent):
    """Reserved for Princeton's SWE-agent"""

    async def fix(
        self,
        problem_statement: str,
        docker_image: str,
        repo_path: Optional[str] = None,
        gold_patch: Optional[str] = None,
        bug_patch: Optional[str] = None,
        base_commit: Optional[str] = None,
    ) -> FixerResult:
        return FixerResult(
            patch="",
            success=False,
            error="SWE-agent support is not yet implemented.",
        )

    def cleanup(self):
        pass


class CodexFixerAgent(BaseFixerAgent):
    """Reserved for Codex-based agents"""

    async def fix(
        self,
        problem_statement: str,
        docker_image: str,
        repo_path: Optional[str] = None,
        gold_patch: Optional[str] = None,
        bug_patch: Optional[str] = None,
        base_commit: Optional[str] = None,
    ) -> FixerResult:
        return FixerResult(
            patch="",
            success=False,
            error="Codex agent support is not yet implemented.",
        )

    def cleanup(self):
        pass
