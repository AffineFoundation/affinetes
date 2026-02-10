"""
Problem Statement Summarizer

Generates user-facing problem statements based on injected bugs.
"""

from pathlib import Path
from typing import List, Optional

import yaml
from litellm import acompletion

from .types import BreakerInput
from .bug_types import BUG_TYPE_DESCRIPTIONS


class ProblemSummarizer:
    """
    Generates problem statements from injected bugs.

    Takes the bug patch, failed tests, and error output to create
    a user-facing bug report that describes symptoms without revealing the fix.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize ProblemSummarizer.

        Args:
            model: Model name for LLM calls
            api_base: API base URL
            api_key: API key
            config_path: Path to config.yaml
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key

        # Load prompt template
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.prompt_template = config.get("summarizer", {}).get(
            "prompt_template", ""
        )

    async def summarize(
        self,
        input: BreakerInput,
        bug_patch: str,
        failed_tests: List[str],
        error_output: str,
        bug_types: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a problem statement for the injected bug.

        Args:
            input: BreakerInput with context
            bug_patch: The injected bug as a diff
            failed_tests: List of tests that failed
            error_output: Test error output
            bug_types: List of bug types that were injected

        Returns:
            Problem statement string
        """
        # Use bug_types from input if not provided
        if bug_types is None:
            bug_types = input.bug_types or []

        # Build the prompt
        prompt = self._build_prompt(
            input=input,
            bug_patch=bug_patch,
            failed_tests=failed_tests,
            error_output=error_output,
            bug_types=bug_types,
        )

        # Prepare model name
        model_name = self._prepare_model_name()

        # Call LLM
        response = await acompletion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0.7,
            timeout=120,
        )

        return response.choices[0].message.content.strip()

    def _build_prompt(
        self,
        input: BreakerInput,
        bug_patch: str,
        failed_tests: List[str],
        error_output: str,
        bug_types: List[str],
    ) -> str:
        """Build the prompt for problem statement generation"""
        # Use Jinja-style template substitution
        prompt = self.prompt_template

        # Build bug type descriptions
        bug_types_str = ", ".join(bug_types) if bug_types else "unknown"
        bug_type_descriptions = self._format_bug_type_descriptions(bug_types)

        # Simple template variable replacement
        replacements = {
            "{{repo}}": input.repo,
            "{{bug_patch}}": bug_patch,
            "{{failed_tests_str}}": "\n".join(f"- {t}" for t in failed_tests),
            "{{error_output}}": error_output[:3000],
            "{{test_patch}}": (input.test_patch or "")[:2000],
            "{{bug_types_str}}": bug_types_str,
            "{{bug_type_descriptions}}": bug_type_descriptions,
        }

        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

        # Handle conditional test_patch section
        if input.test_patch:
            prompt = prompt.replace("{% if test_patch %}", "")
            prompt = prompt.replace("{% endif %}", "")
        else:
            # Remove the test_patch section
            import re
            prompt = re.sub(
                r'\{% if test_patch %\}.*?\{% endif %\}',
                '',
                prompt,
                flags=re.DOTALL
            )

        return prompt

    def _format_bug_type_descriptions(self, bug_types: List[str]) -> str:
        """Format bug type descriptions for the prompt"""
        if not bug_types:
            return ""

        descriptions = []
        for bug_type in bug_types:
            if bug_type in BUG_TYPE_DESCRIPTIONS:
                info = BUG_TYPE_DESCRIPTIONS[bug_type]
                desc = info.get("description", bug_type)
                examples = info.get("examples", [])
                descriptions.append(f"- **{bug_type}**: {desc}")
                if examples:
                    descriptions.append(f"  Examples: {examples[0]}")

        if descriptions:
            return "Bug type characteristics:\n" + "\n".join(descriptions)
        return ""

    def _prepare_model_name(self) -> str:
        """Prepare model name for litellm"""
        if self.model.startswith(("openai/", "anthropic/", "azure/", "bedrock/")):
            return self.model
        elif self.model.startswith("claude"):
            return self.model
        else:
            return f"openai/{self.model}"


def create_summarizer(input: BreakerInput) -> ProblemSummarizer:
    """
    Create a ProblemSummarizer from BreakerInput.

    Args:
        input: BreakerInput with model configuration

    Returns:
        Configured ProblemSummarizer
    """
    return ProblemSummarizer(
        model=input.model,
        api_base=input.api_base,
        api_key=input.api_key,
    )
