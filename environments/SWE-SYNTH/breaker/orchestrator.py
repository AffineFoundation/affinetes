"""
Breaker Orchestrator

Orchestrates the complete bug injection flow:
1. Bug Injector injects a bug
2. Verify tests fail (within injector)
3. If verification fails, retry with feedback
4. On success, Summarizer generates problem statement
5. Return complete BreakerOutput
"""

from dataclasses import replace
from typing import Optional

from .types import BreakerInput, BreakerOutput, InjectionResult, BreakerException
from .injector import BugInjector, create_injector
from .summarizer import ProblemSummarizer, create_summarizer
from .bug_types import get_random_bug_types


class BreakerOrchestrator:
    """
    Orchestrates the complete bug injection workflow.

    Flow:
    1. Injector injects bug and verifies test failures
    2. If not enough tests fail, retry with feedback
    3. On success, Summarizer generates problem statement
    4. Return BreakerOutput that can be cached
    """

    def __init__(
        self,
        injector: BugInjector,
        summarizer: ProblemSummarizer,
        max_retries: int = 5,
        min_failed_tests: int = 1,
        max_failed_ratio: float = 0.2,
    ):
        """
        Initialize BreakerOrchestrator.

        Args:
            injector: Bug injection agent
            summarizer: Problem statement generator
            max_retries: Maximum injection attempts
            min_failed_tests: Minimum tests that must fail
            max_failed_ratio: Maximum ratio of tests that can fail
        """
        self.injector = injector
        self.summarizer = summarizer
        self.max_retries = max_retries
        self.min_failed_tests = min_failed_tests
        self.max_failed_ratio = max_failed_ratio

    async def run(self, input: BreakerInput) -> BreakerOutput:
        """
        Run the complete bug injection workflow.

        Args:
            input: BreakerInput with all configuration

        Returns:
            BreakerOutput with bug patch and problem statement

        Raises:
            BreakerException: If all retries fail
        """
        feedback = None
        last_injection: Optional[InjectionResult] = None
        tried_bug_types: list[str] = []

        for attempt in range(self.max_retries):
            # On retry, try different bug types
            current_input = input
            if attempt > 0:
                new_bug_types = get_random_bug_types(
                    seed=input.seed,
                    attempt=attempt,
                    count=3,
                    exclude=tried_bug_types,
                )
                if new_bug_types:
                    # Create a copy of input with new bug types
                    current_input = replace(input, bug_types=new_bug_types)
                    print(f"Trying different bug types: {new_bug_types}")

            tried_bug_types.extend(current_input.bug_types)
            print(f"Injection attempt {attempt + 1}/{self.max_retries} with bug types: {current_input.bug_types}")

            try:
                # Step 1: Inject bug
                injection = await self.injector.inject(
                    input=current_input,
                    feedback=feedback,
                )
                last_injection = injection

                # Step 2: Verify bug effectiveness
                is_valid, feedback = self._validate_injection(injection, current_input)

                if is_valid:
                    print(f"Bug injection successful! {len(injection.failed_tests)} tests failed.")

                    # Step 3: Generate problem statement
                    # Pass bug_types so summarizer can describe symptoms accurately
                    problem_statement = await self.summarizer.summarize(
                        input=current_input,
                        bug_patch=injection.bug_patch,
                        failed_tests=injection.failed_tests,
                        error_output=injection.error_output,
                        bug_types=current_input.bug_types,
                    )

                    # Step 4: Create and return output (use current_input for accurate bug_types)
                    return BreakerOutput.create(
                        input=current_input,
                        injection=injection,
                        problem_statement=problem_statement,
                    )

                print(f"Injection invalid: {feedback}")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                feedback = f"Previous attempt failed with error: {str(e)}. Try a different bug type."

        # All retries exhausted
        raise BreakerException(
            f"Failed to inject valid bug after {self.max_retries} attempts. "
            f"Tried bug types: {tried_bug_types}. Last feedback: {feedback}"
        )

    def _validate_injection(
        self,
        injection: InjectionResult,
        input: BreakerInput,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that the injection is effective.

        Returns:
            (is_valid, feedback) - feedback is None if valid
        """
        num_failed = len(injection.failed_tests)
        num_passed = len(injection.passed_tests)
        total_tests = num_failed + num_passed

        # Check 1: Must have a valid diff
        if not injection.bug_patch or not injection.bug_patch.startswith("diff"):
            return False, "No valid bug patch was generated. Please create a code change."

        # Check 2: Must have at least min_failed_tests failures
        if num_failed < self.min_failed_tests:
            return False, (
                f"No tests failed. The bug didn't affect test outcomes. "
                f"Try injecting the bug in code that is actually tested - "
                f"focus on return values, conditionals, or error handling."
            )

        # Check 3: Must not fail too many tests
        max_allowed = max(10, int(total_tests * self.max_failed_ratio))
        if num_failed > max_allowed:
            return False, (
                f"Too many tests failed ({num_failed}/{total_tests}). "
                f"The bug is too severe. Please inject a more subtle bug "
                f"that only affects specific functionality."
            )

        return True, None


async def run_breaker(
    input: BreakerInput,
    max_retries: int = 5,
    agent_type: str = "miniswe",
) -> BreakerOutput:
    """
    Convenience function to run the complete breaker workflow.

    Args:
        input: BreakerInput with all configuration
        max_retries: Maximum injection attempts
        agent_type: Agent type to use ("miniswe" or "ridge")

    Returns:
        BreakerOutput with bug patch and problem statement
    """
    # Create components
    injector = create_injector(input, agent_type=agent_type)
    summarizer = create_summarizer(input)

    # Create orchestrator
    orchestrator = BreakerOrchestrator(
        injector=injector,
        summarizer=summarizer,
        max_retries=max_retries,
    )

    # Run workflow
    return await orchestrator.run(input)
