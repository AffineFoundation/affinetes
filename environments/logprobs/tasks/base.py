"""Abstract base class for logprobs task generators."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseTask(ABC):
    """All task types implement this interface.

    generate(seed) -> (prompt, metadata_dict)
    """

    @abstractmethod
    def generate(self, seed: int) -> Tuple[str, dict]:
        """Generate a prompt from seed.

        Args:
            seed: Deterministic seed for this sample.

        Returns:
            (prompt, metadata) where metadata is task-specific extra info.
        """
        ...
