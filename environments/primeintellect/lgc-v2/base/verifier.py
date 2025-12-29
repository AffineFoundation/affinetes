"""Base verifier class"""
from abc import ABC, abstractmethod
from .data import Data


class Verifier(ABC):
    """Abstract base class for verifiers"""

    def __init__(self):
        pass

    @abstractmethod
    def verify(self, data: Data, test_answer: str) -> bool:
        """Verify if the test answer is correct

        Args:
            data: Data object containing the challenge
            test_answer: Answer to verify

        Returns:
            bool: True if correct, False otherwise
        """
        raise NotImplementedError("Verifier.verify() must be implemented")
