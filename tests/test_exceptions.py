"""Tests for custom exception hierarchy."""

import pytest

from affinetes.utils.exceptions import (
    AffinetesError,
    ValidationError,
    ImageBuildError,
    ImageNotFoundError,
    ContainerError,
    ExecutionError,
    BackendError,
    SetupError,
    EnvironmentError,
    NotImplementedError,
)


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions should inherit from AffinetesError."""
        exception_classes = [
            ValidationError,
            ImageBuildError,
            ImageNotFoundError,
            ContainerError,
            ExecutionError,
            BackendError,
            SetupError,
            EnvironmentError,
            NotImplementedError,
        ]
        for exc_cls in exception_classes:
            assert issubclass(exc_cls, AffinetesError), (
                f"{exc_cls.__name__} does not inherit from AffinetesError"
            )

    def test_base_is_exception(self):
        """AffinetesError should inherit from Exception."""
        assert issubclass(AffinetesError, Exception)

    def test_exceptions_can_be_raised_and_caught(self):
        """All exceptions can be raised and caught by their base."""
        with pytest.raises(AffinetesError):
            raise ValidationError("bad input")

        with pytest.raises(AffinetesError):
            raise ContainerError("container failed")

        with pytest.raises(AffinetesError):
            raise BackendError("backend issue")

    def test_exception_message_preserved(self):
        """Exception message should be preserved."""
        msg = "Something went wrong"
        exc = ExecutionError(msg)
        assert str(exc) == msg

    def test_exception_str_representation(self):
        """Exceptions should have correct string representation."""
        exc = ImageNotFoundError("myimage:latest not found")
        assert "myimage:latest not found" in str(exc)
