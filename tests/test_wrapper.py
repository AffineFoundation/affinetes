"""Tests for EnvironmentWrapper."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from affinetes.core.wrapper import EnvironmentWrapper
from affinetes.utils.exceptions import EnvironmentError


class TestEnvironmentWrapper:
    """Tests for EnvironmentWrapper class."""

    def _make_wrapper(self, name="test-env", is_ready=True):
        """Create a wrapper with a mocked backend."""
        backend = MagicMock()
        backend.name = name
        backend.is_ready.return_value = is_ready
        backend.call_method = AsyncMock(return_value={"result": "ok"})
        backend.cleanup = AsyncMock()
        backend.list_methods = AsyncMock(return_value=[])
        return EnvironmentWrapper(backend=backend)

    def test_creation(self):
        """Test wrapper creation stores backend and name."""
        wrapper = self._make_wrapper(name="my-env")
        assert wrapper.name == "my-env"

    def test_is_ready(self):
        """Test is_ready delegates to backend."""
        wrapper = self._make_wrapper(is_ready=True)
        assert wrapper.is_ready() is True

    def test_is_not_ready(self):
        """Test is_ready returns False when backend not ready."""
        wrapper = self._make_wrapper(is_ready=False)
        assert wrapper.is_ready() is False

    def test_getattr_private_raises(self):
        """Accessing private attributes should raise AttributeError."""
        wrapper = self._make_wrapper()
        with pytest.raises(AttributeError):
            _ = wrapper._nonexistent

    def test_getattr_when_not_ready_raises(self):
        """Calling methods when not ready should raise EnvironmentError."""
        wrapper = self._make_wrapper(is_ready=False)
        with pytest.raises(EnvironmentError, match="not ready"):
            _ = wrapper.evaluate

    def test_dynamic_method_returns_callable(self):
        """Dynamic method dispatch should return a callable."""
        wrapper = self._make_wrapper()
        method = wrapper.evaluate
        assert callable(method)

    @pytest.mark.asyncio
    async def test_dynamic_method_calls_backend(self):
        """Dynamic method should call backend.call_method."""
        wrapper = self._make_wrapper()
        result = await wrapper.evaluate(task_id=1)
        wrapper._backend.call_method.assert_called_once_with("evaluate", task_id=1)
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_dynamic_method_with_timeout(self):
        """Dynamic method should support _timeout parameter."""
        wrapper = self._make_wrapper()
        result = await wrapper.evaluate(task_id=1, _timeout=30)
        wrapper._backend.call_method.assert_called_once_with("evaluate", task_id=1)

    @pytest.mark.asyncio
    async def test_dynamic_method_timeout_raises(self):
        """Dynamic method should raise on timeout."""
        wrapper = self._make_wrapper()
        wrapper._backend.call_method = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        # Need to wrap in wait_for to trigger the timeout path
        wrapper._backend.call_method = AsyncMock(
            side_effect=lambda *a, **kw: asyncio.sleep(10)
        )
        with pytest.raises(EnvironmentError, match="timed out"):
            await wrapper.evaluate(task_id=1, _timeout=0.01)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Cleanup should delegate to backend."""
        wrapper = self._make_wrapper()
        await wrapper.cleanup()
        wrapper._backend.cleanup.assert_called_once()
        assert wrapper._is_ready is False

    @pytest.mark.asyncio
    async def test_list_methods(self):
        """list_methods should delegate to backend."""
        wrapper = self._make_wrapper()
        methods = await wrapper.list_methods(print_info=False)
        wrapper._backend.list_methods.assert_called_once()
        assert methods == []

    def test_get_stats_non_pool(self):
        """get_stats returns None for non-pool backends."""
        wrapper = self._make_wrapper()
        assert wrapper.get_stats() is None

    def test_context_manager(self):
        """Wrapper should support context manager protocol."""
        wrapper = self._make_wrapper()
        wrapper._backend._auto_cleanup = False
        with wrapper as w:
            assert w is wrapper
