"""Tests for the EnvironmentRegistry."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from affinetes.core.registry import EnvironmentRegistry


class TestEnvironmentRegistry:
    """Tests for EnvironmentRegistry."""

    def _make_registry(self):
        """Create a fresh registry (bypass singleton for testing)."""
        registry = object.__new__(EnvironmentRegistry)
        registry._environments = {}
        from threading import Lock
        registry._lock = Lock()
        registry._initialized = True
        return registry

    def _make_env(self, name="test"):
        """Create a mock environment."""
        env = MagicMock()
        env.name = name
        env._backend = MagicMock()
        env._backend._auto_cleanup = True
        env.cleanup = AsyncMock()
        return env

    def test_register_and_get(self):
        """Register and retrieve an environment."""
        reg = self._make_registry()
        env = self._make_env()
        reg.register("env1", env)
        assert reg.get("env1") is env

    def test_get_missing_returns_none(self):
        """Getting a non-registered env returns None."""
        reg = self._make_registry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        """Unregistering removes the environment."""
        reg = self._make_registry()
        env = self._make_env()
        reg.register("env1", env)
        reg.unregister("env1")
        assert reg.get("env1") is None

    def test_unregister_missing_is_safe(self):
        """Unregistering a non-existent env doesn't raise."""
        reg = self._make_registry()
        reg.unregister("nonexistent")  # Should not raise

    def test_list_all(self):
        """list_all returns all registered IDs."""
        reg = self._make_registry()
        reg.register("a", self._make_env())
        reg.register("b", self._make_env())
        result = reg.list_all()
        assert sorted(result) == ["a", "b"]

    def test_list_all_empty(self):
        """list_all returns empty list when no envs registered."""
        reg = self._make_registry()
        assert reg.list_all() == []

    def test_count(self):
        """count returns the number of registered environments."""
        reg = self._make_registry()
        assert reg.count() == 0
        reg.register("a", self._make_env())
        assert reg.count() == 1
        reg.register("b", self._make_env())
        assert reg.count() == 2

    def test_register_replaces_existing(self):
        """Re-registering the same ID replaces the environment."""
        reg = self._make_registry()
        env1 = self._make_env("first")
        env2 = self._make_env("second")
        reg.register("env1", env1)
        reg.register("env1", env2)
        assert reg.get("env1") is env2
        assert reg.count() == 1
