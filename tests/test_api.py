"""Tests for the public API input validation."""

import pytest
from unittest.mock import patch, MagicMock

from affinetes.api import load_env, list_active_environments, get_environment
from affinetes.utils.exceptions import ValidationError


class TestLoadEnvValidation:
    """Tests for load_env parameter validation."""

    def test_missing_image_raises(self):
        """Missing image in docker mode should raise ValidationError."""
        with pytest.raises(ValidationError, match="image is required"):
            load_env(image=None, mode="docker")

    def test_connect_only_without_container_name_raises(self):
        """connect_only without container_name should raise."""
        with pytest.raises(ValidationError, match="container_name is required"):
            load_env(image="test", mode="docker", connect_only=True)

    def test_connect_only_with_replicas_raises(self):
        """connect_only with replicas > 1 should raise."""
        with pytest.raises(ValidationError, match="only supports single instance"):
            load_env(
                image="test",
                mode="docker",
                connect_only=True,
                container_name="my-container",
                replicas=2,
            )

    def test_zero_replicas_raises(self):
        """replicas < 1 should raise ValidationError."""
        with pytest.raises(ValidationError, match="replicas must be >= 1"):
            load_env(image="test", mode="docker", replicas=0)

    def test_negative_replicas_raises(self):
        """Negative replicas should raise."""
        with pytest.raises(ValidationError, match="replicas must be >= 1"):
            load_env(image="test", mode="docker", replicas=-1)

    def test_insufficient_hosts_raises(self):
        """Fewer hosts than replicas should raise."""
        with pytest.raises(ValidationError, match="Not enough hosts"):
            load_env(
                image="test",
                mode="docker",
                replicas=3,
                hosts=["host1", "host2"],
            )

    def test_url_mode_without_base_url_raises(self):
        """URL mode without base_url should raise."""
        with pytest.raises(ValidationError, match="base_url"):
            load_env(mode="url", replicas=1)

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid mode"):
            load_env(image="test", mode="kubernetes")


class TestListActiveEnvironments:
    """Tests for list_active_environments."""

    @patch("affinetes.api.get_registry")
    def test_returns_list(self, mock_get_registry):
        """Should return list from registry."""
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = ["env1", "env2"]
        mock_get_registry.return_value = mock_registry
        result = list_active_environments()
        assert result == ["env1", "env2"]


class TestGetEnvironment:
    """Tests for get_environment."""

    @patch("affinetes.api.get_registry")
    def test_returns_env(self, mock_get_registry):
        """Should return environment from registry."""
        mock_registry = MagicMock()
        mock_env = MagicMock()
        mock_registry.get.return_value = mock_env
        mock_get_registry.return_value = mock_registry
        result = get_environment("env1")
        assert result == mock_env

    @patch("affinetes.api.get_registry")
    def test_returns_none_for_missing(self, mock_get_registry):
        """Should return None for non-existent environment."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry
        result = get_environment("nonexistent")
        assert result is None
