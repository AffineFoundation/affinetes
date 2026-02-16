"""Tests for global configuration."""

import os
import pytest

from affinetes.utils.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test default configuration values are set."""
        assert Config.CONTAINER_STARTUP_TIMEOUT == 30
        assert Config.CONTAINER_NAME_PREFIX == "affinetes"
        assert Config.IMAGE_BUILD_TIMEOUT == 600
        assert Config.DEFAULT_IMAGE_PREFIX == "affinetes"
        assert Config.ENV_MODULE_PATH == "/app/env.py"

    def test_default_registry_is_none(self):
        """Default registry should be None."""
        assert Config.DEFAULT_REGISTRY is None

    def test_get_log_level_default(self):
        """get_log_level returns INFO by default."""
        # Remove env var if set, then check
        old = os.environ.pop("AFFINETES_LOG_LEVEL", None)
        try:
            assert Config.get_log_level() == "INFO"
        finally:
            if old is not None:
                os.environ["AFFINETES_LOG_LEVEL"] = old

    def test_get_log_level_from_env(self):
        """get_log_level reads AFFINETES_LOG_LEVEL environment variable."""
        old = os.environ.get("AFFINETES_LOG_LEVEL")
        os.environ["AFFINETES_LOG_LEVEL"] = "DEBUG"
        try:
            assert Config.get_log_level() == "DEBUG"
        finally:
            if old is not None:
                os.environ["AFFINETES_LOG_LEVEL"] = old
            else:
                del os.environ["AFFINETES_LOG_LEVEL"]
