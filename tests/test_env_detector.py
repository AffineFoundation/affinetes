"""Tests for environment type detection."""

import os
import tempfile
import pytest

from affinetes.infrastructure.env_detector import EnvDetector, EnvType, EnvConfig


class TestEnvConfig:
    """Tests for EnvConfig data class."""

    def test_function_based_config(self):
        """Test creating a function-based config."""
        config = EnvConfig(env_type=EnvType.FUNCTION_BASED)
        assert config.env_type == "function_based"
        assert config.server_file is None
        assert config.server_port == 8000

    def test_http_based_config(self):
        """Test creating an HTTP-based config."""
        config = EnvConfig(
            env_type=EnvType.HTTP_BASED,
            server_file="env.py",
            server_port=9000,
        )
        assert config.env_type == "http_based"
        assert config.server_file == "env.py"
        assert config.server_port == 9000


class TestEnvDetector:
    """Tests for EnvDetector.detect()."""

    def _make_env_dir(self, env_py_content: str) -> str:
        """Helper to create a temp directory with an env.py file."""
        tmpdir = tempfile.mkdtemp()
        env_py = os.path.join(tmpdir, "env.py")
        with open(env_py, "w") as f:
            f.write(env_py_content)
        return tmpdir

    def test_detect_function_based(self):
        """Test detection of function-based environment."""
        code = '''
class Actor:
    def evaluate(self, task_id):
        return {"score": 0.5}
'''
        env_dir = self._make_env_dir(code)
        config = EnvDetector.detect(env_dir)
        assert config.env_type == EnvType.FUNCTION_BASED
        assert config.server_file is None

    def test_detect_http_based_fastapi(self):
        """Test detection of HTTP-based environment with FastAPI."""
        code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
'''
        env_dir = self._make_env_dir(code)
        config = EnvDetector.detect(env_dir)
        assert config.env_type == EnvType.HTTP_BASED
        assert config.server_file == "env.py"

    def test_detect_fastapi_without_app_is_function_based(self):
        """FastAPI import but no 'app =' should be function_based."""
        code = '''
from fastapi import FastAPI
# No app = FastAPI() here, just the import
def run():
    pass
'''
        env_dir = self._make_env_dir(code)
        config = EnvDetector.detect(env_dir)
        assert config.env_type == EnvType.FUNCTION_BASED

    def test_detect_missing_env_py_raises(self):
        """Missing env.py should raise ValueError."""
        tmpdir = tempfile.mkdtemp()
        with pytest.raises(ValueError, match="env.py not found"):
            EnvDetector.detect(tmpdir)

    def test_detect_empty_env_py_is_function_based(self):
        """Empty env.py should be function_based."""
        env_dir = self._make_env_dir("")
        config = EnvDetector.detect(env_dir)
        assert config.env_type == EnvType.FUNCTION_BASED

    def test_detect_app_equals_no_spaces(self):
        """Detection should work with 'app=' (no spaces)."""
        code = '''
from fastapi import FastAPI
app=FastAPI()
'''
        env_dir = self._make_env_dir(code)
        config = EnvDetector.detect(env_dir)
        assert config.env_type == EnvType.HTTP_BASED
