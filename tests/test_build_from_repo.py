"""Tests for ImageBuilder repo URL support."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from affinetes.infrastructure.image_builder import ImageBuilder
from affinetes.utils.exceptions import ImageBuildError, ValidationError


# ---------------------------------------------------------------------------
# is_repo_url
# ---------------------------------------------------------------------------

class TestIsRepoUrl:
    @pytest.mark.parametrize("url", [
        "https://github.com/org/repo.git",
        "https://github.com/org/repo.git#main",
        "https://github.com/org/repo.git#main:environments/web",
        "https://github.com/org/repo",
        "http://gitlab.example.com/team/project.git#v2.0",
        "git://github.com/org/repo.git",
        "git@github.com:org/repo.git",
        "git@github.com:org/repo.git#develop:src/env",
    ])
    def test_valid_urls(self, url: str):
        assert ImageBuilder.is_repo_url(url) is True

    @pytest.mark.parametrize("source", [
        "/home/user/my-env",
        "./environments/affine",
        "environments/affine",
        "C:\\Users\\dev\\env",
        "affine:latest",
        "",
        "ftp://example.com/repo.git",
    ])
    def test_local_paths(self, source: str):
        assert ImageBuilder.is_repo_url(source) is False


# ---------------------------------------------------------------------------
# parse_repo_url
# ---------------------------------------------------------------------------

class TestParseRepoUrl:
    def test_plain_url(self):
        url = "https://github.com/org/repo.git"
        clone, ref, sub = ImageBuilder.parse_repo_url(url)
        assert clone == url
        assert ref is None
        assert sub is None

    def test_url_with_ref(self):
        clone, ref, sub = ImageBuilder.parse_repo_url(
            "https://github.com/org/repo.git#main"
        )
        assert clone == "https://github.com/org/repo.git"
        assert ref == "main"
        assert sub is None

    def test_url_with_ref_and_subfolder(self):
        clone, ref, sub = ImageBuilder.parse_repo_url(
            "https://github.com/org/repo.git#v1.0:environments/web"
        )
        assert clone == "https://github.com/org/repo.git"
        assert ref == "v1.0"
        assert sub == "environments/web"

    def test_ssh_url_with_fragment(self):
        clone, ref, sub = ImageBuilder.parse_repo_url(
            "git@github.com:org/repo.git#develop:src/env"
        )
        assert clone == "git@github.com:org/repo.git"
        assert ref == "develop"
        assert sub == "src/env"

    def test_empty_fragment_parts(self):
        clone, ref, sub = ImageBuilder.parse_repo_url(
            "https://github.com/org/repo.git#"
        )
        assert clone == "https://github.com/org/repo.git"
        assert ref is None
        assert sub is None


# ---------------------------------------------------------------------------
# _clone_repo
# ---------------------------------------------------------------------------

class TestCloneRepo:
    @patch("affinetes.infrastructure.image_builder.subprocess.run")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_clone_without_ref(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0)
        builder = ImageBuilder()
        dest = Path("/tmp/test-dest")

        builder._clone_repo("https://github.com/org/repo.git", dest)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:4] == ["git", "clone", "--depth", "1"]
        assert "--branch" not in cmd
        assert "https://github.com/org/repo.git" in cmd
        assert str(dest) in cmd

    @patch("affinetes.infrastructure.image_builder.subprocess.run")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_clone_with_ref(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0)
        builder = ImageBuilder()
        dest = Path("/tmp/test-dest")

        builder._clone_repo("https://github.com/org/repo.git", dest, ref="v2.0")

        cmd = mock_run.call_args[0][0]
        assert "--branch" in cmd
        idx = cmd.index("--branch")
        assert cmd[idx + 1] == "v2.0"

    @patch("affinetes.infrastructure.image_builder.subprocess.run")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_clone_failure_raises(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(
            returncode=128, stderr="fatal: repository not found"
        )
        builder = ImageBuilder()

        with pytest.raises(ImageBuildError, match="git clone failed"):
            builder._clone_repo(
                "https://github.com/org/nonexistent.git", Path("/tmp/x")
            )

    @patch("affinetes.infrastructure.image_builder.subprocess.run")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_clone_timeout_raises_image_build_error(self, mock_run: MagicMock):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git"], timeout=300)
        builder = ImageBuilder()

        with pytest.raises(ImageBuildError, match="timed out"):
            builder._clone_repo(
                "https://github.com/org/huge-repo.git", Path("/tmp/x")
            )

    @patch("affinetes.infrastructure.image_builder.subprocess.run")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_clone_disables_terminal_prompt(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0)
        builder = ImageBuilder()

        builder._clone_repo("https://github.com/org/repo.git", Path("/tmp/x"))

        env_arg = mock_run.call_args[1]["env"]
        assert env_arg["GIT_TERMINAL_PROMPT"] == "0"


# ---------------------------------------------------------------------------
# build_from_repo
# ---------------------------------------------------------------------------

class TestBuildFromRepo:
    @patch.object(ImageBuilder, "build_from_env", return_value="myimage:v1")
    @patch.object(ImageBuilder, "_clone_repo")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_delegates_to_build_from_env(
        self, mock_clone: MagicMock, mock_build: MagicMock
    ):
        builder = ImageBuilder()

        def fake_clone(url, dest, ref=None):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "env.py").write_text("# env")
            (dest / "Dockerfile").write_text("FROM python:3.11")

        mock_clone.side_effect = fake_clone

        result = builder.build_from_repo(
            "https://github.com/org/repo.git#main",
            image_tag="myimage:v1",
        )

        mock_clone.assert_called_once()
        clone_url = mock_clone.call_args[0][0]
        assert clone_url == "https://github.com/org/repo.git"
        assert mock_clone.call_args[1]["ref"] == "main"

        mock_build.assert_called_once()
        assert mock_build.call_args[1]["image_tag"] == "myimage:v1"
        assert result == "myimage:v1"

    @patch.object(ImageBuilder, "build_from_env", return_value="myimage:v1")
    @patch.object(ImageBuilder, "_clone_repo")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_subfolder_routing(
        self, mock_clone: MagicMock, mock_build: MagicMock
    ):
        builder = ImageBuilder()

        def fake_clone(url, dest, ref=None):
            dest.mkdir(parents=True, exist_ok=True)
            sub = dest / "envs" / "web"
            sub.mkdir(parents=True)
            (sub / "env.py").write_text("# env")
            (sub / "Dockerfile").write_text("FROM python:3.11")

        mock_clone.side_effect = fake_clone

        builder.build_from_repo(
            "https://github.com/org/repo.git#main:envs/web",
            image_tag="web:v1",
        )

        build_path = mock_build.call_args[1]["env_path"]
        assert build_path.endswith("envs/web")

    @patch.object(ImageBuilder, "_clone_repo")
    @patch.object(ImageBuilder, "__init__", lambda self: None)
    def test_missing_subfolder_raises(self, mock_clone: MagicMock):
        builder = ImageBuilder()

        def fake_clone(url, dest, ref=None):
            dest.mkdir(parents=True, exist_ok=True)

        mock_clone.side_effect = fake_clone

        with pytest.raises(ValidationError, match="Subfolder.*not found"):
            builder.build_from_repo(
                "https://github.com/org/repo.git#main:nonexistent",
                image_tag="img:v1",
            )


# ---------------------------------------------------------------------------
# Integration: build_image_from_env URL routing
# ---------------------------------------------------------------------------

class TestBuildImageFromEnvRouting:
    @patch("affinetes.api.ImageBuilder")
    def test_url_routes_to_build_from_repo(self, MockBuilder: MagicMock):
        from affinetes.api import build_image_from_env

        instance = MockBuilder.return_value
        instance.build_from_repo.return_value = "img:v1"
        MockBuilder.is_repo_url.return_value = True

        result = build_image_from_env(
            "https://github.com/org/repo.git#main",
            image_tag="img:v1",
        )

        instance.build_from_repo.assert_called_once()
        instance.build_from_env.assert_not_called()
        assert result == "img:v1"

    @patch("affinetes.api.ImageBuilder")
    def test_local_path_routes_to_build_from_env(self, MockBuilder: MagicMock):
        from affinetes.api import build_image_from_env

        instance = MockBuilder.return_value
        instance.build_from_env.return_value = "img:v1"
        MockBuilder.is_repo_url.return_value = False

        result = build_image_from_env(
            "/tmp/my-env",
            image_tag="img:v1",
        )

        instance.build_from_env.assert_called_once()
        instance.build_from_repo.assert_not_called()
        assert result == "img:v1"
