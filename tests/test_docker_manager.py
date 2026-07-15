from unittest.mock import MagicMock

import docker
import pytest

from affinetes.infrastructure.docker_manager import DockerManager
from affinetes.utils.exceptions import ContainerError

SHA256_DIGEST = "a" * 64


def make_manager() -> DockerManager:
    manager = object.__new__(DockerManager)
    manager.client = MagicMock()
    manager.client.api.pull.return_value = iter([{"status": "Pull complete"}])
    return manager


def test_pull_image_passes_digest_reference_through_to_docker_api() -> None:
    manager = make_manager()
    image = f"registry.example.com/team/image@sha256:{SHA256_DIGEST}"

    manager.pull_image(image)

    manager.client.api.pull.assert_called_once_with(
        image,
        stream=True,
        decode=True,
    )


@pytest.mark.parametrize(
    ("image", "repository", "tag"),
    [
        ("team/image:release", "team/image", "release"),
        (
            "registry.example.com:5000/team/image:release",
            "registry.example.com:5000/team/image",
            "release",
        ),
        (
            "registry.example.com:5000/team/image",
            "registry.example.com:5000/team/image",
            "latest",
        ),
        ("team/image", "team/image", "latest"),
    ],
)
def test_pull_image_preserves_tag_and_registry_port_behavior(
    image: str,
    repository: str,
    tag: str,
) -> None:
    manager = make_manager()

    manager.pull_image(image)

    manager.client.api.pull.assert_called_once_with(
        repository,
        stream=True,
        decode=True,
        tag=tag,
    )


@pytest.mark.parametrize(
    "image",
    [
        "registry.example.com/team/image@sha256:abc",
        f"registry.example.com/team/image@sha512:{SHA256_DIGEST}",
        f"registry.example.com/team/image@sha256:{'g' * 64}",
        f"registry.example.com/team/image@sha256:{SHA256_DIGEST}0",
        f"registry.example.com/team/image@sha256:{SHA256_DIGEST}@extra",
    ],
)
def test_pull_image_rejects_malformed_digest_without_fallback(image: str) -> None:
    manager = make_manager()

    with pytest.raises(ContainerError, match="Invalid Docker digest reference"):
        manager.pull_image(image)

    manager.client.api.pull.assert_not_called()
    manager.client.images.get.assert_not_called()


def test_pull_image_uses_local_digest_when_registry_pull_fails() -> None:
    manager = make_manager()
    image = f"registry.example.com/team/image@sha256:{SHA256_DIGEST}"
    manager.client.api.pull.side_effect = docker.errors.APIError("registry unavailable")

    manager.pull_image(image)

    manager.client.images.get.assert_called_once_with(image)


def test_quiet_pull_still_uses_local_fallback_for_stream_error() -> None:
    manager = make_manager()
    image = f"registry.example.com/team/image@sha256:{SHA256_DIGEST}"
    manager.client.api.pull.return_value = iter([{"error": "manifest unavailable"}])

    manager.pull_image(image, quiet=True)

    manager.client.images.get.assert_called_once_with(image)


def test_pull_image_raises_when_registry_and_local_digest_are_unavailable() -> None:
    manager = make_manager()
    image = f"registry.example.com/team/image@sha256:{SHA256_DIGEST}"
    manager.client.api.pull.side_effect = docker.errors.APIError("registry unavailable")
    manager.client.images.get.side_effect = docker.errors.ImageNotFound("missing")

    with pytest.raises(ContainerError, match="doesn't exist locally"):
        manager.pull_image(image)

    manager.client.images.get.assert_called_once_with(image)
