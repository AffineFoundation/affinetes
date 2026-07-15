import pytest

from affinetes.utils.config import image_reference_name


_DIGEST = "a" * 64


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("repo:latest", "repo-latest"),
        ("registry.example:5000/team/repo:1.0", "repo-1.0"),
        (f"registry.example/team/repo@sha256:{_DIGEST}", f"repo-sha256-{_DIGEST}"),
    ],
)
def test_docker_name_accepts_tag_and_digest_references(reference, expected):
    assert image_reference_name(reference) == expected


def test_kubernetes_name_removes_every_digest_separator_and_is_bounded():
    name = image_reference_name(
        f"registry.example/Team_Name/Image.Name@sha256:{_DIGEST}",
        kubernetes=True,
    )
    assert len(name) <= 63
    assert name == name.lower()
    assert all(character.isalnum() or character == "-" for character in name)
    assert "@" not in name and ":" not in name and "_" not in name and "." not in name


@pytest.mark.parametrize("value", ["", None])
def test_empty_image_reference_is_rejected(value):
    with pytest.raises(ValueError, match="non-empty"):
        image_reference_name(value)
