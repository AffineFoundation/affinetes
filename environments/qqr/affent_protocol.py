"""Version-tolerant interpretation of the affentctl batch protocol.

This file is duplicated in each isolated environment build context. Keep the
copies byte-identical; ``test_affent_build_contract.py`` enforces that rule.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class AffentDisposition(str, Enum):
    GRADABLE = "gradable"
    MODEL_FAILURE = "model_failure"
    INFRA_FAILURE = "infra_failure"


@dataclass(frozen=True)
class AffentBuildIdentity:
    revision: str
    sha256: str


_MODEL_FAILURE_KINDS = frozenset({
    "context_overflow",
    "context_window_exceeded",
    "contextWindowExceeded",
    "ContextWindowExceeded",
})
_CONTEXT_ERROR_CODES = frozenset({
    "context_length_exceeded",
    "context_window_exceeded",
})
_AFFENT_HTTP_ERROR = re.compile(
    r"chat http (?P<status>\d{3}): (?P<body>\{.*\})",
    re.DOTALL,
)
_SGLANG_CONTEXT_ERROR = re.compile(
    r"Input length \((?P<input_tokens>\d+) tokens\) exceeds the maximum "
    r"allowed length \((?P<max_tokens>\d+) tokens\)\. Use a shorter input "
    r"or enable --allow-auto-truncate\."
)


def failure_kind_from_provider_error(message: object) -> Optional[str]:
    """Classify only machine-readable provider error contracts."""
    if not isinstance(message, str):
        return None

    raw_body = message.strip()
    http_status: Optional[int] = None
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        match = _AFFENT_HTTP_ERROR.fullmatch(raw_body)
        if match is None:
            return None
        http_status = int(match.group("status"))
        try:
            body = json.loads(match.group("body"))
        except json.JSONDecodeError:
            return None

    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None

    code = error.get("code")
    if code in _CONTEXT_ERROR_CODES:
        return "context_overflow"

    # SGLang currently uses a generic 400 code for context overflow. Require
    # its complete JSON contract and numeric invariant; other 400s remain
    # infrastructure failures and are retried.
    if http_status is None and code in (400, "400"):
        http_status = 400
    if http_status != 400 or error.get("type") != "BadRequestError":
        return None
    detail = error.get("message")
    if not isinstance(detail, str):
        return None
    match = _SGLANG_CONTEXT_ERROR.fullmatch(detail)
    if match is None:
        return None
    if int(match.group("input_tokens")) <= int(match.group("max_tokens")):
        return None
    return "context_overflow"


def verify_affent_binary(
    binary_path: str,
    *,
    metadata_dir: str = "/usr/local/share",
) -> AffentBuildIdentity:
    """Verify that the executed binary matches the image build manifest."""
    binary = Path(binary_path)
    metadata = Path(metadata_dir)
    try:
        revision = (metadata / "affent-ref").read_text().strip()
        expected_sha256 = (metadata / "affent-sha256").read_text().strip()
    except OSError as exc:
        raise RuntimeError(f"affent build metadata unavailable: {exc}") from exc

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"invalid affent revision metadata: {revision!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("invalid affent SHA-256 metadata")

    configured_revision = os.getenv("AFFENT_REF")
    if configured_revision and configured_revision != revision:
        raise RuntimeError(
            "affent revision mismatch: "
            f"AFFENT_REF={configured_revision}, image={revision}"
        )

    try:
        digest = hashlib.sha256()
        with binary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"affent binary unavailable: {exc}") from exc

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "affent binary checksum mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return AffentBuildIdentity(revision=revision, sha256=actual_sha256)


def classify_affent_run(
    exit_code: int,
    *,
    turn_end_reason: Optional[str],
    failure_kind: Optional[str] = None,
) -> AffentDisposition:
    """Classify one affentctl run using its structured terminal event.

    affentctl versions before 8f4f8f6e returned 2 at an action limit; newer
    versions return 0. Both are gradable only when the trace confirms the
    corresponding terminal reason. Exit 2 without that event remains a CLI
    usage/protocol failure.
    """
    if turn_end_reason == "completed" and exit_code == 0:
        return AffentDisposition.GRADABLE
    if turn_end_reason == "max_turns" and exit_code in (0, 2):
        return AffentDisposition.GRADABLE
    if turn_end_reason == "length" and exit_code == 2:
        return AffentDisposition.GRADABLE
    if (
        turn_end_reason == "error"
        and exit_code == 3
        and failure_kind in _MODEL_FAILURE_KINDS
    ):
        return AffentDisposition.MODEL_FAILURE
    return AffentDisposition.INFRA_FAILURE
