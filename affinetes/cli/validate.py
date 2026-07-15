"""Fail-closed lifecycle validation for Affinetes environments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import docker

from ..api import build_image_from_env, load_env
from ..core import get_registry
from ..utils.exceptions import ValidationError

_PROMPT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)",
    re.IGNORECASE,
)
_OWNER_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_OWNER_LABEL = "io.affinetes.validation.owner"
_MAX_VALIDATION_TASKS = 10_000
_MAX_REPORT_STRING = 1024
_UNSAFE_REPORT_ERROR = "UnsafeReportString"


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _contains_secret(value: str, secrets_to_protect: Sequence[str]) -> bool:
    return any(secret and secret in value for secret in secrets_to_protect)


def _report_secrets(
    api_key: str | None,
    env_vars: Mapping[str, str] | None,
) -> tuple[str, ...]:
    values: list[str] = []
    if api_key:
        values.append(api_key)
    for key, value in (env_vars or {}).items():
        if value and _SENSITIVE_ENVIRONMENT_NAME.search(key):
            values.append(value)
    return tuple(dict.fromkeys(values))


def _safe_report_string(
    value: str,
    *,
    secrets_to_protect: Sequence[str],
) -> bool:
    return (
        len(value) <= _MAX_REPORT_STRING
        and not _contains_control_character(value)
        and not _contains_secret(value, secrets_to_protect)
    )


def _sanitize_report(
    report: Mapping[str, Any],
    *,
    secrets_to_protect: Sequence[str],
) -> dict[str, Any]:
    """Fail closed and redact if any report string violates the safe boundary."""

    unsafe = False

    def visit(value: Any) -> Any:
        nonlocal unsafe
        if isinstance(value, str):
            if _safe_report_string(
                value,
                secrets_to_protect=secrets_to_protect,
            ):
                return value
            unsafe = True
            # A fixed textual placeholder can itself equal or contain a
            # supplied secret. ``None`` is an unambiguous JSON-safe redaction
            # that cannot reintroduce secret material or control characters.
            return None
        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if (
                    not isinstance(key, str)
                    or len(key) > _MAX_REPORT_STRING
                    or _contains_control_character(key)
                ):
                    unsafe = True
                    key = f"unsafe_key_{index}"
                sanitized[key] = visit(item)
            return sanitized
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        return value

    sanitized_report = visit(report)
    assert isinstance(sanitized_report, dict)
    if unsafe:
        sanitized_report["passed"] = False
        sanitized_report["operation_error_type"] = (
            _UNSAFE_REPORT_ERROR
            if _safe_report_string(
                _UNSAFE_REPORT_ERROR,
                secrets_to_protect=secrets_to_protect,
            )
            else None
        )
    return sanitized_report


def _seed(namespace: str, task_id: int, variant: int) -> int:
    payload = f"{namespace}:{task_id}:{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _seed_pair(namespace: str, task_id: int) -> tuple[int, int]:
    first = _seed(namespace, task_id, 0)
    second = _seed(namespace, task_id, 1)
    if second == first:
        second = (first + 1) % (2**32)
    return first, second


def _prompt_sha256(result: Mapping[str, Any]) -> str | None:
    """Return a conversation-verified prompt commitment, never bare attestation."""

    extra = result.get("extra")
    if not isinstance(extra, Mapping):
        return None

    committed = extra.get("prompt_sha256")
    if not isinstance(committed, str) or _PROMPT_SHA256.fullmatch(committed) is None:
        return None

    conversation = extra.get("conversation")
    if (
        not isinstance(conversation, Sequence)
        or isinstance(conversation, (str, bytes))
        or not conversation
    ):
        return None
    first_user_content: str | None = None
    for message in conversation:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str) or not content:
                return None
            first_user_content = content
            break
    if first_user_content is None:
        return None
    conversation_sha256 = hashlib.sha256(first_user_content.encode("utf-8")).hexdigest()
    return committed if committed == conversation_sha256 else None


def _error_code(
    result: Mapping[str, Any],
    *,
    secrets_to_protect: Sequence[str],
) -> str | None:
    extra = result.get("extra")
    value = extra.get("error_code") if isinstance(extra, Mapping) else None
    if (
        not isinstance(value, str)
        or _ERROR_CODE.fullmatch(value) is None
        or _contains_secret(value, secrets_to_protect)
    ):
        return None
    return value


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        score = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _task_ids(
    *,
    task_ids: Sequence[int] | None,
    task_id_start: int | None,
    task_id_end: int | None,
    num_tests: int | None,
) -> tuple[int, ...]:
    if task_ids is not None and (
        not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes))
    ):
        raise ValidationError("task_ids must be a sequence of integers")
    explicit_ids = task_ids is not None and len(task_ids) > 0
    if explicit_ids:
        if (
            task_id_start is not None
            or task_id_end is not None
            or num_tests is not None
        ):
            raise ValidationError(
                "--task-id cannot be combined with task range arguments"
            )
        selected = tuple(task_ids)
    else:
        start = 0 if task_id_start is None else task_id_start
        for value, name in (
            (start, "task_id_start"),
            (task_id_end, "task_id_end"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValidationError(f"{name} must be a non-negative integer")
        if task_id_end is not None and num_tests is not None:
            raise ValidationError(
                "choose either --task-id-end or --num-tests for range selection"
            )
        if task_id_end is None:
            count = 2 if num_tests is None else num_tests
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValidationError("num_tests must be a positive integer")
            end = start + count - 1
        else:
            end = task_id_end
        if end < start:
            raise ValidationError(
                "task_id_end must be greater than or equal to task_id_start"
            )
        if end - start + 1 > _MAX_VALIDATION_TASKS:
            raise ValidationError(
                f"validation is limited to {_MAX_VALIDATION_TASKS} valid task IDs"
            )
        selected = tuple(range(start, end + 1))

    if any(
        isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0
        for task_id in selected
    ):
        raise ValidationError("validation task IDs must be non-negative integers")
    if len(selected) != len(set(selected)):
        raise ValidationError("validation task IDs must be unique")
    if len(selected) > _MAX_VALIDATION_TASKS:
        raise ValidationError(
            f"validation is limited to {_MAX_VALIDATION_TASKS} valid task IDs"
        )
    if len(selected) < 2:
        raise ValidationError(
            "at least two valid task IDs are required to verify task diversity"
        )
    return selected


def parse_environment_assignments(values: Sequence[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` without silently dropping malformed assignments."""

    parsed: dict[str, str] = {}
    for assignment in values or ():
        if not isinstance(assignment, str) or "=" not in assignment:
            raise ValidationError("--env values must use KEY=VALUE syntax")
        key, value = assignment.split("=", 1)
        if _ENVIRONMENT_NAME.fullmatch(key) is None:
            raise ValidationError(f"invalid environment variable name: {key!r}")
        if key in parsed:
            raise ValidationError(f"duplicate environment variable: {key}")
        if "\x00" in value:
            raise ValidationError("environment variable values cannot contain NUL")
        parsed[key] = value
    return parsed


def resolve_api_key(
    *,
    api_key: str | None,
    api_key_env: str | None,
) -> str | None:
    """Resolve a credential without including its value in an error message."""

    if api_key is not None and api_key_env is not None:
        raise ValidationError("choose either --api-key or --api-key-env")
    if api_key_env is None:
        return api_key
    if (
        not isinstance(api_key_env, str)
        or _ENVIRONMENT_NAME.fullmatch(api_key_env) is None
    ):
        raise ValidationError("api_key_env must be a valid environment variable name")
    value = os.environ.get(api_key_env)
    if value is None or not value:
        raise ValidationError("the selected API key environment variable is not set")
    return value


def _validate_optional_text(
    value: str | None,
    *,
    name: str,
    max_length: int = _MAX_REPORT_STRING,
) -> None:
    if value is not None and (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
        or _contains_control_character(value)
    ):
        raise ValidationError(
            f"{name} must be a bounded non-empty string without control characters"
        )


def _validate_inputs(
    *,
    env_dir: str | None,
    image: str | None,
    expected_failure_task_ids: Sequence[int],
    expected_failure_error_code: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    timeout: int,
    container_name: str | None,
    env_vars: Mapping[str, str] | None,
    pull: bool,
    host_network: bool,
    no_cache: bool,
) -> tuple[int, ...]:
    _validate_optional_text(env_dir, name="env_dir")
    _validate_optional_text(image, name="image")
    if (env_dir is None) == (image is None):
        raise ValidationError("exactly one of env_dir or image must be supplied")
    if image is not None and no_cache:
        raise ValidationError("no_cache is only valid when building from env_dir")
    if env_dir is not None and pull:
        raise ValidationError("pull is only valid with a pre-built image")

    if not isinstance(expected_failure_task_ids, Sequence) or isinstance(
        expected_failure_task_ids, (str, bytes)
    ):
        raise ValidationError("expected_failure_task_ids must be a sequence")
    expected_failures = tuple(expected_failure_task_ids)
    if any(
        isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0
        for task_id in expected_failures
    ):
        raise ValidationError("expected failure task IDs must be non-negative integers")
    if len(expected_failures) != len(set(expected_failures)):
        raise ValidationError("expected failure task IDs must be unique")
    if len(expected_failures) > _MAX_VALIDATION_TASKS:
        raise ValidationError(
            f"validation is limited to {_MAX_VALIDATION_TASKS} expected failures"
        )
    _validate_optional_text(
        expected_failure_error_code,
        name="expected_failure_error_code",
        max_length=128,
    )
    if expected_failure_error_code is not None and not expected_failures:
        raise ValidationError(
            "expected_failure_error_code requires an expected failure task ID"
        )
    if expected_failures and expected_failure_error_code is None:
        raise ValidationError(
            "expected_failure_error_code is required for expected failure task IDs"
        )
    if (
        expected_failure_error_code is not None
        and _ERROR_CODE.fullmatch(expected_failure_error_code) is None
    ):
        raise ValidationError(
            "expected_failure_error_code must be a structured error identifier"
        )

    _validate_optional_text(model, name="model", max_length=256)
    _validate_optional_text(base_url, name="base_url", max_length=2048)
    _validate_optional_text(api_key, name="api_key", max_length=4096)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0.0 <= float(temperature) <= 2.0
    ):
        raise ValidationError("temperature must be finite and between 0 and 2")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= 3600
    ):
        raise ValidationError("timeout must be an integer between 1 and 3600")
    if container_name is not None and _CONTAINER_NAME.fullmatch(container_name) is None:
        raise ValidationError("container_name is not a valid Docker container name")
    for value, name in (
        (pull, "pull"),
        (host_network, "host_network"),
        (no_cache, "no_cache"),
    ):
        if not isinstance(value, bool):
            raise ValidationError(f"{name} must be a boolean")
    if env_vars is not None:
        if not isinstance(env_vars, Mapping):
            raise ValidationError("env_vars must be a mapping")
        for key, value in env_vars.items():
            if not isinstance(key, str) or _ENVIRONMENT_NAME.fullmatch(key) is None:
                raise ValidationError("env_vars contains an invalid variable name")
            if not isinstance(value, str) or "\x00" in value:
                raise ValidationError("env_vars values must be NUL-free strings")
    secrets_to_protect = _report_secrets(api_key, env_vars)
    for value, name in (
        (image, "image"),
        (container_name, "container_name"),
        (expected_failure_error_code, "expected_failure_error_code"),
    ):
        if isinstance(value, str) and _contains_secret(value, secrets_to_protect):
            raise ValidationError(f"{name} must not contain a supplied secret")
    return expected_failures


def _build_image(env_dir: str, *, no_cache: bool) -> str:
    env_path = Path(env_dir).resolve()
    if not env_path.is_dir():
        raise ValidationError("env_dir must be an existing directory")
    if not (env_path / "env.py").is_file():
        raise ValidationError("env_dir must contain env.py")
    if not (env_path / "Dockerfile").is_file():
        raise ValidationError("env_dir must contain Dockerfile")
    safe_name = re.sub(r"[^a-z0-9_.-]+", "-", env_path.name.lower()).strip("-.")
    safe_name = (safe_name or "environment")[:32]
    source_hash = hashlib.sha256(str(env_path).encode("utf-8")).hexdigest()[:12]
    image_reference = build_image_from_env(
        env_path=str(env_path),
        image_tag=f"affinetes-validate-{safe_name}:{source_hash}",
        nocache=no_cache,
        quiet=True,
    )
    if (
        not isinstance(image_reference, str)
        or not image_reference
        or image_reference != image_reference.strip()
    ):
        raise ValidationError("image build did not return an image reference")
    return image_reference


def _validation_container_name(image_reference: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    image_hash = hashlib.sha256(image_reference.encode("utf-8")).hexdigest()[:12]
    return f"affinetes-validate-{image_hash}-{secrets.token_hex(4)}"


def _assert_container_name_available(container_name: str) -> None:
    """Refuse to reuse or replace a container outside this validation run."""

    try:
        client = docker.from_env()
    except Exception as exc:
        raise ValidationError(
            f"cannot inspect Docker containers ({type(exc).__name__})"
        ) from exc
    try:
        try:
            client.containers.get(container_name)
        except docker.errors.NotFound:
            return
        except Exception as exc:
            raise ValidationError(
                f"cannot verify container name availability ({type(exc).__name__})"
            ) from exc
        raise ValidationError(
            f"container {container_name!r} already exists; refusing to reuse it"
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


def _container_identity(
    container: Any,
) -> tuple[str | None, str | None, str | None]:
    """Return container ID, validation owner, and immutable local image ID."""

    try:
        container.reload()
        container_id = container.id
        labels = container.labels
        attrs = container.attrs
    except Exception:
        return None, None, None
    if (
        not isinstance(container_id, str)
        or _CONTAINER_ID.fullmatch(container_id) is None
    ):
        return None, None, None
    if not isinstance(labels, Mapping):
        labels = {}
    owner = labels.get(_OWNER_LABEL)
    image_id = attrs.get("Image") if isinstance(attrs, Mapping) else None
    if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
        image_id = None
    return container_id, owner if isinstance(owner, str) else None, image_id


def _inspect_owned_container(
    container_name: str,
    *,
    owner_token: str,
) -> tuple[bool, str | None, str | None, str | None]:
    """Bind a loaded environment to the exact labeled Docker object."""

    try:
        client = docker.from_env()
    except Exception:
        return False, None, None, "DockerClientError"
    try:
        try:
            container = client.containers.get(container_name)
        except docker.errors.NotFound:
            return False, None, None, "ContainerMissingAfterLoad"
        except Exception:
            return False, None, None, "ContainerLookupError"
        container_id, observed_owner, image_id = _container_identity(container)
        if container_id is None:
            return False, None, None, "ContainerInspectionError"
        if image_id is None:
            return False, container_id, None, "ContainerImageIdentityError"
        if observed_owner != owner_token:
            return False, container_id, image_id, "ContainerOwnershipMismatch"
        return True, container_id, image_id, None
    finally:
        try:
            client.close()
        except Exception:
            pass


def _quarantine_unverified_environment(environment: Any) -> bool:
    """Prevent registry/destructor cleanup from mutating an unowned object."""

    backend = vars(environment).get("_backend")
    name = vars(environment).get("name")
    if backend is None or not isinstance(name, str) or not name:
        return False
    try:
        setattr(backend, "_auto_cleanup", False)
        get_registry().unregister(name)
    except Exception:
        return False
    return True


def _name_reused(client: Any, container_name: str) -> bool | None:
    """Return whether a different object currently occupies ``container_name``."""

    try:
        client.containers.get(container_name)
    except docker.errors.NotFound:
        return False
    except Exception:
        return None
    return True


def _cleanup_owned_container(
    container_name: str,
    *,
    owner_token: str,
    expected_container_id: str | None,
    expected_image_id: str | None,
) -> tuple[bool, str | None]:
    """Remove only the exact labeled container owned by this validation run."""

    try:
        client = docker.from_env()
    except Exception:
        return False, "DockerClientError"
    try:
        lookup = expected_container_id or container_name
        try:
            container = client.containers.get(lookup)
        except docker.errors.NotFound:
            reused = _name_reused(client, container_name)
            if reused is True:
                return False, "ContainerIdentityMismatch"
            if reused is None:
                return False, "ContainerCleanupVerificationError"
            return True, None
        except Exception:
            return False, "ContainerLookupError"

        container_id, observed_owner, image_id = _container_identity(container)
        if container_id is None:
            return False, "ContainerInspectionError"
        if expected_container_id is not None and container_id != expected_container_id:
            return False, "ContainerIdentityMismatch"
        if expected_image_id is not None and image_id != expected_image_id:
            return False, "ContainerImageIdentityMismatch"
        if image_id is None:
            return False, "ContainerImageIdentityError"
        if observed_owner != owner_token:
            return False, "ContainerOwnershipMismatch"

        try:
            container.remove(force=True)
        except Exception:
            return False, "ContainerRemovalError"

        try:
            client.containers.get(container_id)
        except docker.errors.NotFound:
            reused = _name_reused(client, container_name)
            if reused is True:
                return False, "ContainerIdentityMismatch"
            if reused is None:
                return False, "ContainerCleanupVerificationError"
            return True, None
        except Exception:
            return False, "ContainerCleanupVerificationError"
        return False, "ContainerStillPresent"
    finally:
        try:
            client.close()
        except Exception:
            pass


def _evaluation_kwargs(
    *,
    task_id: int,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "task_id": task_id,
        "temperature": float(temperature),
        "timeout": timeout,
        "_timeout": timeout + 10,
    }
    if model is not None:
        kwargs["model"] = model
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    return kwargs


async def _evaluate_once(
    environment: Any,
    *,
    seed: int,
    kwargs: Mapping[str, Any],
    secrets_to_protect: Sequence[str],
) -> dict[str, Any]:
    try:
        result = await environment.evaluate(seed=seed, **dict(kwargs))
    except Exception as exc:
        return {
            "call_completed": False,
            "exception_type": type(exc).__name__,
            "success": None,
            "score": None,
            "score_is_finite_number": False,
            "error_code": None,
            "prompt_sha256": None,
        }
    if not isinstance(result, Mapping):
        return {
            "call_completed": True,
            "result_is_mapping": False,
            "success": None,
            "score": None,
            "score_is_finite_number": False,
            "error_code": None,
            "prompt_sha256": None,
        }
    success = result.get("success")
    score = _finite_score(result.get("score"))
    extra = result.get("extra")
    return {
        "call_completed": True,
        "result_is_mapping": True,
        "success_is_boolean": isinstance(success, bool),
        "success": success if isinstance(success, bool) else None,
        "score_is_finite_number": score is not None,
        "score": score,
        "error_code_present": isinstance(extra, Mapping) and "error_code" in extra,
        "error_code": _error_code(
            result,
            secrets_to_protect=secrets_to_protect,
        ),
        "prompt_sha256": _prompt_sha256(result),
    }


def _valid_row(
    task_id: int, first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict:
    prompt = first.get("prompt_sha256")
    invariant = prompt is not None and prompt == second.get("prompt_sha256")
    first_passed = (
        first.get("call_completed") is True
        and first.get("result_is_mapping") is True
        and first.get("success") is True
        and first.get("score_is_finite_number") is True
        and first.get("error_code_present") is False
        and prompt is not None
    )
    second_passed = (
        second.get("call_completed") is True
        and second.get("result_is_mapping") is True
        and second.get("success") is True
        and second.get("score_is_finite_number") is True
        and second.get("error_code_present") is False
        and second.get("prompt_sha256") is not None
    )
    return {
        "task_id": task_id,
        "expected_failure": False,
        "first": dict(first),
        "second": dict(second),
        "prompt_sha256": prompt,
        "task_seed_invariant": invariant,
        "passed": first_passed and second_passed and invariant,
    }


def _failure_row(
    task_id: int,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    expected_error_code: str | None,
) -> dict:
    first_code = first.get("error_code")
    second_code = second.get("error_code")
    stable_code = first_code is not None and first_code == second_code
    matches_expected = (
        expected_error_code is not None and first_code == expected_error_code
    )
    rejected = (
        first.get("call_completed") is True
        and second.get("call_completed") is True
        and first.get("result_is_mapping") is True
        and second.get("result_is_mapping") is True
        and first.get("success") is False
        and second.get("success") is False
        and stable_code
        and matches_expected
    )
    return {
        "task_id": task_id,
        "expected_failure": True,
        "first": dict(first),
        "second": dict(second),
        "stable_error_code": stable_code,
        "matches_expected_error_code": matches_expected,
        "rejected": rejected,
        "passed": rejected,
    }


def _write_report(report: Mapping[str, Any], output_dir: str | None) -> None:
    if output_dir is None:
        return
    output = Path(output_dir)
    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ValidationError(
            f"cannot create validation output directory ({type(exc).__name__})"
        ) from exc

    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(output, directory_flags)
    except OSError as exc:
        raise ValidationError(
            "validation output must be a real directory, not a symlink or file"
        ) from exc

    temporary_name = f".summary.json.{secrets.token_hex(16)}.tmp"
    temporary_fd: int | None = None
    temporary_exists = False
    try:
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        temporary_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_exists = True
        os.fchmod(temporary_fd, 0o600)
        payload = (
            json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        with os.fdopen(temporary_fd, "wb") as temporary_file:
            temporary_fd = None
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(
            temporary_name,
            "summary.json",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_exists = False
        os.fsync(directory_fd)
    except OSError as exc:
        raise ValidationError(
            f"cannot publish validation report ({type(exc).__name__})"
        ) from exc
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        os.close(directory_fd)


async def validate_environment(
    *,
    env_dir: str | None = None,
    image: str | None = None,
    task_ids: Sequence[int] | None = None,
    task_id_start: int | None = None,
    task_id_end: int | None = None,
    num_tests: int | None = None,
    expected_failure_task_ids: Sequence[int] = (),
    expected_failure_error_code: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout: int = 60,
    output_dir: str | None = None,
    pull: bool = False,
    host_network: bool = False,
    container_name: str | None = None,
    env_vars: Mapping[str, str] | None = None,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Validate task identity, successful evaluation, rejection, and cleanup."""

    expected_failures = _validate_inputs(
        env_dir=env_dir,
        image=image,
        expected_failure_task_ids=expected_failure_task_ids,
        expected_failure_error_code=expected_failure_error_code,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        container_name=container_name,
        env_vars=env_vars,
        pull=pull,
        host_network=host_network,
        no_cache=no_cache,
    )
    secrets_to_protect = _report_secrets(api_key, env_vars)
    selected_task_ids = _task_ids(
        task_ids=task_ids,
        task_id_start=task_id_start,
        task_id_end=task_id_end,
        num_tests=num_tests,
    )
    if set(selected_task_ids) & set(expected_failures):
        raise ValidationError("valid and expected-failure task IDs must be disjoint")

    image_reference = (
        image if image is not None else _build_image(env_dir, no_cache=no_cache)
    )
    assert image_reference is not None
    _validate_optional_text(image_reference, name="image_reference")
    if _contains_secret(image_reference, secrets_to_protect):
        raise ValidationError("image_reference must not contain a supplied secret")
    owned_container_name = _validation_container_name(image_reference, container_name)
    _assert_container_name_available(owned_container_name)
    owner_token = secrets.token_hex(32)
    if (
        _OWNER_TOKEN.fullmatch(owner_token) is None
    ):  # pragma: no cover - stdlib contract
        raise ValidationError("failed to generate validation ownership token")
    secrets_to_protect = (*secrets_to_protect, owner_token)

    namespace = image_reference
    environment = None
    load_attempted = False
    container_ownership_verified = False
    container_ownership_error_type: str | None = None
    owned_container_id: str | None = None
    owned_image_id: str | None = None
    unverified_environment_quarantined = False
    wrapper_cleanup_skipped = False
    cleanup_attempted = False
    cleanup_completed = False
    cleanup_error_type: str | None = None
    container_cleanup_completed = False
    container_cleanup_error_type: str | None = None
    operation_error_type: str | None = None
    rows: list[dict[str, Any]] = []
    try:
        load_attempted = True
        environment = load_env(
            image=image_reference,
            container_name=owned_container_name,
            env_vars=dict(env_vars or {}),
            cleanup=True,
            force_recreate=False,
            pull=pull,
            host_network=host_network,
            create_only=True,
            expected_owner=(_OWNER_LABEL, owner_token),
            labels={_OWNER_LABEL: owner_token},
        )
        (
            container_ownership_verified,
            inspected_container_id,
            inspected_image_id,
            container_ownership_error_type,
        ) = _inspect_owned_container(
            owned_container_name,
            owner_token=owner_token,
        )
        if not container_ownership_verified:
            unverified_environment_quarantined = _quarantine_unverified_environment(
                environment
            )
            raise ValidationError("loaded container ownership verification failed")
        owned_container_id = inspected_container_id
        owned_image_id = inspected_image_id
        for task_id in selected_task_ids:
            first_seed, second_seed = _seed_pair(namespace, task_id)
            kwargs = _evaluation_kwargs(
                task_id=task_id,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                timeout=timeout,
            )
            first = await _evaluate_once(
                environment,
                seed=first_seed,
                kwargs=kwargs,
                secrets_to_protect=secrets_to_protect,
            )
            second = await _evaluate_once(
                environment,
                seed=second_seed,
                kwargs=kwargs,
                secrets_to_protect=secrets_to_protect,
            )
            rows.append(_valid_row(task_id, first, second))

        for task_id in expected_failures:
            first_seed, second_seed = _seed_pair(namespace, task_id)
            kwargs = _evaluation_kwargs(
                task_id=task_id,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                timeout=timeout,
            )
            first = await _evaluate_once(
                environment,
                seed=first_seed,
                kwargs=kwargs,
                secrets_to_protect=secrets_to_protect,
            )
            second = await _evaluate_once(
                environment,
                seed=second_seed,
                kwargs=kwargs,
                secrets_to_protect=secrets_to_protect,
            )
            rows.append(
                _failure_row(
                    task_id,
                    first,
                    second,
                    expected_error_code=expected_failure_error_code,
                )
            )
    except Exception as exc:
        operation_error_type = type(exc).__name__
    finally:
        wrapper_cleanup_error_type: str | None = None
        cleanup_interrupt: BaseException | None = None
        if environment is not None and container_ownership_verified:
            try:
                await environment.cleanup()
            except BaseException as exc:
                wrapper_cleanup_error_type = type(exc).__name__
                if not isinstance(exc, Exception):
                    cleanup_interrupt = exc
        elif environment is not None:
            # A wrapper that points at an unverified object must never be
            # allowed to stop or remove that object. Label-gated cleanup below
            # remains able to recover a container created during partial load.
            wrapper_cleanup_skipped = True
        if load_attempted:
            cleanup_attempted = True
            (
                container_cleanup_completed,
                container_cleanup_error_type,
            ) = _cleanup_owned_container(
                owned_container_name,
                owner_token=owner_token,
                expected_container_id=owned_container_id,
                expected_image_id=owned_image_id,
            )
            cleanup_error_type = (
                wrapper_cleanup_error_type or container_cleanup_error_type
            )
            cleanup_completed = (
                wrapper_cleanup_error_type is None and container_cleanup_completed
            )
        if cleanup_interrupt is not None:
            raise cleanup_interrupt

    valid_rows = [row for row in rows if not row["expected_failure"]]
    failure_rows = [row for row in rows if row["expected_failure"]]
    prompt_hashes = [row["prompt_sha256"] for row in valid_rows]
    valid_evaluations_passed = len(valid_rows) == len(selected_task_ids) and all(
        row["passed"] for row in valid_rows
    )
    seed_invariants_passed = len(valid_rows) == len(selected_task_ids) and all(
        row["task_seed_invariant"] for row in valid_rows
    )
    task_diversity_passed = (
        len(prompt_hashes) == len(selected_task_ids)
        and None not in prompt_hashes
        and len(prompt_hashes) == len(set(prompt_hashes))
    )
    expected_failures_passed = len(failure_rows) == len(expected_failures) and all(
        row["rejected"] for row in failure_rows
    )
    passed = (
        operation_error_type is None
        and container_ownership_verified
        and cleanup_completed
        and valid_evaluations_passed
        and seed_invariants_passed
        and task_diversity_passed
        and expected_failures_passed
    )
    report: dict[str, Any] = {
        "schema_version": "1.2",
        "image": image_reference,
        "container_name": owned_container_name,
        "passed": passed,
        "operation_error_type": operation_error_type,
        "container_ownership_verified": container_ownership_verified,
        "container_ownership_error_type": container_ownership_error_type,
        "container_id": owned_container_id,
        "container_image_id": owned_image_id,
        "unverified_environment_quarantined": unverified_environment_quarantined,
        "wrapper_cleanup_skipped": wrapper_cleanup_skipped,
        "cleanup_attempted": cleanup_attempted,
        "cleanup_completed": cleanup_completed,
        "cleanup_error_type": cleanup_error_type,
        "container_cleanup_completed": container_cleanup_completed,
        "container_cleanup_error_type": container_cleanup_error_type,
        "valid_evaluations_passed": valid_evaluations_passed,
        "task_seed_invariants_passed": seed_invariants_passed,
        "task_diversity_passed": task_diversity_passed,
        "expected_failures_passed": expected_failures_passed,
        "valid_task_count": len(valid_rows),
        "expected_failure_task_count": len(failure_rows),
        "rows": rows,
    }
    sanitized_report = _sanitize_report(
        report,
        secrets_to_protect=secrets_to_protect,
    )
    _write_report(sanitized_report, output_dir)
    return sanitized_report


__all__ = [
    "parse_environment_assignments",
    "resolve_api_key",
    "validate_environment",
]
