import hashlib
import importlib
import json
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from docker.errors import APIError, NotFound

from affinetes.utils.exceptions import ValidationError

validate_module = importlib.import_module("affinetes.cli.validate")
main_module = importlib.import_module("affinetes.cli.main")
assert_container_name_available = validate_module._assert_container_name_available
cleanup_owned_container = validate_module._cleanup_owned_container
CONTAINER_ID = "c" * 64
FOREIGN_CONTAINER_ID = "f" * 64
REPLACEMENT_CONTAINER_ID = "e" * 64
IMAGE_ID = "sha256:" + "d" * 64


def _successful_result(
    task_id: int,
    *,
    prompt: str | None = None,
    score: object = 1.0,
) -> dict:
    content = prompt if prompt is not None else f"prompt for task {task_id}"
    return {
        "success": True,
        "score": score,
        "extra": {
            "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "conversation": [{"role": "user", "content": content}],
        },
    }


class FakeEnvironment:
    def __init__(
        self,
        *,
        expected_failures: set[int] | None = None,
        valid_failures: set[int] | None = None,
        duplicate_prompt: bool = False,
        mismatched_commitment: bool = False,
        exception_task: int | None = None,
        cleanup_error: Exception | None = None,
        unstable_failure_code: bool = False,
        contradictory_success_error: bool = False,
        score_value: object = 1.0,
        omit_score: bool = False,
        omit_conversation: bool = False,
        expected_failure_code: str = "invalid_task_id",
    ) -> None:
        self.expected_failures = expected_failures or set()
        self.valid_failures = valid_failures or set()
        self.duplicate_prompt = duplicate_prompt
        self.mismatched_commitment = mismatched_commitment
        self.exception_task = exception_task
        self.cleanup_error = cleanup_error
        self.unstable_failure_code = unstable_failure_code
        self.contradictory_success_error = contradictory_success_error
        self.score_value = score_value
        self.omit_score = omit_score
        self.omit_conversation = omit_conversation
        self.expected_failure_code = expected_failure_code
        self.calls: list[dict] = []
        self.cleanup_calls = 0

    async def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        task_id = kwargs["task_id"]
        if task_id == self.exception_task:
            raise RuntimeError("secret transport details")
        if task_id in self.expected_failures:
            code = self.expected_failure_code
            if (
                self.unstable_failure_code
                and len([call for call in self.calls if call["task_id"] == task_id]) > 1
            ):
                code = "different_error"
            return {"success": False, "extra": {"error_code": code}}
        if task_id in self.valid_failures:
            result = _successful_result(task_id)
            result["success"] = False
            result["extra"]["error_code"] = "endpoint_failed"
            return result
        prompt = "duplicate prompt" if self.duplicate_prompt else None
        result = _successful_result(task_id, prompt=prompt, score=self.score_value)
        if self.omit_score:
            result.pop("score")
        if self.omit_conversation:
            result["extra"].pop("conversation")
        if self.mismatched_commitment:
            result["extra"]["prompt_sha256"] = "0" * 64
        if self.contradictory_success_error:
            result["extra"]["error_code"] = 123
        return result

    async def cleanup(self):
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


@pytest.fixture(autouse=True)
def allow_generated_container_name(monkeypatch):
    monkeypatch.setattr(
        validate_module,
        "_assert_container_name_available",
        lambda _name: None,
    )
    monkeypatch.setattr(
        validate_module,
        "_inspect_owned_container",
        lambda _name, *, owner_token: (True, CONTAINER_ID, IMAGE_ID, None),
    )
    monkeypatch.setattr(
        validate_module,
        "_cleanup_owned_container",
        lambda _name, **_kwargs: (True, None),
    )


def _install_environment(monkeypatch, environment: FakeEnvironment) -> dict:
    observed: dict = {}

    def fake_load_env(**kwargs):
        observed.update(kwargs)
        return environment

    monkeypatch.setattr(validate_module, "load_env", fake_load_env)
    return observed


@pytest.mark.asyncio
async def test_image_validation_passes_all_fail_closed_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = FakeEnvironment(expected_failures={99})
    load_kwargs = _install_environment(monkeypatch, environment)
    output = tmp_path / "report"

    report = await validate_module.validate_environment(
        image="registry.example/instruction-gym@sha256:" + "a" * 64,
        task_ids=[0, 1],
        expected_failure_task_ids=[99],
        expected_failure_error_code="invalid_task_id",
        model="evaluation-model",
        base_url="https://model.example/v1",
        api_key="super-secret",
        temperature=0.0,
        timeout=30,
        output_dir=str(output),
        pull=True,
        host_network=True,
        env_vars={"UVICORN_WORKERS": "1"},
    )

    assert report["passed"] is True
    assert report["schema_version"] == "1.2"
    assert report["container_ownership_verified"] is True
    assert report["container_id"] == CONTAINER_ID
    assert report["container_image_id"] == IMAGE_ID
    assert report["unverified_environment_quarantined"] is False
    assert report["valid_evaluations_passed"] is True
    assert report["task_seed_invariants_passed"] is True
    assert report["task_diversity_passed"] is True
    assert report["expected_failures_passed"] is True
    assert report["cleanup_completed"] is True
    assert environment.cleanup_calls == 1
    assert len(environment.calls) == 6
    for task_id in (0, 1, 99):
        seeds = [
            call["seed"] for call in environment.calls if call["task_id"] == task_id
        ]
        assert len(seeds) == 2
        assert seeds[0] != seeds[1]
    assert all(call["model"] == "evaluation-model" for call in environment.calls)
    assert all(
        call["base_url"] == "https://model.example/v1" for call in environment.calls
    )
    assert load_kwargs["pull"] is True
    assert load_kwargs["host_network"] is True
    assert load_kwargs["cleanup"] is True
    assert load_kwargs["force_recreate"] is False
    assert load_kwargs["env_vars"] == {"UVICORN_WORKERS": "1"}
    assert load_kwargs["create_only"] is True
    owner_label, owner_token = load_kwargs["expected_owner"]
    assert owner_label == validate_module._OWNER_LABEL
    assert len(owner_token) == 64
    assert load_kwargs["labels"] == {owner_label: owner_token}

    written = json.loads((output / "summary.json").read_text())
    assert written == report
    serialized = json.dumps(report)
    assert "super-secret" not in serialized
    assert "prompt for task" not in serialized
    assert owner_token not in serialized


@pytest.mark.asyncio
async def test_valid_task_failure_cannot_pass(monkeypatch) -> None:
    environment = FakeEnvironment(valid_failures={0, 1})
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["valid_evaluations_passed"] is False
    assert report["task_seed_invariants_passed"] is True
    assert report["cleanup_completed"] is True


@pytest.mark.asyncio
async def test_prompt_commitment_mismatch_fails_closed(monkeypatch) -> None:
    environment = FakeEnvironment(mismatched_commitment=True)
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["task_seed_invariants_passed"] is False
    assert report["rows"][0]["first"]["prompt_sha256"] is None


@pytest.mark.asyncio
async def test_bare_prompt_commitment_is_not_accepted(monkeypatch) -> None:
    environment = FakeEnvironment(omit_conversation=True)
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["task_seed_invariants_passed"] is False
    assert report["rows"][0]["first"]["prompt_sha256"] is None


@pytest.mark.parametrize(
    "score",
    [True, "1.0", None, float("nan"), float("inf"), float("-inf")],
)
@pytest.mark.asyncio
async def test_valid_tasks_require_finite_non_boolean_numeric_scores(
    monkeypatch,
    score,
) -> None:
    environment = FakeEnvironment(score_value=score)
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["valid_evaluations_passed"] is False
    assert report["rows"][0]["first"]["score_is_finite_number"] is False
    assert report["rows"][0]["first"]["score"] is None


@pytest.mark.asyncio
async def test_missing_score_cannot_pass(monkeypatch) -> None:
    environment = FakeEnvironment(omit_score=True)
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["valid_evaluations_passed"] is False


@pytest.mark.asyncio
async def test_second_evaluation_score_is_checked_independently(monkeypatch) -> None:
    class SecondCallHasInvalidScore(FakeEnvironment):
        async def evaluate(self, **kwargs):
            result = await super().evaluate(**kwargs)
            task_id = kwargs["task_id"]
            task_calls = [call for call in self.calls if call["task_id"] == task_id]
            if len(task_calls) == 2:
                result["score"] = float("nan")
            return result

    environment = SecondCallHasInvalidScore()
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["rows"][0]["first"]["score_is_finite_number"] is True
    assert report["rows"][0]["second"]["score_is_finite_number"] is False


@pytest.mark.asyncio
async def test_duplicate_prompts_across_task_ids_fail(monkeypatch) -> None:
    environment = FakeEnvironment(duplicate_prompt=True)
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["valid_evaluations_passed"] is True
    assert report["task_diversity_passed"] is False


@pytest.mark.asyncio
async def test_success_with_any_error_code_is_contradictory(monkeypatch) -> None:
    environment = FakeEnvironment(contradictory_success_error=True)
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["valid_evaluations_passed"] is False
    assert report["rows"][0]["first"]["error_code_present"] is True


@pytest.mark.asyncio
async def test_evaluation_exception_is_sanitized_and_cleanup_runs(monkeypatch) -> None:
    environment = FakeEnvironment(exception_task=0)
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["cleanup_completed"] is True
    assert environment.cleanup_calls == 1
    serialized = json.dumps(report)
    assert "RuntimeError" in serialized
    assert "secret transport details" not in serialized


@pytest.mark.asyncio
async def test_cleanup_exception_is_a_validation_failure(monkeypatch) -> None:
    environment = FakeEnvironment(cleanup_error=RuntimeError("cleanup details"))
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["cleanup_attempted"] is True
    assert report["cleanup_completed"] is False
    assert report["cleanup_error_type"] == "RuntimeError"
    assert "cleanup details" not in json.dumps(report)


@pytest.mark.asyncio
async def test_cleanup_must_remove_the_owned_container(monkeypatch) -> None:
    environment = FakeEnvironment()
    _install_environment(monkeypatch, environment)
    monkeypatch.setattr(
        validate_module,
        "_cleanup_owned_container",
        lambda _name, **_kwargs: (
            False,
            "ContainerStillPresent",
        ),
    )

    report = await validate_module.validate_environment(image="env:v1")

    assert report["passed"] is False
    assert report["cleanup_completed"] is False
    assert report["cleanup_error_type"] == "ContainerStillPresent"
    assert report["container_cleanup_completed"] is False


@pytest.mark.asyncio
async def test_expected_failure_requires_stable_error_code(monkeypatch) -> None:
    environment = FakeEnvironment(
        expected_failures={99},
        unstable_failure_code=True,
    )
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(
        image="env:v1",
        expected_failure_task_ids=[99],
        expected_failure_error_code="invalid_task_id",
    )

    assert report["passed"] is False
    assert report["expected_failures_passed"] is False
    failure_row = report["rows"][-1]
    assert failure_row["stable_error_code"] is False
    assert failure_row["rejected"] is False


@pytest.mark.parametrize("unsafe_code", ["bad\ncode", "leak_super-secret"])
@pytest.mark.asyncio
async def test_unsafe_or_secret_bearing_error_code_is_rejected_and_not_reported(
    monkeypatch,
    unsafe_code: str,
) -> None:
    environment = FakeEnvironment(
        expected_failures={99},
        expected_failure_code=unsafe_code,
    )
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(
        image="env:v1",
        expected_failure_task_ids=[99],
        expected_failure_error_code="invalid_task_id",
        api_key="super-secret",
    )

    assert report["passed"] is False
    failure_row = report["rows"][-1]
    assert failure_row["first"]["error_code"] is None
    assert failure_row["rejected"] is False
    serialized = json.dumps(report)
    assert "super-secret" not in serialized
    assert "bad\\ncode" not in serialized


def test_report_sanitizer_redacts_and_fails_unsafe_strings() -> None:
    report = validate_module._sanitize_report(
        {
            "passed": True,
            "secret": "prefix-super-secret-suffix",
            "control": "line1\nline2",
            "oversized": "x" * (validate_module._MAX_REPORT_STRING + 1),
        },
        secrets_to_protect=("super-secret",),
    )

    assert report["passed"] is False
    assert report["operation_error_type"] == "UnsafeReportString"
    assert report["secret"] is None
    assert report["control"] is None
    assert report["oversized"] is None
    assert "super-secret" not in json.dumps(report)


def test_report_sanitizer_cannot_reintroduce_a_secret_via_diagnostic_text() -> None:
    report = validate_module._sanitize_report(
        {
            "passed": True,
            "unsafe": "UnsafeReportString",
        },
        secrets_to_protect=("UnsafeReportString",),
    )

    assert report["passed"] is False
    assert report["unsafe"] is None
    assert report["operation_error_type"] is None
    assert "UnsafeReportString" not in json.dumps(report)


def test_report_publish_replaces_summary_symlink_without_writing_through(
    tmp_path: Path,
) -> None:
    external_target = tmp_path / "external-target.json"
    external_target.write_text("external content must remain unchanged\n")
    output = tmp_path / "reports"
    output.mkdir()
    summary = output / "summary.json"
    summary.symlink_to(external_target)

    validate_module._write_report({"passed": True}, str(output))

    assert external_target.read_text() == "external content must remain unchanged\n"
    assert summary.is_symlink() is False
    assert json.loads(summary.read_text()) == {"passed": True}
    assert stat.S_IMODE(summary.stat().st_mode) == 0o600
    assert list(output.glob(".summary.json.*.tmp")) == []


def test_report_publish_rejects_output_directory_symlink(tmp_path: Path) -> None:
    actual_output = tmp_path / "actual-output"
    actual_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(actual_output, target_is_directory=True)

    with pytest.raises(ValidationError, match="real directory"):
        validate_module._write_report({"passed": True}, str(linked_output))

    assert list(actual_output.iterdir()) == []


def test_report_publish_rejects_non_directory_output(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("unchanged\n")

    with pytest.raises(ValidationError, match="real directory"):
        validate_module._write_report({"passed": True}, str(output_file))

    assert output_file.read_text() == "unchanged\n"


@pytest.mark.asyncio
async def test_load_failure_writes_failed_report(monkeypatch, tmp_path: Path) -> None:
    def fail_load(**_kwargs):
        raise RuntimeError("daemon details")

    monkeypatch.setattr(validate_module, "load_env", fail_load)
    output = tmp_path / "output"

    report = await validate_module.validate_environment(
        image="env:v1",
        output_dir=str(output),
    )

    assert report["passed"] is False
    assert report["operation_error_type"] == "RuntimeError"
    assert report["cleanup_attempted"] is True
    assert report["cleanup_completed"] is True
    assert report["container_cleanup_completed"] is True
    assert "daemon details" not in json.dumps(report)
    assert json.loads((output / "summary.json").read_text()) == report


@pytest.mark.asyncio
async def test_directory_mode_builds_required_files_without_pull(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "env.py").write_text("class Actor: pass\n")
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    build_calls: list[dict] = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return "built:v1"

    monkeypatch.setattr(validate_module, "build_image_from_env", fake_build)
    environment = FakeEnvironment()
    _install_environment(monkeypatch, environment)

    report = await validate_module.validate_environment(
        env_dir=str(tmp_path),
        no_cache=True,
    )

    assert report["passed"] is True
    assert report["image"] == "built:v1"
    assert len(build_calls) == 1
    assert build_calls[0]["env_path"] == str(tmp_path.resolve())
    assert build_calls[0]["nocache"] is True
    assert build_calls[0]["image_tag"].startswith("affinetes-validate-")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"image": None, "env_dir": None}, "exactly one"),
        ({"image": "a", "env_dir": "b"}, "exactly one"),
        ({"image": "a", "no_cache": True}, "only valid"),
        ({"env_dir": "/missing", "pull": True}, "only valid"),
        ({"image": "a", "task_ids": [0]}, "at least two"),
        ({"image": "a", "task_ids": [0, 0]}, "unique"),
        (
            {"image": "a", "task_ids": [0, 1], "task_id_start": 0},
            "cannot be combined",
        ),
        (
            {"image": "a", "task_id_end": 2, "num_tests": 3},
            "choose either",
        ),
        (
            {"image": "a", "expected_failure_task_ids": [2, 2]},
            "must be unique",
        ),
        (
            {
                "image": "a",
                "expected_failure_task_ids": [1],
                "expected_failure_error_code": "invalid_task_id",
            },
            "must be disjoint",
        ),
        (
            {"image": "a", "expected_failure_task_ids": [99]},
            "is required",
        ),
        (
            {"image": "a", "expected_failure_error_code": "invalid_task_id"},
            "requires an expected failure",
        ),
        ({"image": "a", "task_id_start": True}, "non-negative integer"),
        ({"image": "a", "num_tests": 10_001}, "limited to"),
        ({"image": "a", "temperature": float("nan")}, "temperature"),
        ({"image": "a", "timeout": 0}, "timeout"),
        (
            {"image": "repo-super-secret", "api_key": "super-secret"},
            "must not contain a supplied secret",
        ),
        ({"image": "bad\nimage"}, "without control characters"),
        (
            {
                "image": "a",
                "expected_failure_task_ids": [99],
                "expected_failure_error_code": "bad code",
            },
            "structured error identifier",
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_inputs_are_rejected_before_loading(
    monkeypatch,
    kwargs: Mapping,
    message: str,
) -> None:
    monkeypatch.setattr(
        validate_module,
        "load_env",
        lambda **_kwargs: pytest.fail("load_env must not be called"),
    )

    with pytest.raises(ValidationError, match=message):
        await validate_module.validate_environment(**kwargs)


def test_environment_assignment_parser_is_strict() -> None:
    assert validate_module.parse_environment_assignments(
        ["A=1", "EMPTY=", "COMPLEX=a=b"]
    ) == {"A": "1", "EMPTY": "", "COMPLEX": "a=b"}

    with pytest.raises(ValidationError, match="KEY=VALUE"):
        validate_module.parse_environment_assignments(["MALFORMED"])
    with pytest.raises(ValidationError, match="invalid environment variable"):
        validate_module.parse_environment_assignments(["1BAD=value"])
    with pytest.raises(ValidationError, match="duplicate environment"):
        validate_module.parse_environment_assignments(["A=1", "A=2"])


def test_seed_pair_is_always_distinct(monkeypatch) -> None:
    monkeypatch.setattr(validate_module, "_seed", lambda *_args: 17)

    assert validate_module._seed_pair("env:v1", 0) == (17, 18)


def test_container_name_preflight_accepts_only_an_absent_name(monkeypatch) -> None:
    class Containers:
        def get(self, _name):
            raise NotFound("absent")

    class Client:
        containers = Containers()

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(validate_module.docker, "from_env", lambda: client)

    assert_container_name_available("dedicated-validation")

    assert client.closed is True


def test_container_name_preflight_refuses_existing_or_unverifiable_name(
    monkeypatch,
) -> None:
    class Client:
        def __init__(self, outcome):
            self.containers = type(
                "Containers",
                (),
                {"get": lambda _self, _name: outcome()},
            )()

        def close(self):
            pass

    monkeypatch.setattr(
        validate_module.docker,
        "from_env",
        lambda: Client(lambda: object()),
    )
    with pytest.raises(ValidationError, match="already exists"):
        assert_container_name_available("occupied")

    def unavailable():
        raise APIError("daemon unavailable")

    monkeypatch.setattr(
        validate_module.docker,
        "from_env",
        lambda: Client(unavailable),
    )
    with pytest.raises(ValidationError, match="cannot verify"):
        assert_container_name_available("unknown")


@pytest.mark.asyncio
async def test_partial_load_failure_force_removes_its_new_container(
    monkeypatch,
) -> None:
    class OwnedContainer:
        def __init__(self, state):
            self.state = state

        @property
        def id(self):
            return self.state["id"]

        @property
        def labels(self):
            return self.state["labels"]

        @property
        def attrs(self):
            return {"Image": self.state["image_id"]}

        def reload(self):
            pass

        def remove(self, *, force):
            assert force is True
            self.state["exists"] = False
            self.state["remove_calls"] += 1

    class Containers:
        def __init__(self, state):
            self.state = state

        def get(self, name):
            if not self.state["exists"] or name not in {
                "dedicated-validation",
                self.state["id"],
            }:
                raise NotFound("absent")
            return OwnedContainer(self.state)

    class Client:
        def __init__(self, state):
            self.containers = Containers(state)

        def close(self):
            pass

    state = {
        "exists": False,
        "id": CONTAINER_ID,
        "image_id": IMAGE_ID,
        "labels": {},
        "remove_calls": 0,
    }

    def partially_start(**kwargs):
        state["exists"] = True
        state["labels"] = kwargs["labels"]
        raise RuntimeError("startup failed after container creation")

    monkeypatch.setattr(validate_module, "load_env", partially_start)
    monkeypatch.setattr(
        validate_module, "_cleanup_owned_container", cleanup_owned_container
    )
    monkeypatch.setattr(
        validate_module.docker,
        "from_env",
        lambda: Client(state),
    )

    report = await validate_module.validate_environment(
        image="env:v1",
        container_name="dedicated-validation",
    )

    assert report["passed"] is False
    assert report["operation_error_type"] == "RuntimeError"
    assert report["cleanup_attempted"] is True
    assert report["cleanup_completed"] is True
    assert report["container_cleanup_completed"] is True
    assert report["container_cleanup_error_type"] is None
    assert state["exists"] is False
    assert state["remove_calls"] == 1
    assert list(state["labels"]) == [validate_module._OWNER_LABEL]
    assert len(state["labels"][validate_module._OWNER_LABEL]) == 64


@pytest.mark.asyncio
async def test_post_load_ownership_mismatch_never_evaluates_or_cleans_wrapper(
    monkeypatch,
) -> None:
    environment = FakeEnvironment()
    _install_environment(monkeypatch, environment)
    monkeypatch.setattr(
        validate_module,
        "_inspect_owned_container",
        lambda _name, *, owner_token: (
            False,
            FOREIGN_CONTAINER_ID,
            IMAGE_ID,
            "ContainerOwnershipMismatch",
        ),
    )
    quarantine_calls: list[object] = []
    monkeypatch.setattr(
        validate_module,
        "_quarantine_unverified_environment",
        lambda value: quarantine_calls.append(value) or True,
    )
    cleanup_calls: list[dict] = []

    def refuse_foreign_cleanup(name, **kwargs):
        cleanup_calls.append({"name": name, **kwargs})
        return False, "ContainerOwnershipMismatch"

    monkeypatch.setattr(
        validate_module,
        "_cleanup_owned_container",
        refuse_foreign_cleanup,
    )

    report = await validate_module.validate_environment(
        image="env:v1",
        container_name="dedicated-validation",
    )

    assert report["passed"] is False
    assert report["operation_error_type"] == "ValidationError"
    assert report["container_ownership_verified"] is False
    assert report["container_ownership_error_type"] == "ContainerOwnershipMismatch"
    assert report["container_id"] is None
    assert report["unverified_environment_quarantined"] is True
    assert report["wrapper_cleanup_skipped"] is True
    assert report["container_cleanup_error_type"] == "ContainerOwnershipMismatch"
    assert environment.calls == []
    assert environment.cleanup_calls == 0
    assert quarantine_calls == [environment]
    assert cleanup_calls[0]["expected_container_id"] is None
    assert cleanup_calls[0]["expected_image_id"] is None
    assert len(cleanup_calls[0]["owner_token"]) == 64


def test_label_gated_cleanup_refuses_a_foreign_container(monkeypatch) -> None:
    class ForeignContainer:
        id = FOREIGN_CONTAINER_ID
        labels = {validate_module._OWNER_LABEL: "b" * 64}
        attrs = {"Image": IMAGE_ID}

        def __init__(self):
            self.remove_calls = 0

        def reload(self):
            pass

        def remove(self, *, force):
            self.remove_calls += 1

    class Client:
        def __init__(self, container):
            self.containers = type(
                "Containers",
                (),
                {"get": lambda _self, _lookup: container},
            )()

        def close(self):
            pass

    foreign = ForeignContainer()
    monkeypatch.setattr(
        validate_module.docker,
        "from_env",
        lambda: Client(foreign),
    )

    completed, error_type = cleanup_owned_container(
        "dedicated-validation",
        owner_token="a" * 64,
        expected_container_id=None,
        expected_image_id=None,
    )

    assert completed is False
    assert error_type == "ContainerOwnershipMismatch"
    assert foreign.remove_calls == 0


def test_image_id_gated_cleanup_refuses_a_different_image(monkeypatch) -> None:
    class Container:
        id = CONTAINER_ID
        labels = {validate_module._OWNER_LABEL: "a" * 64}
        attrs = {"Image": "sha256:" + "e" * 64}

        def __init__(self):
            self.remove_calls = 0

        def reload(self):
            pass

        def remove(self, *, force):
            self.remove_calls += 1

    class Client:
        def __init__(self, container):
            self.containers = type(
                "Containers",
                (),
                {"get": lambda _self, _lookup: container},
            )()

        def close(self):
            pass

    container = Container()
    monkeypatch.setattr(
        validate_module.docker,
        "from_env",
        lambda: Client(container),
    )

    completed, error_type = cleanup_owned_container(
        "dedicated-validation",
        owner_token="a" * 64,
        expected_container_id=CONTAINER_ID,
        expected_image_id=IMAGE_ID,
    )

    assert completed is False
    assert error_type == "ContainerImageIdentityMismatch"
    assert container.remove_calls == 0


def test_quarantine_disables_automatic_cleanup_and_unregisters(monkeypatch) -> None:
    class Backend:
        _auto_cleanup = True

    class Environment:
        def __init__(self):
            self._backend = Backend()
            self.name = "dedicated-validation"

    class Registry:
        def __init__(self):
            self.unregistered = []

        def unregister(self, name):
            self.unregistered.append(name)

    environment = Environment()
    registry = Registry()
    monkeypatch.setattr(validate_module, "get_registry", lambda: registry)

    assert validate_module._quarantine_unverified_environment(environment) is True
    assert environment._backend._auto_cleanup is False
    assert registry.unregistered == ["dedicated-validation"]


def test_id_gated_cleanup_refuses_a_replacement_with_the_same_name(
    monkeypatch,
) -> None:
    class ForeignContainer:
        id = REPLACEMENT_CONTAINER_ID
        labels = {validate_module._OWNER_LABEL: "b" * 64}
        attrs = {"Image": IMAGE_ID}

        def __init__(self):
            self.remove_calls = 0

        def reload(self):
            pass

        def remove(self, *, force):
            self.remove_calls += 1

    class Containers:
        def __init__(self, container):
            self.container = container

        def get(self, lookup):
            if lookup == CONTAINER_ID:
                raise NotFound("original is gone")
            return self.container

    class Client:
        def __init__(self, container):
            self.containers = Containers(container)

        def close(self):
            pass

    replacement = ForeignContainer()
    monkeypatch.setattr(
        validate_module.docker,
        "from_env",
        lambda: Client(replacement),
    )

    completed, error_type = cleanup_owned_container(
        "dedicated-validation",
        owner_token="a" * 64,
        expected_container_id=CONTAINER_ID,
        expected_image_id=IMAGE_ID,
    )

    assert completed is False
    assert error_type == "ContainerIdentityMismatch"
    assert replacement.remove_calls == 0


@pytest.mark.asyncio
async def test_load_race_with_foreign_name_fails_without_deleting_it(
    monkeypatch,
) -> None:
    class ForeignContainer:
        id = FOREIGN_CONTAINER_ID
        labels = {validate_module._OWNER_LABEL: "foreign-owner"}
        attrs = {"Image": IMAGE_ID}

        def __init__(self):
            self.remove_calls = 0

        def reload(self):
            pass

        def remove(self, *, force):
            self.remove_calls += 1

    class Client:
        def __init__(self, container):
            self.containers = type(
                "Containers",
                (),
                {"get": lambda _self, _lookup: container},
            )()

        def close(self):
            pass

    foreign = ForeignContainer()

    def lose_create_race(**_kwargs):
        raise RuntimeError("name conflict after preflight")

    monkeypatch.setattr(validate_module, "load_env", lose_create_race)
    monkeypatch.setattr(
        validate_module,
        "_cleanup_owned_container",
        cleanup_owned_container,
    )
    monkeypatch.setattr(
        validate_module.docker,
        "from_env",
        lambda: Client(foreign),
    )

    report = await validate_module.validate_environment(
        image="env:v1",
        container_name="dedicated-validation",
    )

    assert report["passed"] is False
    assert report["operation_error_type"] == "RuntimeError"
    assert report["container_cleanup_completed"] is False
    assert report["container_cleanup_error_type"] == "ContainerOwnershipMismatch"
    assert foreign.remove_calls == 0


@pytest.mark.asyncio
async def test_preexisting_container_is_never_loaded_or_removed(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_module,
        "_assert_container_name_available",
        lambda _name: (_ for _ in ()).throw(
            ValidationError("container 'occupied' already exists")
        ),
    )
    monkeypatch.setattr(
        validate_module,
        "load_env",
        lambda **_kwargs: pytest.fail("pre-existing container must not be loaded"),
    )
    monkeypatch.setattr(
        validate_module,
        "_cleanup_owned_container",
        lambda _name, **_kwargs: pytest.fail(
            "pre-existing container must not be removed"
        ),
    )

    with pytest.raises(ValidationError, match="already exists"):
        await validate_module.validate_environment(
            image="env:v1",
            container_name="occupied",
        )


def test_validate_parser_supports_image_and_all_runtime_options() -> None:
    args = main_module.create_parser().parse_args(
        [
            "validate",
            "--image",
            "env:v1",
            "--task-id",
            "0",
            "--task-id",
            "2",
            "--expect-failure-task-id",
            "99",
            "--expected-failure-error-code",
            "invalid_task_id",
            "--model",
            "model",
            "--base-url",
            "https://model.example/v1",
            "--api-key",
            "secret",
            "--env",
            "WORKERS=1",
            "--pull",
            "--host-network",
            "--container-name",
            "validate-owned",
            "--output",
            "evidence",
        ]
    )

    assert args.env_dir is None
    assert args.image == "env:v1"
    assert args.task_ids == [0, 2]
    assert args.expected_failure_task_ids == [99]
    assert args.pull is True
    assert args.host_network is True
    assert args.container_name == "validate-owned"
    assert args.api_key == "secret"
    assert args.api_key_env is None


def test_validate_parser_rejects_two_api_key_sources() -> None:
    with pytest.raises(SystemExit):
        main_module.create_parser().parse_args(
            [
                "validate",
                "--image",
                "env:v1",
                "--api-key",
                "legacy-secret",
                "--api-key-env",
                "MODEL_API_KEY",
            ]
        )


def test_api_key_environment_resolution_is_strict_and_secret_safe(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "resolved-secret")
    assert (
        validate_module.resolve_api_key(
            api_key=None,
            api_key_env="MODEL_API_KEY",
        )
        == "resolved-secret"
    )
    with pytest.raises(ValidationError, match="not set") as exc_info:
        validate_module.resolve_api_key(
            api_key=None,
            api_key_env="MISSING_MODEL_API_KEY",
        )
    assert "resolved-secret" not in str(exc_info.value)
    with pytest.raises(ValidationError, match="valid environment variable"):
        validate_module.resolve_api_key(
            api_key=None,
            api_key_env="1INVALID",
        )


def test_main_routes_validation_and_prints_report(monkeypatch, capsys) -> None:
    observed: dict = {}

    async def fake_validate(**kwargs):
        observed.update(kwargs)
        return {"schema_version": "1.0", "passed": True}

    monkeypatch.setattr(main_module, "validate_environment", fake_validate)
    monkeypatch.setenv("MODEL_API_KEY", "resolved-secret")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "afs",
            "validate",
            "--image",
            "env:v1",
            "--task-id",
            "0",
            "--task-id",
            "1",
            "--env",
            "WORKERS=1",
            "--api-key-env",
            "MODEL_API_KEY",
        ],
    )

    main_module.main()

    assert observed["image"] == "env:v1"
    assert observed["task_ids"] == [0, 1]
    assert observed["env_vars"] == {"WORKERS": "1"}
    assert observed["api_key"] == "resolved-secret"
    assert json.loads(capsys.readouterr().out) == {
        "passed": True,
        "schema_version": "1.0",
    }


def test_main_exits_nonzero_when_validation_report_fails(monkeypatch) -> None:
    async def fake_validate(**_kwargs):
        return {"schema_version": "1.0", "passed": False}

    monkeypatch.setattr(main_module, "validate_environment", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["afs", "validate", "--image", "env:v1", "--task-id", "0", "--task-id", "1"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1
