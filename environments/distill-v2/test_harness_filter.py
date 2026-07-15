"""_drop_harness_failures: producer infra deaths must not poison the cell.

Run from this directory: ``pytest test_harness_filter.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from distill_v2_eval.domain.models import Rollout, RolloutStatus  # noqa: E402
from distill_v2_eval.driver import (  # noqa: E402
    _drop_harness_failures,
    _is_harness_failure,
)
from distill_v2_eval.infrastructure.compression import compress_json  # noqa: E402

DAEMON_OUT = (
    "Error response from daemon: container 2b0e9de06995 is not running\n"
)


def _uuid(idx: int) -> str:
    return f"00000000-0000-0000-0000-{idx:012d}"


def _rollout(idx: int, reward: float, breakdown: dict | None) -> Rollout:
    blob, _sha = compress_json({
        "schema_version": "1.0-affent",
        "messages": [{"role": "assistant", "content": "hi"}],
        "reward_breakdown": breakdown,
        "teacher_meta": {},
    })
    return Rollout(
        rollout_id=_uuid(idx),
        env_name="swe-rebench",
        task_id=1,
        teacher_name="t",
        sample_idx=idx,
        temperature=1.0,
        status=RolloutStatus.ok,
        reward=reward,
        schema_version="1.0-affent",
        extra_compressed=blob,
        extra_sha256="x",
        producer_id="test",
    )


def test_daemon_error_rollout_dropped():
    rollouts = [
        _rollout(0, 1.0, {"score": 1.0, "tests_passed": 2, "tests_total": 2,
                          "exit_code": 0, "raw_output": "2 passed"}),
        _rollout(1, 0.0, {"score": 0.0, "tests_passed": 0, "tests_total": 0,
                          "exit_code": 1, "raw_output": DAEMON_OUT}),
    ]
    kept = _drop_harness_failures(rollouts)
    assert [str(r.rollout_id) for r in kept] == [_uuid(0)]


def test_genuine_zero_kept():
    # Agent broke the test run itself (pytest output, no daemon error):
    # a deserved 0/0 stays in the cell.
    rollouts = [
        _rollout(0, 0.0, {"score": 0.0, "tests_passed": 0, "tests_total": 0,
                          "exit_code": 0, "raw_output": "collection error"}),
    ]
    assert len(_drop_harness_failures(rollouts)) == 1


def test_verifiers_breakdown_untouched():
    # verifiers envs have no raw_output/tests_total at all.
    rollouts = [
        _rollout(0, 0.0, {"score": 0.0, "correct_answer": 0.0, "num_turns": 1.0}),
        _rollout(1, 1.0, {"score": 1.0, "correct_answer": 1.0, "num_turns": 1.0}),
    ]
    assert len(_drop_harness_failures(rollouts)) == 2


def test_daemon_string_in_real_test_output_kept():
    # Paranoia: tests ran (tests_total > 0), the marker just appears in
    # their output — must not be dropped.
    b = {"score": 0.5, "tests_passed": 1, "tests_total": 2,
         "exit_code": 0, "raw_output": f"1 passed\n{DAEMON_OUT}"}
    assert not _is_harness_failure(b)


def test_unparseable_extra_kept():
    r = _rollout(0, 1.0, {"score": 1.0})
    object.__setattr__(r, "extra_compressed", b"not-zstd")
    assert len(_drop_harness_failures([r])) == 1
