"""Repository-wide guards for the affentctl build and terminal protocol."""

import hashlib
import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
AFFENT_REF = "8f4f8f6e765ec8faabd2a27a457811f0f2218b74"
DOCKERFILES = (
    ROOT / "environments/SWE-INFINITE/Dockerfile",
    ROOT / "environments/SWE-FRONTIER/Dockerfile",
    ROOT / "environments/qqr/Dockerfile",
)
PROTOCOL_MODULES = (
    ROOT / "environments/SWE-INFINITE/agents/affent_protocol.py",
    ROOT / "environments/SWE-FRONTIER/agents/affent_protocol.py",
    ROOT / "environments/qqr/affent_protocol.py",
)


def _load_protocol_module():
    spec = importlib.util.spec_from_file_location(
        "affent_protocol_contract",
        PROTOCOL_MODULES[0],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_affent_images_use_the_same_pinned_revision() -> None:
    for dockerfile in DOCKERFILES:
        source = dockerfile.read_text()
        refs = re.findall(r"^ARG AFFENT_REF=(\w+)$", source, re.MULTILINE)
        assert refs == [AFFENT_REF], dockerfile
        assert "FROM golang:1.24.13" in source
        assert 'LABEL io.affine.affent.revision="$AFFENT_REF"' in source
        assert 'AFFENT_REF="$AFFENT_REF"' in source
        assert "go version -m" in source
        assert 'grep -Fq "vcs.revision=$AFFENT_REF"' in source
        assert "/usr/local/share/affent-build-info" in source
        assert "/usr/local/share/affent-sha256" in source
        assert "COPY affent/" not in source


def test_affent_protocol_copies_cannot_drift() -> None:
    copies = [path.read_text() for path in PROTOCOL_MODULES]
    assert len(set(copies)) == 1


def test_affent_protocol_version_transition_matrix() -> None:
    protocol = _load_protocol_module()
    gradable = protocol.AffentDisposition.GRADABLE
    model_failure = protocol.AffentDisposition.MODEL_FAILURE
    infra = protocol.AffentDisposition.INFRA_FAILURE

    assert protocol.classify_affent_run(
        0, turn_end_reason="completed"
    ) is gradable
    assert protocol.classify_affent_run(
        0, turn_end_reason="max_turns"
    ) is gradable
    assert protocol.classify_affent_run(
        2, turn_end_reason="max_turns"
    ) is gradable
    assert protocol.classify_affent_run(
        2, turn_end_reason="length"
    ) is gradable
    assert protocol.classify_affent_run(
        3,
        turn_end_reason="error",
        failure_kind="context_overflow",
    ) is model_failure

    for exit_code, reason in (
        (0, None),
        (2, None),
        (2, "completed"),
        (2, "error"),
        (0, "error"),
        (3, "max_turns"),
        (130, "cancelled"),
        (0, "future_reason"),
    ):
        assert protocol.classify_affent_run(
            exit_code, turn_end_reason=reason
        ) is infra
    assert protocol.classify_affent_run(
        3,
        turn_end_reason="error",
        failure_kind="llm_timeout",
    ) is infra


def test_affent_binary_must_match_build_manifest(tmp_path, monkeypatch) -> None:
    protocol = _load_protocol_module()
    binary = tmp_path / "affentctl"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    binary.write_bytes(b"pinned affent binary")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    (metadata / "affent-ref").write_text(AFFENT_REF + "\n")
    (metadata / "affent-sha256").write_text(digest + "\n")
    monkeypatch.setenv("AFFENT_REF", AFFENT_REF)

    identity = protocol.verify_affent_binary(
        str(binary),
        metadata_dir=str(metadata),
    )
    assert identity.revision == AFFENT_REF
    assert identity.sha256 == digest

    binary.write_bytes(b"different binary")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        protocol.verify_affent_binary(
            str(binary),
            metadata_dir=str(metadata),
        )


def test_affent_environment_revision_must_match_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = _load_protocol_module()
    binary = tmp_path / "affentctl"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    binary.write_bytes(b"pinned affent binary")
    (metadata / "affent-ref").write_text(AFFENT_REF + "\n")
    (metadata / "affent-sha256").write_text(
        hashlib.sha256(binary.read_bytes()).hexdigest() + "\n"
    )
    monkeypatch.setenv("AFFENT_REF", "0" * 40)

    with pytest.raises(RuntimeError, match="revision mismatch"):
        protocol.verify_affent_binary(
            str(binary),
            metadata_dir=str(metadata),
        )


def test_affent_is_not_sourced_from_environment_submodules() -> None:
    assert not (ROOT / ".gitmodules").exists()
    assert not (ROOT / "environments/SWE-INFINITE/affent").exists()
    assert not (ROOT / "environments/SWE-FRONTIER/affent").exists()
    for adapter in (
        ROOT / "environments/SWE-INFINITE/agents/affent.py",
        ROOT / "environments/SWE-FRONTIER/agents/affent.py",
    ):
        source = adapter.read_text()
        assert "~/affent-static" not in source
        assert 'return "affent-static"' not in source


def test_qqr_parser_retains_structured_terminal_state() -> None:
    script = textwrap.dedent("""
        import json
        import tempfile
        from pathlib import Path

        from affent_runner import _parse_trace_events

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            trace_path.write_text("\\n".join((
                json.dumps({
                    "type": "error",
                    "data": {
                        "message": "request exceeded context",
                        "failure_kind": "context_overflow",
                        "recoverable": False,
                    },
                }),
                json.dumps({
                    "type": "turn.end",
                    "data": {"reason": "error"},
                }),
            )))
            trace = _parse_trace_events(trace_path)
            assert trace["error_message"] == "request exceeded context"
            assert trace["failure_kind"] == "context_overflow"
            assert trace["turn_end_reason"] == "error"

            trace_path.write_text(json.dumps({
                "type": "error",
                "data": {
                    "message": "request will be retried",
                    "failure_kind": "context_overflow",
                    "recoverable": True,
                },
            }))
            assert _parse_trace_events(trace_path)["failure_kind"] is None
    """)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT / "environments/qqr",
        check=True,
    )


def test_swe_frontier_parser_retains_structured_terminal_state() -> None:
    script = textwrap.dedent("""
        import json

        from agents.affent import AffentAgent

        trace = "\\n".join((
            json.dumps({
                "type": "error",
                "data": {
                    "message": "request exceeded context",
                    "failure_kind": "context_overflow",
                    "recoverable": False,
                },
            }),
            json.dumps({
                "type": "message.done",
                "data": {"text": "final response"},
            }),
            json.dumps({
                "type": "turn.end",
                "data": {"reason": "error"},
            }),
        ))
        _, _, conversation, error, failure_kind, reason = AffentAgent._parse_trace(trace)
        assert {"role": "assistant", "content": "final response"} in conversation
        assert error == "request exceeded context"
        assert failure_kind == "context_overflow"
        assert reason == "error"

        recoverable = json.dumps({
            "type": "error",
            "data": {
                "message": "request will be retried",
                "failure_kind": "context_overflow",
                "recoverable": True,
            },
        })
        assert AffentAgent._parse_trace(recoverable)[4] is None
    """)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT / "environments/SWE-FRONTIER",
        check=True,
    )
