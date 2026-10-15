"""Admission shares complete records while keeping every grammar's owner."""

from __future__ import annotations

import errno
import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import IO

import pytest

from polylogue.archive.raw_payload.decode import scan_jsonl_session_artifact
from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.enums import Provider
from polylogue.sources import detection_projection
from polylogue.sources.dispatch import detect_provider_from_raw_stream_evidence
from polylogue.sources.live.batch_support import _jsonl_provider_and_non_session_proof
from polylogue.sources.origin_specs import path_declaration_refuses_session

_CLAUDE = {"type": "user", "sessionId": "neutral-session", "message": {"role": "user", "content": "hello"}}
_CODEX = {"type": "session_meta", "payload": {"id": "neutral-codex", "cwd": "/neutral"}}


def _records(*records: object) -> bytes:
    return b"\n".join(json.dumps(record).encode() for record in records) + b"\n"


def _separate_owner_result(path: Path) -> tuple[Provider, bool, str | None]:
    with path.open("rb") as handle:
        provider, evidence = detect_provider_from_raw_stream_evidence(
            handle, path.name, Provider.CLAUDE_CODE, truncated_tail_ok=True
        )
    failure = evidence if evidence.startswith("stream decode error") else None
    if path_declaration_refuses_session(provider, path):
        return provider, True, failure
    scan = scan_jsonl_session_artifact(path, provider=provider, source_path=path)
    return provider, scan.proved_non_session, failure


@pytest.mark.parametrize(
    "payload",
    [
        _records(_CLAUDE, _CLAUDE),
        _records([_CLAUDE, _CLAUDE], _CLAUDE),
        b'{"type":"unrecognized","type":"user","sessionId":"neutral","message":{"role":"user","content":"hi"}}\n'
        + _records(_CLAUDE),
        _records(_CLAUDE, _CODEX),
        _records(_CLAUDE, {"pad": "x" * (96 * 1024), **_CODEX}),
        _records(_CLAUDE) + b'{"broken":\n' + _records(_CLAUDE),
        _records(_CLAUDE, _CLAUDE) + b'{"unfinished":',
        json.dumps([_CLAUDE, _CLAUDE], indent=2).encode(),
    ],
)
def test_shared_admission_preserves_separate_complete_grammar_owners(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_bytes(payload)
    expected = _separate_owner_result(path)
    assert _jsonl_provider_and_non_session_proof(path, Provider.CLAUDE_CODE) == expected
    assert path.read_bytes() == payload


def test_complete_admission_decodes_records_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_bytes(_records(_CLAUDE, _CLAUDE, _CLAUDE))
    decoded = projected = 0
    original = detection_projection.iter_decoded_jsonl_records
    original_projected = detection_projection.iter_projected_jsonl_records

    def observe(
        handle: IO[bytes],
        *,
        check_stop: Callable[[], None] | None = None,
        on_decode_failure: Callable[[Exception], None] | None = None,
    ) -> Generator[object, None, None]:
        nonlocal decoded
        decoded += 1
        yield from original(handle, check_stop=check_stop, on_decode_failure=on_decode_failure)

    def observe_projected(
        handle: IO[bytes],
        rule: detection_projection.DetectorProjection,
        *,
        check_stop: Callable[[], None] | None = None,
        on_decode_failure: Callable[[Exception], None] | None = None,
    ) -> Generator[object, None, None]:
        nonlocal projected
        projected += 1
        yield from original_projected(handle, rule, check_stop=check_stop, on_decode_failure=on_decode_failure)

    monkeypatch.setattr(detection_projection, "iter_decoded_jsonl_records", observe)
    monkeypatch.setattr(detection_projection, "iter_projected_jsonl_records", observe_projected)
    assert _jsonl_provider_and_non_session_proof(path, Provider.CLAUDE_CODE) == (Provider.CLAUDE_CODE, False, None)
    assert (decoded, projected) == (1, 0)


def test_shared_admission_refuses_descriptor_replacement(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    payload = _records(_CLAUDE, _CLAUDE)
    path.write_bytes(payload)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(payload)
    changed = False

    def checkpoint() -> None:
        nonlocal changed
        if not changed:
            changed = True
            replacement.replace(path)

    with pytest.raises(OSError) as refused:
        _jsonl_provider_and_non_session_proof(path, Provider.CLAUDE_CODE, checkpoint=checkpoint)
    assert refused.value.errno == errno.ESTALE
    assert changed
    assert path.read_bytes() == payload


def test_shared_admission_preserves_callback_cancellation(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_bytes(_records(_CLAUDE, _CLAUDE))
    cancellation = DaemonOperationCancelled("neutral cancellation")
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        raise cancellation

    with pytest.raises(DaemonOperationCancelled) as refused:
        _jsonl_provider_and_non_session_proof(path, Provider.CLAUDE_CODE, checkpoint=checkpoint)
    assert refused.value is cancellation
    assert calls == 1


def test_baseline_admission_keeps_original_byte_revision(tmp_path: Path) -> None:
    import hashlib

    from polylogue.sources.live.production_baseline import _prepare_file_decision

    path = tmp_path / "transcript.jsonl"
    payload = _records(_CLAUDE, _CLAUDE) + b" \n\t"
    path.write_bytes(payload)
    observed = path.stat()
    decision = _prepare_file_decision(
        (Provider.CLAUDE_CODE.value, path, "accepted", "neutral"),
        expected_observation=(
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        ),
    )
    assert decision.disposition == "accepted"
    assert decision.revision == hashlib.sha256(payload).hexdigest()
    assert decision.material_bytes == len(payload)


def test_record_classifier_state_belongs_to_one_input() -> None:
    from polylogue.sources.origin_specs import detector_registry

    registry = detector_registry()
    first = registry.record_stream_classifier()
    first.observe(_CLAUDE)
    assert first.finish()[0] is Provider.CLAUDE_CODE
    second = registry.record_stream_classifier()
    assert second is not first
    assert second.finish() == (None, None)
    second.observe({"unrelated": True})
    assert second.finish() == (None, None)
    assert first.finish()[0] is Provider.CLAUDE_CODE


def test_admission_open_failure_is_a_source_fault_not_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live.production_baseline import _prepare_file_decision

    path = tmp_path / "transcript.jsonl"
    path.write_bytes(_records(_CLAUDE, _CLAUDE))

    def unavailable(_path: Path, *_args: object, **_kwargs: object) -> IO[bytes]:
        raise OSError(errno.EIO, "neutral source unavailable")

    monkeypatch.setattr(Path, "open", unavailable)
    decision = _prepare_file_decision((Provider.CLAUDE_CODE.value, path, "accepted", "neutral"))
    assert decision.disposition == "fault"
    assert decision.reason.startswith("revision_io_unavailable:")
    assert decision.revision is None
