from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pytest

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.core.timestamps import parse_timestamp_pair
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers import codex
from polylogue.sources.parsers.base_models import AdmissionUnit
from polylogue.sources.prepared_codex_checkpoints import (
    CodexCheckpointArtifactOptions,
    CodexCheckpointDisposition,
    _finalize_codex_prefix,
    _plain_text_header,
    _plain_text_message,
    _prefix_accounting,
    _read_head_and_prove,
    prepare_codex_prefix_checkpoints,
)
from polylogue.sources.prepared_jsonl import PreparedJsonl


def _records(count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": "prefix-session", "timestamp": "2026-06-01T00:00:00Z"}}
    ]
    for index in range(count):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"message-{index}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": [{"type": "input_text", "text": f"content-{index}"}],
                },
            }
        )
    return rows


def test_every_checkpoint_prefix_matches_ordinary_codex_parse(tmp_path: Path) -> None:
    records = _records(8)
    head = codex.parse_stream(records, "fallback")

    assert head.source_name is Provider.CODEX
    for record_count in range(1, len(records) + 1):
        ordinary = codex.parse_stream(records[:record_count], "fallback")
        message_count = record_count - 1
        updated_pair = parse_timestamp_pair(head.created_at)
        for message in head.messages[:message_count]:
            updated_pair = codex._newer_timestamp_pair(updated_pair, parse_timestamp_pair(message.timestamp))
        checkpoint = _finalize_codex_prefix(
            head,
            head.messages,
            message_count,
            _prefix_accounting(message_count),
            updated_pair[1] if updated_pair is not None else None,
        )

        checkpoint_value = checkpoint.model_copy(update={"messages": list(checkpoint.messages)})
        assert checkpoint_value.model_dump(mode="json", exclude={"unit_accounting"}) == ordinary.model_dump(
            mode="json", exclude={"unit_accounting"}
        )
        assert checkpoint.unit_accounting is not None
        assert ordinary.unit_accounting is not None
        assert checkpoint.unit_accounting.model_dump(mode="json") == ordinary.unit_accounting.model_dump(mode="json")
        assert checkpoint.active_leaf_message_provider_id == ordinary.active_leaf_message_provider_id

        if record_count == 5:
            artifact_dir = tmp_path / "private-artifact"
            artifact_dir.mkdir()
            checkpoint.content_hash = session_content_hash(checkpoint)
            artifact = PreparedJsonl.from_sessions(
                (checkpoint,),
                blob_hash="a" * 64,
                artifact_directory=artifact_dir,
                publication_publisher=None,
            )
            try:
                retained = list(artifact.iter_sessions())
                assert len(retained) == 1
                retained_value = retained[0].model_copy(
                    update={
                        "messages": list(retained[0].messages),
                        "session_events": list(retained[0].session_events),
                    }
                )
                assert retained_value.model_dump(mode="json", exclude={"unit_accounting"}) == ordinary.model_dump(
                    mode="json", exclude={"unit_accounting"}
                )
            finally:
                artifact.discard()


def test_header_only_checkpoint_has_zero_messages_and_no_part_denominator() -> None:
    session = codex.parse_stream(_records(0), "fallback")
    accounting = _prefix_accounting(0)

    assert session.messages == []
    assert session.unit_accounting is not None
    assert accounting.model_dump(mode="json") == session.unit_accounting.model_dump(mode="json")
    assert AdmissionUnit.PART not in accounting.expected
    assert AdmissionUnit.BLOCK not in accounting.expected


def test_checkpoint_grammar_falls_back_for_future_sensitive_codex_shapes() -> None:
    assert _plain_text_header(_records(0)[0]) == "prefix-session"
    message = _records(1)[1]
    assert _plain_text_message(message) == ("message-0", "user", "content-0")
    payload = message["payload"]
    assert isinstance(payload, dict)

    event_msg = {"type": "event_msg", "payload": {"type": "exec_command_end", "id": "event"}}
    code_mode_call = {
        "type": "response_item",
        "payload": {"type": "function_call", "id": "call", "name": "exec", "arguments": "{}"},
    }
    compacted = {"type": "compacted", "payload": {"message": "summary"}}
    turn_context = {"type": "turn_context", "payload": {"model": "x"}}
    repeated_header = _records(0)[0]

    for record in (event_msg, code_mode_call, compacted, turn_context, repeated_header):
        assert _plain_text_message(record) is None

    assert _plain_text_message({**message, "extra": "unknown"}) is None
    assert (
        _plain_text_message(
            {
                "type": "response_item",
                "payload": {**payload, "content": [{"type": "output_text", "text": "x"}]},
            }
        )
        is None
    )
    assert (
        _plain_text_message(
            {"type": "response_item", "payload": {**payload, "content": [{"type": "image", "url": "x"}]}}
        )
        is None
    )


class _SourceRead:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = {f"raw-{index}": payload for index, payload in enumerate(payloads)}

    def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
        payload = self.payloads[raw_id]
        return (
            Provider.CODEX,
            hashlib.sha256(payload).hexdigest(),
            "same/path.jsonl",
            RawRevisionKind.FULL,
            len(payload),
        )

    def raw_profile_identity(self, _raw_id: str) -> str:
        return "captured-profile"

    def raw_revision_file_mtime(self, raw_id: str) -> str:
        return f"2026-06-0{int(raw_id.removeprefix('raw-')) + 1}T01:00:00Z"

    @contextmanager
    def open_raw_revision_material(self, raw_id: str) -> Iterator[tuple[Provider, BinaryIO, str, RawRevisionKind]]:
        yield Provider.CODEX, BytesIO(self.payloads[raw_id]), "same/path.jsonl", RawRevisionKind.FULL


def test_source_read_proof_checks_hash_prefix_and_complete_record_boundaries(tmp_path: Path) -> None:
    records = _records(3)
    captures = []
    for count in range(1, 5):
        text = "\n".join(json.dumps(row, separators=(",", ":")) for row in records[:count]) + "\n"
        captures.append(text.encode())
    source_read = _SourceRead(captures)
    head_blob: BinaryIO = tempfile.TemporaryFile(mode="w+b")
    head_blob, counts, hashes, header, message_count, verdicts = _read_head_and_prove(
        source_read,
        tuple(source_read.payloads),
        head_blob,
        validation_mode=ValidationMode.ADVISORY,
        validation_directory=tmp_path,
        schema_registry=None,
    )
    try:
        assert counts == [1, 2, 3, 4]
        assert header == "prefix-session"
        assert message_count == 3
        assert hashes[-1] == hashlib.sha256(captures[-1]).hexdigest()
        assert set(verdicts) == set(tuple(source_read.payloads)[2:-1])
    finally:
        head_blob.close()

    replaced = list(captures)
    replaced[2] = captures[2].replace(b"content-1", b"changed-1")
    replaced_blob = tempfile.TemporaryFile(mode="w+b")
    try:
        with pytest.raises(ValueError, match="exact byte prefixes"):
            _read_head_and_prove(
                _SourceRead(replaced),
                tuple(source_read.payloads),
                replaced_blob,
                validation_mode=ValidationMode.ADVISORY,
                validation_directory=tmp_path,
                schema_registry=None,
            )
    finally:
        replaced_blob.close()

    incomplete = list(captures)
    incomplete[1] = captures[1][:-1]
    incomplete_blob = tempfile.TemporaryFile(mode="w+b")
    try:
        with pytest.raises(ValueError, match="ends inside a JSONL record"):
            _read_head_and_prove(
                _SourceRead(incomplete),
                tuple(source_read.payloads),
                incomplete_blob,
                validation_mode=ValidationMode.ADVISORY,
                validation_directory=tmp_path,
                schema_registry=None,
            )
    finally:
        incomplete_blob.close()


def test_each_prefix_keeps_its_own_fallback_timestamp_provenance() -> None:
    records = _records(3)
    payload = records[0]["payload"]
    assert isinstance(payload, dict)
    payload.pop("timestamp")
    head = codex.parse_stream(records, "fallback")
    checkpoint = _finalize_codex_prefix(head, head.messages, 2, _prefix_accounting(2), head.created_at)

    observed: list[tuple[str | None, str | None, str, str, list[str | None]]] = []
    for fallback in ("2026-06-01T01:00:00Z", "2026-06-02T01:00:00Z"):
        ordinary = normalize_session_timestamps(
            codex.parse_stream(records[:3], "fallback"), fallback_timestamp=fallback
        )
        prepared = normalize_session_timestamps(checkpoint, fallback_timestamp=fallback)
        prepared_value = prepared.model_copy(update={"messages": list(prepared.messages)})
        assert prepared_value.model_dump(mode="json", exclude={"unit_accounting"}) == ordinary.model_dump(
            mode="json", exclude={"unit_accounting"}
        )
        assert prepared.created_at_provenance == "fallback"
        assert prepared.updated_at_provenance == "fallback"
        observed.append(
            (
                prepared.created_at,
                prepared.updated_at,
                prepared.created_at_provenance,
                prepared.updated_at_provenance,
                [message.timestamp for message in prepared.messages],
            )
        )
    assert observed[0][0] != observed[1][0]
    assert observed[0][1] != observed[1][1]
    assert observed[0][2:4] == observed[1][2:4] == ("fallback", "fallback")
    assert observed[0][4] == observed[1][4] == [None, None]


def test_checkpoint_preparation_seals_exact_per_raw_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = _records(3)
    payload = records[0]["payload"]
    assert isinstance(payload, dict)
    payload.pop("timestamp")
    payloads: list[bytes] = []
    for count in range(1, 5):
        text = "\n".join(json.dumps(row, separators=(",", ":")) for row in records[:count]) + "\n"
        payloads.append(text.encode())
    source_read = _SourceRead(payloads)
    raw_ids = tuple(source_read.payloads)
    head = codex.parse_stream(records, "fallback")
    head_hash = hashlib.sha256(payloads[-1]).hexdigest()
    head_dir = tmp_path / "head-artifact"
    head_dir.mkdir()
    head_artifact = PreparedJsonl.from_sessions(
        (head,),
        blob_hash=head_hash,
        artifact_directory=head_dir,
        publication_publisher=None,
        resolved_provider=Provider.CODEX,
        captured_profile_key="captured-profile",
    )
    head_artifact = replace(
        head_artifact,
        parser_stage_artifact=replace(head_artifact, attempt_directory=None),
    )
    interior_dir = tmp_path / "interior-artifacts"
    interior_dir.mkdir()

    with monkeypatch.context() as context:
        context.setattr(
            PreparedJsonl,
            "iter_sessions",
            lambda _self: (_ for _ in ()).throw(AssertionError("checkpoint head must use its paged session sequence")),
        )
        preparation = prepare_codex_prefix_checkpoints(
            source_read,
            raw_ids,
            head_artifact=head_artifact,
            artifact_directory=interior_dir,
            validation_mode=ValidationMode.ADVISORY,
            publication_publisher=None,
            publication_source_read=None,
            prepare_sessions=lambda _raw_id, sessions: sessions,
            artifact_options=lambda raw_id, _record_count: CodexCheckpointArtifactOptions(
                captured_profile_key="captured-profile",
                source_path="same/path.jsonl",
                fallback_timestamp=source_read.raw_revision_file_mtime(raw_id),
            ),
        )
    assert preparation.disposition is CodexCheckpointDisposition.READY
    artifacts = list(preparation.iter_artifacts())
    assert len(artifacts) == 1
    raw_id, artifact = artifacts[0]
    try:
        expected_hash = hashlib.sha256(payloads[2]).hexdigest()
        assert raw_id == raw_ids[2]
        assert artifact.blob_hash == expected_hash
        assert artifact.captured_profile_key == "captured-profile"
        assert artifact.validation_verdict is not None
        assert artifact.validation_verdict.raw_id == raw_id
        assert artifact.validation_verdict.revision_sha256 == expected_hash
        assert artifact.validation_verdict.evidence_id == raw_id
        assert artifact.validation_verdict.mode is ValidationMode.ADVISORY
        sessions = list(artifact.iter_sessions())
        assert len(sessions) == 1
        expected = normalize_session_timestamps(
            codex.parse_stream(records[:3], "fallback"),
            fallback_timestamp=source_read.raw_revision_file_mtime(raw_id),
        )
        actual = sessions[0].model_copy(
            update={"messages": list(sessions[0].messages), "session_events": list(sessions[0].session_events)}
        )
        assert actual.model_dump(mode="json", exclude={"unit_accounting"}) == expected.model_dump(
            mode="json", exclude={"unit_accounting"}
        )
        assert sessions[0].created_at_provenance == "fallback"
        assert sessions[0].updated_at_provenance == "fallback"
        assert all(message.timestamp is None for message in sessions[0].messages)
    finally:
        artifact.discard()
        preparation.close()
        head_artifact.discard()

    normalized_head_dir = tmp_path / "normalized-head-artifact"
    normalized_head_dir.mkdir()
    normalized_head = normalize_session_timestamps(
        head, fallback_timestamp=source_read.raw_revision_file_mtime(raw_ids[-1])
    )
    normalized_head_artifact = PreparedJsonl.from_sessions(
        (normalized_head,),
        blob_hash=head_hash,
        artifact_directory=normalized_head_dir,
        publication_publisher=None,
        resolved_provider=Provider.CODEX,
        captured_profile_key="captured-profile",
    )
    normalized_head_artifact = replace(
        normalized_head_artifact,
        parser_stage_artifact=replace(normalized_head_artifact, attempt_directory=None),
    )
    try:
        rejected = prepare_codex_prefix_checkpoints(
            source_read,
            raw_ids,
            head_artifact=normalized_head_artifact,
            artifact_directory=interior_dir,
            validation_mode=ValidationMode.ADVISORY,
            publication_publisher=None,
            publication_source_read=None,
            prepare_sessions=lambda _raw_id, sessions: sessions,
            artifact_options=lambda raw_id, _record_count: CodexCheckpointArtifactOptions(
                captured_profile_key="captured-profile",
                source_path="same/path.jsonl",
                fallback_timestamp=source_read.raw_revision_file_mtime(raw_id),
            ),
        )
        assert rejected.disposition is CodexCheckpointDisposition.ORDINARY_FALLBACK
        assert list(rejected.iter_artifacts()) == []
        rejected.close()
    finally:
        normalized_head_artifact.discard()
