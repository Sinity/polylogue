"""Regression tests: artifact inspection covers the whole retained stream.

Inspection starts from a 64 KB prefix to bound memory, then uses rolling stream
passes for whole-file loss accounting and positive session evidence. These
tests preserve both duties without weakening definitive sidecar exclusions.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.archive.raw_payload.decode import scan_jsonl_session_artifact
from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.core.json import JSONValue
from polylogue.schemas.observation import schema_cluster_id
from polylogue.storage.artifacts.inspection import (
    _INSPECTION_PREFIX_BYTES,
    inspect_raw_artifact,
)
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord


@pytest.fixture
def blob_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BlobStore]:
    root = tmp_path / "blobs"
    store = BlobStore(root)
    # inspect_raw_artifact resolves the blob store via get_blob_store(), which
    # reads paths.blob_store_root(). Point both at the test root.
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: root, raising=False)
    reset_blob_store()
    yield store
    reset_blob_store()


def _write_record(
    store: BlobStore,
    *,
    content: bytes,
    source_path: str,
    source_name: str = "claude-code",
    provider: Provider = Provider.CLAUDE_CODE,
) -> RawSessionRecord:
    raw_id, blob_size = store.write_from_bytes(content)
    return RawSessionRecord(
        raw_id=raw_id,
        source_name=source_name,
        source_path=source_path,
        payload_provider=provider,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )


def _valid_line(i: int) -> bytes:
    return (
        b'{"type":"user","uuid":"u%d","sessionId":"s","parentUuid":null,'
        b'"cwd":"/tmp","message":{"role":"user","content":"hi %d"}}\n' % (i, i)
    )


def test_malformed_line_past_prefix_is_surfaced(blob_store: BlobStore) -> None:
    """Malformed content past the 64 KB prefix flags the artifact (not silent)."""
    lines: list[bytes] = []
    size = 0
    i = 0
    while size <= _INSPECTION_PREFIX_BYTES * 2:
        line = _valid_line(i)
        lines.append(line)
        size += len(line)
        i += 1
    # Malformed line lives far past the prefix boundary.
    lines.append(b"{ this is not valid json at all\n")
    lines.append(_valid_line(i + 1))
    content = b"".join(lines)
    assert len(content) > _INSPECTION_PREFIX_BYTES

    record = _write_record(blob_store, content=content, source_path="agent-x/session.jsonl")
    observation = inspect_raw_artifact(record)

    assert observation.malformed_jsonl_lines >= 1
    # Valid records coexist with the bad line → partial loss, surfaced.
    assert observation.support_status in {
        ArtifactSupportStatus.PARTIAL_DECODE,
        ArtifactSupportStatus.DECODE_FAILED,
    }


def test_clean_large_jsonl_is_not_flagged(blob_store: BlobStore) -> None:
    """A clean JSONL file larger than the prefix is not falsely flagged."""
    lines = [_valid_line(i) for i in range(2000)]
    content = b"".join(lines)
    assert len(content) > _INSPECTION_PREFIX_BYTES

    record = _write_record(blob_store, content=content, source_path="agent-y/session.jsonl")
    observation = inspect_raw_artifact(record)

    assert observation.malformed_jsonl_lines == 0
    assert observation.support_status not in {
        ArtifactSupportStatus.PARTIAL_DECODE,
        ArtifactSupportStatus.DECODE_FAILED,
    }


def test_large_codex_stream_is_not_terminalized_from_session_meta_prefix(blob_store: BlobStore) -> None:
    session_meta = b'{"type":"session_meta","payload":{"id":"large-codex"}}\n'
    message = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
        b'"role":"user","content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    padding = (
        b'{"type":"response_item","payload":{"type":"token_count","padding":"'
        + (b"x" * (_INSPECTION_PREFIX_BYTES * 2))
        + b'"}}\n'
    )
    content = session_meta + message + padding
    assert len(content) > _INSPECTION_PREFIX_BYTES

    record = _write_record(
        blob_store,
        content=content,
        source_path="codex/large-session.jsonl",
        source_name="codex",
        provider=Provider.CODEX,
    )
    observation = inspect_raw_artifact(record)

    assert observation.parse_as_session is True
    assert observation.artifact_kind == "session_record_stream"
    assert observation.classification_reason == "parser-supported Codex session record stream"


def test_codex_stream_recovers_when_first_record_exceeds_inspection_prefix(blob_store: BlobStore) -> None:
    session_meta = (
        b'{"type":"session_meta","payload":{"id":"large-first-record","base_instructions":{"text":"'
        + (b"x" * (_INSPECTION_PREFIX_BYTES * 2))
        + b'"}}}\n'
    )
    message = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
        b'"role":"user","content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    assert session_meta.find(b"\n") > _INSPECTION_PREFIX_BYTES

    record = _write_record(
        blob_store,
        content=session_meta + message,
        source_path="codex/large-first-record.jsonl",
        source_name="codex",
        provider=Provider.CODEX,
    )
    observation = inspect_raw_artifact(record)

    assert observation.parse_as_session is True
    assert observation.artifact_kind == "session_record_stream"
    assert observation.wire_format == "jsonl"
    assert observation.decode_error is None
    assert observation.malformed_jsonl_lines == 0
    assert observation.support_status is ArtifactSupportStatus.SUPPORTED_PARSEABLE
    assert observation.resolved_package_version == "v1"
    assert observation.resolved_element_kind == "session_record_stream"
    expected_message: JSONValue = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "message-1",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    }
    assert observation.cohort_id == schema_cluster_id([expected_message], "session_record_stream")


def test_recovery_reader_discards_oversized_record_in_bounded_chunks() -> None:
    class BoundedReadlineStream(BytesIO):
        def readline(self, size: int | None = -1, /) -> bytes:
            assert isinstance(size, int)
            assert 0 < size <= _INSPECTION_PREFIX_BYTES + 1
            return super().readline(size)

    oversized = b'{"ignored":"' + (b"x" * (_INSPECTION_PREFIX_BYTES * 3)) + b'"}\n'
    message = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
        b'"role":"user","content":[{"type":"input_text","text":"hello"}]}}\n'
    )

    scan = scan_jsonl_session_artifact(
        BoundedReadlineStream(oversized + message),
        provider=Provider.CODEX,
        source_path="codex/bounded.jsonl",
        max_record_bytes=_INSPECTION_PREFIX_BYTES,
    )

    assert scan.artifact is not None
    assert scan.artifact.parse_as_session is True
    assert scan.oversized_records == 1
    assert len(scan.sample) == 1


@pytest.mark.parametrize(
    ("provider", "source_name"),
    [
        pytest.param(Provider.CLAUDE_CODE, "claude-code", id="claude-code"),
        pytest.param(Provider.CODEX, "codex", id="codex"),
    ],
)
def test_single_oversized_provider_record_under_weak_path_remains_parse_candidate(
    blob_store: BlobStore,
    provider: Provider,
    source_name: str,
) -> None:
    if provider is Provider.CLAUDE_CODE:
        content = (
            b'{"type":"user","uuid":"message-1","sessionId":"oversized-session",'
            b'"parentUuid":null,"message":{"role":"user","content":"'
            + (b"x" * (_INSPECTION_PREFIX_BYTES * 2))
            + b'"}}\n'
        )
    else:
        content = (
            b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
            b'"role":"user","content":[{"type":"input_text","text":"'
            + (b"x" * (_INSPECTION_PREFIX_BYTES * 2))
            + b'"}]}}\n'
        )
    assert content.find(b"\n") > _INSPECTION_PREFIX_BYTES
    record = _write_record(
        blob_store,
        content=content,
        source_path=f"{source_name}/analysis/re-homed-session.jsonl",
        source_name=source_name,
        provider=provider,
    )

    observation = inspect_raw_artifact(record)

    assert observation.parse_as_session is True
    assert observation.schema_eligible is False
    assert observation.artifact_kind == "session_record_stream"
    assert observation.support_status is ArtifactSupportStatus.RECOGNIZED_UNPARSED
    assert observation.malformed_jsonl_lines == 0
    assert observation.decode_error is None


def test_recovered_stream_retains_subagent_artifact_kind(blob_store: BlobStore) -> None:
    oversized = b'{"ignored":"' + (b"x" * (_INSPECTION_PREFIX_BYTES * 2)) + b'"}\n'
    message = (
        b'{"parentUuid":null,"type":"user","sessionId":"agent-session",'
        b'"message":{"role":"user","content":"hello"},'
        b'"uuid":"user-1","timestamp":"2026-01-01T00:00:00Z"}\n'
    )
    record = _write_record(
        blob_store,
        content=oversized + message,
        source_path="projects/project/subagents/agent-abcd.jsonl",
    )

    observation = inspect_raw_artifact(record)

    assert observation.parse_as_session is True
    assert observation.artifact_kind == "agent_transcript"
    assert observation.cohort_id == schema_cluster_id(
        [
            {
                "parentUuid": None,
                "type": "user",
                "sessionId": "agent-session",
                "message": {"role": "user", "content": "hello"},
                "uuid": "user-1",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ],
        "agent_transcript",
    )


def test_rolling_scan_preserves_tool_result_sidecar_exclusion(blob_store: BlobStore) -> None:
    content = (
        b'{"parentUuid":null,"type":"user","sessionId":"embedded",'
        b'"message":{"role":"user","content":"copied transcript"},'
        b'"uuid":"user-1","timestamp":"2026-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"user-1","type":"assistant","sessionId":"embedded",'
        b'"message":{"role":"assistant","content":[{"type":"text","text":"copied reply"}]},'
        b'"uuid":"assistant-1","timestamp":"2026-01-01T00:00:01Z"}\n'
    )
    record = _write_record(
        blob_store,
        content=content,
        source_path="projects/project/session/tool-results/copied-transcript.jsonl",
    )

    observation = inspect_raw_artifact(record)

    assert observation.parse_as_session is False
    assert observation.schema_eligible is False
    assert observation.artifact_kind == "tool_result_sidecar"
