"""Regression tests: artifact DECODE_FAILED/PARTIAL_DECODE covers the whole file (#1745).

The artifact support status is derived from raw inspection. Inspection reads
only a 64 KB prefix to bound memory, so malformed JSONL content *past* the
prefix used to be invisible and the artifact was never flagged. These tests
assert that loss past the prefix is surfaced via a full-scan fallback.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.enums import ArtifactSupportStatus, Provider
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
) -> RawSessionRecord:
    raw_id, blob_size = store.write_from_bytes(content)
    return RawSessionRecord(
        raw_id=raw_id,
        source_name=source_name,
        source_path=source_path,
        payload_provider=Provider.CLAUDE_CODE,
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
