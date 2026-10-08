"""Retained JSONL refusal and full-blob malformed-line inspection agree."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.retained_replay import replay_retained_components


def _valid_line(index: int) -> bytes:
    return (
        b'{"type":"user","uuid":"u%d","sessionId":"s1","parentUuid":null,'
        b'"cwd":"/tmp","message":{"role":"user","content":"hi %d"}}\n' % (index, index)
    )


def test_advisory_retained_jsonl_refusal_keeps_bytes_and_inspection_counts_full_blob(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    lines = [_valid_line(index) for index in range(120)]
    lines.insert(80, b"{ not valid json line A\n")
    lines.insert(101, b"{ not valid json line B\n")
    payload = b"".join(lines)

    def acquire() -> str:
        bootstrap_archive_root(archive)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(archive, read_only=False) as store:
            return store.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=payload,
                source_path="agent-z/session.jsonl",
                canonical_source_path="agent-z/session.jsonl",
                acquired_at_ms=1,
            )

    raw_id = asyncio.run(run_archive_fixture_write(archive, acquire))
    with pytest.raises(RetainedRawDecodeRefusalError, match="malformed JSONL"):
        replay_retained_components(archive, selected_raw_ids=(raw_id,), validation_mode=ValidationMode.ADVISORY)

    with sqlite3.connect(archive / "source.db") as conn:
        row = conn.execute(
            "SELECT blob_hash, blob_size, source_path, canonical_source_path, acquired_at_ms "
            "FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        parse_error = conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    assert row is not None
    raw_record = RawSessionRecord(
        raw_id=raw_id,
        source_name=Provider.CLAUDE_CODE.value,
        payload_provider=Provider.CLAUDE_CODE,
        source_path=row[2],
        canonical_source_path=row[3],
        source_index=0,
        blob_size=int(row[1]),
        blob_hash=bytes(row[0]).hex(),
        acquired_at="2026-01-01T00:00:00+00:00",
    )
    observation = inspect_raw_artifact(raw_record, blob_store=BlobStore(archive / "blob"))
    assert observation.malformed_jsonl_lines == 2
    assert parse_error is not None and "malformed jsonl" in str(parse_error[0]).lower()
    assert BlobStore(archive / "blob").read_all(str(raw_record.blob_hash)) == payload
    with sqlite3.connect(archive / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
