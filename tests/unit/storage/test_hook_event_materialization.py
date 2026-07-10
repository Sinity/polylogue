"""Production source-tier materialization for hook-event artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.core.enums import ArtifactSupportStatus
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.queries.artifacts import save_artifact_observation
from polylogue.storage.sqlite.queries.raw_writes import save_raw_session


@pytest.mark.asyncio
async def test_hook_artifact_materializes_event_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    store = BlobStore(tmp_path / "blobs")
    payload = "\n".join(
        json.dumps(
            {
                "event_type": event,
                "session_id": "session-1",
                "timestamp": f"2026-07-10T10:00:0{index}Z",
                "provider": "claude-code",
                "payload": {"session_id": "session-1"},
            }
        )
        for index, event in enumerate(("SessionStart", "UserPromptSubmit"))
    ).encode("utf-8")
    blob_hash, blob_size = store.write_from_bytes(payload)
    monkeypatch.setattr("polylogue.storage.sqlite.queries.artifacts.get_blob_store", lambda: store)

    async with aiosqlite.connect(source_db) as conn:
        await save_raw_session(
            conn,
            RawSessionRecord(
                raw_id=blob_hash,
                blob_hash=blob_hash,
                source_name="claude-code",
                source_path="/hooks/claude-code-session-1.jsonl",
                source_index=0,
                blob_size=blob_size,
                acquired_at="2026-07-10T10:00:02+00:00",
            ),
            0,
        )
        observation = ArtifactObservationRecord(
            observation_id="obs-hook-session-1",
            raw_id=blob_hash,
            payload_provider="claude-code",
            source_name="claude-code",
            source_path="/hooks/claude-code-session-1.jsonl",
            source_index=0,
            wire_format="jsonl",
            artifact_kind="hook_event",
            classification_reason="hook event stream",
            parse_as_session=False,
            schema_eligible=False,
            support_status=ArtifactSupportStatus.RECOGNIZED_UNPARSED,
            first_observed_at="2026-07-10T10:00:02+00:00",
            last_observed_at="2026-07-10T10:00:02+00:00",
        )
        assert await save_artifact_observation(conn, observation, 0) is True
        assert await save_artifact_observation(conn, observation, 0) is False

    with sqlite3.connect(source_db) as conn:
        rows = conn.execute(
            "SELECT origin, session_native_id, event_type FROM raw_hook_events ORDER BY observed_at_ms"
        ).fetchall()
    assert rows == [
        ("claude-code-session", "session-1", "SessionStart"),
        ("claude-code-session", "session-1", "UserPromptSubmit"),
    ]
