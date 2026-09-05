"""Production acquisition must enforce the durable excision snapshot."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import AssertionKind
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


@pytest.mark.asyncio
async def test_acquire_sources_passes_snapshot_to_raw_admission(tmp_path: Path) -> None:
    """The production acquisition entrypoint refuses an excised source payload."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_path = tmp_path / "source.json"
    payload = b'{"title":"removed","mapping":{}}'
    source_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.execute(
            """INSERT INTO assertions(
                assertion_id, target_ref, kind, value_json, status,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "excision-production-route",
                "source:source.json",
                AssertionKind.EXCISION_RECORD.value,
                json.dumps({"removed_blob_hashes": [digest]}),
                "active",
                1,
                1,
            ),
        )
        conn.commit()

    backend = SQLiteBackend(db_path=archive_root / "index.db")
    try:
        result = await AcquisitionService(backend).acquire_sources([Source(name="chatgpt", path=source_path)])
    finally:
        await backend.close()

    assert result.acquired == 0
    assert result.errors == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
