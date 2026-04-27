"""Storage contracts for persisted publication manifests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import PublicationRecord


def test_record_and_fetch_latest_publication_roundtrip(tmp_path: Path) -> None:
    """Publication records persist and round-trip through the backend query API."""
    db_path = tmp_path / "publications.db"
    backend = SQLiteBackend(db_path=db_path)
    record = PublicationRecord(
        publication_id="site-001",
        publication_kind="site",
        generated_at="2026-03-22T12:00:00+00:00",
        output_dir=str(tmp_path / "site"),
        duration_ms=1234,
        manifest={
            "publication_id": "site-001",
            "publication_kind": "site",
            "outputs": {"total_index_pages": 3},
        },
    )

    try:
        repo = ConversationRepository(backend=backend)
        asyncio.run(repo.record_publication(record))
        loaded = asyncio.run(repo.get_latest_publication("site"))
    finally:
        asyncio.run(backend.close())

    assert loaded is not None
    assert loaded.publication_id == record.publication_id
    assert loaded.duration_ms == 1234
    outputs = loaded.manifest.get("outputs")
    assert isinstance(outputs, dict)
    assert outputs["total_index_pages"] == 3
