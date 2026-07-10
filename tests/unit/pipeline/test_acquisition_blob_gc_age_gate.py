"""Regression harness for blob publication versus durable raw references."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream
from polylogue.storage.blob_gc import MIN_AGE_S, BlobGCResult, run_blob_gc_report
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.frozen_clock import FrozenClock


@pytest.mark.asyncio
@pytest.mark.frozen_clock_modules("polylogue.pipeline.services.acquisition_records")
async def test_slow_following_source_cannot_age_uncommitted_blob_into_gc(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    frozen_clock: FrozenClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later streamed artifact must not expose an earlier blob to GC.

    ``iter_source_raw_stream`` drains a source batch before yielding its first
    record to persistence.  Advancing the controlled clock while the second
    provider-shaped artifact is streamed models an arbitrarily slow read
    without sleeping or allocating a large file.
    """
    del workspace_env
    archive_root = tmp_path / "archive"
    backend = SQLiteBackend(db_path=archive_root / "index.db")
    source_root = tmp_path / "chatgpt-export"
    source_root.mkdir()
    (source_root / "00-first.json").write_text('{"title":"first","mapping":{}}')
    (source_root / "01-slow.json").write_text('{"title":"slow","mapping":{}}')

    original_write = BlobStore.write_from_path
    first_blob_hash: str | None = None
    gc_report: BlobGCResult | None = None
    deleted_before_commit = False

    def measured_write(
        store: BlobStore,
        source: Path,
        *,
        heartbeat: object | None = None,
    ) -> tuple[str, int]:
        nonlocal first_blob_hash, gc_report, deleted_before_commit
        blob_hash, blob_size = original_write(store, source, heartbeat=heartbeat)  # type: ignore[arg-type]
        if source.name == "00-first.json":
            first_blob_hash = blob_hash
            os.utime(store.blob_path(blob_hash), (frozen_clock.time(), frozen_clock.time()))
        elif source.name == "01-slow.json":
            assert first_blob_hash is not None
            frozen_clock.advance(MIN_AGE_S + 1)
            gc_report = run_blob_gc_report(
                archive_root / "source.db",
                store.root,
                max_batch=10,
            )
            deleted_before_commit = not store.exists(first_blob_hash)
        return blob_hash, blob_size

    monkeypatch.setattr(BlobStore, "write_from_path", measured_write)
    try:
        records = [
            record
            async for record in iter_raw_record_stream(
                Source(name="chatgpt", path=source_root),
                blob_root=archive_root / "blob",
            )
        ]
        for record in records:
            await backend.save_raw_session(record)
    finally:
        await backend.close()

    assert len(records) == 2
    assert first_blob_hash is not None
    assert gc_report is not None
    assert gc_report.deleted_count == 0
    assert not deleted_before_commit
    assert BlobStore(archive_root / "blob").exists(first_blob_hash)
