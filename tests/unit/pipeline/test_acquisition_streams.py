from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.pipeline.services import acquisition_streams
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.retained_acquisition import SourceInputRecord
from tests.infra.frozen_clock import FrozenClock


async def test_iter_raw_record_stream_refuses_an_invalid_raw_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid raw record fails the stream instead of being skipped.

    Acquisition fails closed: a record ``make_raw_record`` rejects is surfaced
    to the caller, which records the failure against its source, rather than
    being dropped with a log line. Anti-vacuity: restore the former
    log-and-skip handler and the stream yields nothing without raising.
    """

    async def _raw_stream(*args: object, **kwargs: object) -> AsyncIterator[SourceInputRecord]:
        del args, kwargs
        yield SourceInputRecord(
            '["physical-file-v1",0]',
            RawSessionData(
                raw_bytes=b'{"id":"broken"}',
                source_path=str(tmp_path / "broken.json"),
                provider_hint=Provider.CHATGPT,
            ),
        )

    def _raise(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("boom")

    monkeypatch.setattr(acquisition_streams, "iter_source_raw_stream", _raw_stream)
    monkeypatch.setattr(acquisition_streams, "make_raw_record", _raise)

    with pytest.raises(ValueError, match="boom"):
        _ = [item async for item in acquisition_streams.iter_raw_record_stream(Source(name="chatgpt", path=tmp_path))]


@pytest.mark.asyncio
async def test_iter_raw_record_stream_forwards_source_status_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.pipeline.services import acquisition as acquisition_module

    def _iter_source_acquisition_records(*args: object, **kwargs: object) -> Iterator[SourceInputRecord]:
        del args
        status_callback = kwargs["status_callback"]
        assert callable(status_callback)
        status_callback("Scanning [chatgpt] reading export.json")
        yield SourceInputRecord(
            '["physical-file-v1",0]',
            RawSessionData(
                raw_bytes=b'{"mapping": {}, "id": "ok"}',
                source_path=str(tmp_path / "export.json"),
                provider_hint=Provider.CHATGPT,
            ),
        )

    progress_events: list[tuple[int, str | None]] = []

    def _record_progress(amount: int, desc: str | None = None) -> None:
        progress_events.append((amount, desc))

    monkeypatch.setattr(acquisition_module, "iter_source_acquisition_records", _iter_source_acquisition_records)

    items = [
        item
        async for item in acquisition_streams.iter_raw_record_stream(
            Source(name="chatgpt", path=tmp_path),
            progress_callback=_record_progress,
        )
    ]
    await asyncio.sleep(0)

    assert len(items) == 1
    assert progress_events == [(0, "Scanning [chatgpt] reading export.json")]


@pytest.mark.asyncio
async def test_iter_raw_record_stream_forwards_drive_progress_and_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.sources.drive as drive_module

    observations: list[JSONDocument] = []
    progress_events: list[tuple[int, str | None]] = []

    def _iter_drive_raw_data(*args: object, **kwargs: object) -> Iterator[RawSessionData]:
        del args
        observation_callback = kwargs["observation_callback"]
        status_callback = kwargs["status_callback"]
        assert callable(observation_callback)
        assert callable(status_callback)
        observation_callback({"phase": "drive-test", "source_path": "drive.json"})
        status_callback("Scanning [gemini] reading drive.json")
        yield RawSessionData(
            raw_bytes=b'{"id":"drive"}',
            source_path=str(tmp_path / "drive.json"),
            provider_hint=Provider.GEMINI,
        )

    def _record_progress(amount: int, desc: str | None = None) -> None:
        progress_events.append((amount, desc))

    monkeypatch.setattr(drive_module, "iter_drive_raw_data", _iter_drive_raw_data)

    items = [
        item
        async for item in acquisition_streams.iter_raw_record_stream(
            Source(name="gemini", path=tmp_path, folder="Google AI Studio"),
            observation_callback=observations.append,
            progress_callback=_record_progress,
        )
    ]
    await asyncio.sleep(0)

    assert len(items) == 1
    assert observations == [{"phase": "drive-test", "source_path": "drive.json"}]
    assert progress_events == [(0, "Scanning [gemini] reading drive.json")]


@pytest.mark.asyncio
@pytest.mark.frozen_clock_modules("polylogue.pipeline.services.acquisition_streams")
async def test_ordinary_zip_reobservation_proves_exact_membership_without_clock_order(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    """A returned byte revision restores its exact group and renews custody."""
    import json
    import sqlite3
    import zipfile

    from polylogue.archive.revision_authority import raw_receipt_order_sql
    from polylogue.pipeline.services.acquisition import AcquisitionService
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from tests.infra.archive_templates import bootstrap_archive_root

    archive_root = tmp_path / "archive"
    await asyncio.to_thread(bootstrap_archive_root, archive_root)
    source_root = tmp_path / "input"
    source_root.mkdir()
    bundle = source_root / "duplicates.zip"

    def write_revision(sibling_text: str) -> None:
        with zipfile.ZipFile(bundle, "w") as container:
            for index, text in enumerate(("unchanged", sibling_text)):
                info = zipfile.ZipInfo("sessions/duplicate.jsonl", date_time=(2000, 1, 1, 0, 0, 0))
                payload = (
                    json.dumps({"type": "session_meta", "payload": {"id": f"session-{index}"}})
                    + "\n"
                    + json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            },
                        }
                    )
                    + "\n"
                ).encode()
                if index:
                    with pytest.warns(UserWarning, match="Duplicate name"):
                        container.writestr(info, payload)
                else:
                    container.writestr(info, payload)
            container.writestr(zipfile.ZipInfo("readme.txt", date_time=(2000, 1, 1, 0, 0, 0)), b"unselected")

    def membership() -> tuple[tuple[str, str, int, int], ...]:
        with sqlite3.connect(archive_root / "source.db") as conn:
            return tuple(
                conn.execute(
                    "SELECT m.source_generation_id, m.raw_id, c.entry_ordinal, c.split_index "
                    "FROM source_item_raw_members m JOIN raw_container_coordinates c ON c.raw_id=m.raw_id "
                    "ORDER BY m.source_generation_id, c.entry_ordinal, c.split_index"
                )
            )

    from polylogue.daemon.drive_catchup import DriveCatchupExecution
    from tests.infra.live_ingest import prepared_live_convergence_owner

    backend = SQLiteBackend(db_path=archive_root / "source.db")
    try:
        # Acquisition publishes through the daemon's admitted writer, exactly
        # as configured-source catch-up does.
        async with prepared_live_convergence_owner(archive_root) as owner:
            execution = DriveCatchupExecution(owner._write_coordinator, compute_adapter=owner._compute_adapter)
            service = AcquisitionService(backend, execution=execution)
            write_revision("first")
            first = await service.acquire_sources([Source(name="codex", path=bundle)])
            assert first.errors == 0
            original = membership()
            assert len(original) == 2
            write_revision("changed")
            changed = await service.acquire_sources([Source(name="codex", path=bundle)])
            assert changed.errors == 0
            assert len(membership()) == 4
            with sqlite3.connect(archive_root / "source.db") as conn:
                rank = raw_receipt_order_sql("r")
                before = dict(conn.execute(f"SELECT r.raw_id, {rank} FROM raw_sessions r"))
            write_revision("first")
            returned = await service.acquire_sources([Source(name="codex", path=bundle)])
            assert returned.errors == 0
            assert set(original) <= set(membership())
            assert len(membership()) == 4
            with sqlite3.connect(archive_root / "source.db") as conn:
                assert conn.execute(
                    "SELECT enumerated_record_count, enumerated_member_count FROM source_items "
                    "ORDER BY source_generation_id"
                ).fetchall() == [(2, 3), (2, 3)]
                rank = raw_receipt_order_sql("r")
                after = dict(conn.execute(f"SELECT r.raw_id, {rank} FROM raw_sessions r"))
                assert all(after[raw_id] > before[raw_id] for _group, raw_id, _entry, _split in original)
                assert conn.execute("SELECT DISTINCT acquired_at_ms FROM raw_sessions").fetchall() == [
                    (int(frozen_clock.time() * 1000),)
                ]
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["normal", "operation_cancelled", "task_cancelled"])
async def test_visit_sources_settles_zip_capture_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """Callback cancellation must close the real suspended acquisition carrier."""
    import json
    import zipfile
    from contextlib import contextmanager
    from typing import IO

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.pipeline.services.acquisition import AcquisitionService
    from polylogue.sources import source_acquisition
    from polylogue.sources.acquisition_boundary import BoundContainerCapture
    from polylogue.sources.source_staging import SourceInputBinding
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.runtime import RawSessionRecord
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from tests.infra.archive_templates import bootstrap_archive_root

    archive_root = tmp_path / "archive"
    await asyncio.to_thread(bootstrap_archive_root, archive_root)
    bundle = tmp_path / "neutral.zip"
    # More records than one acquisition batch keep the ZIP carrier suspended.
    with zipfile.ZipFile(bundle, "w") as container:
        for ordinal in range(200):
            container.writestr(
                f"session-{ordinal}.jsonl",
                json.dumps({"type": "session_meta", "payload": {"id": f"neutral-{ordinal}"}}) + "\n",
            )
    streams: list[IO[bytes]] = []
    from polylogue.sources.acquisition_boundary import open_bound_container

    original_open = open_bound_container

    @contextmanager
    def tracked_open(blob_store: BlobStore, source_binding: SourceInputBinding) -> Iterator[BoundContainerCapture]:
        with original_open(blob_store, source_binding) as capture:
            streams.append(capture.stream)
            yield capture

    monkeypatch.setattr(source_acquisition, "open_bound_container", tracked_open)
    received = 0

    async def receive(record: RawSessionRecord) -> None:
        nonlocal received
        assert record.blob_hash is not None
        received += 1
        if outcome == "operation_cancelled":
            raise DaemonOperationCancelled("neutral callback cancellation")
        if outcome == "task_cancelled":
            raise asyncio.CancelledError

    backend = SQLiteBackend(db_path=archive_root / "source.db")
    try:
        service = AcquisitionService(backend)
        if outcome == "normal":
            result = await service.visit_sources(
                [Source(name="codex", path=bundle)], on_record=receive, persist_cursors=False
            )
            assert result.counts == {"scanned": 200, "errors": 0}
        else:
            cancellation = DaemonOperationCancelled if outcome == "operation_cancelled" else asyncio.CancelledError
            with pytest.raises(cancellation):
                await service.visit_sources(
                    [Source(name="codex", path=bundle)], on_record=receive, persist_cursors=False
                )
        assert received == (200 if outcome == "normal" else 1)
        assert streams and all(stream.closed for stream in streams)
    finally:
        await backend.close()
