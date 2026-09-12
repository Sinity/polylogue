"""Real Drive staging must leave the writer available and reject stale work."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.config import Config, Source
from polylogue.daemon.drive_catchup import DriveCatchupExecution
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.pipeline.services.ingest_batch import _core as ingest
from polylogue.pipeline.services.ingest_batch._models import _IngestWorkerRequest, _PreparedIngestUnit
from polylogue.pipeline.services.ingest_worker import IngestRecordResult
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.sources import DriveFile
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement, current_write_lease, require_write_lease
from tests.infra.archive_templates import bootstrap_archive_root

pytestmark = pytest.mark.uses_real_clock("Thread settlement and archive acquisition timestamps are observed.")


class DriveClient:
    def __init__(self, before_download: Callable[[], None] = lambda: None) -> None:
        self.before_download = before_download
        self.attachment_available = True
        self.modified_time = "2026-01-01T00:00:00Z"

    def resolve_folder_id(self, folder_ref: str) -> str:
        return folder_ref

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        yield DriveFile("document", "neutral.json", "application/json", self.modified_time, 100)

    def download_bytes(self, file_id: str) -> bytes:
        self.before_download()
        if file_id == "attachment":
            if not self.attachment_available:
                raise OSError("synthetic unavailable attachment")
            return b"neutral attachment bytes"
        return json.dumps(
            {
                "chunkedPrompt": {
                    "chunks": [
                        {"role": "user", "text": "Neutral question"},
                        {
                            "role": "model",
                            "text": "Neutral answer",
                            "driveDocument": {
                                "id": "attachment",
                                "name": "note.txt",
                                "mimeType": "text/plain",
                            },
                        },
                    ]
                }
            }
        ).encode()


def make_parser(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, phased: bool = True
) -> tuple[ParsingService, Source, DaemonWriteCoordinator]:
    bootstrap_archive_root(root)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda **kwargs: SimpleNamespace(schema_validation="advisory", sinex_mode="off", archive_root=root),
    )
    source = Source(name="gemini", folder="fixture", path=root / "source")
    config = Config(archive_root=root, render_root=root / "render", db_path=root / "index.db", sources=[source])
    backend = SQLiteBackend(db_path=config.db_path)
    repository = SessionRepository(backend=backend, archive_root=root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    parser = ParsingService(
        repository, root, config, ingest_workers=1, execution=DriveCatchupExecution(coordinator) if phased else None
    )
    return parser, source, coordinator


@pytest.mark.parametrize("blocked_phase", ["download", "parser"])
async def test_drive_preparation_leaves_real_writer_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_phase: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fails if either the full pass or parser iteration runs with a lease."""
    parser, source, coordinator = make_parser(tmp_path, monkeypatch)
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        assert current_write_lease() is None
        started.set()
        assert release.wait(15)

    client = DriveClient(block if blocked_phase == "download" else lambda: None)
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: client)
    original = ingest._run_ingest_record

    def parse(record: RawSessionRecord, request: _IngestWorkerRequest) -> IngestRecordResult:
        assert current_write_lease() is None
        if blocked_phase == "parser":
            block()
        return original(record, request)

    monkeypatch.setattr(ingest, "_run_ingest_record", parse)
    from polylogue.storage.artifacts import inspection

    inspect_artifact = inspection.inspect_raw_artifact
    inspected = []

    def inspect(record: RawSessionRecord, *, blob_store: BlobStore | None = None) -> ArtifactObservationRecord:
        assert current_write_lease() is None
        inspected.append(True)
        return inspect_artifact(record, blob_store=blob_store)

    monkeypatch.setattr(inspection, "inspect_raw_artifact", inspect)

    def compete() -> None:
        require_write_lease("competing test mutation", archive_root=tmp_path)
        with sqlite3.connect(tmp_path / "ops.db") as conn:
            conn.execute("CREATE TABLE drive_competing_mutation (completed INTEGER)")
            conn.execute("INSERT INTO drive_competing_mutation VALUES (1)")

    with arm_write_lease_enforcement(process_wide=True):
        task = asyncio.create_task(parser.ingest_sources(sources=[source]))
        try:
            assert await asyncio.to_thread(started.wait, 15)
            await asyncio.wait_for(coordinator.run_sync("test.compete", compete), 10)
        finally:
            release.set()
        result = await task
    await parser.repository.close()
    assert result.acquire_result.acquired == 1
    assert result.parse_result.processed_ids
    assert result.parse_result.parse_failures == 0
    assert inspected == [True]
    assert "UnleasedWriteError" not in caplog.text
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash IS NOT NULL").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'Neutral'").fetchone()[0] == 2


async def test_drive_artifact_inspection_error_retains_partial_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser, source, _ = make_parser(tmp_path, monkeypatch)
    client = DriveClient()
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: client)

    def inspect(record: RawSessionRecord, *, blob_store: BlobStore | None = None) -> ArtifactObservationRecord:
        assert current_write_lease() is None
        raise ValueError("synthetic inspection failure")

    monkeypatch.setattr("polylogue.storage.artifacts.inspection.inspect_raw_artifact", inspect)
    with arm_write_lease_enforcement(process_wide=True):
        result = await parser.ingest_sources(sources=[source])
    await parser.repository.close()
    assert result.acquire_result.errors == 1
    assert result.acquire_result.acquired == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1


@pytest.mark.parametrize("mutation", ["raw", "delete", "authority", "sibling", "policy", "generation"])
async def test_drive_stale_preparation_never_marks_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Changing input, authority, policy or generation invalidates completed parser work."""
    parser, source, coordinator = make_parser(tmp_path, monkeypatch)
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: DriveClient())
    acquired = await parser.ingest_sources(sources=[source], parse_records=False)
    raw_id = acquired.acquire_result.raw_ids[0]
    execution = parser.execution
    assert execution is not None
    original = execution.prepare
    moved = False

    async def prepare(operation: Callable[[], _PreparedIngestUnit | None]) -> _PreparedIngestUnit | None:
        nonlocal moved
        prepared = await original(operation)
        assert prepared is not None
        if moved:
            return prepared
        moved = True

        def mutate() -> None:
            if mutation == "policy":
                with sqlite3.connect(tmp_path / "user.db") as conn:
                    conn.execute("UPDATE query_unit_frame_state SET epoch=epoch+1")
            elif mutation == "generation":
                index = tmp_path / "index.db"
                replacement = tmp_path / "replacement.db"
                with sqlite3.connect(index) as old, sqlite3.connect(replacement) as new:
                    old.backup(new)
                replacement.replace(index)
            else:
                with sqlite3.connect(tmp_path / "source.db") as conn:
                    if mutation == "raw":
                        conn.execute("UPDATE raw_sessions SET file_mtime_ms=file_mtime_ms+1 WHERE raw_id=?", (raw_id,))
                    elif mutation == "delete":
                        conn.execute("DELETE FROM raw_sessions WHERE raw_id=?", (raw_id,))
                    elif mutation == "authority":
                        conn.execute("UPDATE raw_sessions SET revision_authority='asserted' WHERE raw_id=?", (raw_id,))
                    else:
                        # A new cohort member must invalidate the negative sibling comparison.
                        key = prepared.logical_keys[0]
                        columns = [row[1] for row in conn.execute("PRAGMA table_info(raw_sessions)")]
                        row = list(conn.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone())
                        row[columns.index("raw_id")] = "sibling"
                        row[columns.index("logical_source_key")] = key
                        conn.execute(f"INSERT INTO raw_sessions VALUES ({','.join('?' for _ in row)})", row)

        await coordinator.run_sync("test.mutate", mutate)
        return prepared

    monkeypatch.setattr(execution, "prepare", prepare)
    with arm_write_lease_enforcement(process_wide=True):
        result = await parser.parse_from_raw(raw_ids=[raw_id])
    assert not result.processed_ids
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    if mutation in {"raw", "policy", "generation"}:
        result = await parser.parse_from_raw(raw_ids=[raw_id])
        assert result.processed_ids
    await parser.repository.close()


@pytest.mark.parametrize("phase", ["prepare", "publish"])
async def test_drive_cancellation_settles_thread_before_return(
    tmp_path: Path,
    phase: str,
) -> None:
    """Cancellation cannot release a publisher or abandon a preparation thread."""
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    execution = DriveCatchupExecution(coordinator)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def work() -> None:
        assert (current_write_lease() is not None) == (phase == "publish")
        started.set()
        assert release.wait(15)
        finished.set()

    with arm_write_lease_enforcement(process_wide=True):
        task = asyncio.create_task(
            execution.prepare(work) if phase == "prepare" else execution.publish_sync("test", work)
        )
        assert await asyncio.to_thread(started.wait, 15)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        def competing_mutation() -> None:
            require_write_lease("competing publication", archive_root=tmp_path)
            assert phase == "prepare" or finished.is_set()

        competitor = asyncio.create_task(coordinator.run_sync("test.compete", competing_mutation))
        if phase == "prepare":
            await asyncio.wait_for(competitor, 10)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        await competitor
        await coordinator.run_sync("test.after", lambda: require_write_lease("settled", archive_root=tmp_path))


async def test_drive_phased_matches_ordinary_attachment_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real Drive revisions retain the same lineage, content and FTS in both routes."""
    source = Source(name="gemini", folder="fixture", path=tmp_path / "shared-source")
    structural = ingest._drive_structural_growth_predecessor
    structural_calls = 0

    def compare(
        source_conn: sqlite3.Connection,
        blob_publisher: ArchiveBlobPublisher,
        *,
        raw_id: str,
        logical_source_key: str,
    ) -> tuple[str, int, str] | None:
        nonlocal structural_calls
        assert current_write_lease() is None
        structural_calls += 1
        return structural(source_conn, blob_publisher, raw_id=raw_id, logical_source_key=logical_source_key)

    snapshots = []
    for phased in (False, True):
        root = tmp_path / ("phased" if phased else "ordinary")
        parser, _, _ = make_parser(root, monkeypatch, phased=phased)
        client = DriveClient()
        client.attachment_available = False
        monkeypatch.setattr(
            "polylogue.sources.drive._resolved_drive_client", lambda fixture_client=client, **kwargs: fixture_client
        )
        if phased:
            monkeypatch.setattr(ingest, "_drive_structural_growth_predecessor", compare)
        with arm_write_lease_enforcement(armed=phased, process_wide=True):
            first = await parser.ingest_sources(sources=[source])
            assert first.parse_result.processed_ids
            client.attachment_available = True
            client.modified_time = "2026-01-02T00:00:00Z"
            second = await parser.ingest_sources(sources=[source])
            assert second.parse_result.processed_ids
        await parser.repository.close()
        with sqlite3.connect(root / "source.db") as conn:
            raw = conn.execute(
                "SELECT raw_id, blob_hash, logical_source_key, revision_kind, revision_authority, "
                "predecessor_raw_id, baseline_raw_id, parsed_at_ms IS NOT NULL, parse_error "
                "FROM raw_sessions ORDER BY raw_id"
            ).fetchall()
        with sqlite3.connect(root / "index.db") as conn:
            sessions = conn.execute(
                "SELECT session_id, content_hash, raw_id, message_count FROM sessions ORDER BY session_id"
            ).fetchall()
            messages = conn.execute(
                "SELECT message_id, role, content_hash FROM messages ORDER BY message_id"
            ).fetchall()
            attachments = conn.execute(
                "SELECT attachment_id, blob_hash, acquisition_status FROM attachments ORDER BY attachment_id"
            ).fetchall()
            fts = conn.execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'Neutral' ORDER BY rowid"
            ).fetchall()
        assert len(raw) == 2
        assert any(row[4] == "asserted" and row[5] for row in raw)
        snapshots.append((raw, sessions, messages, attachments, fts))
        assert source.path is not None
        (source.path / "neutral.json").unlink()
    assert structural_calls == 2
    assert snapshots[0] == snapshots[1]


async def test_drive_download_cancellation_closes_staging_after_thread_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser, source, _ = make_parser(tmp_path, monkeypatch)
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        assert current_write_lease() is None
        started.set()
        assert release.wait(15)

    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: DriveClient(block))
    with arm_write_lease_enforcement(process_wide=True):
        task = asyncio.create_task(parser.ingest_sources(sources=[source]))
        assert await asyncio.to_thread(started.wait, 15)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not list((tmp_path / "blob" / ".staging").iterdir())
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
    await parser.repository.close()
