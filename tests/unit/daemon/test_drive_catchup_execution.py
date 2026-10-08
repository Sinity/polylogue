"""Real Drive staging must leave the writer available and reject stale work."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import pytest

from polylogue.config import Config, Source
from polylogue.daemon.drive_catchup import DriveCatchupExecution
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.sources import DriveFile
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.write_lease import (
    arm_write_lease_enforcement,
    current_write_lease,
    require_write_lease,
    write_lease,
)
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop

pytestmark = pytest.mark.uses_real_clock("Thread settlement and archive acquisition timestamps are observed.")


class DriveClient:
    def __init__(self, before_download: Callable[[], None] = lambda: None) -> None:
        self.before_download = before_download
        self.grown = False
        self.divergent = False
        self.modified_time = "2026-01-01T00:00:00Z"

    def resolve_folder_id(self, folder_ref: str) -> str:
        return folder_ref

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        yield DriveFile("document", "neutral.json", "application/json", self.modified_time, 100)

    def get_metadata(self, file_id: str, *, refresh: bool = False) -> DriveFile:
        return next(file for file in self.iter_json_files("") if file.file_id == file_id)

    def download_bytes(self, file_id: str) -> bytes:
        self.before_download()
        chunks: list[dict[str, object]] = [
            {"role": "user", "text": "Neutral question"},
            {
                "role": "model",
                "text": "Neutral answer",
                "driveDocument": {"id": "attachment", "name": "note.txt", "mimeType": "text/plain"},
            },
        ]
        if self.grown:
            # The provider re-serializes the whole document on each save, so
            # a conversation that grew is rewritten, not byte-appended.
            chunks.append({"role": "user", "text": "Neutral follow-up"})
            if self.divergent:
                chunks[0] = {"role": "user", "text": "Different neutral question"}
        return json.dumps({"chunkedPrompt": {"chunks": chunks}, "runSettings": {"model": "neutral"}}).encode()

    def download_into(self, file_id: str, handle: IO[bytes]) -> None:
        handle.write(self.download_bytes(file_id))


async def make_parser(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, phased: bool = True
) -> tuple[ParsingService, Source, DaemonWriteCoordinator]:
    await asyncio.to_thread(bootstrap_archive_root, root)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda **kwargs: SimpleNamespace(
            schema_validation="advisory", sinex_mode="off", archive_root=root, backup_verify_tmpdir=None
        ),
    )
    source = Source(name="gemini", folder="fixture", path=root / "source")
    config = Config(archive_root=root, render_root=root / "render", db_path=root / "index.db", sources=[source])
    backend = SQLiteBackend(db_path=config.db_path)
    repository = SessionRepository(backend=backend, archive_root=root)
    coordinator = DaemonWriteCoordinator(archive_root=root)
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge

    execution = DriveCatchupExecution(coordinator)
    raw_owner = RawObservationConvergenceOwner(
        root,
        compute_adapter=execution._compute_adapter,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        write_coordinator=coordinator,
    )
    parser = ParsingService(
        repository,
        root,
        config,
        execution=execution,
        retained_runner=raw_owner.ingest_retained_raw_ids,
    )
    return parser, source, coordinator


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_drive_prepare_cancellation_retains_its_one_worker_until_physical_drain(
    tmp_path: Path, worker_fails: bool
) -> None:
    from polylogue.core.compute import BoundedComputeAdapter, current_cancellation

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    adapter = BoundedComputeAdapter(max_workers=1)
    execution = DriveCatchupExecution(coordinator, compute_adapter=adapter)
    started = threading.Event()
    cancellation_received = threading.Event()
    release = threading.Event()

    def prepare() -> str:
        assert current_write_lease() is None
        cancellation = current_cancellation()
        assert cancellation is not None
        cancellation.add_listener(cancellation_received.set)
        started.set()
        assert release.wait(15)
        if worker_fails:
            raise RuntimeError("synthetic failure after physical drain")
        return "physically drained"

    task = asyncio.create_task(execution.prepare(prepare))
    try:
        assert await asyncio.to_thread(started.wait, 15)
        task.cancel()
        assert await asyncio.to_thread(cancellation_received.wait, 15)
        assert not task.done()
        assert adapter.snapshot().active_units == 1
        release.set()
        if worker_fails:
            with pytest.raises(builtins.BaseExceptionGroup) as raised:
                await task
            assert len(raised.value.exceptions) == 2
            assert isinstance(raised.value.exceptions[0], asyncio.CancelledError)
            assert isinstance(raised.value.exceptions[1], RuntimeError)
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert adapter.snapshot().active_units == 0
        assert await execution.prepare(lambda: "successor") == "successor"
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert await coordinator.shutdown(timeout=30.0)
        assert adapter.close(join_timeout_s=30.0) == ()


@pytest.mark.parametrize("blocked_phase", ["download", "parser"])
async def test_drive_preparation_leaves_real_writer_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_phase: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fails if either the full pass or parser iteration runs with a lease."""
    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        assert current_write_lease() is None
        started.set()
        assert release.wait(15)

    client = DriveClient(block if blocked_phase == "download" else lambda: None)
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: client)
    # Parsing runs in the retained Raw owner's preparation (derivation
    # compute), which must hold no writer lease.
    from polylogue.storage.derived.raw import RawObservationDerivation

    original = RawObservationDerivation.compute

    def parse(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> Any:
        assert current_write_lease() is None
        if blocked_phase == "parser":
            block()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RawObservationDerivation, "compute", parse)
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
        # Drive-hosted bytes are fetched later by the attachment convergence
        # stage; ingest stores the reference only.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM attachments WHERE blob_hash IS NULL AND acquisition_status = 'unfetched'"
            ).fetchone()[0]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'Neutral'").fetchone()[0] == 2


async def test_drive_artifact_inspection_error_retains_partial_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser, source, _ = await make_parser(tmp_path, monkeypatch)
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
    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: DriveClient())
    acquired = await parser.ingest_sources(sources=[source], parse_records=False)
    raw_id = acquired.acquire_result.raw_ids[0]
    from polylogue.storage.derived.raw import RawObservationDerivation

    original_publish = RawObservationDerivation.publish
    moved = False

    def publish_after_mutation(self: RawObservationDerivation, frame: Any, replacement: Any, **kwargs: Any) -> bool:
        # Parser work is complete (``compute`` prepared the replacement); the
        # input, authority, policy or generation moves before publication.
        nonlocal moved
        if not moved and not replacement.already_valid:
            moved = True
            if current_write_lease() is not None:
                mutate()
            else:
                run_off_event_loop(mutate_under_lease)
        return original_publish(self, frame, replacement, **kwargs)

    def mutate_under_lease() -> None:
        with write_lease("test.mutate", archive_root=tmp_path):
            mutate()

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
                    key = conn.execute(
                        "SELECT logical_source_key FROM raw_sessions WHERE raw_id=?", (raw_id,)
                    ).fetchone()[0]
                    columns = [row[1] for row in conn.execute("PRAGMA table_info(raw_sessions)")]
                    row = list(conn.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone())
                    row[columns.index("raw_id")] = "sibling"
                    row[columns.index("logical_source_key")] = key
                    conn.execute(f"INSERT INTO raw_sessions VALUES ({','.join('?' for _ in row)})", row)

    monkeypatch.setattr(RawObservationDerivation, "publish", publish_after_mutation)
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    # The publication's seal refuses the moved input with a typed, retryable
    # refusal; nothing is marked parsed and no session is written.
    with arm_write_lease_enforcement(process_wide=True), pytest.raises(ReferenceSealStaleError):
        await parser.parse_from_raw(raw_ids=[raw_id])
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


async def test_drive_phased_matches_ordinary_document_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real Drive revisions retain the same lineage, content and FTS in both routes.

    Membership governance is the one revision authority for a Drive document:
    the re-serialized larger revision is applied and the earlier capture is
    superseded by it, in both routes.
    """
    source = Source(name="gemini", folder="fixture", path=tmp_path / "shared-source")
    snapshots = []
    for phased in (False, True):
        root = tmp_path / ("phased" if phased else "ordinary")
        parser, _, _ = await make_parser(root, monkeypatch, phased=phased)
        client = DriveClient()
        monkeypatch.setattr(
            "polylogue.sources.drive._resolved_drive_client", lambda fixture_client=client, **kwargs: fixture_client
        )
        with arm_write_lease_enforcement(armed=phased, process_wide=True):
            first = await parser.ingest_sources(sources=[source])
            assert first.parse_result.processed_ids
            client.grown = True
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
            memberships = conn.execute(
                "SELECT m.raw_id, m.logical_source_key, m.decision, m.revision_authority "
                "FROM raw_session_memberships AS m JOIN raw_sessions AS r ON r.raw_id = m.raw_id "
                "ORDER BY r.acquired_at_ms"
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
        assert [(row[1], row[2], row[3]) for row in memberships] == [
            ("aistudio-drive:document", "superseded_prefix", "byte_proven"),
            ("aistudio-drive:document", "applied", "byte_proven"),
        ], (phased, memberships)
        assert [row[2] for row in sessions] == [memberships[1][0]]
        snapshots.append((raw, memberships, sessions, messages, attachments, fts))
        assert source.path is not None
        from polylogue.sources.drive import drive_cache_file_path
        from polylogue.sources.drive.witness import drive_cache_directory

        drive_cache_file_path(drive_cache_directory(source.path, "fixture"), "document").unlink()
    assert snapshots[0] == snapshots[1]


async def test_drive_download_cancellation_closes_staging_after_thread_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser, source, _ = await make_parser(tmp_path, monkeypatch)
    started = threading.Event()
    release = threading.Event()
    from polylogue.sources.drive.witness import DriveListingWitness

    closed_witnesses: list[str] = []
    original_close = DriveListingWitness.close

    def close_witness(witness: DriveListingWitness) -> None:
        original_close(witness)
        closed_witnesses.append(witness.source_name)

    monkeypatch.setattr(DriveListingWitness, "close", close_witness)

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
    assert closed_witnesses == [source.name]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
    await parser.repository.close()


@pytest.mark.parametrize("divergent", [False, True])
async def test_drive_growth_binds_a_raw_owned_by_many_source_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    divergent: bool,
) -> None:
    """Generation ownership cardinality never turns a Drive cohort permanently stale.

    Anti-vacuity: snapshot the cohort's ``source_item_raw_members`` rows into
    the preparation scratch and this raw's 1,001 ownership rows exceed the
    cohort row cap, so every pass discards the grown revision unparsed.
    """
    import ast
    import inspect
    import sys

    from polylogue.storage.derived.raw import RawObservationDerivation

    publish = RawObservationDerivation.publish
    source_lines, source_start = inspect.getsourcelines(publish)
    function_source = "".join(source_lines)
    import textwrap

    false_sites = {
        source_start + node.lineno - 1
        for node in ast.walk(ast.parse(textwrap.dedent(function_source)))
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is False
    }
    observations: list[tuple[bool, bool, bool, tuple[int, ...]]] = []

    def observed_publish(self: Any, frame: Any, replacement: Any, **kwargs: Any) -> bool:
        previous_trace = sys.gettrace()
        reached: list[int] = []

        def trace(actual_frame: Any, event: Any, value: Any) -> Any:
            if actual_frame.f_code is publish.__code__:
                if event == "line" and actual_frame.f_lineno in false_sites:
                    reached.append(actual_frame.f_lineno)
                return trace
            return previous_trace(actual_frame, event, value) if previous_trace is not None else None

        sys.settrace(trace)
        try:
            return publish(self, frame, replacement, **kwargs)
        finally:
            sys.settrace(previous_trace)
            observations.append(
                (
                    replacement.already_valid,
                    replacement.needs_source_census,
                    replacement.needs_source_classification,
                    tuple(reached),
                )
            )

    monkeypatch.setattr(RawObservationDerivation, "publish", observed_publish)
    root = tmp_path / "archive"
    source = Source(name="gemini", folder="fixture", path=tmp_path / "source")
    parser, _, coordinator = await make_parser(root, monkeypatch)
    client = DriveClient()
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: client)
    with arm_write_lease_enforcement(process_wide=True):
        first = await parser.ingest_sources(sources=[source])
        assert first.parse_result.processed_ids
        with contextlib.closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as source_read:
            original_binding = source_read.execute(
                "SELECT raw_id,logical_source_key,revision_kind,source_revision,revision_authority,"
                "predecessor_raw_id,baseline_raw_id,acquisition_generation FROM raw_sessions"
            ).fetchone()
        assert original_binding is not None
        with contextlib.closing(sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)) as index_read:
            original_head = index_read.execute(
                "SELECT accepted_raw_id,accepted_content_hash,accepted_frontier_kind,accepted_frontier "
                "FROM raw_revision_heads"
            ).fetchone()
        assert original_head is not None
        client.grown = True
        client.divergent = divergent
        client.modified_time = "2026-01-02T00:00:00Z"
        acquired = await parser.ingest_sources(sources=[source], parse_records=False)
        (raw_id,) = acquired.acquire_result.raw_ids

        def bind_generations() -> None:
            require_write_lease("bind fixture source generations", archive_root=root)
            with sqlite3.connect(root / "source.db") as conn:
                conn.executemany(
                    "INSERT INTO source_item_raw_members VALUES (?, 'item', 'coordinate', ?, ?)",
                    [(f"generation-{index}", raw_id, bytes(32)) for index in range(1001)],
                )

        await coordinator.run_sync("test.bind_generations", bind_generations)
        try:
            second = await parser.parse_from_raw(raw_ids=[raw_id])
        except BaseException as failure:
            failure.add_note(f"actual Raw publication flags and reached false sites: {observations}")
            raise
        if divergent:
            # An edited earlier turn is not growth: membership governance
            # quarantines both captures as an ambiguous cohort and keeps the
            # accepted head. That is governance evidence, not a parse failure.
            assert not second.processed_ids
            assert second.parse_failures == 0
            with contextlib.closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as decided:
                assert sorted(
                    decided.execute(
                        "SELECT decision, revision_authority FROM raw_session_memberships WHERE logical_source_key=?",
                        ("aistudio-drive:document",),
                    ).fetchall()
                ) == [("ambiguous", "quarantined"), ("ambiguous", "quarantined")]
        else:
            assert second.processed_ids
    await parser.repository.close()
    with contextlib.closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as source_read:
        assert (
            source_read.execute(
                "SELECT raw_id,logical_source_key,revision_kind,source_revision,revision_authority,"
                "predecessor_raw_id,baseline_raw_id,acquisition_generation FROM raw_sessions WHERE raw_id=?",
                (original_binding[0],),
            ).fetchone()
            == original_binding
        )
        assert source_read.execute(
            "SELECT revision_authority,predecessor_raw_id FROM raw_sessions WHERE raw_id=?", (raw_id,)
        ).fetchone() == ("quarantined", None)
    with contextlib.closing(sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)) as index_read:
        assert index_read.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == (2 if divergent else 3)
        current_head = index_read.execute(
            "SELECT accepted_raw_id,accepted_content_hash,accepted_frontier_kind,accepted_frontier "
            "FROM raw_revision_heads"
        ).fetchone()
        if divergent:
            assert current_head == original_head
        else:
            assert current_head is not None and current_head[0] == raw_id


@pytest.mark.parametrize("failure_phase", ["prepare", "publish"])
async def test_prepared_publication_retains_failed_sql_on_its_original_compute_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    """A failed observer or publisher close remains recoverable on the real worker.

    Returning the compute slot or losing a seal during its constructor would
    make successor admission unable to settle the original SQLite handles.
    """
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.write_coordinator import DaemonWriterSettlementError
    from polylogue.storage import io_phase_metrics
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.archive_custody_probe import archive_custody_available
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    class DriveSettlementConnection(ControlledConnection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.owner = self.creator
            self.allow_cleanup = threading.Event()
            self.allow_cleanup.set()
            self.calls: list[tuple[str, threading.Thread]] = []

        def rollback(self) -> None:
            self.calls.append(("rollback", threading.current_thread()))
            assert threading.current_thread() is self.owner
            if not self.allow_cleanup.is_set():
                raise OSError("synthetic native rollback remains unsettled")
            super().rollback()

        def close(self) -> None:
            self.calls.append(("close", threading.current_thread()))
            assert threading.current_thread() is self.owner
            if not self.allow_cleanup.is_set():
                raise OSError("synthetic native close remains unsettled")
            super().close()

    await asyncio.to_thread(bootstrap_archive_root, tmp_path)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    adapter = BoundedComputeAdapter(max_workers=1)
    execution = DriveCatchupExecution(coordinator, compute_adapter=adapter)
    handles: list[DriveSettlementConnection] = []
    original_threads: list[threading.Thread] = []
    read_references = PreparedIndexMutation._read_resolved_references
    # Keep the actual factory, admission/profile hooks and Native registration.
    monkeypatch.setattr(io_phase_metrics, "_MeasuredConnection", DriveSettlementConnection)

    def read_and_fail(seal: PreparedIndexMutation) -> None:
        read_references(seal)
        handle = seal._observers["index"]
        assert isinstance(handle, DriveSettlementConnection)
        handle.allow_cleanup.clear()
        handles.append(handle)
        original_threads.append(threading.current_thread())
        if failure_phase == "prepare":
            raise RuntimeError("synthetic failure after observer acquisition")

    monkeypatch.setattr(PreparedIndexMutation, "_read_resolved_references", read_and_fail)

    def prepare() -> PreparedIndexMutation:
        return PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)

    def publish(_seal: PreparedIndexMutation) -> None:
        store = ArchiveStore(tmp_path, initialize=False)
        store._enter_mutation_lease()
        store._conn.execute("BEGIN IMMEDIATE")
        store._conn.execute("SELECT COUNT(*) FROM sessions")
        handle = store._conn
        assert isinstance(handle, DriveSettlementConnection)
        handle.allow_cleanup.clear()
        handles.append(handle)
        store.close()

    try:
        with pytest.raises(DaemonWriterSettlementError):
            await execution.publish_prepared_sync("test.retained_prepare", prepare, publish)
        assert original_threads[0].is_alive()
        assert coordinator.snapshot().unsettled_writer_workers == 1
        assert adapter.snapshot().active_units == 1
        if failure_phase == "publish":
            assert not archive_custody_available(tmp_path)
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.unsettled_successor", lambda: None)
        assert all(any(name == "close" for name, _thread in handle.calls) for handle in handles)
        for handle in handles:
            handle.allow_cleanup.set()
        await coordinator.run_sync("test.settled_successor", lambda: None)
        assert coordinator.snapshot().unsettled_writer_workers == 0
        assert archive_custody_available(tmp_path)
        assert all(thread is handle.owner for handle in handles for _name, thread in handle.calls)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        assert await coordinator.shutdown(timeout=30.0)
        assert adapter.close(join_timeout_s=30.0) == ()


async def test_prepared_publication_cancellation_reaches_and_settles_the_reference_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the owner stops its off-gate proof and closes every observer."""
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, _check_reference_cancellation

    await asyncio.to_thread(bootstrap_archive_root, tmp_path)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    adapter = BoundedComputeAdapter(max_workers=1)
    execution = DriveCatchupExecution(coordinator, compute_adapter=adapter)
    census_started = threading.Event()
    seals: list[PreparedIndexMutation] = []
    read_references = PreparedIndexMutation._read_resolved_references

    def observe_cancellation(seal: PreparedIndexMutation) -> None:
        read_references(seal)
        seals.append(seal)
        cancellation = compute_cancel.get()
        assert cancellation is not None
        census_started.set()
        assert cancellation.wait(15)
        _check_reference_cancellation()

    monkeypatch.setattr(PreparedIndexMutation, "_read_resolved_references", observe_cancellation)
    published = False

    def publish(_seal: PreparedIndexMutation) -> None:
        nonlocal published
        published = True

    task = asyncio.create_task(
        execution.publish_prepared_sync(
            "test.cancelled_census",
            lambda: PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path),
            publish,
        )
    )
    try:
        assert await asyncio.to_thread(census_started.wait, 15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not published
        assert seals[0]._closed
        assert coordinator.snapshot().unsettled_writer_workers == 0
        await coordinator.run_sync("test.after_cancelled_census", lambda: None)
    finally:
        cancellation = compute_cancel.get()
        if cancellation is not None:
            cancellation.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert await coordinator.shutdown(timeout=30.0)
        assert adapter.close(join_timeout_s=30.0) == ()


def _read_drive_report(parser: ParsingService, source: Source, witnesses: Any, **kwargs: Any) -> Any:
    from polylogue.operations.drive_readiness import inspect_drive_readiness

    with sqlite3.connect(parser.archive_root / "source.db") as raw, sqlite3.connect(parser.config.db_path) as index:
        return inspect_drive_readiness([source], raw, index, witnesses, **kwargs)


@pytest.mark.parametrize("failure", [None, "download", "persist", "revision_race"])
async def test_drive_report_binds_full_listing_to_actual_retained_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from polylogue.operations.drive_readiness import DriveCatchupState

    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)
    client = DriveClient()
    if failure == "download":

        def fail(_file_id: str) -> bytes:
            raise OSError("synthetic transport failure")

        monkeypatch.setattr(client, "download_bytes", fail)
    elif failure == "persist":

        async def fail_persist(*args: Any, **kwargs: Any) -> None:
            raise sqlite3.OperationalError("synthetic persistence failure")

        monkeypatch.setattr(SessionRepository, "admit_raw", fail_persist)
    elif failure == "revision_race":
        client.before_download = lambda: setattr(client, "modified_time", "2026-01-02T00:00:00Z")
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: client)
    result = await parser.ingest_sources(sources=[source])
    report = _read_drive_report(parser, source, result.acquire_result.drive_witnesses)
    expected = (
        DriveCatchupState.COMPLETE
        if failure is None
        else DriveCatchupState.PENDING
        if failure == "revision_race"
        else DriveCatchupState.RETRYABLE
    )
    assert report.state is expected
    assert report.enumerated_count == 1
    assert report.acquired_count == (1 if failure is None else 0)
    assert result.acquire_result.errors == (1 if failure in {"download", "persist"} else 0)
    if failure in {"download", "persist"}:
        assert len(report.failures) == 1
    if failure == "revision_race":
        assert not result.acquire_result.raw_ids
        assert "drive_listing_changed" in report.gaps
    for witness in result.acquire_result.drive_witnesses.values():
        witness.close()
    await parser.repository.close()
    assert await coordinator.shutdown(timeout=30)


async def test_drive_same_names_and_rename_keep_native_identity_and_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations.drive_readiness import DriveCatchupState

    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)

    class TwoFiles(DriveClient):
        renamed = False
        downloaded: list[str] = []

        def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
            for file_id in ("native-one", "native-two"):
                yield DriveFile(
                    file_id,
                    "renamed.json" if self.renamed else "same.json",
                    "application/json",
                    self.modified_time,
                    100,
                )

        def download_bytes(self, file_id: str) -> bytes:
            self.downloaded.append(file_id)
            return super().download_bytes(file_id)

    client = TwoFiles()
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: client)
    first = await parser.ingest_sources(sources=[source])
    assert first.acquire_result.acquired == 2
    assert len(first.parse_result.processed_ids) == 2
    assert _read_drive_report(parser, source, first.acquire_result.drive_witnesses).state is DriveCatchupState.COMPLETE
    client.renamed = True
    second = await parser.ingest_sources(sources=[source])
    assert not second.acquire_result.raw_ids
    assert client.downloaded == ["native-one", "native-two"]
    assert _read_drive_report(parser, source, second.acquire_result.drive_witnesses).state is DriveCatchupState.COMPLETE
    for result in (first, second):
        for witness in result.acquire_result.drive_witnesses.values():
            witness.close()
    await parser.repository.close()
    assert await coordinator.shutdown(timeout=30)


async def test_drive_acquired_backlog_is_pending_and_restart_requires_new_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations.drive_readiness import DriveCatchupState

    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: DriveClient())
    acquired = await parser.ingest_sources(sources=[source], parse_records=False)
    witnesses = acquired.acquire_result.drive_witnesses
    report = _read_drive_report(parser, source, witnesses)
    assert report.state is DriveCatchupState.PENDING
    assert report.materialization_pending == 1
    no_owner = _read_drive_report(parser, source, witnesses, raw_owner_available=False)
    assert no_owner.state is DriveCatchupState.UNKNOWN
    assert no_owner.enumerated_count is no_owner.acquired_count is no_owner.materialization_pending is None
    resumed = await parser.ingest_sources(sources=[source], skip_acquire=True)
    assert resumed.parse_result.processed_ids
    assert _read_drive_report(parser, source, witnesses).state is DriveCatchupState.COMPLETE
    missing = _read_drive_report(parser, source, {})
    assert missing.state is DriveCatchupState.UNKNOWN
    assert missing.enumerated_count is missing.acquired_count is missing.materialization_pending is None
    restarted = await parser.ingest_sources(sources=[source])
    assert (
        _read_drive_report(parser, source, restarted.acquire_result.drive_witnesses).state is DriveCatchupState.COMPLETE
    )
    # A stale bound raw is not completion after its declared acquisition row changes.
    witness = witnesses[source.name]
    path, revision, raw_id, _acquired_revision, _error, _permanent = next(witness.members())
    witness.record_acquired_revision(path, "2025-01-01T00:00:00Z")
    assert _read_drive_report(parser, source, witnesses).state is DriveCatchupState.UNKNOWN
    for result in (acquired, restarted):
        for witness in result.acquire_result.drive_witnesses.values():
            witness.close()
    await parser.repository.close()
    assert await coordinator.shutdown(timeout=30)


@pytest.mark.parametrize("decision,state", [("deferred", "pending"), ("ambiguous", "blocked")])
async def test_drive_membership_debt_keeps_its_canonical_readiness_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str, state: str
) -> None:
    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: DriveClient())
    result = await parser.ingest_sources(sources=[source])
    with sqlite3.connect(tmp_path / "source.db") as raw:
        raw.execute("UPDATE raw_session_memberships SET decision=?", (decision,))
    report = _read_drive_report(parser, source, result.acquire_result.drive_witnesses)
    assert report.state.value == state
    assert report.materialization_pending == 1
    assert not any(failure.permanent for failure in report.failures)
    for witness in result.acquire_result.drive_witnesses.values():
        witness.close()
    await parser.repository.close()
    assert await coordinator.shutdown(timeout=30)


async def test_drive_terminal_no_session_evidence_blocks_even_without_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations.drive_readiness import DriveCatchupState

    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)
    client = DriveClient()
    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: client)
    result = await parser.ingest_sources(sources=[source])
    with sqlite3.connect(tmp_path / "source.db") as raw:
        # A current parser's terminal no-session disposition can have no
        # free-text diagnostic; its typed Source evidence remains authority.
        raw.execute("UPDATE raw_sessions SET parse_error=NULL")
        raw.execute(
            "UPDATE raw_artifacts SET artifact_kind='terminal_unknown_export_no_session',support_status='unsupported_parseable'"
        )
        raw.execute("DELETE FROM raw_session_memberships")
    report = _read_drive_report(parser, source, result.acquire_result.drive_witnesses)
    assert report.state is DriveCatchupState.BLOCKED
    assert report.failures[0].permanent
    for witness in result.acquire_result.drive_witnesses.values():
        witness.close()
    await parser.repository.close()
    assert await coordinator.shutdown(timeout=30)


async def test_drive_interrupted_acquisition_listing_keeps_the_configured_denominator_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations.drive_readiness import DriveCatchupState

    parser, source, coordinator = await make_parser(tmp_path, monkeypatch)

    class Interrupted(DriveClient):
        def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
            yield from super().iter_json_files(folder_id)
            raise OSError("synthetic later-page refusal")

    monkeypatch.setattr("polylogue.sources.drive._resolved_drive_client", lambda **kwargs: Interrupted())
    result = await parser.ingest_sources(sources=[source])
    report = _read_drive_report(parser, source, result.acquire_result.drive_witnesses)
    assert result.acquire_result.errors == 1
    assert report.state is DriveCatchupState.UNKNOWN
    assert report.enumerated_count is report.acquired_count is report.materialization_pending is None
    assert "drive_listing_failed" in report.gaps
    for witness in result.acquire_result.drive_witnesses.values():
        witness.close()
    await parser.repository.close()
    assert await coordinator.shutdown(timeout=30)
