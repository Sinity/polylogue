"""Live archive-write helpers for tests.

Tests seed the archive through the same ``write_parsed_session_to_archive``
path the daemon uses — there is no separate test-only ingest stack. This
helper is the single seam: hand it a :class:`ParsedSession` and a backend (or
a database path) and it writes the canonical session/message/block/attachment/
event rows and returns the archive ``session_id``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.enums import ValidationMode
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.write import ArchiveWriteOutcome
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session

if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult


def write_session_sync(
    db_path: Path,
    session: ParsedSession,
    *,
    raw_id: str | None = None,
    content_hash: str | None = None,
    archive_root: Path | None = None,
) -> str:
    """Write one parsed session through the live archive writer (sync).

    Opens a dedicated sync connection on ``db_path`` so writers and async
    readers do not share a connection. Returns the archive ``session_id``.
    """
    conn = connect_measured(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        return write_fixture_index_session(
            conn,
            session,
            content_hash=content_hash if content_hash is not None else session_content_hash(session),
            raw_id=raw_id,
            archive_root=archive_root,
        )
    finally:
        close_fixture_index_connection(conn)


def write_session_counts_sync(
    db_path: Path,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
) -> dict[str, int]:
    """Write an index-only fixture and return the writer's count contract."""
    conn = connect_measured(str(db_path))
    conn.row_factory = sqlite3.Row
    outcomes: list[ArchiveWriteOutcome] = []
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        write_fixture_index_session(
            conn,
            session,
            content_hash=content_hash if content_hash is not None else session_content_hash(session),
            write_outcome=outcomes,
        )
        skipped = bool(outcomes and getattr(outcomes[0], "stale_skipped", False))
        if skipped:
            counts = {
                "sessions": 0,
                "messages": 0,
                "attachments": 0,
                "session_events": 0,
                "skipped_sessions": 1,
                "skipped_messages": len(session.messages),
                "skipped_attachments": len(session.attachments),
                "skipped_session_events": len(session.session_events),
                "raw_links": 0,
            }
        else:
            counts = {
                "sessions": 1,
                "messages": len(session.messages),
                "attachments": len(session.attachments),
                "session_events": len(session.session_events),
                "skipped_sessions": 0,
                "skipped_messages": 0,
                "skipped_attachments": 0,
                "skipped_session_events": 0,
                "raw_links": 0,
            }
        counts["stale_skipped"] = int(skipped)
        conn.commit()
        return counts
    finally:
        close_fixture_index_connection(conn)


async def ingest_session(
    session: ParsedSession,
    backend: SQLiteBackend,
    *,
    raw_id: str | None = None,
    content_hash: str | None = None,
) -> str:
    """Write one parsed session via the live writer; return the archive ``session_id``.

    Runs the sync writer in a worker thread so it composes with async tests
    that read back through ``backend.connection()``.
    """
    return await asyncio.to_thread(
        write_session_sync,
        backend.db_path,
        session,
        raw_id=raw_id,
        content_hash=content_hash,
        archive_root=backend._source_db_path.parent,
    )


def write_index_session(
    archive: object,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
) -> str:
    """Seed an index-only fixture through the canonical row writer.

    These fixtures intentionally have no raw acquisition to admit. Fixtures that
    need retained evidence use ``retained_replay.publish_retained_payload``.
    """
    db_path = getattr(archive, "index_db_path", None)
    if not isinstance(db_path, Path):
        raise TypeError("index-only fixture requires an ArchiveStore index_db_path")
    own_scope = getattr(archive, "index_mutation_scope", None)
    connection = getattr(archive, "_conn", None)
    if not callable(own_scope) or not isinstance(connection, sqlite3.Connection):
        raise TypeError("index-only fixture requires its ArchiveStore-owned Index transaction")
    acquired = None
    refs = ()
    publisher = getattr(archive, "_blob_publisher", None)
    if any(
        attachment.inline_bytes is not None or attachment.precomputed_blob is not None
        for attachment in session.attachments
    ):
        preacquire = getattr(archive, "_preacquire_attachment_blobs", None)
        if not callable(preacquire) or publisher is None:
            raise TypeError("index-only attachment fixture requires an archive-owned blob publisher")
        acquired, refs = preacquire(session, source_path=f"session:{session.provider_session_id}", acquired_at_ms=0)
        publisher.flush()
    try:
        from contextlib import contextmanager

        @contextmanager
        def publication_scope(seal: object) -> Iterator[object]:
            with own_scope(prepared_seal=seal) as scope:
                yield scope
                if refs:
                    pending = getattr(archive, "_pending_index_blob_receipts", None)
                    if not isinstance(pending, list):
                        raise TypeError("index-only attachment fixture requires archive receipt handling")
                    pending.extend(
                        (ref.publication_receipt_id, ref.blob_hash)
                        for ref in refs
                        if ref.publication_receipt_id is not None
                    )

        session_id = write_fixture_index_session(
            connection,
            session,
            content_hash=content_hash if content_hash is not None else session_content_hash(session),
            preacquired_attachment_blobs=acquired,
            index_scope_factory=publication_scope,
        )
        return session_id
    except BaseException:
        if publisher is not None:
            publisher.discard_pending()
        raise


__all__ = [
    "prepared_live_convergence_owner",
    "ingest_session",
    "write_index_session",
    "write_session_counts_sync",
    "write_session_sync",
]


@asynccontextmanager
async def prepared_live_convergence_owner(
    root: Path,
    *,
    compute_adapter: BoundedComputeAdapter | None = None,
    write_coordinator: DaemonWriteCoordinator | None = None,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
) -> AsyncIterator[RawObservationConvergenceOwner]:
    """Borrow the actual daemon preparation owner for a complete live pass."""
    import sys

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge

    compute = compute_adapter if compute_adapter is not None else BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = write_coordinator
    try:
        if coordinator is None:
            coordinator = DaemonWriteCoordinator(archive_root=root)
        owner = RawObservationConvergenceOwner(
            root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            write_coordinator=coordinator,
            validation_mode=validation_mode,
        )
        yield owner
    finally:
        primary = sys.exception()
        cleanup_failures: list[BaseException] = []
        coordinator_settled = True
        if write_coordinator is None and coordinator is not None:
            try:
                # This owner sends cleanup requests to the original creator's
                # terminal mailbox. Settle it before joining that creator.
                coordinator_settled = await coordinator.shutdown(timeout=float("inf"))
                if not coordinator_settled:
                    raise RuntimeError("prepared live coordinator did not settle")
            except BaseException as failure:
                coordinator_settled = False
                cleanup_failures.append(failure)
        if compute_adapter is None:
            try:
                closing = asyncio.create_task(asyncio.to_thread(compute.shutdown, wait=coordinator_settled))
                while not closing.done():
                    try:
                        await asyncio.shield(closing)
                    except asyncio.CancelledError as failure:
                        cleanup_failures.append(failure)
                closing.result()
            except BaseException as failure:
                cleanup_failures.append(failure)
        if cleanup_failures:
            if primary is not None:
                cleanup_failures.insert(0, primary)
            raise BaseExceptionGroup("prepared live fixture settlement failed", cleanup_failures)


def run_owned_append_plans(root: Path, owner: object, plans: list[_AppendPlan]) -> _AppendResult:
    """Acquire and converge append plans through the canonical daemon owner.

    The owner prepares and publishes each acquired raw on its admitted compute
    creator, exactly as the watcher's append runner does; no fixture-local
    convergence callback is substituted.
    """
    from typing import Any, cast

    async def run() -> _AppendResult:
        async with prepared_live_convergence_owner(root) as convergence:
            return await convergence.ingest_append_plans(cast(Any, owner), plans)

    return asyncio.run(run())
