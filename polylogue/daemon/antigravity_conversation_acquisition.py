"""Periodic daemon acquisition of Antigravity's real ``.pb`` conversations.

The live daemon's file-watcher model (``sources/live/watcher.py``) only
observes ``*.metadata.json`` sidecar changes for the ``antigravity`` source
(``WatchSource(name="antigravity", ..., suffixes=(".metadata.json",))``) --
the real ``conversations/*.pb`` trajectory files are structurally invisible
to it (no suffix match), and even if they were watched, the watcher's
per-file streaming-JSON model has no hook to drive the local Antigravity
language-server subprocess the ``.pb`` -> markdown conversion requires.

The batch importer (``pipeline/services/archive_ingest.py``) already has
real acquisition code for this
(``sources/parsers/antigravity.py::iter_language_server_exports``, wired via
PR #3441 / polylogue-eo81), but the live daemon never calls it: only a
one-shot ``polylogue import`` run does. Result (polylogue-3m3de, verified
live 2026-08-03): the archive kept admitting metadata-only fragment sessions
forever (116 -> 232 rows and still growing after #3441 merged) while the 44
real conversations (314MB of ``.pb`` trajectories) sat permanently
un-acquired by the running daemon.

This module closes that gap with a bounded, quiet-cadence reconciliation
loop -- the same shape as ``embedding_backlog.py``'s periodic drains -- that
finds cascades not yet represented in ``raw_sessions`` and acquires only
those, instead of re-running the language-server conversion for the whole
corpus on every tick.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import Origin
from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

logger = get_logger(__name__)

#: Reconciliation cadence. Antigravity conversation growth happens at human
#: pace (a chat session ends, a ``.pb`` file is written), not a hot ingest
#: source, so a long interval keeps the periodic subprocess-driven export
#: cheap in aggregate.
ANTIGRAVITY_ACQUISITION_INTERVAL_SECONDS = 1800

#: Bound the number of cascades converted per tick so one reconciliation
#: pass never issues an unbounded number of language-server RPCs (each
#: starts/queries a local HTTP loopback subprocess) in a single call.
ANTIGRAVITY_ACQUISITION_MAX_PER_TICK = 25


async def periodic_antigravity_conversation_acquisition_check(
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically acquire not-yet-acquired Antigravity ``.pb`` conversations."""
    from polylogue.daemon.cli import _await_catch_up_gate
    from polylogue.paths import archive_root

    root = archive_root()
    await _await_catch_up_gate(catch_up_complete, loop_name="antigravity conversation acquisition")
    while True:
        await asyncio.sleep(ANTIGRAVITY_ACQUISITION_INTERVAL_SECONDS)
        try:
            from polylogue.daemon.write_coordinator import daemon_write_coordinator

            acquired = await daemon_write_coordinator().run_sync(
                "maintenance.antigravity_conversation_acquisition",
                acquire_antigravity_conversations_once,
                root,
            )
            if acquired:
                logger.info(
                    "antigravity: acquired %d real conversation(s) from conversations/*.pb",
                    acquired,
                )
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info(
                    "antigravity: archive busy; retrying conversation acquisition on next tick: %s",
                    exc,
                )
                continue
            logger.warning("antigravity: conversation acquisition check failed", exc_info=True)
        except Exception:
            logger.warning("antigravity: conversation acquisition check failed", exc_info=True)


def _acquired_pb_cascade_ids(source_db: Path) -> frozenset[str]:
    """Return cascade ids already present in ``raw_sessions`` for antigravity ``.pb`` sources."""
    if not source_db.exists():
        return frozenset()
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(source_db)
    try:
        rows = conn.execute(
            "SELECT source_path FROM raw_sessions WHERE origin = ? AND source_path LIKE ?",
            (Origin.ANTIGRAVITY_SESSION.value, "%.pb"),
        ).fetchall()
    finally:
        conn.close()
    return frozenset(Path(row[0]).stem for row in rows)


def acquire_antigravity_conversations_once(archive_root: Path) -> int:
    """Acquire one bounded batch of not-yet-acquired ``.pb`` conversations.

    Returns the number of conversations newly written to ``raw_sessions``/
    ``index.db``. A ``0`` return is the common steady-state case (nothing
    new since the last tick, or Antigravity/its language server not present
    on this host), not a failure.
    """
    from polylogue.paths import antigravity_path
    from polylogue.pipeline.services.archive_ingest import (
        _archive_raw_payload,
        _archive_raw_source_index,
        _archive_raw_source_path,
    )
    from polylogue.sources.source_parsing import iter_antigravity_language_server_sessions
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    root = antigravity_path()
    conversations_dir = root / "conversations"
    if not conversations_dir.is_dir():
        return 0
    all_ids = frozenset(path.stem for path in conversations_dir.glob("*.pb"))
    if not all_ids:
        return 0

    acquired_ids = _acquired_pb_cascade_ids(archive_root / "source.db")
    missing_ids = sorted(all_ids - acquired_ids)
    if not missing_ids:
        return 0
    batch_ids = frozenset(missing_ids[:ANTIGRAVITY_ACQUISITION_MAX_PER_TICK])

    source = Source(name="antigravity", path=root)
    blob_root = archive_root / "blob"
    blob_publisher = ArchiveBlobPublisher(archive_root / "source.db", blob_root)
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)

    admitted = 0
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for raw_data, session in iter_antigravity_language_server_sessions(
            source,
            capture_raw=True,
            blob_root=blob_root,
            blob_store=blob_publisher,
            only_cascade_ids=batch_ids,
        ):
            payload = _archive_raw_payload(raw_data, session, blob_root=blob_root)
            source_path = _archive_raw_source_path(raw_data, source)
            source_index = _archive_raw_source_index(raw_data)
            try:
                archive.write_raw_and_parsed_result(
                    session,
                    payload=payload,
                    source_path=source_path,
                    acquired_at_ms=acquired_at_ms,
                    source_index=source_index,
                    blob_publication_receipt_id=(
                        raw_data.blob_publication_receipt_id if raw_data is not None else None
                    ),
                )
                admitted += 1
            except ContentExcisedError as exc:
                # The archive can forget on purpose (polylogue-27m): skip
                # only this one cascade and continue the batch.
                logger.info("antigravity: skipping durably excised conversation: %s", exc)
                continue
    return admitted


__all__ = [
    "ANTIGRAVITY_ACQUISITION_INTERVAL_SECONDS",
    "ANTIGRAVITY_ACQUISITION_MAX_PER_TICK",
    "acquire_antigravity_conversations_once",
    "periodic_antigravity_conversation_acquisition_check",
]
