"""Compose the storage-owned message FTS derivation for daemon convergence.

The operations boundary owns active-generation routing.  The FTS adapter stays
in ``storage`` and speaks the daemon kernel protocol structurally, so neither
the FTS SQL nor its output relation depends on daemon implementation details.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from polylogue.daemon.derivation import DerivationFrame
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.fts.derivation import FtsDerivationAdapter
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection

__all__ = ["FTS_ORPHAN_INTERVAL_S", "make_fts_derivation", "make_fts_frame"]


# Global residue is defense-in-depth for interrupted/malformed historical
# writes. Session partitions are the normal path, so this process-local cadence
# does not create durable freshness authority or suppress required work.
FTS_ORPHAN_INTERVAL_S = 60.0 * 60.0


def make_fts_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
    orphan_interval_s: float | None = FTS_ORPHAN_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
) -> FtsDerivationAdapter:
    """Return one active-generation FTS adapter for the shared daemon owner."""
    del index_db_path

    def active_generation_path() -> Path:
        return resolve_active_index_path(archive_root).resolve()

    def read_connection() -> sqlite3.Connection:
        return open_readonly_connection(active_generation_path(), timeout_class="background-read")

    def write_connection() -> sqlite3.Connection:
        return open_daemon_connection(active_generation_path(), archive_root=archive_root)

    def generation_binding() -> str:
        return str(active_generation_path())

    return FtsDerivationAdapter(
        read_connection,
        write_connection,
        generation_binding=generation_binding,
        orphan_interval_s=orphan_interval_s,
        monotonic=monotonic,
    )


def make_fts_frame(
    index_db_path: Path,
    *,
    archive_root: Path,
    scope: Sequence[str] | None = None,
) -> DerivationFrame:
    """Bind an FTS pass to the active path, generation and identity recipe."""
    del index_db_path
    adapter = FtsDerivationAdapter()
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=f"index-generation:{resolve_active_index_path(archive_root).resolve()}",
        recipe_versions={adapter.domain: adapter.recipe_id},
        scope=None if scope is None else tuple(dict.fromkeys(str(session_id) for session_id in scope)),
    )
