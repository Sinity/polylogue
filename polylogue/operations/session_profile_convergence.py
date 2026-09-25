"""Compose storage-backed session-profile derivation for the daemon owner."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

from polylogue.daemon.convergence_stages import _archive_hot_insight_session_ids
from polylogue.daemon.derivation import DerivationFrame
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.derived.session.derivation import (
    SESSION_PROFILE_DOMAIN,
    SESSION_PROFILE_RECIPE_VERSION,
    SessionProfileDerivation,
    bound_session_profile_partitions,
)
from polylogue.storage.derived.session.marker_domain import (
    SESSION_MARKER_DOMAIN,
    SESSION_MARKER_RECIPE_VERSION,
    SessionMarkerDerivation,
)
from polylogue.storage.derived.session.summary import (
    SESSION_SUMMARY_DOMAIN,
    SESSION_SUMMARY_RECIPE_VERSION,
    SessionSummaryDerivation,
)
from polylogue.storage.derived.session.usage_rollup import (
    SESSION_USAGE_ROLLUP_DOMAIN,
    SessionUsageRollupDerivation,
    session_usage_rollup_recipe_version,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection, read_frame

__all__ = [
    "make_session_marker_derivation",
    "make_session_profile_derivation",
    "make_session_profile_frame",
    "make_session_summary_derivation",
    "make_session_usage_rollup_derivation",
    "session_profile_partition_status",
]


def session_profile_partition_status(archive_root: Path, session_id: str) -> tuple[str, str | None, int | None]:
    """Classify one profile and return its row identity from one read frame."""
    with read_frame(resolve_active_index_path(archive_root)) as frame:
        conn = frame.connection
        conn.execute("BEGIN")
        status = bound_session_profile_partitions(
            conn,
            (session_id,),
            materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        )[session_id]
        row = conn.execute(
            "SELECT input_content_hash, materializer_version FROM session_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()
        return (
            status,
            None if row is None or row[0] is None else str(row[0]),
            None if row is None else int(row[1]),
        )


def _session_derivation_connections(
    archive_root: Path,
) -> tuple[
    Callable[[], sqlite3.Connection],
    Callable[[], sqlite3.Connection],
    Callable[[], str],
]:
    """Open the active index generation for every session-partition adapter."""

    def active_generation_path() -> Path:
        return resolve_active_index_path(archive_root).resolve()

    def read_connection() -> sqlite3.Connection:
        return open_readonly_connection(active_generation_path(), timeout_class="background-read")

    def write_connection() -> sqlite3.Connection:
        return open_daemon_connection(active_generation_path(), archive_root=archive_root)

    def generation_binding() -> str:
        return str(active_generation_path())

    return read_connection, write_connection, generation_binding


def _session_scope(frame: object) -> Sequence[str] | None:
    value = getattr(frame, "scope", None)
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise TypeError("session derivation frame scope must be a tuple of session ids or None")
    return tuple(str(item) for item in value)


def make_session_summary_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
) -> SessionSummaryDerivation:
    """Build the canonical counter partition adapter for the active generation."""
    del index_db_path
    read_connection, write_connection, generation_binding = _session_derivation_connections(archive_root)
    return SessionSummaryDerivation(
        read_connection,
        write_connection,
        session_scope=_session_scope,
        generation_binding=generation_binding,
    )


def _hot_session_quiet_key(
    read_connection: Callable[[], sqlite3.Connection],
    *,
    archive_root: Path,
    now: Callable[[], float],
) -> Callable[[object, str], bool]:
    """Defer a session whose source file is still being written.

    Shared by the profile and its usage-rollup prerequisite so the two agree
    about a hot session. If only one of them deferred, the other would churn
    on every pass while the pair could never converge.
    """

    def quiet_key(frame: object, session_id: str) -> bool:
        del frame
        conn = read_connection()
        try:
            return session_id in _archive_hot_insight_session_ids(
                conn,
                (session_id,),
                now=now(),
                archive_root=archive_root,
            )
        finally:
            conn.close()

    return quiet_key


def make_session_usage_rollup_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
    now: Callable[[], float],
) -> SessionUsageRollupDerivation:
    """Build the canonical usage reconciliation that precedes profile preparation."""
    del index_db_path
    read_connection, write_connection, generation_binding = _session_derivation_connections(archive_root)
    return SessionUsageRollupDerivation(
        read_connection,
        write_connection,
        session_scope=_session_scope,
        quiet_key=_hot_session_quiet_key(read_connection, archive_root=archive_root, now=now),
        generation_binding=generation_binding,
    )


def make_session_profile_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
    materializer_version: int | None = None,
    now: Callable[[], float],
) -> SessionProfileDerivation:
    """Build the storage adapter that the daemon owner drives for one generation."""
    del index_db_path
    if materializer_version is None:
        materializer_version = SESSION_INSIGHT_MATERIALIZER_VERSION

    read_connection, write_connection, generation_binding = _session_derivation_connections(archive_root)

    quiet_key = _hot_session_quiet_key(read_connection, archive_root=archive_root, now=now)

    return SessionProfileDerivation(
        read_connection,
        write_connection,
        materializer_version=materializer_version,
        session_scope=_session_scope,
        quiet_key=quiet_key,
        generation_binding=generation_binding,
    )


def make_session_marker_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
) -> SessionMarkerDerivation:
    """Build the marker-lowering adapter the daemon owner drives after profiles.

    polylogue-ylh7v: marker delivery is its own domain rather than a tail on
    profile publication. Marker availability is resolved when a connection is
    opened, never when this adapter is constructed. The daemon owner is
    composed once, before the first tier-creating call, so a construction-time
    ``exists()`` check would blind the whole process run to markers whenever
    ``user.db`` is created afterwards -- and ``user.db`` is the durable,
    irreplaceable tier, so silently deriving without it is the wrong failure
    direction.
    """
    del index_db_path
    source_db = archive_root / "source.db"
    user_db = archive_root / "user.db"

    def source_read_connection() -> sqlite3.Connection:
        return open_readonly_connection(source_db, timeout_class="background-read")

    def marker_read_connection() -> sqlite3.Connection:
        return open_readonly_connection(user_db, timeout_class="background-read")

    def marker_write_connection() -> sqlite3.Connection:
        return open_daemon_connection(user_db, archive_root=archive_root)

    return SessionMarkerDerivation(
        source_read_connection,
        marker_read_connection,
        marker_write_connection,
    )


def make_session_profile_frame(
    index_db_path: Path,
    *,
    archive_root: Path,
    scope: Sequence[str] | None,
    profile_demand_only: bool = False,
) -> DerivationFrame:
    """Describe one bounded pass against the active index generation."""
    del index_db_path
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=f"index-generation:{resolve_active_index_path(archive_root).resolve()}",
        recipe_versions={
            SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
            SESSION_USAGE_ROLLUP_DOMAIN: session_usage_rollup_recipe_version(),
            SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
            SESSION_MARKER_DOMAIN: SESSION_MARKER_RECIPE_VERSION,
        },
        scope=None if scope is None else tuple(dict.fromkeys(str(session_id) for session_id in scope)),
        profile_demand_only=profile_demand_only,
    )
