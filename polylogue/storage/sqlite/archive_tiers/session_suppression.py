"""Durable session suppression consulted at the index write choke point.

An identity-preserving reset (``polylogue reset --session`` /
``--source``) deliberately leaves the acquired raw evidence in the durable
``source.db`` and records the operator's deletion as a durable *suppression
assertion* in ``user.db``. The rebuildable ``index.db`` row is dropped.

That split is only a privacy guarantee if every route that can put a
session row back into ``index.db`` consults the suppression first.
``write_parsed_session_to_archive`` is the single choke point shared by
live ingest and full replay/reindex (see ``CLAUDE.md``), so the check lives
there and this module owns the lookup.

Resolution is *ambient on the connection* rather than an extra writer
argument on purpose: a required argument would be forgettable by exactly
the kind of new replay path that caused this defect, and an optional one
defaulting to "no guard" reproduces it. The user tier is found from the
index connection itself:

1. an already-``ATTACH``ed ``user_tier`` schema (``ArchiveStore``'s own
   read path attaches it), otherwise
2. a ``user.db`` sitting beside the connection's ``main`` database file,
   or one directory up (the promoted-generation layout, where the active
   index lives in a generation directory under the archive root).

A connection with no reachable ``user.db`` (``:memory:`` archives, bare
index fixtures) has no durable tier that could hold a tombstone, so it is
resolved as "nothing suppressed" -- not as a refusal. A ``user.db`` that
exists but cannot be read is a refusal: failing open there would resurrect
content exactly when the durable tier is in trouble.

Refusals are counted, never silent. :func:`suppression_refusal_scope`
gives a replay run a totals window, and every refusal also emits
``storage.session_suppression.write_refused``.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.core.enums import AssertionKind
from polylogue.logging import WARNING, emit

__all__ = [
    "SuppressionRefusal",
    "SuppressionRefusalTotals",
    "SuppressionTierUnreadableError",
    "record_suppression_refusal",
    "reset_suppression_caches",
    "session_write_is_suppressed",
    "suppression_refusal_scope",
    "suppression_resolution_for_connection",
]


class SuppressionTierUnreadableError(RuntimeError):
    """A durable ``user.db`` exists but its suppressions cannot be read.

    Raised instead of proceeding: a write that cannot prove the operator did
    not delete this session is the resurrection defect this module exists to
    prevent.
    """


@dataclass(frozen=True)
class SuppressionRefusal:
    """One index write refused because the operator tombstoned the session."""

    session_id: str
    reason: str = "durable user.db suppression assertion"


@dataclass
class SuppressionRefusalTotals:
    """Counted refusals inside one :func:`suppression_refusal_scope`."""

    refusals: list[SuppressionRefusal] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.refusals)

    @property
    def session_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for refusal in self.refusals:
            seen.setdefault(refusal.session_id, None)
        return tuple(seen)

    def describe(self) -> str:
        """One line naming what was skipped, for a run's own receipt."""
        if not self.refusals:
            return "0 sessions skipped as tombstoned"
        ids = self.session_ids
        sample = ", ".join(ids[:5])
        more = f" (+{len(ids) - 5} more)" if len(ids) > 5 else ""
        return f"{len(ids)} session(s) skipped as tombstoned: {sample}{more}"


_state = threading.local()


def _scopes() -> list[SuppressionRefusalTotals]:
    scopes: list[SuppressionRefusalTotals] | None = getattr(_state, "scopes", None)
    if scopes is None:
        scopes = []
        _state.scopes = scopes
    return scopes


@contextmanager
def suppression_refusal_scope() -> Iterator[SuppressionRefusalTotals]:
    """Collect every suppression refusal raised on this thread in the body.

    A replay/rebuild run wraps its writes in this so it can report what it
    deliberately did not restore. Scopes nest; an inner scope's refusals are
    also recorded by every enclosing scope.
    """
    totals = SuppressionRefusalTotals()
    scopes = _scopes()
    scopes.append(totals)
    try:
        yield totals
    finally:
        scopes.remove(totals)


def record_suppression_refusal(session_id: str, *, route: str) -> SuppressionRefusal:
    """Count and log one refusal. ``route`` names the write path that asked."""
    refusal = SuppressionRefusal(session_id=session_id)
    for totals in _scopes():
        totals.refusals.append(refusal)
    emit(
        "storage.session_suppression.write_refused",
        level=WARNING,
        outcome="degraded",
        session_id=session_id,
        route=route,
        reason=refusal.reason,
    )
    return refusal


def _attached_schemas(conn: sqlite3.Connection) -> dict[str, str]:
    # Deliberately not guarded: a connection that cannot answer
    # ``PRAGMA database_list`` cannot perform the write this check gates
    # either, and swallowing the error here would resolve as "nothing
    # suppressed" -- the fail-open shape this module exists to remove.
    rows = conn.execute("PRAGMA database_list").fetchall()
    schemas: dict[str, str] = {}
    for row in rows:
        name = str(row[1])
        filename = str(row[2] or "")
        schemas[name] = filename
    return schemas


def _user_db_path_beside(main_file: str) -> Path | None:
    if not main_file:
        return None
    main_path = Path(main_file)
    for candidate in (main_path.parent / "user.db", main_path.parent.parent / "user.db"):
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class SuppressionResolution:
    """Where this connection's durable suppressions are read from."""

    #: ``ATTACH`` alias to query, when the user tier is already mounted.
    schema: str | None = None
    #: Standalone ``user.db`` to open read-only, otherwise.
    path: Path | None = None

    @property
    def reachable(self) -> bool:
        return self.schema is not None or self.path is not None


def suppression_resolution_for_connection(conn: sqlite3.Connection) -> SuppressionResolution:
    """Locate the durable user tier for an index connection."""
    schemas = _attached_schemas(conn)
    for alias in ("user_tier", "user_debt"):
        if alias in schemas:
            return SuppressionResolution(schema=alias)
    # Deliberately not memoized: an archive initialized tier by tier can gain
    # its user.db after this connection's first write, and a stat is nothing
    # beside the session write it guards.
    path = _user_db_path_beside(schemas.get("main", ""))
    if path is None:
        return SuppressionResolution()
    return SuppressionResolution(path=path)


def reset_suppression_caches() -> None:
    """Close and drop memoized read-only ``user.db`` handles.

    Tests build and tear down archive roots at the same temporary paths, so
    the pooled handles have to be droppable.
    """
    with _connection_cache_lock:
        for conn in _connection_cache.values():
            conn.close()
        _connection_cache.clear()


#: Matched on ``target_ref``/``kind`` rather than the derived assertion id so
#: the index writer never has to import the user-tier id recipe (which would
#: pull ``user_write`` into its schema-identity closure). ``target_ref`` is
#: written as ``session:<session_id>`` by ``upsert_suppression`` and is the
#: leading column of ``idx_assertions_target_kind_status_visibility``.
_SUPPRESSION_LOOKUP_SQL = (
    "SELECT 1 FROM {schema}assertions WHERE target_ref = ? AND kind = ? AND COALESCE(status, '') != 'deleted' LIMIT 1"
)

_connection_cache: dict[str, sqlite3.Connection] = {}
_connection_cache_lock = threading.Lock()


def _cached_user_connection(path: Path) -> sqlite3.Connection:
    key = str(path)
    with _connection_cache_lock:
        cached = _connection_cache.get(key)
        if cached is not None:
            return cached
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0, check_same_thread=False)
        except sqlite3.Error as exc:
            raise SuppressionTierUnreadableError(
                f"cannot open the durable user tier at {path} to check session suppressions: {exc}"
            ) from exc
        _connection_cache[key] = conn
        return conn


def _assertions_table_present(conn: sqlite3.Connection, schema: str) -> bool:
    """Whether this tier holds the assertions table at all.

    An absent table is a user tier that was never initialized -- there is no
    tombstone it could be hiding. A *failing* catalog read is not that, so it
    is raised as a refusal rather than resolved as "nothing suppressed".
    """
    prefix = f"{schema}." if schema else ""
    try:
        row = conn.execute(
            f"SELECT 1 FROM {prefix}sqlite_master WHERE type = 'table' AND name = 'assertions' LIMIT 1"
        ).fetchone()
    except sqlite3.Error as exc:
        raise SuppressionTierUnreadableError(
            f"cannot read the user tier catalog to check session suppressions: {exc}"
        ) from exc
    return row is not None


def session_write_is_suppressed(conn: sqlite3.Connection, session_id: str) -> bool:
    """Return whether ``session_id`` carries a live durable suppression."""
    target_ref = f"session:{session_id}"
    resolution = suppression_resolution_for_connection(conn)
    if not resolution.reachable:
        return False
    if resolution.schema is not None:
        if not _assertions_table_present(conn, resolution.schema):
            return False
        sql = _SUPPRESSION_LOOKUP_SQL.format(schema=f"{resolution.schema}.")
        try:
            row = conn.execute(sql, (target_ref, str(AssertionKind.SUPPRESSION))).fetchone()
        except sqlite3.Error as exc:
            raise SuppressionTierUnreadableError(
                f"cannot read session suppressions from the attached user tier: {exc}"
            ) from exc
        return row is not None

    assert resolution.path is not None
    user_conn = _cached_user_connection(resolution.path)
    if not _assertions_table_present(user_conn, ""):
        return False
    try:
        row = user_conn.execute(
            _SUPPRESSION_LOOKUP_SQL.format(schema=""),
            (target_ref, str(AssertionKind.SUPPRESSION)),
        ).fetchone()
    except sqlite3.Error as exc:
        raise SuppressionTierUnreadableError(f"cannot read session suppressions from {resolution.path}: {exc}") from exc
    return row is not None
