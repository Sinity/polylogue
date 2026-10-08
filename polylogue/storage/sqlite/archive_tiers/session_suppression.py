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

The matching Index mutation scope owns its User reader for one commit window.
Active scopes borrow their already-retained reference-seal observer; inactive
generations open one creator-owned reader against their declared archive root.
A genuinely standalone scope has no durable tier. Other direct callers use a
single readonly context against an attached or neighboring User tier. A
canonical archive missing its required User tier refuses before a write.

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
from polylogue.storage.sqlite.connection_profile import readonly_connection_context
from polylogue.storage.sqlite.reference_seal import current_index_mutation_scope

__all__ = [
    "SuppressionRefusal",
    "SuppressionRefusalTotals",
    "SuppressionTierUnreadableError",
    "record_suppression_refusal",
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
        if any((candidate.parent / marker).exists() for marker in ("source.db", "audit.db", ".polylogue-format.json")):
            raise SuppressionTierUnreadableError("declared archive is missing its required durable User tier")
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


#: Matched on ``target_ref``/``kind`` rather than the derived assertion id so
#: the index writer never has to import the user-tier id recipe (which would
#: pull ``user_write`` into its schema-identity closure). ``target_ref`` is
#: written as ``session:<session_id>`` by ``upsert_suppression`` and is the
#: leading column of ``idx_assertions_target_kind_status_visibility``.
_SUPPRESSION_LOOKUP_SQL = (
    "SELECT 1 FROM {schema}assertions WHERE target_ref = ? AND kind = ? AND COALESCE(status, '') != 'deleted' LIMIT 1"
)


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
    scope = current_index_mutation_scope()
    if scope is not None:
        scope.require_new_work(conn)
        reader = scope.user_reader()
        return _reader_suppresses(reader, session_id) if reader is not None else False
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
    try:
        with readonly_connection_context(resolution.path, validate_schema=False) as user_conn:
            return _reader_suppresses(user_conn, session_id)
    except sqlite3.Error as exc:
        raise SuppressionTierUnreadableError(f"cannot read session suppressions from {resolution.path}: {exc}") from exc


def _reader_suppresses(user_conn: sqlite3.Connection, session_id: str) -> bool:
    if not _assertions_table_present(user_conn, ""):
        return False
    try:
        row = user_conn.execute(
            _SUPPRESSION_LOOKUP_SQL.format(schema=""),
            (f"session:{session_id}", str(AssertionKind.SUPPRESSION)),
        ).fetchone()
    except sqlite3.Error as exc:
        raise SuppressionTierUnreadableError(
            f"cannot read session suppressions from the owned User reader: {exc}"
        ) from exc
    return row is not None
