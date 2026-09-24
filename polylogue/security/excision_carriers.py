"""Declared excision reach for session-addressable source-tier relations.

Session excision has one recurring defect shape (polylogue-14ucm, and the
seven routes closed before it): a durable relation carries evidence that is
addressable by a session key, ``apply_session_excision`` does not reach it,
and the receipt still reads complete. Every instance so far --- hook events
(polylogue-bhhsa), fact-tier TODO snapshots (polylogue-si5kj), container
payloads held live through ``source_items`` (polylogue-q4f6d) --- was found
by a human reading the schema, not by the archive refusing.

The membership list was hand-maintained, so a *new* session-keyed relation
was silently exempt (polylogue-9lrqs). This module inverts that: the live
source-tier schema is scanned for the two declared session-key shapes, and
any table carrying one that this registry does not classify makes excision
**refuse** rather than report success.

Two session-key shapes, both structural:

``raw_id``
    The relation hangs off one raw acquisition. Reached transitively when
    ``raw_sessions`` is deleted *if and only if* its foreign key to
    ``raw_sessions(raw_id)`` carries ``ON DELETE CASCADE``. A ``SET NULL``
    foreign key leaves the row --- and any blob it owns --- behind, which is
    exactly how ``source_items`` kept excised bytes GC-rooted.

``(origin, session_native_id)``
    The relation is addressed by the session key directly and has no
    ``raw_sessions`` row at all, so no raw target can ever reach it. Every
    such relation must be deleted explicitly by the apply.

The cascade claim is not taken on trust: :func:`audit_session_carriers`
re-derives it from the live ``PRAGMA foreign_key_list`` and reports a
declaration that no longer matches as a misdeclaration, so replacing a
``CASCADE`` with ``SET NULL`` refuses instead of quietly stranding rows.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from polylogue.storage.sqlite.archive_tiers.source import RETIRED_SOURCE_SCHEMA_OBJECTS

#: Column that makes a relation reachable through a raw acquisition.
RAW_KEY_COLUMN: Final = "raw_id"

#: Columns that together address a relation by session key with no raw row.
SESSION_KEY_COLUMNS: Final = frozenset({"origin", "session_native_id"})


class CarrierReach(StrEnum):
    """How an excision of one session reaches a session-keyed relation."""

    #: Deleted by the database when the raw acquisition row goes, through an
    #: ``ON DELETE CASCADE`` foreign key to ``raw_sessions(raw_id)``.
    RAW_CASCADE = "raw-cascade"
    #: Deleted by ``apply_session_excision`` itself, by name.
    EXCISED = "excised"
    #: A container that can cover records of several sessions. Its members
    #: are excised per member; the container row survives only while another
    #: member is still live, and the receipt names it when it does.
    CONTAINER = "container"
    #: A retired schema object that only a migrated historical source tier
    #: still carries. Fresh generations omit it, so it never appears in a
    #: fresh tier's audit; it holds derived bookkeeping receipts or telemetry,
    #: not acquired payload. Where a migrated tier does carry it,
    #: ``apply_session_excision`` still deletes its rows by name.
    RETIRED = "retired"
    #: Content-free terminal evidence deliberately retained after carrier
    #: bytes are erased, so a retry cannot recreate them.
    TOMBSTONE = "tombstone"


@dataclass(frozen=True, slots=True)
class SessionCarrier:
    """One session-addressable source-tier relation and its declared reach."""

    table: str
    reach: CarrierReach
    reason: str


def _carriers(*specs: SessionCarrier) -> dict[str, SessionCarrier]:
    return {spec.table: spec for spec in specs}


#: The declared reach of every session-keyed source-tier relation.
#:
#: Adding a table with a ``raw_id`` column or an ``(origin,
#: session_native_id)`` pair without adding it here makes excision refuse ---
#: that refusal is the point (polylogue-9lrqs).
SESSION_CARRIERS: Final[dict[str, SessionCarrier]] = _carriers(
    SessionCarrier(
        "raw_sessions",
        CarrierReach.EXCISED,
        "the acquisition itself; deleted with its blob_refs and an excised_content marker",
    ),
    SessionCarrier(
        "pending_accepted_marker_inputs",
        CarrierReach.EXCISED,
        "sealed pending marker bytes are explicitly erased by session excision",
    ),
    SessionCarrier(
        "accepted_marker_inputs",
        CarrierReach.EXCISED,
        "sealed accepted marker bytes are explicitly erased by session excision",
    ),
    SessionCarrier(
        "excised_marker_inputs",
        CarrierReach.TOMBSTONE,
        "content-free terminal marker-carrier evidence is deliberately retained",
    ),
    SessionCarrier(
        "raw_hook_events",
        CarrierReach.EXCISED,
        "hook payloads are addressed by (origin, session_native_id) and carry no raw row (polylogue-bhhsa)",
    ),
    SessionCarrier(
        "otlp_spans",
        CarrierReach.RETIRED,
        "retired inbound span storage (polylogue-enrpa); excised by name where a migrated tier still carries it",
    ),
    SessionCarrier(
        "source_items",
        CarrierReach.CONTAINER,
        "one manifest item can cover many sessions; blob-liveness owner (polylogue-q4f6d)",
    ),
    SessionCarrier(
        "source_item_raw_members",
        CarrierReach.CONTAINER,
        "per-record members of a container item; the member row is excised with its raw",
    ),
    SessionCarrier(
        "raw_artifacts",
        CarrierReach.RAW_CASCADE,
        "artifact taxonomy rows for one acquisition",
    ),
    SessionCarrier(
        "raw_session_memberships",
        CarrierReach.RAW_CASCADE,
        "logical-source membership of one acquisition",
    ),
    SessionCarrier(
        "raw_membership_census",
        CarrierReach.RAW_CASCADE,
        "derived membership census for one acquisition",
    ),
    SessionCarrier(
        "raw_container_coordinates",
        CarrierReach.RAW_CASCADE,
        "container coordinates of one acquisition",
    ),
    SessionCarrier(
        "raw_capture_observations",
        CarrierReach.RAW_CASCADE,
        "capture-time observations of one acquisition",
    ),
    SessionCarrier(
        "raw_authority_verdicts",
        CarrierReach.RAW_CASCADE,
        "derived raw-authority verdict for one acquisition",
    ),
    SessionCarrier(
        "raw_authority_parser_census",
        CarrierReach.RAW_CASCADE,
        "derived parser census for one acquisition",
    ),
    SessionCarrier(
        "raw_legacy_append_resynthesis_receipts",
        CarrierReach.RAW_CASCADE,
        "resynthesis receipt for one acquisition",
    ),
)


class UnclassifiedSessionCarrierError(RuntimeError):
    """Raised when the live source tier carries an unreachable session key.

    Excision must not report success over a relation whose reach nobody
    declared. The message names each offending table so the fix is to
    classify it in :data:`SESSION_CARRIERS` (and, for anything but
    ``raw-cascade``, to make the apply actually reach it).
    """

    def __init__(self, *, undeclared: tuple[str, ...], misdeclared: tuple[str, ...]) -> None:
        self.undeclared = undeclared
        self.misdeclared = misdeclared
        parts: list[str] = []
        if undeclared:
            parts.append("carrier(s) not excisable (no declared excision reach): " + ", ".join(undeclared))
        if misdeclared:
            parts.append(
                "carrier(s) declared raw-cascade without an ON DELETE CASCADE "
                "foreign key to raw_sessions(raw_id): " + ", ".join(misdeclared)
            )
        super().__init__(
            "; ".join(parts) + ". Declare each in polylogue.security.excision_carriers.SESSION_CARRIERS "
            "and make apply_session_excision reach it before excising."
        )


@dataclass(frozen=True, slots=True)
class CarrierAudit:
    """Result of comparing the live source schema against the registry."""

    declared: tuple[str, ...]
    undeclared: tuple[str, ...]
    misdeclared: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.undeclared or self.misdeclared)

    def raise_if_unreachable(self) -> None:
        if not self.ok:
            raise UnclassifiedSessionCarrierError(undeclared=self.undeclared, misdeclared=self.misdeclared)


def _table_columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall())


def _cascades_from_raw_sessions(conn: sqlite3.Connection, table: str) -> bool:
    for row in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall():
        if str(row[2]) == "raw_sessions" and str(row[3]) == RAW_KEY_COLUMN and str(row[6]).upper() == "CASCADE":
            return True
    return False


def _live_tables(conn: sqlite3.Connection) -> Iterable[str]:
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall():
        yield str(name)


def session_keyed_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Live source-tier tables carrying one of the declared session-key shapes."""
    found: list[str] = []
    for table in _live_tables(conn):
        columns = _table_columns(conn, table)
        if RAW_KEY_COLUMN in columns or columns >= SESSION_KEY_COLUMNS:
            found.append(table)
    return tuple(found)


def audit_session_carriers(conn: sqlite3.Connection) -> CarrierAudit:
    """Compare every live session-keyed relation against its declared reach.

    Derived from the live schema, not from a copy of the DDL, so a table that
    only a migrated historical tier carries is classified too --- and a table
    created outside the declared DDL (the anti-vacuity case) is reported as
    undeclared rather than skipped.
    """
    declared: list[str] = []
    undeclared: list[str] = []
    misdeclared: list[str] = []
    for table in session_keyed_tables(conn):
        carrier = SESSION_CARRIERS.get(table)
        if carrier is None:
            if f"table:{table}" in RETIRED_SOURCE_SCHEMA_OBJECTS:
                declared.append(table)
                continue
            undeclared.append(table)
            continue
        declared.append(table)
        if carrier.reach is CarrierReach.RAW_CASCADE and not _cascades_from_raw_sessions(conn, table):
            misdeclared.append(table)
    return CarrierAudit(
        declared=tuple(declared),
        undeclared=tuple(undeclared),
        misdeclared=tuple(misdeclared),
    )


__all__ = [
    "RAW_KEY_COLUMN",
    "SESSION_CARRIERS",
    "SESSION_KEY_COLUMNS",
    "CarrierAudit",
    "CarrierReach",
    "SessionCarrier",
    "UnclassifiedSessionCarrierError",
    "audit_session_carriers",
    "session_keyed_tables",
]
