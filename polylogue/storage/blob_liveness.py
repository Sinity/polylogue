"""Canonical, descriptor-driven decisions about retained blob bytes.

The source tier owns raw and hook payloads plus the ``blob_refs`` ledger;
the active index owns attachment payloads. ``blob_refs`` is only evidence when
its typed referent still resolves. Receipt caches and observation IDs are not
part of this relation.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence, Set
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from polylogue.core.evidence import Measured, Unavailable
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.tier_access import capture_sqlite_read


#: An attachment the writer retained with an ambiguous owner is unreferenced by
#: construction: ``_write_attachments`` inserts it with ``ref_count`` 0 and
#: deliberately keeps it out of the ref-count sweep, so it never had a ref to
#: lose. Reference-closure debt and unreachable-coverage debt both mean "refs
#: went away without the sweep running", which only a non-zero ``ref_count``
#: witnesses. Every site that counts ref-less acquired attachments as debt uses
#: this predicate, so the two states cannot be conflated on one route.
def acquired_attachment_missing_ref_predicate(alias: str = "a", *, refs_table: str = "attachment_refs") -> str:
    """SQL predicate for an acquired attachment whose refs went away."""
    return (
        f"{alias}.acquisition_status = 'acquired'\n"
        f"              AND {alias}.ref_count != 0\n"
        f"              AND NOT EXISTS (SELECT 1 FROM {refs_table} r "
        f"WHERE r.attachment_id = {alias}.attachment_id)"
    )


class LivenessState(str, Enum):
    LIVE = "live"
    UNREFERENCED = "unreferenced"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class BlobOwner:
    """One authoritative blob-bearing owner or typed ledger referent."""

    tier: str
    table: str
    blob_column: str | None = None
    ref_type: str | None = None
    referent_column: str | None = None


# The sole map for per-hash inspection, bulk projection, schema preflight,
# integrity, sealing, GC, and blob-ref reconciliation.
BLOB_OWNERS: tuple[BlobOwner, ...] = (
    BlobOwner("source", "raw_sessions", blob_column="blob_hash"),
    BlobOwner("source", "raw_hook_events", blob_column="blob_hash"),
    # Linked materials retain their bytes independently of session parsing.
    BlobOwner("source", "material_observations", blob_column="blob_hash"),
    # Frozen inputs remain live before any decoder has admitted a raw record.
    BlobOwner("source", "source_items", blob_column="blob_hash"),
    # polylogue-8v4rm: an acquired attachment reference states, durably, that
    # these exact bytes were fetched and stored -- the source tier's own claim,
    # independent of whether a derived message ever linked them. Without this
    # owner the only liveness surface for those bytes was ``index.attachments``,
    # so a rebuildable tier decided whether durable bytes survived.
    BlobOwner("source", "source_attachments", blob_column="blob_hash"),
    BlobOwner("index", "attachments", blob_column="blob_hash"),
    BlobOwner("source", "raw_sessions", ref_type="raw_payload", referent_column="raw_id"),
    BlobOwner("source", "raw_sessions", ref_type="attachment", referent_column="raw_id"),
    BlobOwner("source", "raw_hook_events", ref_type="hook_payload", referent_column="hook_event_id"),
    BlobOwner("source", "history_sidecars", ref_type="sidecar", referent_column="sidecar_id"),
)

# A source owner introduced by an additive migration must not make older
# archives unreadable to backup/GC before that migration is applied.
_OPTIONAL_OWNER_TABLES = frozenset({"material_observations", "source_items"})


def validated_blob_ref_liveness_joins() -> tuple[tuple[str, str, str], ...]:
    """Return the canonical ledger map, rejecting ambiguous descriptors.

    Every ref type must have exactly one referent relation.
    """

    joins: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for owner in BLOB_OWNERS:
        if owner.tier != "source" or owner.ref_type is None:
            continue
        if not owner.ref_type or not owner.referent_column:
            raise ValueError(f"invalid blob owner descriptor: {owner!r}")
        if owner.ref_type in seen:
            raise ValueError(
                f"ambiguous blob_refs ref_type mapping for {owner.ref_type!r}: "
                f"duplicate referent {owner.table}.{owner.referent_column}"
            )
        seen.add(owner.ref_type)
        joins.append((owner.ref_type, owner.table, owner.referent_column))
    return tuple(joins)


@dataclass(frozen=True, slots=True)
class BlobLiveness:
    """One structured, destructive-safe liveness decision."""

    state: LivenessState
    surfaces: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BlobLivenessProjection:
    """Bulk projection generated from the same descriptor as single lookup."""

    live_hashes: frozenset[str]
    blockers: tuple[str, ...] = ()
    owner_hashes: tuple[tuple[str, frozenset[str]], ...] = ()


def blob_hash_bytes(blob_hash: str) -> bytes | None:
    if len(blob_hash) != 64:
        return None
    try:
        return bytes.fromhex(blob_hash)
    except ValueError:
        return None


def _owners(*, tier: str | None = None, ledger: bool | None = None) -> tuple[BlobOwner, ...]:
    return tuple(
        owner
        for owner in BLOB_OWNERS
        if (tier is None or owner.tier == tier) and (ledger is None or (owner.ref_type is not None) == ledger)
    )


def _known_ref_types() -> frozenset[str]:
    return frozenset(owner.ref_type for owner in _owners(tier="source", ledger=True) if owner.ref_type is not None)


def _schema_blockers(conn: sqlite3.Connection, *, tier: str, required: bool) -> list[str]:
    if not required:
        return []
    blockers: list[str] = []
    for owner in _owners(tier=tier, ledger=False):
        assert owner.blob_column is not None
        if not _table_exists(conn, owner.table):
            if owner.table in _OPTIONAL_OWNER_TABLES:
                continue
            blockers.append(f"{tier}.{owner.table} is missing")
    if tier == "source":
        if not _table_exists(conn, "blob_refs"):
            blockers.append("source.blob_refs is missing")
        else:
            for owner in _owners(tier="source", ledger=True):
                assert owner.referent_column is not None
                if not _table_exists(conn, owner.table):
                    blockers.append(f"source.{owner.table} is missing")
    return blockers


def _source_global_blockers(source_conn: sqlite3.Connection) -> list[str]:
    blockers = _schema_blockers(source_conn, tier="source", required=True)
    if blockers:
        return blockers
    close_failures: list[sqlite3.Error] = []

    def scan() -> list[str]:
        scan_complete = False
        try:
            with connection_cursor(source_conn, "SELECT DISTINCT ref_type FROM blob_refs") as cursor:
                rows = sorted(str(row[0]) for row in cursor if str(row[0]) not in _known_ref_types())
                scan_complete = True
        except sqlite3.Error as exc:
            # A completed read can fail here only while physically closing
            # its cursor. That cleanup failure must retain custody and escape,
            # never become a schema blocker.
            if scan_complete:
                close_failures.append(exc)
            raise
        return rows

    evidence = capture_sqlite_read(scan)
    if close_failures:
        raise close_failures[0]
    if not isinstance(evidence, Measured):
        detail = evidence.detail if isinstance(evidence, Unavailable) else None
        return [f"source.blob_refs is unreadable: {detail or 'sqlite_read_failed'}"]
    unknown = evidence.value
    if unknown:
        blockers.append(f"unknown blob_refs ref_type(s): {', '.join(unknown)}")
    return blockers


def _ledger_surfaces(source_conn: sqlite3.Connection, blob_bytes: bytes, *, prefix: str) -> list[str]:
    if not _table_exists(source_conn, "blob_refs"):
        return []
    surfaces: list[str] = []
    for owner in _owners(tier="source", ledger=True):
        assert owner.ref_type is not None and owner.referent_column is not None
        if not _table_exists(source_conn, owner.table):
            continue
        row = source_conn.execute(
            f"""SELECT 1 FROM blob_refs AS ref WHERE ref.blob_hash = ? AND ref.ref_type = ?
            AND EXISTS (SELECT 1 FROM {owner.table} AS owner WHERE owner.{owner.referent_column} = ref.ref_id)
            LIMIT 1""",
            (blob_bytes, owner.ref_type),
        ).fetchone()
        if row is not None:
            surfaces.append(f"{prefix}.blob_refs")
    return surfaces


def _direct_surfaces(conn: sqlite3.Connection, blob_bytes: bytes, *, tier: str, prefix: str) -> list[str]:
    surfaces: list[str] = []
    for owner in _owners(tier=tier, ledger=False):
        assert owner.blob_column is not None
        if not _table_exists(conn, owner.table):
            continue
        if (
            conn.execute(f"SELECT 1 FROM {owner.table} WHERE {owner.blob_column} = ? LIMIT 1", (blob_bytes,)).fetchone()
            is not None
        ):
            surfaces.append(f"{prefix}.{owner.table}")
    return surfaces


def inspect_blob_liveness(
    source_conn: sqlite3.Connection,
    blob_hash: str,
    *,
    index_conn: sqlite3.Connection | None = None,
    require_index: bool = False,
) -> BlobLiveness:
    """Return ``live``, ``unreferenced``, or typed ``blocked`` for one hash.

    Durable Source owners retain acquired bytes independently of Index
    reconstruction. The current Index can withhold collection for an existing
    reference; its historical population cannot authorize or block collection.
    """
    blockers = _source_global_blockers(source_conn)
    if index_conn is None:
        if require_index:
            blockers.append("index tier is unavailable")
    else:
        blockers.extend(_schema_blockers(index_conn, tier="index", required=True))
    if blockers:
        return BlobLiveness(LivenessState.BLOCKED, blockers=tuple(dict.fromkeys(blockers)))
    blob_bytes = blob_hash_bytes(blob_hash)
    if blob_bytes is None:
        return BlobLiveness(LivenessState.UNREFERENCED)
    try:
        surfaces = _direct_surfaces(source_conn, blob_bytes, tier="source", prefix="source.db")
        surfaces.extend(_ledger_surfaces(source_conn, blob_bytes, prefix="source.db"))
        if index_conn is not None:
            surfaces.extend(_direct_surfaces(index_conn, blob_bytes, tier="index", prefix="index.db"))
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        return BlobLiveness(LivenessState.BLOCKED, blockers=(f"blob liveness query is unreadable: {exc}",))
    if surfaces:
        return BlobLiveness(LivenessState.LIVE, tuple(surfaces))
    return BlobLiveness(LivenessState.UNREFERENCED)


#: Source owners that are not a session's reference to its bytes.
#: ``source_attachments`` is a generation-local census of what an acquisition
#: fetched, keyed by (generation, reference); no session key reaches it, so
#: session excision never removes it, and counting it would make every blob
#: it names look shared with a session that does not exist.
_CENSUS_OWNER_TABLES: frozenset[str] = frozenset({"source_attachments"})


#: Hashes per ``IN (...)`` query: one scan per owner per chunk, well under
#: SQLite's bound-parameter limit.
_REFERENCE_QUERY_CHUNK = 500


class SessionBlobLivenessSourceRead(Protocol):
    """Finite Source observations consumed by the canonical session classifier."""

    def session_blob_global_blockers(self) -> tuple[str, ...]: ...
    def session_blob_owner_available(self, owner: BlobOwner) -> bool: ...
    def session_blob_direct_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]: ...
    def session_blob_ledger_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]: ...


def _require_session_blob_owner(owner: BlobOwner, *, ledger: bool) -> None:
    if owner not in _owners(tier="source", ledger=ledger):
        raise ValueError(f"noncanonical session blob owner: {owner!r}")


def session_blob_direct_query(owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[str, tuple[object, ...]]:
    """Build the sole direct-owner query, including an empty selection."""
    _require_session_blob_owner(owner, ledger=False)
    marks = ",".join("?" for _ in hashes)
    predicate = f"{owner.blob_column} IN ({marks})" if hashes else "0"
    return f"SELECT DISTINCT {owner.blob_column} FROM {owner.table} WHERE {predicate}", hashes


def session_blob_ledger_query(owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[str, tuple[object, ...]]:
    """Build the sole typed-ledger query with its actual surviving referent."""
    _require_session_blob_owner(owner, ledger=True)
    marks = ",".join("?" for _ in hashes)
    predicate = f"ref.blob_hash IN ({marks})" if hashes else "0"
    return (
        f"SELECT DISTINCT ref.blob_hash FROM blob_refs AS ref WHERE {predicate} AND ref.ref_type = ? "
        f"AND EXISTS (SELECT 1 FROM {owner.table} AS owner WHERE owner.{owner.referent_column} = ref.ref_id)",
        (*hashes, owner.ref_type),
    )


@dataclass(frozen=True, slots=True)
class ConnectionSessionBlobLivenessRead:
    """Borrow an ordinary Source connection without taking its write/lifetime authority."""

    connection: sqlite3.Connection

    def session_blob_global_blockers(self) -> tuple[str, ...]:
        return tuple(_source_global_blockers(self.connection))

    def session_blob_owner_available(self, owner: BlobOwner) -> bool:
        _require_session_blob_owner(owner, ledger=False)
        assert owner.blob_column is not None
        return _table_exists(self.connection, owner.table)

    def session_blob_direct_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        with connection_cursor(self.connection, *session_blob_direct_query(owner, hashes)) as cursor:
            return tuple(bytes(row[0]) for row in cursor)

    def session_blob_ledger_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        with connection_cursor(self.connection, *session_blob_ledger_query(owner, hashes)) as cursor:
            return tuple(bytes(row[0]) for row in cursor)


def inspect_session_blob_references(
    source: SessionBlobLivenessSourceRead,
    blob_hashes: Sequence[bytes],
    *,
    index_conn: sqlite3.Connection | None,
    excluding_session_ids: Set[str],
) -> dict[bytes, BlobLiveness]:
    """Whether a session outside ``excluding_session_ids`` still references each blob.

    Session excision owns a blob only while no other session references it:
    blobs are content-addressed, so two sessions with the same tool output or
    the same attachment share one hash. Excision calls this after deleting
    the excised session's source rows, in the same transaction, and marks a
    hash forgotten only when its answer is ``unreferenced``.

    Source owners come from :data:`BLOB_OWNERS` (direct columns and ledger
    rows whose referent still exists), minus :data:`_CENSUS_OWNER_TABLES`.
    The live batch records attachments only in ``index.attachments``, so the
    index is asked too: an attachment still linked through ``attachment_refs``
    to a session outside the excision is a live reference. The index only
    withholds a marker here; it never causes one.

    ``blocked`` means the answer cannot be decided (an unknown ``blob_refs``
    type, a missing owner table), and the caller must refuse rather than
    guess in either direction.

    Each owner is queried once per chunk of hashes, not once per hash: an
    owner without a ``blob_hash`` index is scanned per query. A query that
    fails propagates, so the caller's transaction rolls back.
    """
    hashes = tuple(dict.fromkeys(blob_hashes))
    blockers = list(source.session_blob_global_blockers())
    if index_conn is not None:
        blockers.extend(_schema_blockers(index_conn, tier="index", required=True))
        if not _table_exists(index_conn, "attachment_refs"):
            blockers.append("index.attachment_refs is missing")
    if blockers:
        blocked = BlobLiveness(LivenessState.BLOCKED, blockers=tuple(dict.fromkeys(blockers)))
        return dict.fromkeys(hashes, blocked)
    surfaces: dict[bytes, list[str]] = {blob_hash: [] for blob_hash in hashes}
    for start in range(0, len(hashes), _REFERENCE_QUERY_CHUNK):
        chunk = hashes[start : start + _REFERENCE_QUERY_CHUNK]
        marks = ",".join("?" for _ in chunk)
        for owner in _owners(tier="source", ledger=False):
            assert owner.blob_column is not None
            if owner.table in _CENSUS_OWNER_TABLES or not source.session_blob_owner_available(owner):
                continue
            for found in source.session_blob_direct_hashes(owner, chunk):
                surfaces[found].append(f"source.db.{owner.table}")
        for owner in _owners(tier="source", ledger=True):
            assert owner.ref_type is not None and owner.referent_column is not None
            for found in source.session_blob_ledger_hashes(owner, chunk):
                surfaces[found].append("source.db.blob_refs")
        if index_conn is not None:
            with connection_cursor(
                index_conn,
                "SELECT DISTINCT a.blob_hash,r.session_id FROM attachments AS a "
                "JOIN attachment_refs AS r ON r.attachment_id = a.attachment_id "
                f"WHERE a.blob_hash IN ({marks})",
                chunk,
            ) as cursor:
                for found, session_id in cursor:
                    found_surfaces = surfaces[bytes(found)]
                    if "index.db.attachment_refs" not in found_surfaces and session_id not in excluding_session_ids:
                        found_surfaces.append("index.db.attachment_refs")
    decisions: dict[bytes, BlobLiveness] = {}
    for blob_hash, found_surfaces in surfaces.items():
        if found_surfaces:
            decisions[blob_hash] = BlobLiveness(LivenessState.LIVE, tuple(dict.fromkeys(found_surfaces)))
        else:
            decisions[blob_hash] = BlobLiveness(LivenessState.UNREFERENCED)
    return decisions


def inspect_blob_reservation(source_conn: sqlite3.Connection, blob_hash: str) -> BlobLiveness:
    """Return an exact-ID protocol decision for a hash's remaining receipts.

    This is only a GC protection query. Publication reconciliation consumes by
    ``(publication_id, blob_hash)`` in the transaction that creates the
    referent; it must never use this hash-level answer to consume a receipt.
    """
    if not _table_exists(source_conn, "blob_publication_reservations"):
        return BlobLiveness(LivenessState.UNREFERENCED)
    blob_bytes = blob_hash_bytes(blob_hash)
    if blob_bytes is None:
        return BlobLiveness(LivenessState.UNREFERENCED)
    try:
        row = source_conn.execute(
            "SELECT 1 FROM blob_publication_reservations WHERE blob_hash = ? LIMIT 1", (blob_bytes,)
        ).fetchone()
    except sqlite3.Error as exc:
        return BlobLiveness(LivenessState.BLOCKED, blockers=(f"publication reservation query is unreadable: {exc}",))
    return BlobLiveness(
        LivenessState.LIVE if row is not None else LivenessState.UNREFERENCED,
        ("source.db.blob_publication_reservations",) if row is not None else (),
    )


def project_live_blob_hashes(
    source_conn: sqlite3.Connection,
    *,
    index_conn: sqlite3.Connection | None = None,
    require_index: bool = False,
) -> BlobLivenessProjection:
    """Project every live hash across all source generations.

    A source tier keeps every generation's rows, so the projection keeps every
    generation's blobs: a caller that narrowed it to one generation would treat
    an earlier generation's still-referenced bytes as dead.
    """
    blockers = _source_global_blockers(source_conn)
    if index_conn is None:
        if require_index:
            blockers.append("index tier is unavailable")
    else:
        blockers.extend(_schema_blockers(index_conn, tier="index", required=True))
    if blockers:
        return BlobLivenessProjection(frozenset(), tuple(dict.fromkeys(blockers)))
    hashes: set[str] = set()
    owner_hashes: dict[str, set[str]] = {}
    try:
        for conn, tier in ((source_conn, "source"), (index_conn, "index")):
            if conn is None:
                continue
            for owner in _owners(tier=tier, ledger=False):
                assert owner.blob_column is not None
                if not _table_exists(conn, owner.table):
                    continue
                owner_name = f"{tier}.db.{owner.table}"
                for row in conn.execute(f"SELECT DISTINCT {owner.blob_column} FROM {owner.table}"):
                    if isinstance(row[0], bytes) and len(row[0]) == 32:
                        blob_hash = row[0].hex()
                        hashes.add(blob_hash)
                        owner_hashes.setdefault(owner_name, set()).add(blob_hash)
        if _table_exists(source_conn, "blob_refs"):
            for owner in _owners(tier="source", ledger=True):
                assert owner.ref_type is not None and owner.referent_column is not None
                if not _table_exists(source_conn, owner.table):
                    continue
                for row in source_conn.execute(
                    f"""SELECT DISTINCT ref.blob_hash FROM blob_refs AS ref WHERE ref.ref_type = ? AND EXISTS (
                    SELECT 1 FROM {owner.table} AS owner WHERE owner.{owner.referent_column} = ref.ref_id)""",
                    (owner.ref_type,),
                ):
                    if isinstance(row[0], bytes) and len(row[0]) == 32:
                        blob_hash = row[0].hex()
                        hashes.add(blob_hash)
                        owner_hashes.setdefault("source.db.blob_refs", set()).add(blob_hash)
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        return BlobLivenessProjection(frozenset(), (f"blob liveness query is unreadable: {exc}",))
    return BlobLivenessProjection(
        frozenset(hashes),
        owner_hashes=tuple((owner, frozenset(values)) for owner, values in sorted(owner_hashes.items())),
    )


__all__ = [
    "BLOB_OWNERS",
    "SessionBlobLivenessSourceRead",
    "ConnectionSessionBlobLivenessRead",
    "session_blob_direct_query",
    "session_blob_ledger_query",
    "acquired_attachment_missing_ref_predicate",
    "BlobLiveness",
    "BlobLivenessProjection",
    "LivenessState",
    "blob_hash_bytes",
    "inspect_blob_liveness",
    "inspect_blob_reservation",
    "inspect_session_blob_references",
    "project_live_blob_hashes",
    "validated_blob_ref_liveness_joins",
]
