"""Canonical, descriptor-driven decisions about retained blob bytes.

The source tier owns raw and hook payloads plus the ``blob_refs`` ledger;
the active index owns attachment payloads. ``blob_refs`` is only evidence when
its typed referent still resolves. Receipt caches and observation IDs are not
part of this relation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum

from polylogue.storage.hook_payload_ref_reconciliation import (
    HookPayloadRefMatchStage,
    ensure_current_match_stage,
    prepare_match_stage,
)
from polylogue.storage.introspection import column_exists as _column_exists
from polylogue.storage.introspection import table_exists as _table_exists


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
    rekeyable_legacy_ref: bool = False


# The sole map for per-hash inspection, bulk projection, schema preflight,
# integrity, sealing, GC, and blob-ref reconciliation.
BLOB_OWNERS: tuple[BlobOwner, ...] = (
    BlobOwner("source", "raw_sessions", blob_column="blob_hash"),
    BlobOwner("source", "raw_hook_events", blob_column="blob_hash"),
    BlobOwner("index", "attachments", blob_column="blob_hash"),
    BlobOwner("source", "raw_sessions", ref_type="raw_payload", referent_column="raw_id"),
    BlobOwner("source", "raw_sessions", ref_type="attachment", referent_column="raw_id"),
    BlobOwner("source", "raw_hook_events", ref_type="hook_payload", referent_column="hook_event_id"),
    # Before source schema v22, hook payloads were recorded as raw_payload
    # refs keyed by a deterministic raw id, even though hooks never create a
    # raw_sessions row. The rekey matcher proves the actual hook referent.
    BlobOwner(
        "source",
        "raw_hook_events",
        ref_type="raw_payload",
        referent_column="hook_event_id",
        rekeyable_legacy_ref=True,
    ),
)


def validated_blob_ref_liveness_joins() -> tuple[tuple[str, str, str], ...]:
    """Return the canonical ledger map, rejecting ambiguous descriptors.

    Legacy rekeyable refs are intentionally outside this direct join map: the
    matcher proves their ownership separately.  Every ordinary ref type must
    have exactly one referent relation.
    """

    joins: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for owner in BLOB_OWNERS:
        if owner.tier != "source" or owner.ref_type is None:
            continue
        if owner.rekeyable_legacy_ref:
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


def blob_refs_has_ref_type_column(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "blob_refs"):
        return False
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(blob_refs)")}
    return {"blob_hash", "ref_id", "ref_type"}.issubset(columns)


def _legacy_hook_rekey_supported(conn: sqlite3.Connection) -> bool:
    """Whether this source schema carries all evidence the legacy matcher needs."""

    return (
        blob_refs_has_ref_type_column(conn)
        and all(_column_exists(conn, "blob_refs", column) for column in ("source_path", "size_bytes", "acquired_at_ms"))
        and _table_exists(conn, "raw_hook_events")
        and all(
            _column_exists(conn, "raw_hook_events", column)
            for column in ("hook_event_id", "origin", "native_id", "source_path", "blob_hash")
        )
        and _table_exists(conn, "raw_sessions")
        and _column_exists(conn, "raw_sessions", "raw_id")
    )


def _legacy_hook_rekey_schema_blockers(conn: sqlite3.Connection) -> list[str]:
    """Describe every missing fact needed to prove legacy hook ownership."""

    required_columns = {
        "blob_refs": ("blob_hash", "ref_id", "ref_type", "source_path", "size_bytes", "acquired_at_ms"),
        "raw_hook_events": ("hook_event_id", "origin", "native_id", "source_path", "blob_hash"),
        "raw_sessions": ("raw_id",),
    }
    blockers: list[str] = []
    for table, columns in required_columns.items():
        if not _table_exists(conn, table):
            blockers.append(f"source.{table} is missing")
            continue
        missing = [column for column in columns if not _column_exists(conn, table, column)]
        if missing:
            blockers.append(f"source.{table} is missing columns: {', '.join(missing)}")
    if blockers or not _legacy_hook_rekey_supported(conn):
        blockers.append("source.legacy hook rekey evidence is unavailable")
    return blockers


def _schema_blockers(conn: sqlite3.Connection, *, tier: str, required: bool) -> list[str]:
    if not required:
        return []
    blockers: list[str] = []
    for owner in _owners(tier=tier, ledger=False):
        assert owner.blob_column is not None
        if not _table_exists(conn, owner.table):
            blockers.append(f"{tier}.{owner.table} is missing")
        elif not _column_exists(conn, owner.table, owner.blob_column):
            blockers.append(f"{tier}.{owner.table} is missing columns: {owner.blob_column}")
    if tier == "source":
        if not _table_exists(conn, "blob_refs"):
            blockers.append("source.blob_refs is missing")
        elif not blob_refs_has_ref_type_column(conn):
            blockers.append("source.blob_refs is missing columns: blob_hash, ref_id, ref_type")
        else:
            for owner in _owners(tier="source", ledger=True):
                assert owner.referent_column is not None
                if not _table_exists(conn, owner.table):
                    blockers.append(f"source.{owner.table} is missing")
                elif not _column_exists(conn, owner.table, owner.referent_column):
                    blockers.append(f"source.{owner.table} is missing columns: {owner.referent_column}")
        if any(owner.rekeyable_legacy_ref for owner in _owners(tier="source", ledger=True)):
            blockers.extend(_legacy_hook_rekey_schema_blockers(conn))
    return blockers


def _source_global_blockers(source_conn: sqlite3.Connection) -> list[str]:
    blockers = _schema_blockers(source_conn, tier="source", required=True)
    if blockers:
        return blockers
    try:
        unknown = sorted(
            str(row[0])
            for row in source_conn.execute("SELECT DISTINCT ref_type FROM blob_refs")
            if str(row[0]) not in _known_ref_types()
        )
    except sqlite3.Error as exc:
        return [f"source.blob_refs is unreadable: {exc}"]
    if unknown:
        blockers.append(f"unknown blob_refs ref_type(s): {', '.join(unknown)}")
    return blockers


def _ledger_surfaces(source_conn: sqlite3.Connection, blob_bytes: bytes, *, prefix: str) -> list[str]:
    if not blob_refs_has_ref_type_column(source_conn):
        return []
    surfaces: list[str] = []
    for owner in _owners(tier="source", ledger=True):
        assert owner.ref_type is not None and owner.referent_column is not None
        if owner.rekeyable_legacy_ref:
            continue
        if not _table_exists(source_conn, owner.table) or not _column_exists(
            source_conn, owner.table, owner.referent_column
        ):
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


def _rekeyable_legacy_hook_surfaces(
    source_conn: sqlite3.Connection,
    blob_bytes: bytes,
    *,
    prefix: str,
    stage: HookPayloadRefMatchStage | None = None,
) -> list[str]:
    """Return legacy hook refs whose deterministic rekey proof is current.

    The matcher is the same all-or-nothing stage consumed by blob-ref
    reconciliation. Both deterministic matches and ambiguous candidates own
    bytes: ambiguity blocks attribution rewrites, not retention.
    """

    if not any(owner.rekeyable_legacy_ref for owner in _owners(tier="source", ledger=True)):
        return []
    try:
        # A caller holding source/index writer exclusion passes one token for
        # its whole candidate batch.  The cheap generation marker refreshes
        # after local writes, external commits, or schema drift; it never
        # lets a stale match stage decide deletion.
        if stage is None:
            prepare_match_stage(source_conn)
        else:
            ensure_current_match_stage(source_conn, stage)
        row = source_conn.execute(
            """SELECT 1 FROM (
                SELECT blob_hash FROM temp.hook_payload_ref_reconciliation_matches
                UNION
                SELECT blob_hash FROM temp.hook_payload_ref_reconciliation_ambiguous
            ) WHERE blob_hash = ? LIMIT 1""",
            (blob_bytes,),
        ).fetchone()
    except Exception as exc:
        raise RuntimeError(f"legacy hook rekey matcher failed: {exc}") from exc
    return [f"{prefix}.rekeyable_hook_payload"] if row is not None else []


def _direct_surfaces(conn: sqlite3.Connection, blob_bytes: bytes, *, tier: str, prefix: str) -> list[str]:
    surfaces: list[str] = []
    for owner in _owners(tier=tier, ledger=False):
        assert owner.blob_column is not None
        if not _table_exists(conn, owner.table) or not _column_exists(conn, owner.table, owner.blob_column):
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
    legacy_hook_stage: HookPayloadRefMatchStage | None = None,
) -> BlobLiveness:
    """Return ``live``, ``unreferenced``, or typed ``blocked`` for one hash."""
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
        surfaces.extend(
            _rekeyable_legacy_hook_surfaces(
                source_conn,
                blob_bytes,
                prefix="source.db",
                stage=legacy_hook_stage,
            )
        )
        if index_conn is not None:
            surfaces.extend(_direct_surfaces(index_conn, blob_bytes, tier="index", prefix="index.db"))
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        return BlobLiveness(LivenessState.BLOCKED, blockers=(f"blob liveness query is unreadable: {exc}",))
    return BlobLiveness(LivenessState.LIVE if surfaces else LivenessState.UNREFERENCED, tuple(surfaces))


def inspect_blob_reservation(source_conn: sqlite3.Connection, blob_hash: str) -> BlobLiveness:
    """Return an exact-ID protocol decision for a hash's remaining receipts.

    This is only a GC protection query. Publication reconciliation consumes by
    ``(publication_id, blob_hash)`` in the transaction that creates the
    referent; it must never use this hash-level answer to consume a receipt.
    """
    if not _table_exists(source_conn, "blob_publication_reservations"):
        return BlobLiveness(LivenessState.UNREFERENCED)
    if not _column_exists(source_conn, "blob_publication_reservations", "blob_hash"):
        return BlobLiveness(
            LivenessState.BLOCKED, blockers=("source.blob_publication_reservations is missing columns: blob_hash",)
        )
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
    source_generation_id: str | None = None,
) -> BlobLivenessProjection:
    """Project live hashes, optionally scoped to one source generation."""
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
                if not _table_exists(conn, owner.table) or not _column_exists(conn, owner.table, owner.blob_column):
                    continue
                owner_name = f"{tier}.db.{owner.table}"
                generation_filter = ""
                params: tuple[object, ...] = ()
                if source_generation_id is not None and owner.table == "raw_sessions":
                    generation_filter = (
                        " WHERE EXISTS (SELECT 1 FROM source_items si "
                        "WHERE si.source_generation_id = ? AND si.raw_id = raw_sessions.raw_id)"
                    )
                    params = (source_generation_id,)
                for row in conn.execute(
                    f"SELECT DISTINCT {owner.blob_column} FROM {owner.table}{generation_filter}", params
                ):
                    if isinstance(row[0], bytes) and len(row[0]) == 32:
                        blob_hash = row[0].hex()
                        hashes.add(blob_hash)
                        owner_hashes.setdefault(owner_name, set()).add(blob_hash)
        if blob_refs_has_ref_type_column(source_conn):
            for owner in _owners(tier="source", ledger=True):
                assert owner.ref_type is not None and owner.referent_column is not None
                if owner.rekeyable_legacy_ref:
                    continue
                if not _table_exists(source_conn, owner.table) or not _column_exists(
                    source_conn, owner.table, owner.referent_column
                ):
                    continue
                generation_filter = ""
                params = (owner.ref_type,)
                if source_generation_id is not None and owner.table == "raw_sessions":
                    generation_filter = (
                        " AND EXISTS (SELECT 1 FROM source_items si "
                        "WHERE si.source_generation_id = ? AND si.raw_id = owner.raw_id)"
                    )
                    params += (source_generation_id,)
                for row in source_conn.execute(
                    f"""SELECT DISTINCT ref.blob_hash FROM blob_refs AS ref WHERE ref.ref_type = ? AND EXISTS (
                    SELECT 1 FROM {owner.table} AS owner WHERE owner.{owner.referent_column} = ref.ref_id{generation_filter})""",
                    params,
                ):
                    if isinstance(row[0], bytes) and len(row[0]) == 32:
                        blob_hash = row[0].hex()
                        hashes.add(blob_hash)
                        owner_hashes.setdefault("source.db.blob_refs", set()).add(blob_hash)
        if any(owner.rekeyable_legacy_ref for owner in _owners(tier="source", ledger=True)):
            from polylogue.storage.hook_payload_ref_reconciliation import _create_match_stage

            try:
                _create_match_stage(source_conn)
                rows = source_conn.execute(
                    """SELECT DISTINCT blob_hash FROM (
                        SELECT blob_hash FROM temp.hook_payload_ref_reconciliation_matches
                        UNION
                        SELECT blob_hash FROM temp.hook_payload_ref_reconciliation_ambiguous
                    )"""
                )
                for row in rows:
                    if isinstance(row[0], bytes) and len(row[0]) == 32:
                        blob_hash = row[0].hex()
                        hashes.add(blob_hash)
                        owner_hashes.setdefault("source.db.rekeyable_hook_payload", set()).add(blob_hash)
            except Exception as exc:
                raise RuntimeError(f"legacy hook rekey matcher failed: {exc}") from exc
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        return BlobLivenessProjection(frozenset(), (f"blob liveness query is unreadable: {exc}",))
    return BlobLivenessProjection(
        frozenset(hashes),
        owner_hashes=tuple((owner, frozenset(values)) for owner, values in sorted(owner_hashes.items())),
    )


def project_index_blob_hashes(index_conn: sqlite3.Connection) -> BlobLivenessProjection:
    """Project the active index's direct blob owners without source evidence.

    Historical source fallback uses this narrow descriptor-owned projection to
    retain readable index attachments. It intentionally cannot decide source
    ledger ownership, which remains with :func:`project_live_blob_hashes`.
    """

    blockers = _schema_blockers(index_conn, tier="index", required=True)
    if blockers:
        return BlobLivenessProjection(frozenset(), tuple(dict.fromkeys(blockers)))
    hashes: set[str] = set()
    owner_hashes: dict[str, set[str]] = {}
    try:
        for owner in _owners(tier="index", ledger=False):
            assert owner.blob_column is not None
            owner_name = f"index.db.{owner.table}"
            for row in index_conn.execute(f"SELECT DISTINCT {owner.blob_column} FROM {owner.table}"):
                if isinstance(row[0], bytes) and len(row[0]) == 32:
                    blob_hash = row[0].hex()
                    hashes.add(blob_hash)
                    owner_hashes.setdefault(owner_name, set()).add(blob_hash)
    except sqlite3.Error as exc:
        return BlobLivenessProjection(frozenset(), (f"index blob liveness query is unreadable: {exc}",))
    return BlobLivenessProjection(
        frozenset(hashes),
        owner_hashes=tuple((owner, frozenset(values)) for owner, values in sorted(owner_hashes.items())),
    )


__all__ = [
    "BLOB_OWNERS",
    "BlobLiveness",
    "BlobLivenessProjection",
    "LivenessState",
    "blob_hash_bytes",
    "blob_refs_has_ref_type_column",
    "inspect_blob_liveness",
    "inspect_blob_reservation",
    "project_index_blob_hashes",
    "project_live_blob_hashes",
    "validated_blob_ref_liveness_joins",
]
