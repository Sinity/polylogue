"""Canonical blob live-hash relation.

A blob is live when it has at least one of two kinds of protection:

1. A *direct referent* -- a row in ``raw_sessions``, ``attachments``, or
   ``raw_hook_events`` whose own ``blob_hash`` column names the content
   directly. This is protective even when no ``blob_refs`` ledger row exists
   for the same content (``DIRECT_BLOB_HASH_TABLES``).
2. A ``blob_refs`` row whose ``ref_type`` names a referent table/column
   (``BLOB_REF_LIVENESS_JOIN``) that still contains the row identified by
   ``ref_id``. ``blob_refs`` is provenance/indexing evidence, not an
   independent liveness grant: a row whose declared referent is absent
   ("dangling") contributes no liveness, and a row whose ``ref_type`` is
   unknown or whose referent table/column is unavailable is retained because
   liveness cannot be disproven, never treated as dead.
3. A durable publication reservation (``blob_publication_reservations``),
   protecting the publish-to-reference crash window until the exact
   reference transaction consumes it.

Historical authority verdicts, quarantine state, excision rows, source-path
existence, and age never grant liveness through this module. Durable excision
(``excised_content``) may explain intentional non-reacquisition but is
deliberately never consulted here: it suppresses reacquisition, it does not
retain bytes.

This module is the single owner of the reference-kind map and the joins that
implement it. Blob GC, integrity scanning, residue census, and source sealing
all consume this module instead of hand-rolling their own reference union;
see ``polylogue/storage/blob_gc.py`` for the GC-specific orchestration
(candidate walking, locking, unlink) built on top of it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.introspection import column_exists as _column_exists
from polylogue.storage.introspection import table_exists as _table_exists

logger = __import__("logging").getLogger(__name__)


class BlobLivenessError(RuntimeError):
    """Raised when the closed reference-kind map itself is invalid."""


@dataclass(frozen=True, slots=True)
class BlobLiveness:
    """The complete, explainable answer for one blob hash.

    ``blockers`` are deliberately protective.  A destructive caller must not
    turn an incomplete archive shape into an empty reference set; diagnostic
    callers can expose them without pretending a dangling ledger row is live.
    """

    surfaces: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    @property
    def protected(self) -> bool:
        return bool(self.surfaces or self.blockers)


# blob_refs.ref_type -> the table/column a ref_id must actually resolve to for
# that row to mean the blob is still live. 'attachment' refs are keyed by the
# *parent session's* raw_id (write_source_raw_session/
# write_source_raw_session_blob_ref always pass ref.raw_id=resolved_raw_id for
# every entry in additional_blob_refs, regardless of ref_type -- verified
# against every production call site, polylogue-tfzw0), not a raw_artifacts
# row, so it joins the same table as 'raw_payload'. 'sidecar' has no
# production writer today; it is included for completeness/forward-safety
# against history_sidecars.sidecar_id, its only plausible referent.
BLOB_REF_LIVENESS_JOIN: tuple[tuple[str, str, str], ...] = (
    ("raw_payload", "raw_sessions", "raw_id"),
    ("attachment", "raw_sessions", "raw_id"),
    ("hook_payload", "raw_hook_events", "hook_event_id"),
    ("sidecar", "history_sidecars", "sidecar_id"),
)

# Row surfaces whose own ``blob_hash`` column is a first-class, direct
# referent to the CAS payload -- independent of, and protective without, any
# ``blob_refs`` ledger row for the same content.
DIRECT_BLOB_HASH_TABLES: tuple[str, ...] = ("raw_sessions", "attachments", "raw_hook_events")

_SOURCE_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "raw_sessions": frozenset(("raw_id", "blob_hash")),
    "raw_hook_events": frozenset(("hook_event_id", "blob_hash")),
    "history_sidecars": frozenset(("sidecar_id",)),
    "blob_refs": frozenset(("blob_hash", "ref_id", "ref_type")),
    "blob_publication_reservations": frozenset(("blob_hash",)),
}
_INDEX_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {"attachments": frozenset(("blob_hash",))}


def liveness_schema_blockers(conn: sqlite3.Connection, *, tier: str, current_source: bool = False) -> tuple[str, ...]:
    """Return typed-to-the-caller schema gaps that make a destructive answer unsafe."""
    required: dict[str, frozenset[str]] = {}
    if tier == "source" and current_source:
        required.update(_SOURCE_REQUIRED_COLUMNS)
    if tier == "index" and current_source:
        required.update(_INDEX_REQUIRED_COLUMNS)
    if tier == "source" and _table_exists(conn, "blob_refs") and blob_refs_has_ref_type_column(conn):
        try:
            ref_types = {str(row[0]) for row in conn.execute("SELECT DISTINCT ref_type FROM blob_refs")}
        except sqlite3.Error as exc:
            return (f"source.blob_refs is unreadable: {exc}",)
        known = {ref_type: (table, column) for ref_type, table, column in BLOB_REF_LIVENESS_JOIN}
        for ref_type in ref_types & set(known):
            table, column = known[ref_type]
            required[table] = required.get(table, frozenset()) | frozenset((column,))
    blockers: list[str] = []
    for table, columns in required.items():
        try:
            if not _table_exists(conn, table):
                blockers.append(f"{tier}.{table} is missing")
                continue
            missing = sorted(column for column in columns if not _column_exists(conn, table, column))
        except sqlite3.Error as exc:
            blockers.append(f"{tier}.{table} is unreadable: {exc}")
            continue
        if missing:
            blockers.append(f"{tier}.{table} is missing columns: {', '.join(missing)}")
    return tuple(blockers)


def validated_blob_liveness_joins() -> tuple[tuple[str, str, str], ...]:
    """Return the closed ref-type map, rejecting duplicate or invalid entries."""
    seen: set[str] = set()
    for ref_type, referent_table, referent_column in BLOB_REF_LIVENESS_JOIN:
        if ref_type in seen:
            raise BlobLivenessError(
                f"ambiguous blob_refs ref_type mapping for {ref_type!r}: "
                f"duplicate referent {referent_table}.{referent_column}"
            )
        if not ref_type or not referent_table or not referent_column:
            raise BlobLivenessError(
                f"invalid blob_refs ref_type mapping: {ref_type!r} -> {referent_table!r}.{referent_column!r}"
            )
        seen.add(ref_type)
    return BLOB_REF_LIVENESS_JOIN


def blob_hash_bytes(blob_hash: str) -> bytes | None:
    """Decode a 64-character lowercase-hex blob hash, or ``None`` if malformed."""
    if len(blob_hash) != 64:
        return None
    try:
        return bytes.fromhex(blob_hash)
    except ValueError:
        return None


def blob_refs_has_ref_type_column(conn: sqlite3.Connection) -> bool:
    """Guard for legacy/alternate ``blob_refs`` shapes lacking ``ref_type``/``ref_id``.

    Some pre-#1743 fixtures and test doubles model ``blob_refs`` with a
    different column set entirely (e.g. ``owner_kind``/``owner_id`` instead of
    ``ref_type``/``ref_id``). The liveness join is meaningless there; callers
    fall back to treating ``blob_refs`` as contributing nothing rather than
    raising ``OperationalError``.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(blob_refs)")}
    return "ref_type" in columns and "ref_id" in columns


def prepare_legacy_hook_liveness(conn: sqlite3.Connection) -> str:
    """Prepare the canonical legacy-hook reconciliation stage for this connection.

    Returns ``"not_applicable"`` when ``raw_hook_events`` is absent or lacks
    the required columns, ``"unavailable"`` when staging fails, or
    ``"ready"`` when the match tables are staged and queryable.
    """
    if not _table_exists(conn, "raw_hook_events"):
        return "not_applicable"
    hook_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(raw_hook_events)")}
    required = {"hook_event_id", "origin", "native_id", "source_path", "blob_hash"}
    if required - hook_columns:
        return "unavailable"
    try:
        # Reuse the same bounded, ambiguity-aware stage that the offline
        # reconciliation classifier uses. Liveness only reads its match tables.
        from polylogue.storage.hook_payload_ref_reconciliation import _create_match_stage

        _create_match_stage(conn)
    except Exception:
        logger.warning("Could not stage legacy hook evidence for blob liveness; retaining candidate blobs")
        return "unavailable"
    return "ready"


def legacy_hook_liveness_status(conn: sqlite3.Connection) -> str:
    """Prepare legacy-hook evidence once when this connection needs it."""
    if not _table_exists(conn, "blob_refs") or not blob_refs_has_ref_type_column(conn):
        return "not_applicable"
    if conn.execute("SELECT 1 FROM blob_refs WHERE ref_type = 'raw_payload' LIMIT 1").fetchone() is None:
        return "not_applicable"
    return prepare_legacy_hook_liveness(conn)


def blob_refs_row_is_live(
    conn: sqlite3.Connection,
    blob_bytes: bytes,
    *,
    legacy_hook_status: str | None = None,
) -> bool:
    """Return True if some ``blob_refs`` row for this hash has a live referent.

    A ``blob_refs`` row is only real evidence of liveness when the table its
    own ``ref_type`` names (see ``BLOB_REF_LIVENESS_JOIN``) still has the row
    identified by ``ref_id``. A hook-event blob ref whose ``raw_hook_events``
    row was deleted, or a stale ref left behind by a since-reverted write, no
    longer joins to anything and is correctly treated as dead here. Unknown
    ref types and known types whose referent table or key column is
    unavailable are retained because liveness cannot be disproven.
    """
    if not _table_exists(conn, "blob_refs"):
        return False
    if not blob_refs_has_ref_type_column(conn):
        # A legacy or malformed reference table cannot prove that its rows are
        # dead. Keep the blob until an offline integrity tool records a typed
        # disposition; liveness must never turn an unknown schema into
        # evidence loss.
        return True
    ref_types = {
        str(row[0])
        for row in conn.execute("SELECT DISTINCT ref_type FROM blob_refs WHERE blob_hash = ?", (blob_bytes,))
    }
    if not ref_types:
        return False
    known_ref_types = {ref_type for ref_type, _table, _column in BLOB_REF_LIVENESS_JOIN}
    if ref_types - known_ref_types:
        # Unknown ref types are a blocker, never evidence of death. Liveness
        # follows the same fail-closed rule as offline reconciliation.
        return True
    if "raw_payload" in ref_types:
        legacy_hook_status = (
            legacy_hook_status if legacy_hook_status is not None else prepare_legacy_hook_liveness(conn)
        )
        if legacy_hook_status == "unavailable":
            return True
        if legacy_hook_status == "ready":
            legacy_hook_row = conn.execute(
                """
                SELECT 1
                FROM temp.hook_payload_ref_reconciliation_matches
                WHERE blob_hash = ?
                UNION ALL
                SELECT 1
                FROM temp.hook_payload_ref_reconciliation_ambiguous
                WHERE blob_hash = ?
                LIMIT 1
                """,
                (blob_bytes, blob_bytes),
            ).fetchone()
            if legacy_hook_row is not None:
                return True
    clauses = [
        f"(ref_type = ? AND EXISTS (SELECT 1 FROM {referent_table} WHERE {referent_table}.{referent_column} = blob_refs.ref_id))"
        for ref_type, referent_table, referent_column in BLOB_REF_LIVENESS_JOIN
        if ref_type in ref_types
    ]
    params: list[str] = []
    for ref_type, referent_table, referent_column in BLOB_REF_LIVENESS_JOIN:
        if ref_type not in ref_types:
            continue
        if not _table_exists(conn, referent_table):
            # The type is known, but this archive cannot evaluate its join.
            # Keep the blob until the missing referent surface is restored or
            # an integrity tool records a disposition.
            return True
        if not _column_exists(conn, referent_table, referent_column):
            return True
        params.append(ref_type)
    if not clauses:
        return False
    query = f"SELECT 1 FROM blob_refs WHERE blob_hash = ? AND ({' OR '.join(clauses)}) LIMIT 1"
    row = conn.execute(query, (blob_bytes, *params)).fetchone()
    return row is not None


def archive_reference_surfaces(
    conn: sqlite3.Connection,
    blob_hash: str,
    *,
    surface_prefix: str,
    legacy_hook_status: str | None = None,
) -> list[str]:
    """Return the named surfaces (within one connection) that protect *blob_hash*."""
    blob_bytes = blob_hash_bytes(blob_hash)
    if blob_bytes is None:
        return []

    surfaces: list[str] = []
    for table in DIRECT_BLOB_HASH_TABLES:
        if not _table_exists(conn, table) or not _column_exists(conn, table, "blob_hash"):
            continue
        row = conn.execute(f"SELECT 1 FROM {table} WHERE blob_hash = ? LIMIT 1", (blob_bytes,)).fetchone()
        if row is not None:
            surfaces.append(f"{surface_prefix}.{table}")
    if blob_refs_row_is_live(conn, blob_bytes, legacy_hook_status=legacy_hook_status):
        surfaces.append(f"{surface_prefix}.blob_refs")
    return surfaces


def reference_surfaces(
    conn: sqlite3.Connection,
    blob_hash: str,
    *,
    source_db_path: Path | None = None,
    source_conn: sqlite3.Connection | None = None,
    index_conn: sqlite3.Connection | None = None,
    legacy_hook_status: str | None = None,
    source_legacy_hook_status: str | None = None,
    index_legacy_hook_status: str | None = None,
) -> list[str]:
    """Return every named surface across the control/source/index tiers.

    ``conn`` is the control (currently open) tier. ``source_conn``/
    ``index_conn`` are optional sibling tiers already open for this pass;
    when absent and ``source_db_path`` is given, the source sibling is opened
    read-only for the duration of this call.
    """
    surfaces = archive_reference_surfaces(
        conn,
        blob_hash,
        surface_prefix="current",
        legacy_hook_status=legacy_hook_status,
    )

    if source_conn is not None:
        source_prefix = source_db_path.name if source_db_path is not None else "source"
        surfaces.extend(
            archive_reference_surfaces(
                source_conn,
                blob_hash,
                surface_prefix=source_prefix,
                legacy_hook_status=source_legacy_hook_status,
            )
        )
    elif source_db_path is not None and source_db_path.exists():
        try:
            opened_source_conn = sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)
            try:
                surfaces.extend(
                    archive_reference_surfaces(
                        opened_source_conn,
                        blob_hash,
                        surface_prefix=source_db_path.name,
                        legacy_hook_status=source_legacy_hook_status,
                    )
                )
            finally:
                opened_source_conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not inspect archive source blob references in %s: %s", source_db_path, exc)

    if index_conn is not None:
        surfaces.extend(
            archive_reference_surfaces(
                index_conn,
                blob_hash,
                surface_prefix="index.db",
                legacy_hook_status=index_legacy_hook_status,
            )
        )

    if _table_exists(conn, "raw_sessions"):
        row = conn.execute(
            "SELECT 1 FROM raw_sessions WHERE raw_id = ? LIMIT 1",
            (blob_hash,),
        ).fetchone()
        if row is not None:
            surfaces.append("current.raw_sessions")

    return surfaces


def inspect_blob_liveness(
    source_conn: sqlite3.Connection,
    blob_hash: str,
    *,
    index_conn: sqlite3.Connection | None = None,
    require_index: bool = False,
    include_reservations: bool = True,
    strict: bool = True,
) -> BlobLiveness:
    """Resolve current ownership and reservations without hiding uncertainty.

    The source tier owns direct raw and hook payloads, joined ledger rows, and
    publication reservations.  The index tier owns direct attachment payloads.
    A caller that requires both tiers, such as GC, receives a blocker instead
    of an accidentally permissive answer when the index connection is absent.
    """
    current_source = _table_exists(source_conn, "blob_publication_reservations")
    blockers = list(liveness_schema_blockers(source_conn, tier="source", current_source=current_source))
    if index_conn is None:
        if require_index and current_source:
            blockers.append("index tier is unavailable")
    else:
        blockers.extend(liveness_schema_blockers(index_conn, tier="index", current_source=current_source))

    if blockers and strict:
        return BlobLiveness((), tuple(blockers))
    if not strict:
        blockers = []

    try:
        blob_bytes = blob_hash_bytes(blob_hash)
        if blob_bytes is None:
            return BlobLiveness(())
        ref_types = {
            str(row[0])
            for row in source_conn.execute("SELECT DISTINCT ref_type FROM blob_refs WHERE blob_hash = ?", (blob_bytes,))
        }
        known_ref_types = {ref_type for ref_type, _table, _column in BLOB_REF_LIVENESS_JOIN}
        unknown = sorted(ref_types - known_ref_types)
        if unknown:
            return BlobLiveness((), (f"unknown blob_refs ref_type(s): {', '.join(unknown)}",))
        surfaces = archive_reference_surfaces(
            source_conn,
            blob_hash,
            surface_prefix="source.db",
        )
        # ``archive_reference_surfaces`` also understands source-only tables,
        # but the index schema is intentionally limited here to its actual
        # direct owner. Do not make every tier impersonate every other tier.
        if index_conn is not None and (
            index_conn.execute("SELECT 1 FROM attachments WHERE blob_hash = ? LIMIT 1", (blob_bytes,)).fetchone()
            is not None
        ):
            surfaces.append("index.db.attachments")
        if include_reservations and has_publication_reservation(source_conn, blob_hash):
            surfaces.append("source.db.blob_publication_reservations")
    except sqlite3.Error as exc:
        return BlobLiveness((), (f"blob liveness query is unreadable: {exc}",))
    return BlobLiveness(tuple(surfaces), tuple(blockers))


def blob_hash_is_referenced(
    conn: sqlite3.Connection,
    blob_hash: str,
    *,
    source_db_path: Path | None = None,
) -> bool:
    """Return True if the blob hash is referenced by any archive row."""
    return bool(reference_surfaces(conn, blob_hash, source_db_path=source_db_path))


def has_publication_reservation(conn: sqlite3.Connection, blob_hash: str) -> bool:
    """Return True if a durable publication reservation still protects this hash."""
    blob_bytes = blob_hash_bytes(blob_hash)
    if blob_bytes is None or not _table_exists(conn, "blob_publication_reservations"):
        return False
    return (
        conn.execute(
            "SELECT 1 FROM blob_publication_reservations WHERE blob_hash = ? LIMIT 1",
            (blob_bytes,),
        ).fetchone()
        is not None
    )


def _hex_hashes(conn: sqlite3.Connection, query: str, params: tuple[object, ...] = ()) -> set[str]:
    hashes: set[str] = set()
    for row in conn.execute(query, params):
        value = row[0]
        if isinstance(value, bytes) and len(value) == 32:
            hashes.add(value.hex())
    return hashes


def live_blob_ref_hashes(conn: sqlite3.Connection) -> set[str]:
    """Return hashes whose ``blob_refs`` row(s) join to a live referent.

    A dangling ``blob_refs`` row -- one whose declared referent is absent --
    contributes no hash here. Unknown ``ref_type`` values and known types
    whose referent table/column is unavailable are retained (their hashes
    included, not excluded) so this bulk form fails closed the same way the
    per-row ``blob_refs_row_is_live`` check does.
    """
    if not _table_exists(conn, "blob_refs"):
        return set()
    if not blob_refs_has_ref_type_column(conn):
        # A legacy or malformed reference table cannot prove its rows are
        # dead: fail closed by keeping every hash it names, matching
        # ``blob_refs_row_is_live``'s per-hash stance for the same shape.
        return _hex_hashes(conn, "SELECT blob_hash FROM blob_refs")
    ref_type_counts = {
        str(row[0]): int(row[1]) for row in conn.execute("SELECT ref_type, COUNT(*) FROM blob_refs GROUP BY ref_type")
    }
    known = {ref_type: (table, column) for ref_type, table, column in validated_blob_liveness_joins()}
    legacy_hook_status = "not_applicable"
    if ref_type_counts.get("raw_payload"):
        legacy_hook_status = prepare_legacy_hook_liveness(conn)

    live: set[str] = set()
    for ref_type, count in ref_type_counts.items():
        if not count:
            continue
        if ref_type not in known:
            # Unknown ref_type: fail closed by keeping every hash it names.
            live |= _hex_hashes(conn, "SELECT blob_hash FROM blob_refs WHERE ref_type = ?", (ref_type,))
            continue
        referent_table, referent_column = known[ref_type]
        if not _table_exists(conn, referent_table) or not _column_exists(conn, referent_table, referent_column):
            live |= _hex_hashes(conn, "SELECT blob_hash FROM blob_refs WHERE ref_type = ?", (ref_type,))
            continue
        if ref_type == "raw_payload" and legacy_hook_status in {"ready", "unavailable"}:
            if legacy_hook_status == "unavailable":
                live |= _hex_hashes(conn, "SELECT blob_hash FROM blob_refs WHERE ref_type = ?", (ref_type,))
                continue
            live |= _hex_hashes(
                conn,
                """
                SELECT b.blob_hash
                FROM blob_refs AS b
                WHERE b.ref_type = ?
                  AND (
                      EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = b.ref_id)
                      OR EXISTS (
                          SELECT 1 FROM temp.hook_payload_ref_reconciliation_matches AS m
                          WHERE m.blob_hash = b.blob_hash AND m.orphaned_ref_id = b.ref_id
                      )
                      OR EXISTS (
                          SELECT 1 FROM temp.hook_payload_ref_reconciliation_ambiguous AS a
                          WHERE a.blob_hash = b.blob_hash AND a.orphaned_ref_id = b.ref_id
                      )
                  )
                """,
                (ref_type,),
            )
            continue
        live |= _hex_hashes(
            conn,
            f"""
            SELECT b.blob_hash
            FROM blob_refs AS b
            WHERE b.ref_type = ?
              AND EXISTS (SELECT 1 FROM {referent_table} AS r WHERE r.{referent_column} = b.ref_id)
            """,
            (ref_type,),
        )
    return live


__all__ = [
    "BLOB_REF_LIVENESS_JOIN",
    "DIRECT_BLOB_HASH_TABLES",
    "BlobLiveness",
    "BlobLivenessError",
    "archive_reference_surfaces",
    "blob_hash_bytes",
    "blob_hash_is_referenced",
    "blob_refs_has_ref_type_column",
    "blob_refs_row_is_live",
    "has_publication_reservation",
    "inspect_blob_liveness",
    "legacy_hook_liveness_status",
    "live_blob_ref_hashes",
    "liveness_schema_blockers",
    "prepare_legacy_hook_liveness",
    "reference_surfaces",
    "validated_blob_liveness_joins",
]
