"""Read-only classification: fully-quarantined raw_sessions groups sharing (source_path, blob_hash).

polylogue-zm4w8 (measured live 2026-08-03, source.db mode=ro): of 5,203
quarantined ``codex-session`` raw_sessions rows (45.73 GB), 3,426 distinct
``(source_path, blob_hash)`` pairs exist but 1,777 rows are pure redundant
duplicates -- same ``source_path`` AND same ``blob_hash`` as another already-
counted row -- 22.19 GB reclaimable. Sample: one file has NINE separate
``raw_id`` rows, all byte-identical, all ``revision_kind='unknown'``,
``revision_authority='quarantined'``.

This is a distinct gap from :mod:`polylogue.storage.raw_byte_duplicate_supersession`
(polylogue-6753s): that module matches a quarantined row against an
**already-indexed** twin (a different ``raw_id`` with a materialized
``sessions`` row in ``index.db``). The population here never has an indexed
twin at all -- every member of a qualifying group starts out quarantined,
with nothing yet materialized for any of them. Confirmed live: ZERO of these
duplicate ``blob_hash`` values have any non-quarantined twin anywhere in
``raw_sessions``, so the existing actuator's classifier (whose universe is
exactly "quarantined rows with an indexed twin") returns zero candidates for
this class by construction.

This module answers exactly one question per ``(source_path, blob_hash)``
group among quarantined rows: does more than one raw_sessions row share this
exact source_path and blob_hash, and does NONE of them (nor any other raw
sharing that blob_hash, regardless of source_path) already have a
materialized session in ``index.db`` or a non-quarantined
``revision_authority``? If so, this group is a genuine unresolved cluster of
repeated acquisitions of the same content -- real, legitimate content that
was simply never chosen as the group's representative and materialized.

Deterministic representative selection: the lowest (lexicographically
smallest) ``raw_id`` in each group is the representative a corresponding
actuator would materialize; the rest are the group's duplicates. This
mirrors ``raw_byte_duplicate_supersession``'s own "first (lowest) id wins"
precedent for a tie with no other governing signal.

:mod:`polylogue.maintenance.raw_quarantine_group_dedup_apply` is the "act"
half: it promotes the representative through the real materialization path
(``ParsingService.parse_from_raw`` -> ``write_parsed_session_to_archive`` ->
``refresh_session_insights_bulk``) and marks the rest ``byte_proven`` with a
receipt, exactly the same safety pattern as every other actuator in this
family (dry-run by default, verified-backup-required-to-apply, immutable
receipt, never touches blob storage or runs GC).

This module itself is **strictly read-only**: it never mutates ``source.db``
or ``index.db``, and both connections are safe to open ``mode=ro``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawQuarantineGroup:
    """One (source_path, blob_hash) group of >1 fully-quarantined byte-identical raws."""

    source_path: str
    blob_hash: bytes
    blob_size: int
    #: All raw_ids in the group, sorted ascending -- deterministic.
    raw_ids: tuple[str, ...]

    @property
    def representative_raw_id(self) -> str:
        """The deterministic (lowest raw_id) member an actuator would materialize."""
        return self.raw_ids[0]

    @property
    def duplicate_raw_ids(self) -> tuple[str, ...]:
        """Every group member other than the representative."""
        return self.raw_ids[1:]


@dataclass(frozen=True, slots=True)
class RawQuarantineGroupDedupPlan:
    """Read-only projection: how much quarantined authority is a within-group duplicate cluster."""

    #: Total quarantined, source_path-bearing rows examined.
    scanned_count: int
    groups: tuple[RawQuarantineGroup, ...]
    #: (source_path, blob_hash) pairs with count > 1 among quarantined rows
    #: that were EXCLUDED because some raw sharing that blob_hash already has
    #: a materialized session or a non-quarantined revision_authority --
    #: already resolved, or raw_byte_duplicate_supersession's own territory.
    already_resolved_group_count: int

    @property
    def duplicate_count(self) -> int:
        return sum(len(group.duplicate_raw_ids) for group in self.groups)

    @property
    def duplicate_bytes(self) -> int:
        return sum(group.blob_size * len(group.duplicate_raw_ids) for group in self.groups)


def plan_raw_quarantine_group_dedup(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> RawQuarantineGroupDedupPlan:
    """Read-only: classify quarantined raws into fully-unresolved duplicate groups.

    ``source_conn`` and ``index_conn`` are both used strictly for reads --
    safe to pass connections opened ``file:...?mode=ro``. Never mutates
    either database. ``limit`` caps the number of qualifying *groups*
    returned (not the number of rows scanned).
    """
    original_source_row_factory = source_conn.row_factory
    original_index_row_factory = index_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    index_conn.row_factory = sqlite3.Row
    try:
        candidate_rows = source_conn.execute(
            """
            SELECT raw_id, source_path, blob_hash, blob_size
            FROM raw_sessions
            WHERE revision_authority = 'quarantined' AND source_path IS NOT NULL
            ORDER BY source_path, blob_hash, raw_id
            """
        ).fetchall()
        if not candidate_rows:
            return RawQuarantineGroupDedupPlan(scanned_count=0, groups=(), already_resolved_group_count=0)

        grouped: dict[tuple[str, bytes], list[sqlite3.Row]] = {}
        for row in candidate_rows:
            key = (str(row["source_path"]), bytes(row["blob_hash"]))
            grouped.setdefault(key, []).append(row)

        multi_member_keys = [key for key, rows in grouped.items() if len(rows) > 1]
        if not multi_member_keys:
            return RawQuarantineGroupDedupPlan(
                scanned_count=len(candidate_rows), groups=(), already_resolved_group_count=0
            )

        # Which blob_hash values (across ALL raw_sessions rows, any
        # source_path/revision_authority) already have a materialized
        # index.db session or a non-quarantined revision_authority -- those
        # groups are out of scope: either already resolved, or
        # raw_byte_duplicate_supersession's own territory.
        distinct_hashes = sorted({blob_hash for _, blob_hash in multi_member_keys})
        hash_already_resolved: dict[bytes, bool] = {}
        for chunk_start in range(0, len(distinct_hashes), 500):
            hash_chunk = distinct_hashes[chunk_start : chunk_start + 500]
            placeholders = ", ".join("?" for _ in hash_chunk)
            rows = source_conn.execute(
                f"SELECT raw_id, blob_hash, revision_authority FROM raw_sessions WHERE blob_hash IN ({placeholders})",
                hash_chunk,
            ).fetchall()
            raw_ids_by_hash: dict[bytes, list[str]] = {}
            non_quarantined_hashes: set[bytes] = set()
            for row in rows:
                blob_hash = bytes(row["blob_hash"])
                raw_ids_by_hash.setdefault(blob_hash, []).append(str(row["raw_id"]))
                if str(row["revision_authority"]) != "quarantined":
                    non_quarantined_hashes.add(blob_hash)

            all_raw_ids = sorted({raw_id for hash_raw_ids in raw_ids_by_hash.values() for raw_id in hash_raw_ids})
            indexed_raw_ids: set[str] = set()
            for id_chunk_start in range(0, len(all_raw_ids), 500):
                id_chunk = all_raw_ids[id_chunk_start : id_chunk_start + 500]
                id_placeholders = ", ".join("?" for _ in id_chunk)
                for index_row in index_conn.execute(
                    f"SELECT DISTINCT raw_id FROM sessions WHERE raw_id IN ({id_placeholders})", id_chunk
                ):
                    indexed_raw_ids.add(str(index_row[0]))

            for blob_hash, hash_raw_ids in raw_ids_by_hash.items():
                hash_already_resolved[blob_hash] = blob_hash in non_quarantined_hashes or any(
                    raw_id in indexed_raw_ids for raw_id in hash_raw_ids
                )

        groups: list[RawQuarantineGroup] = []
        already_resolved_group_count = 0
        for key in multi_member_keys:
            source_path, blob_hash = key
            if hash_already_resolved.get(blob_hash, False):
                already_resolved_group_count += 1
                continue
            rows = grouped[key]
            raw_ids = tuple(sorted(str(row["raw_id"]) for row in rows))
            blob_size = int(rows[0]["blob_size"])
            groups.append(
                RawQuarantineGroup(
                    source_path=source_path,
                    blob_hash=blob_hash,
                    blob_size=blob_size,
                    raw_ids=raw_ids,
                )
            )
            if limit is not None and len(groups) >= limit:
                break

        return RawQuarantineGroupDedupPlan(
            scanned_count=len(candidate_rows),
            groups=tuple(groups),
            already_resolved_group_count=already_resolved_group_count,
        )
    finally:
        source_conn.row_factory = original_source_row_factory
        index_conn.row_factory = original_index_row_factory


__all__ = [
    "RawQuarantineGroup",
    "RawQuarantineGroupDedupPlan",
    "plan_raw_quarantine_group_dedup",
]
