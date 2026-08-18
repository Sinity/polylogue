"""Read-only classification: quarantined, logical-key-less raws byte-identical to an already-indexed raw.

polylogue-6753s (corrected finding, 2026-08-03): of 7,200 unindexed logical-
source heads in the live archive, 4,305 (17.6 of 22.9 GiB, 77% of the bytes)
are ``revision_authority='quarantined'`` with ``logical_source_key IS NULL``
-- never touched by the reconciler, because every reconciliation path
(``classify_raw_revision_cohort``, the membership pipeline, the append-chain
backfill) keys off ``logical_source_key`` and a row with none is invisible to
all of them. That population is not homogeneous: 4,305 of those rows turn out
to be **byte-identical re-acquisitions** of content the archive already
indexed under a *different* ``raw_id`` -- a re-synced export, a re-run
capture, a duplicate upload -- not missing or novel content at all. The
genuinely novel remainder (~2,895 rows, ~5.3 GiB) is explicitly out of scope
here; it is polylogue-lkrc/polylogue-hjpx's reconciler work, not this bead's.

This module answers exactly one question per candidate row, read-only: does
this raw's ``blob_hash`` match some *other* ``raw_sessions`` row whose content
is already materialized as an indexed ``sessions`` row in ``index.db``? If so,
this raw is a pure duplicate of already-archived content and its terminal
disposition is "superseded by an already-indexed byte-identical raw" -- it
carries nothing the archive doesn't already have.

Terminal-disposition mechanism reused, not invented: this follows the exact
precedent ``polylogue.storage.raw_membership_writeback`` (polylogue-lb39z item
2) already established for "pure fact transfer, not new judgment" promotions.
``revision_authority`` promotes from ``'quarantined'`` to ``'byte_proven'``
(the only two-value axis this closed vocabulary offers for "provenance no
longer contested"); ``revision_authority_evidence`` is deliberately left
``NULL`` rather than widening its closed CHECK vocabulary (widening that
column requires a full ``raw_sessions`` table rebuild, migration 021's own
precedent for *why* that is expensive -- unwarranted for a mechanism whose
real evidence belongs in its own dedicated, independently-verifiable receipt
table anyway, exactly as item 2's writeback module already decided for its
own distinct evidence mechanism). The real evidence -- *which* indexed raw_id
and session this duplicate matches -- is recorded in a dedicated
``raw_byte_duplicate_supersession_receipts`` row (migration 022), never
silently.

This is **strictly read-then-classify-then-mark**: it never touches, re-
parses, or writes to ``index.db``, and it never mutates the indexed twin's
``raw_sessions`` row (its own ``revision_authority``/``logical_source_key``
stay exactly as they were). ``index.db`` is opened read-only, purely to
determine which raw_ids already have a materialized session; nothing here
ever writes to it.

:mod:`polylogue.maintenance.raw_byte_duplicate_supersession_apply` is the
"act" half, following the identical dry-run-by-default /
verified-backup-required-to-apply / immutable-receipt pattern as every other
actuator in this family (``raw_live_source_reconciliation_apply``,
``raw_membership_writeback_apply``, ``raw_append_chain_backfill_apply``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ByteDuplicateSupersessionCandidate:
    """One quarantined, logical-key-less raw proven byte-identical to an indexed twin."""

    raw_id: str
    blob_hash: bytes
    blob_size: int
    #: The OTHER raw_sessions row (never this row) whose bytes are identical
    #: and which already has a materialized session in index.db.
    duplicate_of_raw_id: str
    #: The indexed session_id materialized from duplicate_of_raw_id -- proof,
    #: not just an assertion, that the twin really is indexed.
    duplicate_of_session_id: str


@dataclass(frozen=True, slots=True)
class ByteDuplicateSupersessionPlan:
    """Read-only projection: how much quarantined, logical-key-less authority is a pure byte duplicate.

    Mirrors the report-first shape every sibling classifier in this family
    uses (``LiveSourceReconciliationPlan``, ``AppendChainBackfillPlan``) --
    nothing here marks a row's authority or touches index.db.
    """

    scanned_count: int
    duplicates: tuple[ByteDuplicateSupersessionCandidate, ...]
    #: Scanned rows with no byte-identical indexed twin -- left untouched;
    #: this is the genuinely-novel remainder lkrc/hjpx's reconciler owns.
    novel_count: int

    @property
    def duplicate_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.duplicates)


def plan_byte_duplicate_supersession(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    after_raw_id: str | None = None,
    raw_ids: Sequence[str] | None = None,
) -> ByteDuplicateSupersessionPlan:
    """Read-only: classify quarantined, logical-key-less raws against already-indexed twins.

    ``source_conn`` and ``index_conn`` are both used strictly for reads --
    safe to pass connections opened ``file:...?mode=ro``. Never mutates
    either database.
    """
    original_source_row_factory = source_conn.row_factory
    original_index_row_factory = index_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    index_conn.row_factory = sqlite3.Row
    try:
        if raw_ids is not None and limit is not None:
            raise ValueError("raw_ids cannot be combined with limit")
        query = """
            SELECT raw_id, blob_hash, blob_size, origin, source_path, source_index
            FROM raw_sessions
            WHERE revision_authority = 'quarantined'
              AND logical_source_key IS NULL
              AND parse_error IS NULL
        """
        params: list[object] = []
        if raw_ids is not None:
            selected = tuple(dict.fromkeys(raw_ids))
            if not selected:
                return ByteDuplicateSupersessionPlan(scanned_count=0, duplicates=(), novel_count=0)
            placeholders = ", ".join("?" for _ in selected)
            query += f" AND raw_id IN ({placeholders})"
            params.extend(selected)
        elif after_raw_id is not None:
            query += " AND raw_id > ?"
            params.append(after_raw_id)
        query += " ORDER BY raw_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        candidate_rows = source_conn.execute(query, params).fetchall()
        if not candidate_rows:
            return ByteDuplicateSupersessionPlan(scanned_count=0, duplicates=(), novel_count=0)

        candidate_hashes = sorted({bytes(row["blob_hash"]) for row in candidate_rows})

        # Every raw_sessions row sharing one of the candidate hashes --
        # including other candidates themselves (a candidate's duplicate
        # twin need not have a logical_source_key or a non-quarantined
        # authority; it only needs to already be indexed) -- lets us find
        # each candidate's "other raws with this same blob_hash" without one
        # query per candidate. Self-exclusion happens per-candidate below,
        # not here, since a raw_id can legitimately be the "other" raw for
        # every other candidate sharing its hash while still being a
        # candidate in its own right.
        hash_to_raws: dict[bytes, dict[str, sqlite3.Row]] = {}
        for chunk_start in range(0, len(candidate_hashes), 500):
            chunk = candidate_hashes[chunk_start : chunk_start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            rows = source_conn.execute(
                f"SELECT raw_id, blob_hash, origin, source_path, source_index "
                f"FROM raw_sessions WHERE blob_hash IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                hash_to_raws.setdefault(bytes(row["blob_hash"]), {})[str(row["raw_id"])] = row

        # Which of those raw_ids are actually materialized in index.db --
        # the one live-tier fact this classifier is allowed to read (never
        # write).
        all_raw_ids = sorted({raw_id for raws in hash_to_raws.values() for raw_id in raws})
        indexed_session_by_raw_id: dict[str, str] = {}
        for chunk_start in range(0, len(all_raw_ids), 500):
            raw_id_chunk = all_raw_ids[chunk_start : chunk_start + 500]
            placeholders = ", ".join("?" for _ in raw_id_chunk)
            rows = index_conn.execute(
                f"SELECT raw_id, session_id FROM sessions WHERE raw_id IN ({placeholders}) ORDER BY session_id",
                raw_id_chunk,
            ).fetchall()
            for row in rows:
                raw_id = str(row["raw_id"])
                # Deterministic: first (lowest) session_id wins if a raw_id
                # somehow materialized more than one session row.
                indexed_session_by_raw_id.setdefault(raw_id, str(row["session_id"]))

        duplicates: list[ByteDuplicateSupersessionCandidate] = []
        novel_count = 0
        for row in candidate_rows:
            raw_id = str(row["raw_id"])
            blob_hash = bytes(row["blob_hash"])
            # Same origin, but deliberately NOT the same source coordinate.
            # #3949 narrowed this to require an identical
            # (origin, source_path, source_index) twin, which made the whole
            # class unreachable: raw_artifacts carries a UNIQUE index on exactly
            # that triple for every non-deferred artifact kind, so two raws
            # sharing one coordinate cannot both exist. The duplicates that do
            # exist -- and that polylogue-vzn6 measures at ~46% of stored blob
            # bytes -- are identical bytes acquired at DIFFERENT paths, e.g. the
            # same export re-downloaded. Origin equality keeps a codex raw from
            # superseding a chatgpt one; the real safety gate is below, where a
            # twin only counts if it is already materialized in index.db.
            other_raws_for_hash = [
                other
                for other_id, other in hash_to_raws.get(blob_hash, {}).items()
                if other_id != raw_id and other["origin"] == row["origin"]
            ]
            indexed_matches = sorted(
                str(other["raw_id"])
                for other in other_raws_for_hash
                if str(other["raw_id"]) in indexed_session_by_raw_id
            )
            if not indexed_matches:
                novel_count += 1
                continue
            duplicate_of_raw_id = indexed_matches[0]
            duplicates.append(
                ByteDuplicateSupersessionCandidate(
                    raw_id=raw_id,
                    blob_hash=blob_hash,
                    blob_size=int(row["blob_size"]),
                    duplicate_of_raw_id=duplicate_of_raw_id,
                    duplicate_of_session_id=indexed_session_by_raw_id[duplicate_of_raw_id],
                )
            )

        return ByteDuplicateSupersessionPlan(
            scanned_count=len(candidate_rows),
            duplicates=tuple(duplicates),
            novel_count=novel_count,
        )
    finally:
        source_conn.row_factory = original_source_row_factory
        index_conn.row_factory = original_index_row_factory


__all__ = [
    "ByteDuplicateSupersessionCandidate",
    "ByteDuplicateSupersessionPlan",
    "plan_byte_duplicate_supersession",
]
