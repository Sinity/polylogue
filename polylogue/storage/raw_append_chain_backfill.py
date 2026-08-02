"""Read-only classification: membershipless append raws provable against live sources.

polylogue-lb39z (Phase 1, item 3): a live, read-only audit found 2,712
``raw_sessions`` rows that are ``revision_kind='append'``,
``revision_authority='quarantined'``, and have NO corresponding
``raw_session_memberships`` row at all -- not ``ambiguous``, not
``deferred``, genuinely absent. This is a distinct population from the two
already-landed Phase-1 actuators:

* :mod:`polylogue.storage.raw_membership_writeback` acts on quarantined raws
  whose membership row already carries a *decided* verdict.
* :mod:`polylogue.storage.live_source_reconciliation` (polylogue-u19l) acts on
  every quarantined raw regardless of membership presence, using a pure
  byte/prefix/offset-range comparison against the live source file.

The membershipless population never reaches either mechanism because
``_promote_contiguous_append_evidence``
(``storage/sqlite/archive_tiers/revision_governance.py``) -- the *only*
mechanism that ever promotes an append raw's ``revision_authority`` -- requires
its byte-contiguous *predecessor* row to already be ``revision_authority=
'byte_proven'``. When the predecessor itself is stuck ``quarantined`` (a
chain of quarantined dominoes), the child can never be promoted through that
path, no matter how many times it runs: it is a genuine fixed point, not a
transient backlog.

This module proves a stuck append raw's *own* claimed byte range
(``[append_start_offset:append_end_offset)``) directly against its live
source file's *current* bytes -- a proof that does not depend on any
ancestor's authority at all, reusing the identical byte-window comparison
:mod:`polylogue.storage.live_source_reconciliation` already implements for
``revision_kind == RawRevisionKind.APPEND`` (see that module's
``compare_raw_against_live_source`` docstring). Promoting this raw's own
``revision_authority`` to ``byte_proven`` does not by itself establish
``predecessor_raw_id``/``baseline_raw_id``/``acquisition_generation`` --
exactly like polylogue-u19l's actuator, that revision-graph linkage remains
``classify_raw_revision_cohort``'s/``_promote_contiguous_append_evidence``'s
concern, and the existing cascade picks it up for free on the next normal
convergence pass, either as the resolved child of its own true predecessor
(once that ancestor is itself proven) or as a newly eligible *parent* for
whatever append fragment sits downstream of it -- unblocking the rest of an
otherwise permanently stuck chain one proven link at a time.

Scope is deliberately narrower than polylogue-u19l's: only rows with
``revision_kind='append'`` AND zero ``raw_session_memberships`` rows. A
membershipless row already covered by u19l's broader classifier is still
independently discoverable here (the two populations overlap), but this
module's own query filters explicitly to the population named by this bead
item, "append-chain backfill" -- it is not a replacement for the broader
live-source reconciliation actuator.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import provider_from_origin
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.live_source_reconciliation import (
    LiveSourceComparison,
    LiveSourceVerdict,
    compare_raw_against_live_source,
)


@dataclass(frozen=True, slots=True)
class AppendChainBackfillCandidate:
    """One membershipless append raw's live-source classification."""

    raw_id: str
    logical_source_key: str | None
    provider: Provider
    source_path: str | None
    blob_hash: bytes
    blob_size: int
    append_start_offset: int
    append_end_offset: int
    comparison: LiveSourceComparison


@dataclass(frozen=True, slots=True)
class AppendChainBackfillPlan:
    """Read-only projection: how much of the membershipless population is provable.

    Mirrors ``LiveSourceReconciliationPlan``'s report-first shape -- nothing
    here marks a row's authority or drops a blob.
    """

    scanned_count: int
    exact_match: tuple[AppendChainBackfillCandidate, ...]
    diverged: tuple[AppendChainBackfillCandidate, ...]
    source_missing: tuple[AppendChainBackfillCandidate, ...]

    @property
    def exact_match_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.exact_match)

    @property
    def diverged_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.diverged)

    @property
    def source_missing_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.source_missing)


def plan_append_chain_backfill(
    source_conn: sqlite3.Connection,
    *,
    blob_store: BlobStore,
    limit: int | None = None,
) -> AppendChainBackfillPlan:
    """Read-only: classify every membershipless quarantined append row.

    Never mutates ``source.db`` and never touches the blob store beyond
    reading. Safe to run against a live archive opened read-only
    (``file:...?mode=ro``).
    """
    original_row_factory = source_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT r.raw_id, r.origin, r.capture_mode, r.logical_source_key,
                   r.source_path, r.blob_size, r.blob_hash,
                   lower(hex(r.blob_hash)) AS blob_hash_hex,
                   r.append_start_offset, r.append_end_offset
            FROM raw_sessions AS r
            WHERE r.revision_authority = 'quarantined'
              AND r.revision_kind = 'append'
              AND r.append_start_offset IS NOT NULL
              AND r.append_end_offset IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM raw_session_memberships AS m WHERE m.raw_id = r.raw_id
              )
            ORDER BY r.raw_id
        """
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = source_conn.execute(query, params).fetchall()

        exact: list[AppendChainBackfillCandidate] = []
        diverged: list[AppendChainBackfillCandidate] = []
        missing: list[AppendChainBackfillCandidate] = []
        for row in rows:
            provider = provider_from_origin(
                Origin.from_string(str(row["origin"])),
                family_hint=row["capture_mode"],
            )
            source_path_value = row["source_path"]
            blob_size = int(row["blob_size"] or 0)
            append_start_offset = int(row["append_start_offset"])
            append_end_offset = int(row["append_end_offset"])

            if source_path_value is None:
                comparison = LiveSourceComparison(
                    verdict=LiveSourceVerdict.SOURCE_MISSING,
                    detail="no recorded source_path",
                )
            else:
                blob_hash_hex = row["blob_hash_hex"]
                if blob_hash_hex is None or not blob_store.exists(blob_hash_hex):
                    comparison = LiveSourceComparison(
                        verdict=LiveSourceVerdict.SOURCE_MISSING,
                        detail="archived blob unavailable",
                    )
                else:
                    raw_payload = blob_store.read_all(blob_hash_hex)
                    comparison = compare_raw_against_live_source(
                        raw_payload=raw_payload,
                        provider=provider,
                        revision_kind=RawRevisionKind.APPEND,
                        source_path=Path(source_path_value),
                        append_start_offset=append_start_offset,
                        append_end_offset=append_end_offset,
                    )

            candidate = AppendChainBackfillCandidate(
                raw_id=str(row["raw_id"]),
                logical_source_key=row["logical_source_key"],
                provider=provider,
                source_path=source_path_value,
                blob_hash=bytes(row["blob_hash"]),
                blob_size=blob_size,
                append_start_offset=append_start_offset,
                append_end_offset=append_end_offset,
                comparison=comparison,
            )
            if comparison.verdict == LiveSourceVerdict.EXACT_MATCH:
                exact.append(candidate)
            elif comparison.verdict == LiveSourceVerdict.SOURCE_MISSING:
                missing.append(candidate)
            else:
                diverged.append(candidate)

        return AppendChainBackfillPlan(
            scanned_count=len(rows),
            exact_match=tuple(exact),
            diverged=tuple(diverged),
            source_missing=tuple(missing),
        )
    finally:
        source_conn.row_factory = original_row_factory


__all__ = [
    "AppendChainBackfillCandidate",
    "AppendChainBackfillPlan",
    "plan_append_chain_backfill",
]
