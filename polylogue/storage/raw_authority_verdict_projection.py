"""Read-only projection of live ``raw_sessions`` evidence onto ``RawAuthorityVerdict``.

Phase 2 of the raw-authority redesign (polylogue-w6hql). This is a **read-only
consumer** of the existing revision-authority evidence -- it issues no writes
to ``raw_sessions`` or any of the fragmented census/blocker/plan tables, and
reuses the exact same byte-proof machinery (``classify_historical_full_
revision_streams``) the real classifier (``classify_raw_revision_cohort`` in
``storage/sqlite/archive_tiers/revision_governance.py``) already uses, so its
verdicts cannot silently diverge from what governance actually proved.

Scope of this phase: ``revision_kind = 'full'`` cohorts only, matching
``classify_raw_revision_cohort``'s own current scope. Append-only cohorts
(``revision_kind = 'append'``) are out of scope for this projection and raise
``NotImplementedError`` rather than silently returning a wrong verdict --
their authority is governed by contiguous-append promotion
(``_promote_contiguous_append_evidence``), a different proof shape that this
phase does not attempt to collapse (see polylogue-w6hql notes for the
follow-up this is filed under).
"""

from __future__ import annotations

import sqlite3
from typing import BinaryIO, Protocol

from polylogue.archive.raw_authority_verdict import derive_raw_authority_verdict
from polylogue.archive.revision_authority import (
    HistoricalRawRevisionStream,
    classify_historical_full_revision_streams,
)
from polylogue.core.enums import RawAuthorityVerdict
from polylogue.storage.blob_publication import ArchiveBlobPublisher


class RawAuthorityVerdictProjectionHost(Protocol):
    """The narrow read-only slice this module needs from ``ArchiveStore``."""

    _blob_publisher: ArchiveBlobPublisher | None

    def _ensure_source_conn(self) -> sqlite3.Connection: ...


def project_raw_authority_verdicts(
    store: RawAuthorityVerdictProjectionHost,
    logical_source_key: str,
) -> dict[str, RawAuthorityVerdict]:
    """Return the closed verdict for every ``revision_kind='full'`` raw in a cohort.

    Read-only: opens the source connection and blob store for reading only,
    issues no writes. Raises ``NotImplementedError`` if the cohort contains
    any ``revision_kind != 'full'`` row (see module docstring).
    """
    if store._blob_publisher is None:
        raise RuntimeError("raw authority verdict projection requires a readable blob publisher")
    source_conn = store._ensure_source_conn()
    rows = source_conn.execute(
        """
        SELECT raw_id, lower(hex(blob_hash)) AS blob_hash, blob_size, revision_kind
        FROM raw_sessions
        WHERE logical_source_key = ?
        """,
        (logical_source_key,),
    ).fetchall()
    if not rows:
        return {}

    # 'unknown' is the pre-governance default (DDL:
    # revision_kind NOT NULL DEFAULT 'unknown') -- identity/kind resolution
    # has not run over this raw yet, so no verdict can be proven.
    unresolved = [str(row[0]) for row in rows if str(row[3]) == "unknown"]
    non_full_resolved = [str(row[0]) for row in rows if str(row[3]) not in ("full", "unknown")]
    if non_full_resolved:
        raise NotImplementedError(
            f"raw authority verdict projection is scoped to revision_kind='full' cohorts "
            f"this phase; logical_source_key={logical_source_key!r} contains non-full raw_ids "
            f"{non_full_resolved!r} (see polylogue-w6hql)"
        )

    verdicts: dict[str, RawAuthorityVerdict] = dict.fromkeys(unresolved, RawAuthorityVerdict.UNCHECKED)
    provable: list[HistoricalRawRevisionStream] = []
    for row in rows:
        raw_id, blob_hash, blob_size, revision_kind = row
        if str(revision_kind) != "full":
            continue

        def open_payload(blob_hash: str = str(blob_hash)) -> BinaryIO:
            assert store._blob_publisher is not None
            handle: BinaryIO = store._blob_publisher.open(blob_hash)
            return handle

        provable.append(
            HistoricalRawRevisionStream(
                raw_id=str(raw_id),
                payload_size=int(blob_size),
                open_payload=open_payload,
            )
        )

    if provable:
        decisions = classify_historical_full_revision_streams(provable)
        verdicts.update(derive_raw_authority_verdict(decisions))
    return verdicts


__all__ = ["RawAuthorityVerdictProjectionHost", "project_raw_authority_verdicts"]
