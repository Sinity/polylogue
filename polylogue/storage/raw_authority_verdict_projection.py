"""Read-only projection of live ``raw_sessions`` evidence onto ``RawAuthorityVerdict``.

Phase 2 of the raw-authority redesign (polylogue-w6hql). This is a **read-only
consumer** of the existing revision-authority evidence -- it issues no writes
to ``raw_sessions`` or any of the fragmented census/blocker/plan tables. Full
snapshots reuse the exact byte-proof machinery (``classify_historical_full_
revision_streams``) used by the real classifier. Append fragments reuse the
same persisted byte-proven predecessor links written by contiguous-append
promotion in that classifier, so verdicts cannot silently diverge from the
authority that replay already accepts.
"""

from __future__ import annotations

import sqlite3
from typing import BinaryIO, Protocol

from polylogue.archive.raw_authority_verdict import derive_raw_authority_verdict
from polylogue.archive.revision_authority import (
    HistoricalRawRevisionStream,
    RawRevisionAuthority,
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
    """Return the closed verdict for every raw in a cohort without writing it.

    Full snapshots are reclassified from their bytes. Append fragments cannot
    be compared as complete snapshots, so their verdict comes from the
    byte-proven predecessor chain the production classifier already persisted:
    an accepted fragment is VERIFIED at the chain head and SUPERSEDED once a
    later accepted fragment names it as predecessor. ASSERTED fragments remain
    UNCHECKED; QUARANTINED fragments are DIVERGED.
    """
    if store._blob_publisher is None:
        raise RuntimeError("raw authority verdict projection requires a readable blob publisher")
    source_conn = store._ensure_source_conn()
    rows = source_conn.execute(
        """
        SELECT raw_id, lower(hex(blob_hash)) AS blob_hash, blob_size, revision_kind,
               revision_authority, predecessor_raw_id
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
    verdicts: dict[str, RawAuthorityVerdict] = {
        str(row[0]): RawAuthorityVerdict.UNCHECKED for row in rows if str(row[3]) == "unknown"
    }
    byte_proven_successors = {
        str(row[5]) for row in rows if str(row[4]) == RawRevisionAuthority.BYTE_PROVEN.value and row[5] is not None
    }
    provable: list[HistoricalRawRevisionStream] = []
    for row in rows:
        raw_id, blob_hash, blob_size, revision_kind, _authority, _predecessor_raw_id = row
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
    for raw_id, _blob_hash, _blob_size, revision_kind, authority, _predecessor_raw_id in rows:
        raw_id_text = str(raw_id)
        kind = str(revision_kind)
        if kind == "append":
            if str(authority) == RawRevisionAuthority.BYTE_PROVEN.value:
                verdicts[raw_id_text] = (
                    RawAuthorityVerdict.SUPERSEDED
                    if raw_id_text in byte_proven_successors
                    else RawAuthorityVerdict.VERIFIED
                )
            elif str(authority) == RawRevisionAuthority.QUARANTINED.value:
                verdicts[raw_id_text] = RawAuthorityVerdict.DIVERGED
            else:
                verdicts[raw_id_text] = RawAuthorityVerdict.UNCHECKED
        elif (
            kind == "full"
            and str(authority) == RawRevisionAuthority.BYTE_PROVEN.value
            and raw_id_text in byte_proven_successors
            and verdicts.get(raw_id_text) in (RawAuthorityVerdict.SOLE_COPY, RawAuthorityVerdict.VERIFIED)
        ):
            verdicts[raw_id_text] = RawAuthorityVerdict.SUPERSEDED
    return verdicts


__all__ = ["RawAuthorityVerdictProjectionHost", "project_raw_authority_verdicts"]
