"""Read-only reclassification report for browser captures stamped ``unknown-export``.

polylogue-mvq8: a capture larger than ``_STREAMING_FULL_INGEST_BYTES`` (8MiB)
detects its provider via ``sources/live/batch_support.py``'s
``_browser_capture_prefix_probe``, which used to read only the first 1MiB
(``_BROWSER_CAPTURE_PREFIX_PROBE_BYTES``) of the envelope. Because the
receiver serializes envelopes with ``sort_keys=True``
(``browser_capture/receiver.py``), ``raw_provider_payload`` (an unbounded
copy of the provider's own wire payload) sorts alphabetically *before*
``session`` -- so once ``raw_provider_payload`` alone exceeds that 1MiB
window, ``session.provider`` never appeared in the prefix at all and the
capture was permanently stamped ``unknown-export``.

Fixing the probe (this bead's detection-side fix, now landed) only prevents
*new* misclassification -- ``origin`` is stamped durably on the ``raw_sessions``
row at acquisition time, so already-committed rows stay wrong until something
re-detects them. This module is deliberately NOT a mutator. It mirrors
``live_source_reconciliation.py``'s shape: given one ``unknown-export`` raw
row's already-archived blob bytes, does the fixed detection logic
(``_stream_browser_capture_provider``, a memory-bounded ``ijson`` scan
identical to the one the acquisition path now uses) recover a real provider?
Acting on a "reclassifiable" verdict -- rewriting ``origin``/``capture_mode``
on the raw row, re-parsing, re-indexing -- is an explicitly separate,
operator-authorized follow-up; nothing here mutates ``source.db`` or the
blob store beyond reading.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import origin_from_provider
from polylogue.sources.source_acquisition_components import _stream_browser_capture_provider
from polylogue.storage.blob_store import BlobStore

#: Default source-path scope: the browser-capture spool shape the mvq8 audit
#: measured (23 rows / 641,613,073 bytes). Pass ``source_path_like=None`` to
#: scan every ``unknown-export`` row regardless of where it came from.
DEFAULT_SOURCE_PATH_LIKE: Final = "%browser-capture%"

#: The durable repair is intentionally narrower than the report. The bead's
#: measured population is the ChatGPT browser-capture spool, and the actuator
#: must not reclassify another provider merely because its envelope is valid.
CHATGPT_BROWSER_CAPTURE_SOURCE_PATH_LIKE: Final = "%/browser-capture/chatgpt/%"


class UnknownExportReclassificationVerdict:
    """Closed vocabulary for one ``unknown-export`` raw row's re-detection outcome."""

    RECLASSIFIABLE: Final = "reclassifiable"
    STILL_UNKNOWN: Final = "still_unknown"
    BLOB_MISSING: Final = "blob_missing"


@dataclass(frozen=True, slots=True)
class UnknownExportReclassificationCandidate:
    """One ``unknown-export`` raw row's re-detection classification."""

    raw_id: str
    source_path: str
    blob_size: int
    verdict: str
    recovered_provider: Provider | None = None
    recovered_origin: Origin | None = None
    previous_capture_mode: Provider | None = None


@dataclass(frozen=True, slots=True)
class UnknownExportReclassificationPlan:
    """Read-only projection: how much stored ``unknown-export`` evidence is re-detectable.

    Mirrors ``live_source_reconciliation.py``'s ``LiveSourceReconciliationPlan``
    shape (report-first, no mutation) -- this is the report a genuinely
    separate, explicitly operator-authorized pass would act on.
    """

    scanned_count: int
    reclassifiable: tuple[UnknownExportReclassificationCandidate, ...]
    still_unknown: tuple[UnknownExportReclassificationCandidate, ...]
    blob_missing: tuple[UnknownExportReclassificationCandidate, ...]

    @property
    def reclassifiable_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.reclassifiable)

    @property
    def still_unknown_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.still_unknown)

    @property
    def blob_missing_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.blob_missing)

    @property
    def reclassifiable_by_origin(self) -> dict[str, int]:
        """Count reclassifiable rows by their recovered origin, for a quick summary."""
        counts: dict[str, int] = {}
        for candidate in self.reclassifiable:
            key = candidate.recovered_origin.value if candidate.recovered_origin is not None else "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def chatgpt_reclassifiable(self) -> tuple[UnknownExportReclassificationCandidate, ...]:
        """Return only rows proven by an embedded ChatGPT provider marker."""
        return tuple(
            candidate
            for candidate in self.reclassifiable
            if candidate.recovered_provider is Provider.CHATGPT and candidate.recovered_origin is Origin.CHATGPT_EXPORT
        )

    @property
    def non_chatgpt_reclassifiable(self) -> tuple[UnknownExportReclassificationCandidate, ...]:
        """Return valid envelopes the ChatGPT-only repair deliberately leaves alone."""
        chatgpt_ids = {candidate.raw_id for candidate in self.chatgpt_reclassifiable}
        return tuple(candidate for candidate in self.reclassifiable if candidate.raw_id not in chatgpt_ids)


def plan_unknown_export_reclassification(
    source_conn: sqlite3.Connection,
    *,
    blob_store: BlobStore,
    source_path_like: str | None = DEFAULT_SOURCE_PATH_LIKE,
    limit: int | None = None,
) -> UnknownExportReclassificationPlan:
    """Read-only: re-run the fixed provider probe against stored ``unknown-export`` rows.

    Never mutates ``source.db`` and never touches the blob store beyond
    reading -- safe to run against a live archive opened read-only
    (``file:...?mode=ro``).
    """
    original_row_factory = source_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    try:
        query = (
            "SELECT raw_id, source_path, blob_size, capture_mode, "
            "lower(hex(blob_hash)) AS blob_hash_hex "
            "FROM raw_sessions WHERE origin = 'unknown-export'"
        )
        params: list[object] = []
        if source_path_like is not None:
            query += " AND source_path LIKE ?"
            params.append(source_path_like)
        query += " ORDER BY raw_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = source_conn.execute(query, tuple(params)).fetchall()

        reclassifiable: list[UnknownExportReclassificationCandidate] = []
        still_unknown: list[UnknownExportReclassificationCandidate] = []
        blob_missing: list[UnknownExportReclassificationCandidate] = []
        for row in rows:
            raw_id = str(row["raw_id"])
            source_path = str(row["source_path"])
            blob_size = int(row["blob_size"] or 0)
            previous_capture_mode = (
                Provider.from_string(str(row["capture_mode"])) if row["capture_mode"] is not None else None
            )
            blob_hash_hex = row["blob_hash_hex"]
            if blob_hash_hex is None or not blob_store.exists(blob_hash_hex):
                blob_missing.append(
                    UnknownExportReclassificationCandidate(
                        raw_id=raw_id,
                        source_path=source_path,
                        blob_size=blob_size,
                        verdict=UnknownExportReclassificationVerdict.BLOB_MISSING,
                        previous_capture_mode=previous_capture_mode,
                    )
                )
                continue

            provider = _stream_browser_capture_provider(blob_store, blob_hash_hex)
            if provider is Provider.UNKNOWN:
                still_unknown.append(
                    UnknownExportReclassificationCandidate(
                        raw_id=raw_id,
                        source_path=source_path,
                        blob_size=blob_size,
                        verdict=UnknownExportReclassificationVerdict.STILL_UNKNOWN,
                        previous_capture_mode=previous_capture_mode,
                    )
                )
                continue

            reclassifiable.append(
                UnknownExportReclassificationCandidate(
                    raw_id=raw_id,
                    source_path=source_path,
                    blob_size=blob_size,
                    verdict=UnknownExportReclassificationVerdict.RECLASSIFIABLE,
                    recovered_provider=provider,
                    recovered_origin=origin_from_provider(provider),
                    previous_capture_mode=previous_capture_mode,
                )
            )

        return UnknownExportReclassificationPlan(
            scanned_count=len(rows),
            reclassifiable=tuple(reclassifiable),
            still_unknown=tuple(still_unknown),
            blob_missing=tuple(blob_missing),
        )
    finally:
        source_conn.row_factory = original_row_factory


__all__ = [
    "CHATGPT_BROWSER_CAPTURE_SOURCE_PATH_LIKE",
    "DEFAULT_SOURCE_PATH_LIKE",
    "UnknownExportReclassificationCandidate",
    "UnknownExportReclassificationPlan",
    "UnknownExportReclassificationVerdict",
    "plan_unknown_export_reclassification",
]
