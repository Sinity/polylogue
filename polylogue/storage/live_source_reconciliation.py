"""Live-source-verification-based classification for quarantined raw evidence.

polylogue-u19l: ~59GB of ``raw_sessions`` rows sit at
``revision_authority='quarantined'`` -- an absorbing state the existing
``RawAuthorityReconciler`` revision-graph heuristics never resolve for most
of them. The prior investigation on this bead found a strictly stronger and
cheaper proof available for the common case: "the live source file still
contains these exact bytes" -- no revision-graph reasoning required at all.

This module is deliberately NOT an extension of ``RawAuthorityReconciler`` or
``storage/raw_retention.py``'s revision-authority machinery. It is a small,
independent, read-only classifier: given one quarantined raw row's archived
bytes and its recorded ``source_path``, does the live file still contain
those bytes (optionally, for Codex append captures, modulo the synthetic
``session_meta`` header historical rows had spliced in before hashing --
see ``sources/live/batch.py``'s ``_append_payload_for_provider``)?

Scope, deliberately narrow:

* Pure byte/prefix/offset-range comparison against a real on-disk file.
  Zip-bundle exports (ChatGPT/Claude-ai GDPR exports) and other non-flat-file
  sources need a parsed-object-identity comparator, not a byte comparator --
  out of scope here (the prior investigation covered those separately and
  found them 100% reconcilable by parsed conversation id, not by bytes).
* Read-only. Nothing here writes to ``source.db``, marks a row
  ``byte_proven``, or deletes a blob. Acting on a "reconcilable" classification
  (marking authority, dropping a redundant archived duplicate, or anything
  else that mutates live authority state) is an explicitly separate,
  operator-authorized follow-up -- this module only classifies and reports.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path
from typing import Final

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import provider_from_origin
from polylogue.storage.blob_store import BlobStore

#: The exact single-line synthetic header shape
#: ``sources/live/batch.py::_append_payload_for_provider`` used to prepend
#: ahead of every Codex append-mode capture before hashing/storing it (retired
#: for NEW writes by polylogue-u19l; historical rows still carry it).
_CODEX_SYNTHETIC_HEADER_RECORD_TYPE: Final = "session_meta"


class LiveSourceVerdict:
    """Closed vocabulary for a raw row's live-source reconciliation outcome."""

    EXACT_MATCH: Final = "exact_match"
    DIVERGED: Final = "diverged"
    SOURCE_MISSING: Final = "source_missing"


@dataclass(frozen=True, slots=True)
class LiveSourceComparison:
    """Result of comparing one raw row's archived bytes to its live source."""

    verdict: str
    #: True when the match only succeeded after stripping the historical
    #: Codex synthetic ``session_meta`` header from the archived payload --
    #: i.e. the live file's real bytes matched, but the stored blob did not,
    #: purely because of the header artifact this bead's part (B) retires.
    matched_after_codex_header_strip: bool = False
    detail: str | None = None


def _strip_codex_synthetic_header(payload: bytes) -> bytes | None:
    """Remove a leading synthetic ``session_meta`` line, if present.

    Only strips a header matching the *exact* deterministic shape
    ``_append_payload_for_provider`` used to emit
    (``json.dumps({"type": "session_meta", "payload": {"id": <id>}},
    separators=(",", ":"))``) -- not any arbitrary first-line JSON object, so
    a live file whose own real first line legitimately happens to be a
    ``session_meta`` record is never mistaken for the synthetic artifact.
    Returns ``None`` when there is nothing matching that exact shape to
    strip.
    """
    newline_at = payload.find(b"\n")
    if newline_at < 0:
        return None
    first_line = payload[:newline_at]
    try:
        record = json_loads(first_line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("type") != _CODEX_SYNTHETIC_HEADER_RECORD_TYPE:
        return None
    inner = record.get("payload")
    if not isinstance(inner, dict) or "id" not in inner or set(inner.keys()) != {"id"}:
        return None
    if set(record.keys()) != {"type", "payload"}:
        return None
    reencoded = json_dumps(
        {"type": "session_meta", "payload": {"id": inner["id"]}},
        separators=(",", ":"),
    ).encode()
    if first_line != reencoded:
        return None
    return payload[newline_at + 1 :]


def compare_raw_against_live_source(
    *,
    raw_payload: bytes,
    provider: Provider,
    revision_kind: RawRevisionKind,
    source_path: Path,
    append_start_offset: int | None = None,
    append_end_offset: int | None = None,
) -> LiveSourceComparison:
    """Classify one raw row's archived bytes against its live source file.

    Comparison window:

    * ``revision_kind == APPEND`` with both offsets recorded: compare against
      the live file's exact ``[append_start_offset:append_end_offset)`` byte
      range -- the offsets the append plan itself recorded at capture time
      (``sources/live/batch.py::_AppendPlan``), independent of any header
      that plan's stored payload may carry.
    * Anything else (``FULL``/``UNKNOWN``, or an append row missing offsets):
      prefix-compare the live file's first ``len(raw_payload)`` bytes.

    For a Codex append row, a second candidate is tried after stripping the
    historical synthetic ``session_meta`` header (see
    :func:`_strip_codex_synthetic_header`) -- this is the one provider/kind
    combination the investigation found permanently defeats a naive
    byte-identity check for reasons unrelated to real data loss.

    Returns ``SOURCE_MISSING`` when the source file cannot be opened at all
    (deleted, permission denied, not a regular file) -- this function never
    raises for an ordinary missing-file condition.
    """
    candidates: list[tuple[bytes, bool]] = [(raw_payload, False)]
    if provider is Provider.CODEX and revision_kind is RawRevisionKind.APPEND:
        stripped = _strip_codex_synthetic_header(raw_payload)
        if stripped is not None:
            candidates.append((stripped, True))

    read_start = 0
    fixed_window: int | None = None
    if (
        revision_kind is RawRevisionKind.APPEND
        and append_start_offset is not None
        and append_end_offset is not None
        and append_end_offset >= append_start_offset
    ):
        read_start = append_start_offset
        fixed_window = append_end_offset - append_start_offset

    try:
        with source_path.open("rb") as handle:
            for candidate, header_stripped in candidates:
                length = fixed_window if fixed_window is not None else len(candidate)
                handle.seek(read_start)
                live_slice = handle.read(length)
                if live_slice == candidate:
                    return LiveSourceComparison(
                        verdict=LiveSourceVerdict.EXACT_MATCH,
                        matched_after_codex_header_strip=header_stripped,
                    )
    except FileNotFoundError:
        return LiveSourceComparison(verdict=LiveSourceVerdict.SOURCE_MISSING, detail="source file no longer exists")
    except OSError as exc:
        return LiveSourceComparison(verdict=LiveSourceVerdict.SOURCE_MISSING, detail=f"source file unreadable: {exc}")
    return LiveSourceComparison(verdict=LiveSourceVerdict.DIVERGED)


@dataclass(frozen=True, slots=True)
class LiveSourceReconciliationCandidate:
    """One quarantined raw row's live-source classification."""

    raw_id: str
    logical_source_key: str | None
    provider: Provider
    revision_kind: RawRevisionKind
    source_path: str | None
    blob_size: int
    comparison: LiveSourceComparison


@dataclass(frozen=True, slots=True)
class LiveSourceReconciliationPlan:
    """Read-only projection: how much quarantined authority is live-source-verifiable.

    Mirrors ``raw_retention.py``'s ``plan_stale_supersession_reissue`` shape
    (report-first, no mutation) -- this is the report a genuinely separate,
    explicitly operator-authorized pass would act on; nothing here marks a
    row's authority or drops a blob.
    """

    scanned_count: int
    exact_match: tuple[LiveSourceReconciliationCandidate, ...]
    diverged: tuple[LiveSourceReconciliationCandidate, ...]
    source_missing: tuple[LiveSourceReconciliationCandidate, ...]

    @property
    def exact_match_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.exact_match)

    @property
    def diverged_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.diverged)

    @property
    def source_missing_bytes(self) -> int:
        return sum(candidate.blob_size for candidate in self.source_missing)

    @property
    def codex_header_strip_match_count(self) -> int:
        return sum(1 for candidate in self.exact_match if candidate.comparison.matched_after_codex_header_strip)


def plan_quarantined_live_source_reconciliation(
    source_conn: sqlite3.Connection,
    *,
    blob_store: BlobStore,
    limit: int | None = None,
) -> LiveSourceReconciliationPlan:
    """Read-only: classify every quarantined raw row against its live source.

    Never mutates ``source.db`` and never touches the blob store beyond
    reading. Safe to run against a live archive opened read-only
    (``file:...?mode=ro``).
    """
    original_row_factory = source_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT raw_id, origin, capture_mode, logical_source_key, revision_kind,
                   source_path, blob_size, lower(hex(blob_hash)) AS blob_hash_hex,
                   append_start_offset, append_end_offset
            FROM raw_sessions
            WHERE revision_authority = 'quarantined'
        """
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = source_conn.execute(query, params).fetchall()

        exact: list[LiveSourceReconciliationCandidate] = []
        diverged: list[LiveSourceReconciliationCandidate] = []
        missing: list[LiveSourceReconciliationCandidate] = []
        for row in rows:
            provider = provider_from_origin(
                Origin.from_string(str(row["origin"])),
                family_hint=row["capture_mode"],
            )
            kind = RawRevisionKind(str(row["revision_kind"])) if row["revision_kind"] else RawRevisionKind.UNKNOWN
            source_path_value = row["source_path"]
            blob_size = int(row["blob_size"] or 0)

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
                        revision_kind=kind,
                        source_path=Path(source_path_value),
                        append_start_offset=row["append_start_offset"],
                        append_end_offset=row["append_end_offset"],
                    )

            candidate = LiveSourceReconciliationCandidate(
                raw_id=str(row["raw_id"]),
                logical_source_key=row["logical_source_key"],
                provider=provider,
                revision_kind=kind,
                source_path=source_path_value,
                blob_size=blob_size,
                comparison=comparison,
            )
            if comparison.verdict == LiveSourceVerdict.EXACT_MATCH:
                exact.append(candidate)
            elif comparison.verdict == LiveSourceVerdict.SOURCE_MISSING:
                missing.append(candidate)
            else:
                diverged.append(candidate)

        return LiveSourceReconciliationPlan(
            scanned_count=len(rows),
            exact_match=tuple(exact),
            diverged=tuple(diverged),
            source_missing=tuple(missing),
        )
    finally:
        source_conn.row_factory = original_row_factory


__all__ = [
    "LiveSourceComparison",
    "LiveSourceReconciliationCandidate",
    "LiveSourceReconciliationPlan",
    "LiveSourceVerdict",
    "compare_raw_against_live_source",
    "plan_quarantined_live_source_reconciliation",
]
