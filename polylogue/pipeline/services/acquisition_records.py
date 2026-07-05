"""Acquisition record models and raw-record preparation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typing_extensions import TypedDict

from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.sources.parsers.base import RawSessionData
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.runtime import RawSessionRecord


class ScanCounts(TypedDict):
    scanned: int
    errors: int


class ScanResult:
    """Result of scanning raw payloads from sources without persisting them."""

    def __init__(self) -> None:
        self.counts: ScanCounts = {
            "scanned": 0,
            "errors": 0,
        }
        self.cursors: dict[str, CursorStatePayload] = {}


def make_raw_record(
    raw_data: RawSessionData,
    source_name: str,
    *,
    blob_root: Path | None = None,
) -> RawSessionRecord:
    """Prepare a raw session record from acquisition data.

    When ``blob_hash`` is set on the data (content already in blob store),
    uses it as raw_id directly. Otherwise falls back to hashing raw_bytes.
    """
    if raw_data.blob_hash is not None:
        raw_id = raw_data.blob_hash
        blob_size = raw_data.blob_size or 0
    elif raw_data.raw_bytes:
        # Bytes provided without pre-computed blob hash (e.g. from tests
        # or legacy callers). Write to blob store and use the hash.
        from polylogue.storage.blob_store import BlobStore, get_blob_store

        blob_store = BlobStore(blob_root) if blob_root is not None else get_blob_store()
        raw_id, blob_size = blob_store.write_from_bytes(raw_data.raw_bytes)
    else:
        raise ValueError("RawSessionData has neither blob_hash nor raw_bytes")

    acquired_at = datetime.now(timezone.utc).isoformat()
    source_name = canonical_acquisition_provider(
        str(raw_data.provider_hint) if raw_data.provider_hint is not None else None,
        source_name=source_name,
    )

    return RawSessionRecord(
        raw_id=raw_id,
        source_name=source_name,
        source_path=raw_data.source_path,
        source_index=raw_data.source_index,
        blob_size=blob_size,
        acquired_at=acquired_at,
        file_mtime=raw_data.file_mtime,
    )


__all__ = ["ScanCounts", "ScanResult", "make_raw_record"]
