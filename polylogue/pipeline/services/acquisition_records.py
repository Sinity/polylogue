"""Acquisition record models and raw-record preparation helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from polylogue.lib.provider_identity import canonical_acquisition_provider
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.store import MAX_RAW_CONTENT_SIZE, RawConversationRecord


class ScanResult:
    """Result of scanning raw payloads from sources without persisting them."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "scanned": 0,
            "errors": 0,
        }
        self.cursors: dict[str, dict[str, object]] = {}


def make_raw_record(
    raw_data: RawConversationData,
    source_name: str,
) -> RawConversationRecord:
    """Prepare a raw conversation record from scanned payload bytes."""
    size = len(raw_data.raw_bytes)
    if size > MAX_RAW_CONTENT_SIZE:
        raise ValueError(
            f"Oversized source file at {raw_data.source_path} "
            f"({size} bytes > {MAX_RAW_CONTENT_SIZE} max)"
        )

    raw_id = hashlib.sha256(raw_data.raw_bytes).hexdigest()
    acquired_at = datetime.now(timezone.utc).isoformat()
    provider_name = canonical_acquisition_provider(
        str(raw_data.provider_hint) if raw_data.provider_hint is not None else None,
        source_name=source_name,
    )

    return RawConversationRecord(
        raw_id=raw_id,
        provider_name=provider_name,
        source_name=source_name,
        source_path=raw_data.source_path,
        source_index=raw_data.source_index,
        raw_content=raw_data.raw_bytes,
        acquired_at=acquired_at,
        file_mtime=raw_data.file_mtime,
    )


__all__ = ["ScanResult", "make_raw_record"]
