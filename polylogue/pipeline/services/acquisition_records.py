"""Acquisition record models and raw-record preparation helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from polylogue.lib.provider_identity import canonical_acquisition_provider
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.store import MAX_RAW_CONTENT_SIZE, RawConversationRecord

_HASH_CHUNK_SIZE = 65536  # 64 KB chunks for streaming hash


def hash_file_streaming(path: Path) -> str:
    """Compute SHA-256 hash by reading in chunks. Never holds full file in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> int:
    """Get file size without reading content."""
    return path.stat().st_size


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
    *,
    keep_content: bool = True,
) -> RawConversationRecord:
    """Prepare a raw conversation record from scanned payload bytes.

    Args:
        keep_content: If False, raw_content is set to b"" after hashing.
            The raw_id (content hash) is still computed from the bytes,
            but the bytes themselves are not stored in the record.
            Used by lightweight/preview mode to prevent OOM.
    """
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
        raw_content=raw_data.raw_bytes if keep_content else b"",
        acquired_at=acquired_at,
        file_mtime=raw_data.file_mtime,
    )


def make_lightweight_record(
    source_path: str,
    source_name: str,
    *,
    file_mtime: str | None = None,
    provider_hint: str | None = None,
) -> RawConversationRecord:
    """Create a record by streaming-hashing a file. Never holds full bytes.

    Used for preview/planning where we need the content hash (raw_id)
    but not the actual bytes. Prevents OOM on large archives.
    """
    p = Path(source_path)
    size = p.stat().st_size
    if size > MAX_RAW_CONTENT_SIZE:
        raise ValueError(f"Oversized source file at {source_path} ({size} bytes)")

    raw_id = hash_file_streaming(p)
    provider_name = canonical_acquisition_provider(provider_hint, source_name=source_name)

    return RawConversationRecord(
        raw_id=raw_id,
        provider_name=provider_name,
        source_name=source_name,
        source_path=source_path,
        source_index=None,
        raw_content=b"",
        acquired_at=datetime.now(timezone.utc).isoformat(),
        file_mtime=file_mtime,
    )


__all__ = ["ScanResult", "make_lightweight_record", "make_raw_record"]
