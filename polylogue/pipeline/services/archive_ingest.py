"""source parsing helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.config import Source
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.sources.source_parsing import iter_source_sessions_with_raw
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


async def parse_sources_archive(archive_root: Path, sources: list[Source]) -> ParseResult:
    """Parse configured sources directly into archive source/index tiers."""
    result = ParseResult()
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    raw_rows_written = 0
    index_rows_written = 0
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for source in sources:
            for raw_data, session in iter_source_sessions_with_raw(source, capture_raw=True):
                payload = _archive_raw_payload(raw_data, session)
                source_path = _archive_raw_source_path(raw_data, source)
                source_index = _archive_raw_source_index(raw_data)
                _raw_id, session_id = archive.write_raw_and_parsed(
                    session,
                    payload=payload,
                    source_path=source_path,
                    acquired_at_ms=acquired_at_ms,
                    source_index=source_index,
                )
                raw_rows_written += 1
                index_rows_written += 1
                await result.merge_result(
                    session_id,
                    {
                        "sessions": 1,
                        "messages": len(session.messages),
                        "attachments": len(session.attachments),
                        "skipped_sessions": 0,
                        "skipped_messages": 0,
                        "skipped_attachments": 0,
                    },
                    content_changed=True,
                )
    result.batch_observations.append(
        {
            "primary_ingest_store": "archive_file_set",
            "archive_primary_write": True,
            "archive_write_mode": "archive",
            "archive_root": str(archive_root),
            "archive_write_targets": ["source.db", "index.db"],
            "archive_source_rows": raw_rows_written,
            "archive_index_rows": index_rows_written,
            "records": len(result.processed_ids),
            "sessions": result.counts["sessions"],
            "messages": result.counts["messages"],
            "changed_sessions": result.changed_counts["sessions"],
        }
    )
    return result


def _archive_raw_payload(raw_data: object, session: Any) -> bytes:
    from polylogue.storage.blob_store import get_blob_store

    raw_bytes = getattr(raw_data, "raw_bytes", None)
    if isinstance(raw_bytes, bytes) and raw_bytes:
        return raw_bytes
    blob_hash = getattr(raw_data, "blob_hash", None)
    if isinstance(blob_hash, str) and blob_hash:
        return get_blob_store().read_all(blob_hash)
    if callable(getattr(session, "model_dump_json", None)):
        return str(session.model_dump_json()).encode("utf-8")
    return json.dumps(str(session), sort_keys=True).encode("utf-8")


def _archive_raw_source_path(raw_data: object, source: Source) -> str:
    source_path = getattr(raw_data, "source_path", None)
    if source_path is not None:
        return str(source_path)
    return str(source.path)


def _archive_raw_source_index(raw_data: object) -> int:
    source_index = getattr(raw_data, "source_index", None)
    return int(source_index) if source_index is not None else 0


__all__ = ["parse_sources_archive"]
