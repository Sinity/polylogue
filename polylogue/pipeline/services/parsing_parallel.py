"""Process-parallel parse worker for CPU-bound decode + parse.

Runs in a subprocess via ProcessPoolExecutor to bypass the GIL.
JSON decode (orjson) and provider parsing are CPU-bound; running
them in separate processes achieves true parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polylogue.storage.store import RawConversationRecord


@dataclass
class ParseWorkerResult:
    """Result of parsing one raw record in a subprocess."""

    raw_id: str
    payload_provider: str | None = None
    conversations: list[Any] = field(default_factory=list)  # list[ParsedConversation]
    source_name: str | None = None
    error: str | None = None


def parse_record_sync(raw_record: RawConversationRecord) -> ParseWorkerResult:
    """Parse a single raw record (runs in subprocess)."""
    from polylogue.lib.raw_payload import build_raw_payload_envelope
    from polylogue.schemas.runtime_registry import SchemaRegistry
    from polylogue.sources.dispatch import parse_payload
    from polylogue.storage.blob_store import get_blob_store

    stored_payload_provider = raw_record.payload_provider
    if not isinstance(stored_payload_provider, str) or not stored_payload_provider.strip():
        stored_payload_provider = None

    blob_store = get_blob_store()
    raw_source = blob_store.blob_path(raw_record.raw_id)

    try:
        envelope = build_raw_payload_envelope(
            raw_source,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.provider_name,
            payload_provider=stored_payload_provider,
        )
    except Exception as exc:
        return ParseWorkerResult(
            raw_id=raw_record.raw_id,
            payload_provider=stored_payload_provider,
            error=f"decode: {exc}",
        )

    if not envelope.artifact.parse_as_conversation:
        return ParseWorkerResult(
            raw_id=raw_record.raw_id,
            payload_provider=str(envelope.provider),
        )

    import re

    _SOURCE_HASH_SUFFIX = re.compile(r"-(?:[0-9a-f]{16,64})$", re.IGNORECASE)

    def _fallback_id(source_path: str | None, raw_id: str) -> str:
        if not source_path:
            return raw_id
        from pathlib import Path

        normalized = source_path.replace("\\", "/")
        entry_path = normalized.rsplit(":", 1)[-1]
        stem = Path(entry_path).stem
        if not stem:
            return raw_id
        cleaned = _SOURCE_HASH_SUFFIX.sub("", stem).strip("._- ")
        return cleaned or stem

    registry = SchemaRegistry()
    schema_resolution = registry.resolve_payload(
        envelope.provider,
        envelope.payload,
        source_path=raw_record.source_path,
    )

    try:
        conversations = parse_payload(
            envelope.provider,
            envelope.payload,
            _fallback_id(raw_record.source_path, raw_record.raw_id),
            schema_resolution=schema_resolution,
        )
    except Exception as exc:
        return ParseWorkerResult(
            raw_id=raw_record.raw_id,
            payload_provider=str(envelope.provider),
            error=f"parse: {exc}",
        )

    source_name = raw_record.source_name or raw_record.source_path
    return ParseWorkerResult(
        raw_id=raw_record.raw_id,
        payload_provider=str(envelope.provider),
        conversations=conversations,
        source_name=source_name,
    )


__all__ = ["ParseWorkerResult", "parse_record_sync"]
