"""Shared helpers for payload -> parse -> archive-write -> hydration laws."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.session.domain_models import Session
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue, loads
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.hydrators import session_from_records
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.archive_scenarios import read_session_records


@dataclass(frozen=True, slots=True)
class PipelineRoundtrip:
    """Parsed representation of one source payload and its content hash."""

    parsed: ParsedSession
    content_hash: str


def decode_source_payload(raw_bytes: bytes) -> JSONValue:
    """Decode JSON or JSONL provider payload bytes."""
    text = raw_bytes.decode("utf-8")
    try:
        return loads(text)
    except ValueError:
        return [loads(line) for line in text.strip().splitlines() if line.strip()]


def parse_payload_roundtrip(
    source_name: str,
    raw_bytes: bytes,
    unique_id: str = "default",
) -> PipelineRoundtrip:
    """Run detect -> parse for one payload and compute its content hash."""
    payload = decode_source_payload(raw_bytes)
    detected = detect_provider(payload)
    assert detected is not None, f"Provider detection failed for {source_name}"

    parsed_list = parse_payload(detected, payload, f"rt-{unique_id}")
    assert parsed_list, "Parser returned no sessions"
    parsed = parsed_list[0]
    return PipelineRoundtrip(parsed=parsed, content_hash=session_content_hash(parsed))


def write_and_hydrate(roundtrip: PipelineRoundtrip, db_conn: sqlite3.Connection) -> Session:
    """Write the parsed session via the live archive writer and hydrate it back."""
    parsed = roundtrip.parsed
    origin = origin_from_provider(parsed.source_name)
    session_id = archive_session_id(origin.value, parsed.provider_session_id)
    database_path = next(
        Path(str(path)) for _sequence, name, path in db_conn.execute("PRAGMA database_list") if name == "main" and path
    )
    blob_store = BlobStore(database_path.parent / "blob")
    preacquired: dict[int, tuple[bytes | None, int, str]] = {}
    for attachment in parsed.attachments:
        if attachment.inline_bytes is None:
            continue
        blob_hash, size = blob_store.write_from_bytes(attachment.inline_bytes)
        preacquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    write_parsed_session_to_archive(
        db_conn,
        parsed,
        content_hash=roundtrip.content_hash,
        preacquired_attachment_blobs=preacquired,
    )
    conv_record, msg_records, attachment_records = read_session_records(db_conn, session_id)
    return session_from_records(conv_record, msg_records, attachment_records)


__all__ = [
    "PipelineRoundtrip",
    "decode_source_payload",
    "parse_payload_roundtrip",
    "write_and_hydrate",
]
