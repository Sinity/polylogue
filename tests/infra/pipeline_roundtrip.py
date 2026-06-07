"""Shared helpers for payload -> transform -> storage -> hydration laws."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.session.domain_models import Session
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue, loads
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.prepare_models import TransformResult
from polylogue.pipeline.prepare_transform import transform_to_records
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.hydrators import session_from_records
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.archive_scenarios import read_session_records


@dataclass(frozen=True, slots=True)
class PipelineRoundtrip:
    """Parsed and transformed representation of one source payload."""

    parsed: ParsedSession
    transform: TransformResult


def decode_source_payload(raw_bytes: bytes) -> JSONValue:
    """Decode JSON or JSONL provider payload bytes."""
    text = raw_bytes.decode("utf-8")
    try:
        return loads(text)
    except ValueError:
        return [loads(line) for line in text.strip().splitlines() if line.strip()]


def parse_and_transform_payload(
    source_name: str,
    raw_bytes: bytes,
    archive_root: Path,
    unique_id: str = "default",
) -> PipelineRoundtrip:
    """Run parse -> transform for one payload."""
    payload = decode_source_payload(raw_bytes)
    detected = detect_provider(payload)
    assert detected is not None, f"Provider detection failed for {source_name}"

    parsed_list = parse_payload(detected, payload, f"rt-{unique_id}")
    assert parsed_list, "Parser returned no sessions"
    parsed = parsed_list[0]
    return PipelineRoundtrip(
        parsed=parsed,
        transform=transform_to_records(parsed, f"test-{source_name}", archive_root=archive_root),
    )


def save_transform_and_hydrate(result: TransformResult, db_conn: sqlite3.Connection) -> Session:
    """Persist a transform's parsed session and hydrate the stored ``Session``."""
    parsed = result.session
    origin = origin_from_provider(parsed.source_name)
    session_id = archive_session_id(origin.value, parsed.provider_session_id)
    write_parsed_session_to_archive(db_conn, parsed, content_hash=str(result.content_hash))
    conv_record, msg_records, attachment_records = read_session_records(db_conn, session_id)
    return session_from_records(conv_record, msg_records, attachment_records)


__all__ = [
    "PipelineRoundtrip",
    "decode_source_payload",
    "parse_and_transform_payload",
    "save_transform_and_hydrate",
]
