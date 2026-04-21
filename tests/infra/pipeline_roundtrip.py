"""Shared helpers for payload -> transform -> storage -> hydration laws."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.lib.conversation_models import Conversation
from polylogue.lib.json import JSONValue, loads
from polylogue.pipeline.prepare_models import TransformResult
from polylogue.pipeline.prepare_transform import transform_to_records
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.base import ParsedConversation
from polylogue.storage.hydrators import conversation_from_records
from tests.infra.storage_records import store_records


@dataclass(frozen=True, slots=True)
class PipelineRoundtrip:
    """Parsed and transformed representation of one source payload."""

    parsed: ParsedConversation
    transform: TransformResult


def decode_source_payload(raw_bytes: bytes) -> JSONValue:
    """Decode JSON or JSONL provider payload bytes."""
    text = raw_bytes.decode("utf-8")
    try:
        return loads(text)
    except ValueError:
        return [loads(line) for line in text.strip().splitlines() if line.strip()]


def parse_and_transform_payload(
    provider_name: str,
    raw_bytes: bytes,
    archive_root: Path,
    unique_id: str = "default",
) -> PipelineRoundtrip:
    """Run parse -> transform for one payload."""
    payload = decode_source_payload(raw_bytes)
    detected = detect_provider(payload)
    assert detected is not None, f"Provider detection failed for {provider_name}"

    parsed_list = parse_payload(detected, payload, f"rt-{unique_id}")
    assert parsed_list, "Parser returned no conversations"
    parsed = parsed_list[0]
    return PipelineRoundtrip(
        parsed=parsed,
        transform=transform_to_records(parsed, f"test-{provider_name}", archive_root=archive_root),
    )


def save_transform_and_hydrate(result: TransformResult, db_conn: sqlite3.Connection) -> Conversation:
    """Persist transform records and hydrate the resulting bundle."""
    bundle = result.bundle
    store_records(
        conversation=bundle.conversation,
        messages=bundle.messages,
        attachments=bundle.attachments,
        conn=db_conn,
    )
    return conversation_from_records(bundle.conversation, bundle.messages, bundle.attachments)


__all__ = [
    "PipelineRoundtrip",
    "decode_source_payload",
    "parse_and_transform_payload",
    "save_transform_and_hydrate",
]
