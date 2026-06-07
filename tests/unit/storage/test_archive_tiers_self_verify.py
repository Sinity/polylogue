from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.self_verify import build_archive_session_self_verify_envelope
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    upsert_insight_materialization,
    upsert_session_phase,
    upsert_session_tag,
    upsert_session_work_event,
    write_parsed_session_to_archive,
)
from polylogue.types import ContentBlockType, Provider


def test_archive_tiers_archive_self_verify_envelope_is_stable(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "index.db")
    initialize_archive_tier(conn, ArchiveTier.INDEX)

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-self-verify-1",
        title="Self verify",
        messages=[
            ParsedMessage(
                provider_message_id="msg-user",
                role=Role.USER,
                text="seed baseline",
                position=0,
                content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text="seed baseline")],
            ),
            ParsedMessage(
                provider_message_id="msg-assistant",
                role=Role.ASSISTANT,
                text="query token",
                position=1,
                is_active_path=True,
                is_active_leaf=True,
                content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text="query token")],
            ),
        ],
        active_leaf_message_provider_id="msg-assistant",
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="msg-user",
                name="note.txt",
                upload_origin="paste",
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                payload={"summary": "compressed context"},
                source_message_provider_id="msg-assistant",
                timestamp="2026-01-01T00:00:03+00:00",
            )
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)
    upsert_session_tag(
        conn,
        session_id=session_id,
        tag="Self-Verify",
        tag_source="auto",
        method="fixture",
        confidence=0.9,
    )
    upsert_insight_materialization(
        conn,
        insight_type="session_profile",
        session_id=session_id,
        materializer_version=2,
        materialized_at_ms=1_767_225_604_000,
        input_row_count=2,
    )
    upsert_session_work_event(
        conn,
        session_id=session_id,
        position=0,
        work_event_type="analysis",
        summary="Self-verify event",
        start_index=0,
        end_index=1,
    )
    upsert_session_phase(
        conn,
        session_id=session_id,
        position=0,
        phase_type="inspect",
        start_index=0,
        end_index=1,
    )
    envelope = build_archive_session_self_verify_envelope(
        conn,
        session_id,
        query="token",
    )

    assert envelope == {
        "session_id": "chatgpt-export:chatgpt-self-verify-1",
        "origin": "chatgpt-export",
        "active_leaf_message_id": "chatgpt-export:chatgpt-self-verify-1:msg-assistant",
        "counts": {
            "messages": 2,
            "blocks": 2,
            "session_events": 1,
            "session_links_outbound": 0,
            "attachments": 1,
            "attachment_refs": 1,
            "tags": 1,
            "insight_materializations": 1,
            "work_events": 1,
            "phases": 1,
        },
        "ordered_message_ids": [
            "chatgpt-export:chatgpt-self-verify-1:msg-user",
            "chatgpt-export:chatgpt-self-verify-1:msg-assistant",
        ],
        "ordered_block_ids": [
            "chatgpt-export:chatgpt-self-verify-1:msg-user:0",
            "chatgpt-export:chatgpt-self-verify-1:msg-assistant:0",
        ],
        "session_event_ids": ["chatgpt-export:chatgpt-self-verify-1:0"],
        # Attachments are content-addressed: attachment_id is the deterministic
        # blob hash over the attachment identity fields, not a generated token.
        "attachment_ids": ["7851ae7672e51f490ce4aaaa7b082dc562e93a5b26f9c0cf40eb46be4cca80c7"],
        "tag_keys": ["auto:self-verify"],
        "insight_materialization_keys": ["session_profile:2"],
        "work_event_ids": ["chatgpt-export:chatgpt-self-verify-1:work_event:0"],
        "phase_ids": ["chatgpt-export:chatgpt-self-verify-1:phase:0"],
        "fts_hits": {
            "query": "token",
            "block_ids": ["chatgpt-export:chatgpt-self-verify-1:msg-assistant:0"],
        },
    }
