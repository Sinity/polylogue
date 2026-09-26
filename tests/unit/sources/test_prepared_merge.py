"""Parity checks for worker-side composition of retained revisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role, TitleSource
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import merge_parsed_session_chunks
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact
from polylogue.sources.prepared_merge import prepare_retained_cohort_artifact, prepared_cohort_source_hash
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard


def _chunk_artifact(directory: Path, session: ParsedSession, source_hash: str) -> PreparedJsonl:
    directory.mkdir(parents=True, exist_ok=True)
    store = SqliteMessageStore(directory / f"{source_hash}.db")
    try:
        session.content_hash = session_content_hash(session)
        shard = prepare_session_shard(directory, [session])
        _write_artifact(store, source_hash, [session], enrichment_digest=None, enrichment_index_path=None)
    finally:
        store.close()
    return PreparedJsonl.seal(source_hash, store.path, shard.path)


def test_prepared_cohort_preserves_merge_order_title_and_duplicate_leaf(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="same",
        title="weak",
        title_source=TitleSource.HEURISTIC,
        title_ref="message:dup",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:01:00Z",
        models_used=["alpha"],
        working_directories=["/first"],
        messages=[
            ParsedMessage(provider_message_id="dup", role=Role.USER, text="first", position=9, is_active_leaf=True)
        ],
        session_events=[ParsedSessionEvent(event_type="turn_context", payload={"chunk": 1})],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="same",
        title="strong",
        title_source=TitleSource.ORIGIN,
        title_ref="codex-thread-name:same",
        created_at="2026-01-02T00:00:00Z",
        updated_at="2026-01-02T00:01:00Z",
        models_used=["beta"],
        working_directories=["/second"],
        messages=[
            ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="last", position=0, is_active_leaf=True)
        ],
        session_events=[ParsedSessionEvent(event_type="turn_context", payload={"chunk": 2})],
    )
    ordered = [
        ("raw-0", _chunk_artifact(tmp_path, first, "a" * 64)),
        ("raw-1", _chunk_artifact(tmp_path, second, "b" * 64)),
    ]

    aggregate = prepare_retained_cohort_artifact(ordered, tmp_path)
    aggregate.verify_files(full=True)
    session = aggregate.load_sessions()[0]
    expected = merge_parsed_session_chunks([first, second])[0]

    assert aggregate.blob_hash == prepared_cohort_source_hash(ordered)
    assert session.content_hash == session_content_hash(expected)
    assert session.title == expected.title
    assert session.title == "strong"
    assert session.title_source is TitleSource.ORIGIN
    assert session.created_at == expected.created_at
    assert session.updated_at == expected.updated_at
    assert session.models_used == expected.models_used
    assert session.working_directories == expected.working_directories
    assert [
        (message.provider_message_id, message.text, message.position, message.is_active_leaf)
        for message in session.messages
    ] == [
        ("dup", "first", 0, False),
        ("dup", "last", 1, True),
    ]
    assert [event.payload["chunk"] for event in session.session_events] == [1, 2]
    assert prepared_cohort_source_hash(list(reversed(ordered))) != aggregate.blob_hash


def test_prepared_claude_cohort_reduces_chunk_summaries(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="same",
        updated_at="2026-01-01T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="zero")],
        session_events=[
            ParsedSessionEvent(event_type="point", payload={"chunk": 1}),
            ParsedSessionEvent(event_type="claude_session_environment", payload={"entrypoints": {"cli": 1}}),
            ParsedSessionEvent(event_type="claude_parse_coverage", payload={"sidecar_seen": {"user": 1}}),
        ],
    )
    second = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="same",
        updated_at="2026-01-02T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.ASSISTANT, text="one")],
        session_events=[
            ParsedSessionEvent(event_type="point", payload={"chunk": 2}),
            ParsedSessionEvent(event_type="claude_session_environment", payload={"entrypoints": {"cli": 2, "hook": 1}}),
            ParsedSessionEvent(event_type="claude_parse_coverage", payload={"sidecar_seen": {"user": 3}}),
        ],
    )
    ordered = [
        ("raw-0", _chunk_artifact(tmp_path, first, "a" * 64)),
        ("raw-1", _chunk_artifact(tmp_path, second, "b" * 64)),
    ]

    aggregate = prepare_retained_cohort_artifact(ordered, tmp_path)
    session = aggregate.load_sessions()[0]
    expected = merge_parsed_session_chunks([first, second])[0]

    assert [event.event_type for event in session.session_events] == [
        event.event_type for event in expected.session_events
    ]
    assert [event.payload for event in session.session_events] == [event.payload for event in expected.session_events]
    assert session.content_hash == session_content_hash(expected)


def test_prepared_cohort_rejects_conflicting_session_identity(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="first",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="first")],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="second",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="second")],
    )
    ordered = [
        ("raw-0", _chunk_artifact(tmp_path, first, "a" * 64)),
        ("raw-1", _chunk_artifact(tmp_path, second, "b" * 64)),
    ]

    with pytest.raises(ValueError, match="provider-native session identity"):
        prepare_retained_cohort_artifact(ordered, tmp_path)
