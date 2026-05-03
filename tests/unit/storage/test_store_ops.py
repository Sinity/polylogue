"""Focused roundtrip and validation contracts for storage record helpers."""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from typing_extensions import TypedDict, Unpack

from polylogue.archive.conversation.models import Conversation
from polylogue.archive.message.messages import MessageCollection
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    _json_or_none,
    _make_ref_id,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
)
from tests.infra.storage_records import (
    _prune_attachment_refs,
    make_attachment,
    make_conversation,
    make_message,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)
from tests.infra.strategies.messages import conversation_strategy
from tests.infra.strategies.storage import (
    TagAssignmentSpec,
    TitleSearchSpec,
    expected_tag_counts,
    seed_conversation_graph,
)
from tests.infra.strategies.storage import (
    literal_title_search_strategy as infra_title_search_strategy,
)
from tests.infra.strategies.storage import (
    tag_assignment_strategy as infra_tag_assignment_strategy,
)


class SimpleTagSpec(TypedDict):
    conversation_id: str
    tags: list[str]


class SimpleTitleSearchSpec(TypedDict):
    title: str
    search_term: str


class RecordQueryKwargs(TypedDict, total=False):
    provider: str
    referenced_path: tuple[str, ...]
    action_terms: tuple[str, ...]
    excluded_action_terms: tuple[str, ...]
    tool_terms: tuple[str, ...]
    excluded_tool_terms: tuple[str, ...]
    limit: int


def _conversation_id(value: str) -> ConversationId:
    return ConversationId(value)


def _message_id(value: str) -> MessageId:
    return MessageId(value)


def _attachment_id(value: str) -> AttachmentId:
    return AttachmentId(value)


def _content_hash(value: str) -> ContentHash:
    return ContentHash(value)


def _content_block(
    *,
    block_id: str,
    message_id: str,
    conversation_id: str,
    block_index: int,
    block_type: str,
    text: str | None = None,
    tool_name: str | None = None,
    tool_id: str | None = None,
    tool_input: str | None = None,
    media_type: str | None = None,
    metadata: str | None = None,
    semantic_type: str | None = None,
) -> ContentBlockRecord:
    return ContentBlockRecord(
        block_id=block_id,
        message_id=_message_id(message_id),
        conversation_id=_conversation_id(conversation_id),
        block_index=block_index,
        type=ContentBlockType.from_string(block_type),
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        media_type=media_type,
        metadata=metadata,
        semantic_type=None if semantic_type is None else SemanticBlockType.from_string(semantic_type),
    )


def _conversation_record(
    *,
    conversation_id: str,
    provider_name: str,
    provider_conversation_id: str,
    content_hash: str,
    title: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
    provider_meta: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=_conversation_id(conversation_id),
        provider_name=provider_name,
        provider_conversation_id=provider_conversation_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        content_hash=_content_hash(content_hash),
        provider_meta=provider_meta,
        metadata=metadata,
    )


def _attachment_record(
    *,
    attachment_id: str,
    conversation_id: str,
    message_id: str | None,
    mime_type: str,
    size_bytes: int | None,
    provider_meta: dict[str, object] | None = None,
) -> AttachmentRecord:
    return AttachmentRecord(
        attachment_id=_attachment_id(attachment_id),
        conversation_id=_conversation_id(conversation_id),
        message_id=None if message_id is None else _message_id(message_id),
        mime_type=mime_type,
        size_bytes=size_bytes,
        provider_meta=provider_meta,
    )


def _ref_id(attachment_id: str, conversation_id: str, message_id: str | None) -> str:
    return _make_ref_id(
        _attachment_id(attachment_id),
        _conversation_id(conversation_id),
        None if message_id is None else _message_id(message_id),
    )


def _conversation_model(conversation_id: str) -> Conversation:
    return Conversation(
        id=_conversation_id(conversation_id),
        provider=Provider.CHATGPT,
        messages=MessageCollection.empty(),
    )


def _conversation_row(conn: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if row is not None and not isinstance(row, sqlite3.Row):
        raise TypeError(f"expected sqlite3.Row, got {type(row).__name__}")
    return row


def _message_count(conn: sqlite3.Connection, conversation_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _attachment_row(conn: sqlite3.Connection, attachment_id: str) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM attachments WHERE attachment_id = ?",
        (attachment_id,),
    ).fetchone()
    if row is not None and not isinstance(row, sqlite3.Row):
        raise TypeError(f"expected sqlite3.Row, got {type(row).__name__}")
    return row


def _message_payloads(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    payloads: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"expected message payload dict, got {type(item).__name__}")
        payloads.append(dict(item))
    return payloads


def _record_query(**kwargs: Unpack[RecordQueryKwargs]) -> ConversationRecordQuery:
    return ConversationRecordQuery(**kwargs)


@pytest.mark.asyncio
async def test_aggregate_message_stats_reports_role_counts_and_words(tmp_path: Path) -> None:
    from tests.infra.storage_records import ConversationBuilder

    db_path = tmp_path / "stats.db"

    (
        ConversationBuilder(db_path, "conv-stats-a")
        .provider("chatgpt")
        .add_message("m-user-a", role="user", text="hello world")
        .add_message("m-assistant-a", role="assistant", text="answer words here")
        .add_attachment(message_id="m-assistant-a", path="spec.pdf")
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-stats-b")
        .provider("codex")
        .add_message("m-system-b", role="system", text="system note")
        .add_message("m-user-b", role="user", text="follow up")
        .save()
    )

    backend = SQLiteBackend(db_path=db_path)
    try:
        unfiltered = await backend.queries.aggregate_message_stats()
        filtered = await backend.queries.aggregate_message_stats(["conv-stats-a"])
    finally:
        await backend.close()

    assert unfiltered["total"] == 4
    assert unfiltered["user"] == 2
    assert unfiltered["assistant"] == 1
    assert unfiltered["system"] == 1
    assert unfiltered["words_approx"] > 0
    assert unfiltered["attachment_refs"] == 1
    assert unfiltered["distinct_attachments"] == 1
    assert unfiltered["providers"] == {"chatgpt": 1, "codex": 1}

    assert filtered["total"] == 2
    assert filtered["user"] == 1
    assert filtered["assistant"] == 1
    assert filtered["system"] == 0
    assert filtered["words_approx"] > 0
    assert filtered["attachment_refs"] == 1
    assert filtered["distinct_attachments"] == 1
    assert filtered["providers"] == {"chatgpt": 1}


@pytest.mark.asyncio
async def test_backend_referenced_path_filter_contract(tmp_path: Path) -> None:
    """Low-level list/count filters must honor persisted semantic paths."""
    from polylogue.storage.action_events.rebuild_runtime import rebuild_action_event_read_model_sync
    from tests.infra.storage_records import ConversationBuilder

    db_path = tmp_path / "path-filter.db"
    target_path = "/workspace/polylogue/README.md"
    other_path = "/workspace/polylogue/docs/cli-reference.md"

    (
        ConversationBuilder(db_path, "conv-readme")
        .provider("claude-code")
        .title("README work")
        .add_message(
            "m1",
            role="assistant",
            text="Inspecting the repository README",
            content_blocks=[
                _content_block(
                    block_id="blk-readme-0",
                    message_id="m1",
                    conversation_id="conv-readme",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Read",
                    metadata=f'{{"path":"{target_path}"}}',
                    semantic_type="file_read",
                )
            ],
        )
        .save()
    )

    (
        ConversationBuilder(db_path, "conv-other")
        .provider("claude-code")
        .title("CLI docs work")
        .add_message(
            "m2",
            role="assistant",
            text="Inspecting docs",
            content_blocks=[
                _content_block(
                    block_id="blk-other-0",
                    message_id="m2",
                    conversation_id="conv-other",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Read",
                    metadata=f'{{"path":"{other_path}"}}',
                    semantic_type="file_read",
                )
            ],
        )
        .save()
    )

    with open_connection(db_path) as conn:
        rebuild_action_event_read_model_sync(conn)
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    try:
        matches = await backend.queries.list_conversations(_record_query(referenced_path=(target_path,), limit=10))
        assert [record.conversation_id for record in matches] == ["conv-readme"]
        assert await backend.queries.count_conversations(_record_query(referenced_path=(target_path,))) == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_list_summaries_by_query_omits_large_provider_meta_payloads(tmp_path: Path) -> None:
    db_path = tmp_path / "summary-provider-meta.db"
    large_meta = {"raw": "x" * 100_000, "source": "codex"}

    with open_connection(db_path) as conn:
        upsert_conversation(
            conn,
            make_conversation(
                "conv-large-meta",
                provider_name="codex",
                title="Large Meta Conversation",
                provider_meta=large_meta,
                metadata={"tag": "kept"},
            ),
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        summaries = await repo.list_summaries_by_query(_record_query(provider="codex", limit=1))
        assert len(summaries) == 1
        summary = summaries[0]
        assert str(summary.id) == "conv-large-meta"
        assert summary.title == "Large Meta Conversation"
        assert summary.metadata == {"tag": "kept"}
        assert summary.provider_meta is None
    finally:
        await repo.close()


def test_action_event_rebuild_omits_large_provider_meta_payloads(tmp_path: Path) -> None:
    from polylogue.storage.action_events.rebuild_runtime import rebuild_action_event_read_model_sync
    from tests.infra.storage_records import ConversationBuilder

    db_path = tmp_path / "action-event-provider-meta.db"

    (
        ConversationBuilder(db_path, "conv-heavy-action")
        .provider("codex")
        .title("Heavy Action Event Source")
        .add_message(
            "m-heavy-action",
            role="assistant",
            text="Searching the repo for provider-meta handling.",
            content_blocks=[
                _content_block(
                    block_id="blk-heavy-action-0",
                    message_id="m-heavy-action",
                    conversation_id="conv-heavy-action",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Grep",
                    semantic_type="search",
                )
            ],
        )
        .save()
    )

    huge_provider_meta = {
        "cwd": "/realm/project/sinex",
        "raw": {"payload": "x" * 200_000},
        "git": {"branch": "master", "repository_url": "git@github.com:Sinity/sinex.git"},
    }

    with open_connection(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET provider_meta = ? WHERE conversation_id = ?",
            (json.dumps(huge_provider_meta), "conv-heavy-action"),
        )
        replaced = rebuild_action_event_read_model_sync(conn, page_size=1)
        row = conn.execute(
            """
            SELECT action_kind, normalized_tool_name
            FROM action_events
            WHERE conversation_id = ?
            """,
            ("conv-heavy-action",),
        ).fetchone()

    assert replaced == 1
    assert row is not None
    assert row["action_kind"] == "search"
    assert row["normalized_tool_name"] == "grep"


@pytest.mark.asyncio
async def test_filter_referenced_path_apply_after_fts_search(tmp_path: Path) -> None:
    """Combined FTS + path queries must keep the path constraint after search ranking."""
    from polylogue.archive.filter.filters import ConversationFilter
    from tests.infra.storage_records import ConversationBuilder

    db_path = tmp_path / "path-fts.db"
    target_path = "/workspace/polylogue/README.md"

    (
        ConversationBuilder(db_path, "conv-match")
        .provider("claude-code")
        .title("Path match")
        .add_message(
            "m1",
            role="assistant",
            text="Investigating the same parser regression",
            content_blocks=[
                _content_block(
                    block_id="blk-match-0",
                    message_id="m1",
                    conversation_id="conv-match",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Read",
                    metadata=f'{{"path":"{target_path}"}}',
                    semantic_type="file_read",
                )
            ],
        )
        .save()
    )

    (
        ConversationBuilder(db_path, "conv-no-path")
        .provider("claude-code")
        .title("No path match")
        .add_message(
            "m2",
            role="assistant",
            text="Investigating the same parser regression",
            content_blocks=[
                _content_block(
                    block_id="blk-nopath-0",
                    message_id="m2",
                    conversation_id="conv-no-path",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Read",
                    metadata='{"path":"/workspace/polylogue/docs/cli-reference.md"}',
                    semantic_type="file_read",
                )
            ],
        )
        .save()
    )

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        results = await ConversationFilter(repo).contains("parser regression").referenced_path(target_path).list()
        assert [str(conversation.id) for conversation in results] == ["conv-match"]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_backend_action_terms_filter_contract(tmp_path: Path) -> None:
    """Low-level list/count filters must honor semantic action categories."""
    from polylogue.storage.action_events.rebuild_runtime import rebuild_action_event_read_model_sync
    from tests.infra.storage_records import ConversationBuilder

    db_path = tmp_path / "action-filter.db"

    (
        ConversationBuilder(db_path, "conv-search")
        .provider("claude-code")
        .title("Search work")
        .add_message(
            "m1",
            role="assistant",
            text="Searching for parser code",
            content_blocks=[
                _content_block(
                    block_id="blk-search-0",
                    message_id="m1",
                    conversation_id="conv-search",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Grep",
                    semantic_type="search",
                )
            ],
        )
        .save()
    )

    (
        ConversationBuilder(db_path, "conv-git")
        .provider("claude-code")
        .title("Git work")
        .add_message(
            "m2",
            role="assistant",
            text="Checking git status",
            content_blocks=[
                _content_block(
                    block_id="blk-git-0",
                    message_id="m2",
                    conversation_id="conv-git",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Bash",
                    tool_input='{"command":"git status"}',
                    semantic_type="git",
                )
            ],
        )
        .save()
    )

    (
        ConversationBuilder(db_path, "conv-other")
        .provider("claude-code")
        .title("Other tool work")
        .add_message(
            "m3",
            role="assistant",
            text="Using an unknown tool",
            content_blocks=[
                _content_block(
                    block_id="blk-other-0",
                    message_id="m3",
                    conversation_id="conv-other",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Mystery",
                    semantic_type="other",
                )
            ],
        )
        .save()
    )

    (
        ConversationBuilder(db_path, "conv-none")
        .provider("claude-code")
        .title("Plain dialogue")
        .add_message(
            "m4",
            role="assistant",
            text="No tool use here",
        )
        .save()
    )

    with open_connection(db_path) as conn:
        rebuild_action_event_read_model_sync(conn)
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    try:
        matches = await backend.queries.list_conversations(_record_query(action_terms=("search",), limit=10))
        assert [record.conversation_id for record in matches] == ["conv-search"]
        assert await backend.queries.count_conversations(_record_query(action_terms=("search",))) == 1

        other_matches = await backend.queries.list_conversations(_record_query(action_terms=("other",), limit=10))
        assert [record.conversation_id for record in other_matches] == ["conv-other"]

        none_matches = await backend.queries.list_conversations(_record_query(action_terms=("none",), limit=10))
        assert [record.conversation_id for record in none_matches] == ["conv-none"]
        assert await backend.queries.count_conversations(_record_query(action_terms=("none",))) == 1

        grep_tool_matches = await backend.queries.list_conversations(_record_query(tool_terms=("grep",), limit=10))
        assert [record.conversation_id for record in grep_tool_matches] == ["conv-search"]

        none_tool_matches = await backend.queries.list_conversations(_record_query(tool_terms=("none",), limit=10))
        assert [record.conversation_id for record in none_tool_matches] == ["conv-none"]

        filtered = await backend.queries.list_conversations(
            _record_query(
                action_terms=("search",),
                excluded_action_terms=("git",),
                limit=10,
            )
        )
        assert [record.conversation_id for record in filtered] == ["conv-search"]

        non_grep = await backend.queries.list_conversations(
            _record_query(
                excluded_tool_terms=("grep",),
                limit=10,
            )
        )
        assert sorted(record.conversation_id for record in non_grep) == ["conv-git", "conv-none", "conv-other"]

        non_none = await backend.queries.list_conversations(
            _record_query(
                excluded_action_terms=("none",),
                limit=10,
            )
        )
        assert sorted(record.conversation_id for record in non_none) == ["conv-git", "conv-other", "conv-search"]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_filter_action_terms_apply_after_fts_search(tmp_path: Path) -> None:
    """Combined FTS + action queries must preserve action constraints after search ranking."""
    from polylogue.archive.filter.filters import ConversationFilter
    from tests.infra.storage_records import ConversationBuilder

    db_path = tmp_path / "action-fts.db"

    (
        ConversationBuilder(db_path, "conv-search")
        .provider("claude-code")
        .title("Search match")
        .add_message(
            "m1",
            role="assistant",
            text="Investigating the same parser regression",
            content_blocks=[
                _content_block(
                    block_id="blk-search-0",
                    message_id="m1",
                    conversation_id="conv-search",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Grep",
                    semantic_type="search",
                )
            ],
        )
        .save()
    )

    (
        ConversationBuilder(db_path, "conv-shell")
        .provider("claude-code")
        .title("Shell only")
        .add_message(
            "m2",
            role="assistant",
            text="Investigating the same parser regression",
            content_blocks=[
                _content_block(
                    block_id="blk-shell-0",
                    message_id="m2",
                    conversation_id="conv-shell",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="Bash",
                    tool_input='{"command":"python -m pytest"}',
                    semantic_type="shell",
                )
            ],
        )
        .save()
    )

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        results = await (
            ConversationFilter(repo).contains("parser regression").action("search").exclude_action("git").list()
        )
        assert [str(conversation.id) for conversation in results] == ["conv-search"]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_filter_action_terms_reconcile_runtime_semantics_after_sql_candidate_fetch(tmp_path: Path) -> None:
    """Runtime semantic facts must outrank stale persisted semantic_type labels."""
    from polylogue.archive.filter.filters import ConversationFilter
    from tests.infra.storage_records import ConversationBuilder

    db_path = tmp_path / "action-runtime-reconcile.db"

    (
        ConversationBuilder(db_path, "conv-stale-other")
        .provider("claude-code")
        .title("Stale semantic label")
        .add_message(
            "m1",
            role="assistant",
            text="Create a task for the next review pass",
            content_blocks=[
                _content_block(
                    block_id="blk-stale-0",
                    message_id="m1",
                    conversation_id="conv-stale-other",
                    block_index=0,
                    block_type="tool_use",
                    tool_name="TaskCreate",
                    semantic_type="other",
                )
            ],
        )
        .save()
    )

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        stale_other = await ConversationFilter(repo).action("other").list()
        upgraded_agent = await ConversationFilter(repo).action("agent").list()

        assert [str(conversation.id) for conversation in stale_other] == []
        assert [str(conversation.id) for conversation in upgraded_agent] == ["conv-stale-other"]
        assert await ConversationFilter(repo).action("agent").count() == 1
    finally:
        await backend.close()


def test_store_records_roundtrip_contract(test_conn: sqlite3.Connection) -> None:
    """store_records() must insert, skip, update, and handle sparse payloads coherently."""
    initial = make_conversation("conv-create", content_hash="hash-create")
    created = store_records(
        conversation=initial,
        messages=[make_message("msg-create", "conv-create", text="Hello")],
        attachments=[],
        conn=test_conn,
    )
    assert created == {
        "conversations": 1,
        "messages": 1,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    created_row = _conversation_row(test_conn, "conv-create")
    assert created_row is not None
    assert created_row["title"] == "Test Conversation"
    assert _message_count(test_conn, "conv-create") == 1

    duplicate = store_records(
        conversation=initial,
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert duplicate["conversations"] == 0
    assert duplicate["skipped_conversations"] == 1

    updated = store_records(
        conversation=make_conversation("conv-create", title="Updated Title", content_hash="hash-updated"),
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert updated["conversations"] == 1
    updated_row = _conversation_row(test_conn, "conv-create")
    assert updated_row is not None
    assert updated_row["title"] == "Updated Title"
    assert updated_row["content_hash"] == "hash-updated"

    multi = store_records(
        conversation=make_conversation("conv-multi", title="Multi Message"),
        messages=[
            make_message(
                f"msg-multi-{idx}", "conv-multi", role="user" if idx % 2 == 0 else "assistant", text=f"Message {idx}"
            )
            for idx in range(5)
        ],
        attachments=[],
        conn=test_conn,
    )
    assert multi["messages"] == 5
    assert _message_count(test_conn, "conv-multi") == 5

    sparse = store_records(
        conversation=make_conversation("conv-empty", title="Empty Conversation"),
        messages=[],
        attachments=[
            make_attachment(
                "att-empty",
                "conv-empty",
                message_id=None,
                mime_type="application/pdf",
                size_bytes=5000,
            )
        ],
        conn=test_conn,
    )
    assert sparse["conversations"] == 1
    assert sparse["messages"] == 0
    assert sparse["attachments"] == 1
    sparse_attachment = _attachment_row(test_conn, "att-empty")
    assert sparse_attachment is not None
    assert sparse_attachment["ref_count"] == 1


def test_prune_attachment_refs_contract(test_conn: sqlite3.Connection) -> None:
    """Pruning refs must keep requested refs, recalculate counts, and delete zero-ref attachments."""
    conv = make_conversation("conv-prune", title="Prune Test")
    msg1 = make_message("msg-prune-1", "conv-prune", provider_message_id="ext-1", text="First")
    msg2 = make_message("msg-prune-2", "conv-prune", provider_message_id="ext-2", text="Second")
    att1 = make_attachment("att-prune-1", "conv-prune", "msg-prune-1", mime_type="image/png")
    att2 = make_attachment("att-prune-2", "conv-prune", "msg-prune-2", mime_type="image/jpeg", size_bytes=2048)
    shared_att_1 = make_attachment("att-shared", "conv-prune", "msg-prune-1", mime_type="image/png")
    shared_att_2 = make_attachment("att-shared", "conv-prune", "msg-prune-2", mime_type="image/png")
    store_records(
        conversation=conv,
        messages=[msg1, msg2],
        attachments=[att1, att2, shared_att_1, shared_att_2],
        conn=test_conn,
    )

    keep_ref = _ref_id("att-prune-1", "conv-prune", "msg-prune-1")
    keep_shared = _ref_id("att-shared", "conv-prune", "msg-prune-1")
    _prune_attachment_refs(test_conn, "conv-prune", {keep_ref, keep_shared})

    remaining_refs = test_conn.execute(
        "SELECT ref_id FROM attachment_refs WHERE conversation_id = ? ORDER BY ref_id",
        ("conv-prune",),
    ).fetchall()
    assert [row["ref_id"] for row in remaining_refs] == sorted([keep_ref, keep_shared])
    pruned_attachment = _attachment_row(test_conn, "att-prune-1")
    shared_attachment = _attachment_row(test_conn, "att-shared")
    assert pruned_attachment is not None
    assert shared_attachment is not None
    assert pruned_attachment["ref_count"] == 1
    assert shared_attachment["ref_count"] == 1
    assert _attachment_row(test_conn, "att-prune-2") is None


def test_upsert_optional_and_attachment_contracts(test_conn: sqlite3.Connection) -> None:
    """Optional-field upserts and attachment metadata updates must round-trip cleanly."""
    conversation = _conversation_record(
        conversation_id="conv-optional",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title=None,
        created_at=None,
        updated_at=None,
        content_hash="hash1",
        provider_meta=None,
    )
    assert upsert_conversation(test_conn, conversation) is True
    conv_row = _conversation_row(test_conn, "conv-optional")
    assert conv_row is not None
    assert conv_row["title"] is None
    assert conv_row["created_at"] is None
    assert conv_row["provider_meta"] is None

    message = MessageRecord(
        message_id=_message_id("msg-optional"),
        conversation_id=_conversation_id("conv-optional"),
        provider_message_id=None,
        role=None,
        text=None,
        sort_key=None,
        content_hash=_content_hash("msg-optional-hash"),
        provider_name="",
        word_count=0,
        has_tool_use=0,
        has_thinking=0,
    )
    assert upsert_message(test_conn, message) is True
    msg_row = test_conn.execute(
        "SELECT * FROM messages WHERE message_id = ?",
        ("msg-optional",),
    ).fetchone()
    assert msg_row is not None
    assert msg_row["role"] is None
    assert msg_row["text"] is None
    assert msg_row["provider_message_id"] is None

    msg2 = make_message("msg-attachment-2", "conv-optional", provider_message_id="ext-msg-2", text="Second")
    assert upsert_message(test_conn, msg2) is True
    first = make_attachment("att-meta", "conv-optional", "msg-optional", mime_type="image/png")
    second = make_attachment(
        "att-meta",
        "conv-optional",
        "msg-attachment-2",
        mime_type="image/jpeg",
        size_bytes=2048,
        path="new/path.jpg",
    )
    assert upsert_attachment(test_conn, first) is True
    assert upsert_attachment(test_conn, first) is False
    assert upsert_attachment(test_conn, second) is True
    att_row = _attachment_row(test_conn, "att-meta")
    assert att_row is not None
    assert att_row["mime_type"] == "image/jpeg"
    assert att_row["size_bytes"] == 2048
    assert att_row["path"] == "new/path.jpg"
    assert att_row["ref_count"] == 2


def test_json_or_none_contract() -> None:
    """JSON serialization helper must preserve mappings and None."""
    import json

    payloads: list[tuple[dict[str, object] | None, dict[str, object] | None]] = [
        ({"key": "value"}, {"key": "value"}),
        ({"nested": {"key": "value"}, "list": [1, 2, 3]}, {"nested": {"key": "value"}, "list": [1, 2, 3]}),
        (None, None),
    ]
    for input_val, expected in payloads:
        result = _json_or_none(input_val)
        if expected is None:
            assert result is None
        else:
            assert result is not None
            assert json.loads(result) == expected


def test_make_ref_id_contract() -> None:
    """Attachment ref IDs must be deterministic and sensitive to attachment, conversation, and message."""
    same_1 = _ref_id("att1", "conv1", "msg1")
    same_2 = _ref_id("att1", "conv1", "msg1")
    different_attachment = _ref_id("att2", "conv1", "msg1")
    different_conversation = _ref_id("att1", "conv2", "msg1")
    none_message_1 = _ref_id("att1", "conv1", None)
    none_message_2 = _ref_id("att1", "conv1", None)

    assert same_1 == same_2
    assert same_1 != different_attachment
    assert same_1 != different_conversation
    assert none_message_1 == none_message_2
    assert none_message_1 != same_1
    assert same_1.startswith("ref-")
    assert len(same_1) == len("ref-") + 16


@pytest.mark.slow
def test_write_lock_prevents_concurrent_writes(test_db: Path) -> None:
    """Threaded store_records() calls must complete without corrupting conversation or message counts."""
    results = []
    errors = []

    def write_conversation(conv_id: int) -> None:
        try:
            conv = make_conversation(f"conv{conv_id}", title=f"Conversation {conv_id}")
            messages = [make_message(f"msg{conv_id}-{i}", f"conv{conv_id}", text=f"Message {i}") for i in range(3)]
            with open_connection(test_db) as conn:
                results.append(store_records(conversation=conv, messages=messages, attachments=[], conn=conn))
        except Exception as exc:  # pragma: no cover - failure path assertion target
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_conversation, idx) for idx in range(10)]
        for future in as_completed(futures):
            future.result()

    assert errors == []
    assert len(results) == 10
    with open_connection(test_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 10
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 30


def test_store_records_without_connection_creates_own(
    test_db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """store_records() must honor the default DB path when no connection is supplied."""
    import polylogue.paths
    import polylogue.storage.sqlite.connection as connection_module
    from polylogue.storage.sqlite.connection import _clear_connection_cache

    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    _clear_connection_cache()
    importlib.reload(polylogue.paths)
    importlib.reload(connection_module)

    default_path = polylogue.paths.db_path()
    default_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(test_db), str(default_path))

    counts = store_records(
        conversation=make_conversation("conv-default", title="No Conn Test"),
        messages=[],
        attachments=[],
    )
    assert counts["conversations"] == 1

    with open_connection(default_path) as conn:
        assert _conversation_row(conn, "conv-default") is not None


@pytest.mark.slow
def test_concurrent_upsert_same_attachment_ref_count_correct(test_db: Path) -> None:
    """Concurrent upserts of the same attachment must keep ref_count equal to actual refs."""
    shared_attachment_id = "shared-attachment-race-test"

    def create_conversation(index: int) -> None:
        conv = make_conversation(
            f"race-conv-{index}",
            title=f"Race Test {index}",
            created_at=None,
            updated_at=None,
            content_hash=f"hash-{index}",
        )
        msg = make_message(
            f"race-msg-{index}",
            f"race-conv-{index}",
            text="test",
            timestamp=None,
            provider_meta=None,
        )
        attachment = make_attachment(
            shared_attachment_id,
            f"race-conv-{index}",
            f"race-msg-{index}",
            mime_type="text/plain",
            size_bytes=100,
            provider_meta=None,
        )
        with open_connection(test_db) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[attachment], conn=conn)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(create_conversation, range(10)))

    with open_connection(test_db) as conn:
        stored_ref_count = conn.execute(
            "SELECT ref_count FROM attachments WHERE attachment_id = ?",
            (shared_attachment_id,),
        ).fetchone()[0]
        actual_refs = conn.execute(
            "SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?",
            (shared_attachment_id,),
        ).fetchone()[0]

    assert stored_ref_count == 10
    assert actual_refs == 10
    assert stored_ref_count == actual_refs


@pytest.mark.parametrize(
    ("size_bytes", "valid"),
    [
        (0, True),
        (1_000_000_000, True),
        (None, True),
        (-100, False),
    ],
    ids=["zero", "large", "unknown", "negative"],
)
def test_attachment_size_bytes_contract(size_bytes: int | None, valid: bool) -> None:
    """Attachment size validation must accept non-negative bounds."""
    if valid:
        record = _attachment_record(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=size_bytes,
            provider_meta=None,
        )
        assert record.size_bytes == size_bytes
    else:
        with pytest.raises(ValidationError):
            AttachmentRecord(
                attachment_id=_attachment_id("test"),
                conversation_id=_conversation_id("conv1"),
                message_id=_message_id("msg1"),
                mime_type="text/plain",
                size_bytes=size_bytes,
                provider_meta=None,
            )


@pytest.mark.parametrize("name", ["claude-ai", "claude-code", "Provider123"])
def test_provider_name_accepts_valid(name: str) -> None:
    """Representative provider-name formats should validate."""
    record = _conversation_record(
        conversation_id="test",
        provider_name=name,
        provider_conversation_id="ext1",
        title="Test",
        content_hash="hash123",
    )
    assert record.provider_name == name


# ============================================================================
# CRUD Laws (from test_crud_laws.py)
# ============================================================================


class TestCrudLaws:
    """Property-based CRUD round-trip laws."""

    @given(conversation_strategy(min_messages=1, max_messages=5))
    @settings(max_examples=30, deadline=None)
    async def test_save_retrieve_roundtrip(self, conv_data: dict[str, object]) -> None:
        """Saving a strategy-generated conversation and retrieving it preserves identity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "roundtrip.db"
            backend = SQLiteBackend(db_path=db_path)

            raw_id = conv_data["id"]
            assert isinstance(raw_id, str)
            conv_id = f"test-{raw_id[:16]}"
            raw_provider = conv_data.get("provider", "test")
            assert isinstance(raw_provider, str)
            provider = raw_provider
            raw_title = conv_data.get("title", "Generated")
            assert isinstance(raw_title, str)
            raw_created_at = conv_data.get("created_at")
            assert raw_created_at is None or isinstance(raw_created_at, str)

            conv = make_conversation(
                conversation_id=conv_id,
                provider_name=provider,
                title=raw_title,
                created_at=raw_created_at,
            )

            messages: list[MessageRecord] = []
            message_payloads = _message_payloads(conv_data.get("messages", []))
            for i, msg_data in enumerate(message_payloads):
                raw_role = msg_data.get("role", "user")
                assert isinstance(raw_role, str)
                raw_text = msg_data.get("text", "")
                assert isinstance(raw_text, str)
                msg = make_message(
                    message_id=f"{conv_id}-m{i}",
                    conversation_id=conv_id,
                    role=raw_role,
                    text=raw_text,
                )
                messages.append(msg)

            await backend.save_conversation_record(conv)
            if messages:
                await backend.save_messages(messages)

            retrieved = await backend.get_conversation(conv_id)
            assert retrieved is not None
            assert retrieved.conversation_id == conv_id
            assert retrieved.provider_name == provider

            retrieved_msgs = await backend.get_messages(conv_id)
            assert len(retrieved_msgs) == len(messages)

            await backend.close()

    @given(conversation_strategy(min_messages=1, max_messages=3))
    @settings(max_examples=20, deadline=None)
    async def test_save_is_idempotent(self, conv_data: dict[str, object]) -> None:
        """Saving the same conversation twice yields the same stored data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "idempotent.db"
            backend = SQLiteBackend(db_path=db_path)

            raw_id = conv_data["id"]
            assert isinstance(raw_id, str)
            conv_id = f"idem-{raw_id[:16]}"
            raw_provider = conv_data.get("provider", "test")
            assert isinstance(raw_provider, str)
            raw_title = conv_data.get("title", "Idempotent")
            assert isinstance(raw_title, str)
            conv = make_conversation(
                conversation_id=conv_id,
                provider_name=raw_provider,
                title=raw_title,
            )

            # Save twice
            await backend.save_conversation_record(conv)
            await backend.save_conversation_record(conv)

            # Should still be exactly one conversation
            all_convs = await backend.queries.list_conversations(_record_query(limit=100))
            matching = [c for c in all_convs if c.conversation_id == conv_id]
            assert len(matching) == 1

            await backend.close()


# ============================================================================
# Repository Laws (from test_repository_laws.py)
# ============================================================================


@st.composite
def simple_tag_spec(draw: st.DrawFn) -> SimpleTagSpec:
    """Generate a tag assignment spec: conversation ID + list of tags."""
    conv_suffix = draw(
        st.text(
            min_size=3,
            max_size=12,
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        ).filter(lambda s: s[0].isalpha())
    )
    tags = draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=15,
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            ),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    return {"conversation_id": f"tag-{conv_suffix}", "tags": tags}


@st.composite
def simple_title_search_spec(draw: st.DrawFn) -> SimpleTitleSearchSpec:
    """Generate a title search spec: title and search substring."""
    words = draw(
        st.lists(
            st.text(min_size=3, max_size=12, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            min_size=2,
            max_size=5,
        )
    )
    title = " ".join(words)
    search_word = draw(st.sampled_from(words))
    return {"title": title, "search_term": search_word}


class TestTagAssignmentLaws:
    """Property-based tests for tag operations on repository."""

    @given(simple_tag_spec())
    @settings(max_examples=15, deadline=None)
    async def test_add_tag_is_retrievable(self, spec: SimpleTagSpec) -> None:
        """Adding a tag to a conversation makes it appear in metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "tags.db"
            conv_id = spec["conversation_id"]

            (
                ConversationBuilder(db_path, conv_id)
                .provider("test")
                .title("Tag Test")
                .add_message("m1", text="Hello")
                .save()
            )

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            tag = spec["tags"][0]
            await repo.add_tag(conv_id, tag)

            conv = await repo.get(conv_id)
            assert conv is not None
            assert tag in conv.tags

            await backend.close()

    @given(simple_tag_spec())
    @settings(max_examples=15, deadline=None)
    async def test_remove_tag_is_idempotent(self, spec: SimpleTagSpec) -> None:
        """Removing a tag that doesn't exist doesn't crash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "rmtags.db"
            conv_id = spec["conversation_id"]

            (
                ConversationBuilder(db_path, conv_id)
                .provider("test")
                .title("Remove Tag Test")
                .add_message("m1", text="Hello")
                .save()
            )

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            tag = spec["tags"][0]
            await repo.remove_tag(conv_id, tag)

            conv = await repo.get(conv_id)
            assert conv is not None
            assert tag not in conv.tags

            await backend.close()


class TestTitleSearchLaws:
    """Property-based tests for title-based search."""

    @given(simple_title_search_spec())
    @settings(max_examples=15, deadline=None)
    async def test_title_search_finds_matching(self, spec: SimpleTitleSearchSpec) -> None:
        """Searching by title substring finds the matching conversation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.index import rebuild_index
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "search.db"
            conv_id = "search-conv-1"

            (
                ConversationBuilder(db_path, conv_id)
                .provider("test")
                .title(spec["title"])
                .add_message("m1", text="Search test content")
                .save()
            )

            with open_connection(db_path) as conn:
                rebuild_index(conn)

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec["search_term"])
            found_ids = [str(c.id) for c in results]
            assert conv_id in found_ids, (
                f"Expected to find '{conv_id}' when searching title='{spec['title']}' for term='{spec['search_term']}'"
            )

            await backend.close()

    @given(simple_title_search_spec())
    @settings(max_examples=15, deadline=None)
    async def test_title_search_excludes_non_matching(self, spec: SimpleTitleSearchSpec) -> None:
        """Title search doesn't return conversations with unrelated titles."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "nomatch.db"

            (
                ConversationBuilder(db_path, "match-conv")
                .provider("test")
                .title(spec["title"])
                .add_message("m1", text="Content")
                .save()
            )

            (
                ConversationBuilder(db_path, "nomatch-conv")
                .provider("test")
                .title("Zzqxjk Wvpnrl Tmygbs")
                .add_message("m2", text="Other content")
                .save()
            )

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec["search_term"])
            found_ids = {str(c.id) for c in results}
            assert "match-conv" in found_ids

            await backend.close()


# ============================================================================
# Search Cache Tests (from test_cache.py)
# ============================================================================


class TestSearchCacheKey:
    """Tests for SearchCacheKey creation and behavior."""

    def test_create_basic(self, tmp_path: Path) -> None:
        """Create a basic cache key."""
        from polylogue.storage.search.cache import SearchCacheKey

        key = SearchCacheKey.create(
            query="hello",
            archive_root=tmp_path,
        )
        assert key.query == "hello"
        assert key.archive_root == str(tmp_path)
        assert key.limit == 20  # default
        assert key.source is None
        assert key.since is None

    def test_create_with_all_params(self, tmp_path: Path) -> None:
        """Create a cache key with all parameters."""
        from polylogue.storage.search.cache import SearchCacheKey

        key = SearchCacheKey.create(
            query="test query",
            archive_root=tmp_path / "archive",
            render_root_path=tmp_path / "render",
            db_path=tmp_path / "test.db",
            limit=50,
            source="claude-ai",
            since="2024-01-01",
        )
        assert key.query == "test query"
        assert key.limit == 50
        assert key.source == "claude-ai"
        assert key.since == "2024-01-01"
        assert key.render_root_path == str(tmp_path / "render")
        assert key.db_path == str(tmp_path / "test.db")

    def test_key_is_frozen(self, tmp_path: Path) -> None:
        """Cache key is immutable (frozen dataclass)."""
        from polylogue.storage.search.cache import SearchCacheKey

        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        attr_name = "query"
        with pytest.raises(AttributeError):
            setattr(key, attr_name, "changed")

    def test_same_params_same_key(self, tmp_path: Path) -> None:
        """Same parameters produce equal keys (same cache version)."""
        from polylogue.storage.search.cache import SearchCacheKey

        key1 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        key2 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        assert key1 == key2

    def test_different_query_different_key(self, tmp_path: Path) -> None:
        """Different queries produce different keys."""
        from polylogue.storage.search.cache import SearchCacheKey

        key1 = SearchCacheKey.create(query="hello", archive_root=tmp_path)
        key2 = SearchCacheKey.create(query="world", archive_root=tmp_path)
        assert key1 != key2

    def test_different_limit_different_key(self, tmp_path: Path) -> None:
        """Different limits produce different keys."""
        from polylogue.storage.search.cache import SearchCacheKey

        key1 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        key2 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=20)
        assert key1 != key2

    def test_none_render_root(self, tmp_path: Path) -> None:
        """None render_root_path stored as None."""
        from polylogue.storage.search.cache import SearchCacheKey

        key = SearchCacheKey.create(query="test", archive_root=tmp_path, render_root_path=None)
        assert key.render_root_path is None

    def test_key_is_hashable(self, tmp_path: Path) -> None:
        """Cache key can be used as dict key (hashable)."""
        from polylogue.storage.search.cache import SearchCacheKey

        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        d = {key: "result"}
        assert d[key] == "result"


class TestInvalidateSearchCache:
    """Tests for cache invalidation."""

    def test_invalidation_increments_version(self, tmp_path: Path) -> None:
        """Invalidation changes cache version."""
        from polylogue.storage.search.cache import SearchCacheKey, invalidate_search_cache

        key_before = SearchCacheKey.create(query="test", archive_root=tmp_path)
        invalidate_search_cache()
        key_after = SearchCacheKey.create(query="test", archive_root=tmp_path)

        # Keys should differ due to version change
        assert key_before != key_after
        assert key_before.cache_version < key_after.cache_version

    def test_multiple_invalidations(self, tmp_path: Path) -> None:
        """Multiple invalidations increment version each time."""
        from polylogue.storage.search.cache import SearchCacheKey, invalidate_search_cache

        v1 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v2 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v3 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version

        assert v1 < v2 < v3


class TestCacheStats:
    """Tests for cache statistics."""

    def test_stats_returns_dict(self) -> None:
        """get_cache_stats returns a dictionary."""
        from polylogue.storage.search.cache import get_cache_stats

        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert "cache_version" in stats

    def test_stats_version_matches_current(self, tmp_path: Path) -> None:
        """Stats version matches what keys use."""
        from polylogue.storage.search.cache import SearchCacheKey, get_cache_stats

        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        stats = get_cache_stats()
        assert stats["cache_version"] == key.cache_version


# ============================================================================
# Repository Tests (relocated from test_json.py)
# ============================================================================


class TestRepositoryOperations:
    """ConversationRepository CRUD operations."""

    async def test_repository_basic_operations(self, test_db: Path) -> None:
        """Test ConversationRepository basic get/list operations."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        factory.create_conversation(
            id="c1", provider="chatgpt", messages=[{"id": "m1", "role": "user", "text": "hello world"}]
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)

        conv = await repo.get("c1")
        assert conv is not None
        assert conv.id == "c1"
        messages = conv.messages.to_list()
        assert len(messages) == 1
        assert messages[0].text == "hello world"

        lst = await repo.list()
        assert len(lst) == 1
        assert lst[0].id == "c1"

    async def test_get_eager_includes_attachment_conversation_id(self, test_db: Path) -> None:
        """ConversationRepository.get_eager() returns attachments with conversation_id field."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        factory.create_conversation(
            id="c-with-att",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "message with attachment",
                    "attachments": [
                        {
                            "id": "att1",
                            "mime_type": "image/png",
                            "size_bytes": 2048,
                            "path": "/path/to/image.png",
                        }
                    ],
                }
            ],
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)
        conv = await repo.get_eager("c-with-att")

        assert conv is not None
        messages = conv.messages.to_list()
        assert len(messages) == 1
        msg = messages[0]
        assert len(msg.attachments) == 1
        att = msg.attachments[0]
        assert att.id == "att1"
        assert att.mime_type == "image/png"

    async def test_get_eager_multiple_attachments(self, test_db: Path) -> None:
        """get_eager() correctly groups multiple attachments per message."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        factory.create_conversation(
            id="c-multi-att",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "first message",
                    "attachments": [
                        {"id": "att1", "mime_type": "image/png"},
                        {"id": "att2", "mime_type": "image/jpeg"},
                    ],
                },
                {
                    "id": "m2",
                    "role": "assistant",
                    "text": "second message",
                    "attachments": [
                        {"id": "att3", "mime_type": "application/pdf"},
                    ],
                },
            ],
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)
        conv = await repo.get_eager("c-multi-att")

        assert conv is not None
        messages = conv.messages.to_list()
        assert len(messages) == 2

        m1 = messages[0]
        assert len(m1.attachments) == 2
        m1_att_ids = {a.id for a in m1.attachments}
        assert m1_att_ids == {"att1", "att2"}

        m2 = messages[1]
        assert len(m2.attachments) == 1
        assert m2.attachments[0].id == "att3"

    async def test_get_eager_attachment_metadata_decoded(self, test_db: Path) -> None:
        """Attachment provider_meta JSON is properly decoded."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        meta = {"original_name": "photo.png", "source": "upload"}
        factory.create_conversation(
            id="c-att-meta",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "with meta",
                    "attachments": [
                        {
                            "id": "att-meta",
                            "mime_type": "image/png",
                            "meta": meta,
                        }
                    ],
                }
            ],
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)
        conv = await repo.get_eager("c-att-meta")

        assert conv is not None
        messages = conv.messages.to_list()
        assert len(messages) == 1
        msg = messages[0]
        assert len(msg.attachments) == 1
        att = msg.attachments[0]
        assert att.provider_meta == meta or att.provider_meta is None


class TestCacheThreadSafety:
    """Thread safety tests for cache invalidation."""

    def test_concurrent_invalidation(self) -> None:
        """Concurrent invalidation doesn't corrupt state."""
        import threading

        from polylogue.storage.search.cache import get_cache_stats, invalidate_search_cache

        initial_stats = get_cache_stats()
        initial_version = initial_stats["cache_version"]

        errors: list[Exception] = []
        num_threads = 10
        invalidations_per_thread = 100

        def invalidate_many() -> None:
            try:
                for _ in range(invalidations_per_thread):
                    invalidate_search_cache()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=invalidate_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        final_stats = get_cache_stats()
        expected_version = initial_version + (num_threads * invalidations_per_thread)
        assert final_stats["cache_version"] == expected_version


class _VectorSpy:
    model = "test-model"

    def __init__(self) -> None:
        self.query_calls: list[tuple[str, int]] = []
        self.upsert_calls: list[tuple[str, list[MessageRecord]]] = []

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        self.query_calls.append((text, limit))
        return [("msg-1", 0.125)]

    def upsert(self, conversation_id: str, messages: list[MessageRecord]) -> None:
        self.upsert_calls.append((conversation_id, messages))


class TestRepositoryVectorAsyncBoundary:
    async def test_search_similar_offloads_vector_query(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "vectors.db")
        repo = ConversationRepository(backend=backend)
        provider = _VectorSpy()
        monkeypatch.setattr(repo, "_get_message_conversation_mapping", AsyncMock(return_value={"msg-1": "conv-1"}))
        monkeypatch.setattr(repo, "get_many", AsyncMock(return_value=[_conversation_model("conv-1")]))

        to_thread_calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []

        async def fake_to_thread(func: Callable[..., object], /, *args: object, **kwargs: object) -> object:
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        result = await repo.search_similar("semantic query", limit=4, vector_provider=provider)

        assert [str(conversation.id) for conversation in result] == ["conv-1"]
        assert provider.query_calls == [("semantic query", 12)]
        assert len(to_thread_calls) == 1
        assert getattr(to_thread_calls[0][0], "__self__", None) is provider
        assert getattr(to_thread_calls[0][0], "__name__", "") == "query"

    async def test_embed_conversation_offloads_vector_upsert(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        messages = [make_message("msg-embed", "conv-embed", text="Message long enough to embed.")]
        backend = SQLiteBackend(db_path=tmp_path / "vectors.db")
        repo = ConversationRepository(backend=backend)
        monkeypatch.setattr(repo.queries, "get_messages", AsyncMock(return_value=messages))
        provider = _VectorSpy()

        to_thread_calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []

        async def fake_to_thread(func: Callable[..., object], /, *args: object, **kwargs: object) -> object:
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        result = await repo.embed_conversation("conv-embed", vector_provider=provider)

        assert result == 1
        assert provider.upsert_calls == [("conv-embed", messages)]
        assert len(to_thread_calls) == 1
        assert getattr(to_thread_calls[0][0], "__self__", None) is provider
        assert getattr(to_thread_calls[0][0], "__name__", "") == "upsert"

    async def test_similarity_search_offloads_vector_query(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "vectors.db")
        repo = ConversationRepository(backend=backend)
        provider = _VectorSpy()
        monkeypatch.setattr(repo, "_get_message_conversation_mapping", AsyncMock(return_value={"msg-1": "conv-1"}))

        to_thread_calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []

        async def fake_to_thread(func: Callable[..., object], /, *args: object, **kwargs: object) -> object:
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        result = await repo.similarity_search("semantic query", limit=4, vector_provider=provider)

        assert result == [("conv-1", "msg-1", 0.125)]
        assert provider.query_calls == [("semantic query", 4)]
        assert len(to_thread_calls) == 1
        assert getattr(to_thread_calls[0][0], "__self__", None) is provider
        assert getattr(to_thread_calls[0][0], "__name__", "") == "query"


# ============================================================================
# TagAssignmentSpec / TitleSearchSpec — infra strategy activation (B5)
# ============================================================================


class TestInfraTagAssignment:
    """Property-based tests using the full TagAssignmentSpec strategy."""

    @given(infra_tag_assignment_strategy(min_conversations=2, max_conversations=4))
    @settings(max_examples=10, deadline=None)
    async def test_tag_assignment_roundtrip(self, spec: TagAssignmentSpec) -> None:
        """Tags assigned via strategy-generated specs are retrievable and consistent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository

            db_path = Path(tmp_dir) / "tags-infra.db"
            seed_conversation_graph(db_path, spec.conversations)

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            # Assign all tags
            for conv, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                for tag in tags:
                    await repo.add_tag(conv.conversation_id, tag)

            # Verify each conversation has the expected tags
            for conv, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                stored = await repo.get(conv.conversation_id)
                assert stored is not None
                stored_tags = set(stored.tags)
                for tag in set(tags):
                    assert tag in stored_tags, f"Tag '{tag}' missing from conv '{conv.conversation_id}'"

            await backend.close()

    @given(infra_tag_assignment_strategy(min_conversations=2, max_conversations=4))
    @settings(max_examples=10, deadline=None)
    async def test_tag_counts_match_expected(self, spec: TagAssignmentSpec) -> None:
        """Tag counts computed from strategy match actual stored tag counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository

            db_path = Path(tmp_dir) / "tag-counts.db"
            seed_conversation_graph(db_path, spec.conversations)

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            for conv, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                for tag in tags:
                    await repo.add_tag(conv.conversation_id, tag)

            expected = expected_tag_counts(spec)
            actual: dict[str, int] = {}
            for conv, _tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                stored = await repo.get(conv.conversation_id)
                if stored:
                    for tag in stored.tags:
                        actual[tag] = actual.get(tag, 0) + 1

            assert actual == expected

            await backend.close()


class TestInfraTitleSearch:
    """Property-based tests using the full TitleSearchSpec strategy."""

    @given(infra_title_search_strategy())
    @settings(max_examples=15, deadline=None)
    async def test_literal_title_search_finds_matching_with_special_chars(self, spec: TitleSearchSpec) -> None:
        """Title search with wildcard-sensitive characters finds exact matches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "title-search.db"

            (
                ConversationBuilder(db_path, "match-conv")
                .provider("test")
                .title(spec.matching_title)
                .add_message("m1", text="Content")
                .save()
            )

            (
                ConversationBuilder(db_path, "decoy-conv")
                .provider("test")
                .title(spec.decoy_title)
                .add_message("m2", text="Other")
                .save()
            )

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec.needle)
            found_ids = {str(c.id) for c in results}
            assert "match-conv" in found_ids, (
                f"Expected 'match-conv' for needle='{spec.needle}' in title='{spec.matching_title}'"
            )

            await backend.close()

    @given(infra_title_search_strategy())
    @settings(max_examples=15, deadline=None)
    async def test_literal_title_search_excludes_decoy(self, spec: TitleSearchSpec) -> None:
        """Title search with special characters does not match the decoy title."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "decoy-search.db"

            (
                ConversationBuilder(db_path, "decoy-only")
                .provider("test")
                .title(spec.decoy_title)
                .add_message("m1", text="Content")
                .save()
            )

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec.needle)
            found_ids = {str(c.id) for c in results}
            assert "decoy-only" not in found_ids, f"Decoy '{spec.decoy_title}' should not match needle='{spec.needle}'"
