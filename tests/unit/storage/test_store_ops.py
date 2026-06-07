"""Focused roundtrip and validation contracts for storage record helpers."""

from __future__ import annotations

import asyncio
import hashlib
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

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.core.sources import origin_from_provider
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    MessageRecord,
    SessionRecord,
    _json_or_none,
    _make_ref_id,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    MessageId,
    Provider,
    SemanticBlockType,
    SessionId,
)
from tests.infra.storage_records import (
    _prune_attachment_refs,
    make_attachment,
    make_message,
    make_session,
    store_records,
    upsert_attachment,
    upsert_message,
    upsert_session,
)
from tests.infra.strategies.messages import session_strategy
from tests.infra.strategies.storage import (
    TagAssignmentSpec,
    TitleSearchSpec,
    expected_tag_counts,
    seed_session_graph,
)
from tests.infra.strategies.storage import (
    literal_title_search_strategy as infra_title_search_strategy,
)
from tests.infra.strategies.storage import (
    tag_assignment_strategy as infra_tag_assignment_strategy,
)


class SimpleTagSpec(TypedDict):
    session_id: str
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


def _session_id(value: str) -> SessionId:
    return SessionId(value)


def _message_id(value: str) -> MessageId:
    return MessageId(value)


def _attachment_id(value: str) -> AttachmentId:
    return AttachmentId(value)


def _content_hash(value: str) -> ContentHash:
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        return ContentHash(hashlib.sha256(value.encode("utf-8")).hexdigest())
    return ContentHash(value if len(raw) == 32 else hashlib.sha256(value.encode("utf-8")).hexdigest())


def _content_block(
    *,
    block_id: str,
    message_id: str,
    session_id: str,
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
    # #1240: media_type now lives inside the metadata JSON envelope.
    if media_type:
        from polylogue.core.json import dumps as _json_dumps
        from polylogue.core.json import loads as _json_loads

        base: dict[str, object] = {}
        if metadata:
            try:
                parsed = _json_loads(metadata)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                base.update(parsed)
        base.setdefault("media_type", media_type)
        metadata = _json_dumps(base)
    return ContentBlockRecord(
        block_id=block_id,
        message_id=_message_id(message_id),
        session_id=_session_id(session_id),
        block_index=block_index,
        type=ContentBlockType.from_string(block_type),
        text=text,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input,
        metadata=metadata,
        semantic_type=None if semantic_type is None else SemanticBlockType.from_string(semantic_type),
    )


def _session_record(
    *,
    session_id: str,
    origin: str,
    native_id: str,
    content_hash: str,
    title: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
    metadata: dict[str, object] | None = None,
) -> SessionRecord:
    return SessionRecord(
        session_id=_session_id(session_id),
        origin=Origin.from_string(origin),
        native_id=native_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        content_hash=_content_hash(content_hash),
        metadata=metadata,
    )


def _attachment_record(
    *,
    attachment_id: str,
    session_id: str,
    message_id: str | None,
    mime_type: str,
    size_bytes: int | None,
    display_name: str | None = None,
) -> AttachmentRecord:
    return AttachmentRecord(
        attachment_id=_attachment_id(attachment_id),
        session_id=_session_id(session_id),
        message_id=None if message_id is None else _message_id(message_id),
        mime_type=mime_type,
        size_bytes=size_bytes,
        display_name=display_name,
        attachment_native_id=attachment_id,
    )


def _ref_id(attachment_id: str, session_id: str, message_id: str | None) -> str:
    return _make_ref_id(
        _attachment_id(attachment_id),
        _session_id(session_id),
        None if message_id is None else _message_id(message_id),
    )


def _session_model(session_id: str) -> Session:
    return Session(
        id=_session_id(session_id),
        origin=Origin.CHATGPT_EXPORT,
        messages=MessageCollection.empty(),
    )


def _session_row(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ? OR native_id = ? ORDER BY session_id LIMIT 1",
        (session_id, session_id),
    ).fetchone()
    if row is not None and not isinstance(row, sqlite3.Row):
        raise TypeError(f"expected sqlite3.Row, got {type(row).__name__}")
    return row


def _message_count(conn: sqlite3.Connection, session_id: str) -> int:
    session_row = _session_row(conn, session_id)
    resolved_session_id = str(session_row["session_id"]) if session_row is not None else session_id
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (resolved_session_id,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _attachment_row(conn: sqlite3.Connection, attachment_id: str) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT
            a.*,
            a.media_type AS mime_type,
            a.byte_count AS size_bytes,
            COALESCE(r.source_url, a.display_name) AS path
        FROM attachments a
        LEFT JOIN attachment_refs r ON r.attachment_id = a.attachment_id
        LEFT JOIN attachment_native_ids ani ON ani.ref_id = r.ref_id
        WHERE a.attachment_id = ? OR (ani.id_kind = 'attachment' AND ani.native_id = ?)
        ORDER BY a.attachment_id
        LIMIT 1
        """,
        (attachment_id, attachment_id),
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


def _record_query(**kwargs: Unpack[RecordQueryKwargs]) -> SessionRecordQuery:
    return SessionRecordQuery(**kwargs)


class _MessageStats(TypedDict):
    total: int
    user: int
    assistant: int
    system: int
    words_approx: int
    attachment_refs: int
    distinct_attachments: int
    providers: dict[str, int]


def _aggregate_message_stats_native(db_path: Path, session_ids: list[str] | None = None) -> _MessageStats:
    """Aggregate message stats read directly from the archive `index.db`.

    Mirrors the legacy ``backend.queries.aggregate_message_stats`` contract over
    the archive ``messages`` / ``sessions`` / ``attachment_refs`` tables.
    """
    from tests.infra.archive_scenarios import open_index_db

    with open_index_db(db_path) as conn:
        where = ""
        params: tuple[str, ...] = ()
        if session_ids is not None:
            placeholders = ", ".join("?" for _ in session_ids)
            where = f" WHERE m.session_id IN ({placeholders})"
            params = tuple(session_ids)
        rows = conn.execute(
            f"SELECT m.role AS role, m.word_count AS word_count FROM messages m{where}",
            params,
        ).fetchall()
        total = len(rows)
        by_role = {"user": 0, "assistant": 0, "system": 0, "tool": 0}
        words = 0
        for row in rows:
            role = str(row["role"])
            if role in by_role:
                by_role[role] += 1
            words += int(row["word_count"] or 0)

        ref_where = ""
        if session_ids is not None:
            placeholders = ", ".join("?" for _ in session_ids)
            ref_where = f" WHERE ar.session_id IN ({placeholders})"
        attachment_refs = int(
            conn.execute(
                f"SELECT COUNT(*) FROM attachment_refs ar{ref_where}",
                params,
            ).fetchone()[0]
        )
        distinct_attachments = int(
            conn.execute(
                f"SELECT COUNT(DISTINCT ar.attachment_id) FROM attachment_refs ar{ref_where}",
                params,
            ).fetchone()[0]
        )

        provider_where = ""
        if session_ids is not None:
            placeholders = ", ".join("?" for _ in session_ids)
            provider_where = f" WHERE s.session_id IN ({placeholders})"
        provider_rows = conn.execute(
            f"SELECT s.origin AS origin, COUNT(*) AS count FROM sessions s{provider_where} GROUP BY s.origin",
            params,
        ).fetchall()
        providers = {_origin_to_provider(str(row["origin"])): int(row["count"]) for row in provider_rows}

    return {
        "total": total,
        "user": by_role["user"],
        "assistant": by_role["assistant"],
        "system": by_role["system"],
        "words_approx": words,
        "attachment_refs": attachment_refs,
        "distinct_attachments": distinct_attachments,
        "providers": providers,
    }


def _origin_to_provider(origin: str) -> str:
    from polylogue.api.archive import _provider_for_archive_origin

    return _provider_for_archive_origin(origin).value


@pytest.mark.asyncio
async def test_aggregate_message_stats_reports_role_counts_and_words(workspace_env: dict[str, Path]) -> None:
    from tests.infra.archive_scenarios import native_session_id_for
    from tests.infra.storage_records import SessionBuilder, db_setup

    db_path = db_setup(workspace_env)

    (
        SessionBuilder(db_path, "conv-stats-a")
        .provider("chatgpt")
        .add_message("m-user-a", role="user", text="hello world")
        .add_message("m-assistant-a", role="assistant", text="answer words here")
        .add_attachment(message_id="m-assistant-a", path="spec.pdf")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-stats-b")
        .provider("codex")
        .add_message("m-system-b", role="system", text="system note")
        .add_message("m-user-b", role="user", text="follow up")
        .save()
    )

    unfiltered = _aggregate_message_stats_native(db_path)
    filtered = _aggregate_message_stats_native(db_path, [native_session_id_for("chatgpt", "conv-stats-a")])

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


def _file_read_block(*, message_id: str, session_id: str, path: str, block_index: int = 0) -> ContentBlockRecord:
    """A native ``Read`` tool_use block whose path lives in ``tool_input.file_path``.

    Native ``tool_path`` (and therefore referenced-path filtering) is the
    generated column ``json_extract(tool_input, '$.file_path' | '$.path')``, so
    the path must be carried in ``tool_input`` rather than block metadata.
    """
    return _content_block(
        block_id=f"{message_id}-{block_index}",
        message_id=message_id,
        session_id=session_id,
        block_index=block_index,
        block_type="tool_use",
        tool_name="Read",
        tool_input=json.dumps({"file_path": path}),
        semantic_type="file_read",
    )


@pytest.mark.asyncio
async def test_backend_referenced_path_filter_contract(workspace_env: dict[str, Path]) -> None:
    """Native substrate list/count filters must honor persisted semantic paths."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_scenarios import native_session_id_for
    from tests.infra.storage_records import SessionBuilder, db_setup

    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    target_path = "/workspace/polylogue/README.md"
    other_path = "/workspace/polylogue/docs/cli-reference.md"

    (
        SessionBuilder(db_path, "conv-readme")
        .provider("claude-code")
        .title("README work")
        .add_message(
            "m1",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Inspecting the repository README"},
                _file_read_block(message_id="m1", session_id="conv-readme", path=target_path),
            ],
        )
        .save()
    )

    (
        SessionBuilder(db_path, "conv-other")
        .provider("claude-code")
        .title("CLI docs work")
        .add_message(
            "m2",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Inspecting docs"},
                _file_read_block(message_id="m2", session_id="conv-other", path=other_path),
            ],
        )
        .save()
    )

    expected_id = native_session_id_for("claude-code", "conv-readme")
    with ArchiveStore.open_existing(archive_root) as archive:
        matches = archive.list_summaries(referenced_paths=(target_path,), limit=10)
        assert [summary.session_id for summary in matches] == [expected_id]
        assert archive.count_sessions(referenced_paths=(target_path,)) == 1


@pytest.mark.asyncio
async def test_list_summaries_by_query_uses_current_session_columns(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        upsert_session(
            conn,
            make_session(
                "conv-large-meta",
                source_name="codex",
                title="Large Meta Session",
                metadata={"tag": "kept"},
            ),
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    repo = SessionRepository(backend=backend)
    try:
        summaries = await repo.list_summaries_by_query(_record_query(provider="codex", limit=1))
        assert len(summaries) == 1
        summary = summaries[0]
        assert str(summary.id) == "codex-session:conv-large-meta"
        assert summary.origin.value == "codex-session"
        assert summary.title == "Large Meta Session"
        assert summary.metadata == {}
        assert "provider_meta" not in summary.model_dump()
    finally:
        await repo.close()


def test_actions_view_uses_blocks_without_session_payload_bloat(workspace_env: dict[str, Path]) -> None:
    """The archive actions surface derives from blocks and has no session payload column.

    The archive ``actions`` view derives tool/semantic facts directly from
    ``blocks``. Session working-directory and git facts live in typed columns
    and child tables, so large raw payloads stay in source storage.
    """
    from tests.infra.archive_scenarios import native_session_id_for, open_index_db
    from tests.infra.storage_records import SessionBuilder, db_setup

    db_path = db_setup(workspace_env)

    (
        SessionBuilder(db_path, "conv-heavy-action")
        .provider("codex")
        .title("Heavy Action Event Source")
        .working_directories(["/realm/project/sinex"])
        .git_branch("master")
        .git_repository_url("git@github.com:Sinity/sinex.git")
        .add_message(
            "m-heavy-action",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Searching the repo for current archive handling."},
                _tool_block(
                    message_id="m-heavy-action",
                    session_id="conv-heavy-action",
                    tool_name="Grep",
                    semantic_type="search",
                ),
            ],
        )
        .save()
    )

    session_id = native_session_id_for("codex", "conv-heavy-action")
    with open_index_db(db_path) as conn:
        action_columns = {row["name"] for row in conn.execute("PRAGMA table_info(actions)")}
        assert "payload" not in action_columns
        row = conn.execute(
            "SELECT tool_name, semantic_type FROM actions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    assert row is not None
    assert row["semantic_type"] == "search"
    assert row["tool_name"] == "Grep"


@pytest.mark.asyncio
async def test_filter_referenced_path_apply_after_fts_search(workspace_env: dict[str, Path]) -> None:
    """Combined FTS + path queries must keep the path constraint after search ranking."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_scenarios import native_session_id_for
    from tests.infra.storage_records import SessionBuilder, db_setup

    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    target_path = "/workspace/polylogue/README.md"

    (
        SessionBuilder(db_path, "conv-match")
        .provider("claude-code")
        .title("Path match")
        .add_message(
            "m1",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Investigating the same parser regression"},
                _file_read_block(message_id="m1", session_id="conv-match", path=target_path),
            ],
        )
        .save()
    )

    (
        SessionBuilder(db_path, "conv-no-path")
        .provider("claude-code")
        .title("No path match")
        .add_message(
            "m2",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Investigating the same parser regression"},
                _file_read_block(
                    message_id="m2",
                    session_id="conv-no-path",
                    path="/workspace/polylogue/docs/cli-reference.md",
                ),
            ],
        )
        .save()
    )

    with ArchiveStore.open_existing(archive_root) as archive:
        hits = archive.search_summaries("parser regression", referenced_paths=(target_path,), limit=10)
        assert [hit.session_id for hit in hits] == [native_session_id_for("claude-code", "conv-match")]


def _tool_block(
    *, message_id: str, session_id: str, tool_name: str, semantic_type: str, tool_input: str | None = None
) -> ContentBlockRecord:
    return _content_block(
        block_id=f"{message_id}-0",
        message_id=message_id,
        session_id=session_id,
        block_index=0,
        block_type="tool_use",
        tool_name=tool_name,
        tool_input=tool_input,
        semantic_type=semantic_type,
    )


@pytest.mark.asyncio
async def test_backend_action_terms_filter_contract(workspace_env: dict[str, Path]) -> None:
    """Native substrate list/count filters must honor semantic action categories."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_scenarios import native_session_id_for
    from tests.infra.storage_records import SessionBuilder, db_setup

    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    (
        SessionBuilder(db_path, "conv-search")
        .provider("claude-code")
        .title("Search work")
        .add_message(
            "m1",
            role="assistant",
            content_blocks=[
                _tool_block(message_id="m1", session_id="conv-search", tool_name="Grep", semantic_type="search")
            ],
        )
        .save()
    )
    (
        SessionBuilder(db_path, "conv-git")
        .provider("claude-code")
        .title("Git work")
        .add_message(
            "m2",
            role="assistant",
            content_blocks=[
                _tool_block(
                    message_id="m2",
                    session_id="conv-git",
                    tool_name="Bash",
                    semantic_type="git",
                    tool_input='{"command":"git status"}',
                )
            ],
        )
        .save()
    )
    # ``Edit`` classifies to the ``file_edit`` semantic category directly. (The
    # archive writer derives semantic_type from tool name + input via
    # ``classify_tool`` and stores OTHER as NULL, so there is no queryable
    # "other" action category — it is replaced here with a real category.)
    (
        SessionBuilder(db_path, "conv-edit")
        .provider("claude-code")
        .title("Edit tool work")
        .add_message(
            "m3",
            role="assistant",
            content_blocks=[
                _tool_block(message_id="m3", session_id="conv-edit", tool_name="Edit", semantic_type="file_edit")
            ],
        )
        .save()
    )
    (
        SessionBuilder(db_path, "conv-none")
        .provider("claude-code")
        .title("Plain dialogue")
        .add_message("m4", role="assistant", text="No tool use here")
        .save()
    )

    def nid(conv_id: str) -> str:
        return native_session_id_for("claude-code", conv_id)

    with ArchiveStore.open_existing(archive_root) as archive:

        def ids(**kwargs: object) -> list[str]:
            return [summary.session_id for summary in archive.list_summaries(limit=10, **kwargs)]  # type: ignore[arg-type]

        assert ids(action_terms=("search",)) == [nid("conv-search")]
        assert archive.count_sessions(action_terms=("search",)) == 1

        assert ids(action_terms=("file_edit",)) == [nid("conv-edit")]

        assert ids(action_terms=("none",)) == [nid("conv-none")]
        assert archive.count_sessions(action_terms=("none",)) == 1

        assert ids(tool_terms=("grep",)) == [nid("conv-search")]
        assert ids(tool_terms=("none",)) == [nid("conv-none")]

        assert ids(action_terms=("search",), excluded_action_terms=("git",)) == [nid("conv-search")]

        assert sorted(ids(excluded_tool_terms=("grep",))) == sorted(
            [nid("conv-git"), nid("conv-none"), nid("conv-edit")]
        )
        assert sorted(ids(excluded_action_terms=("none",))) == sorted(
            [nid("conv-git"), nid("conv-edit"), nid("conv-search")]
        )


@pytest.mark.asyncio
async def test_filter_action_terms_apply_after_fts_search(workspace_env: dict[str, Path]) -> None:
    """Combined FTS + action queries must preserve action constraints after search ranking."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_scenarios import native_session_id_for
    from tests.infra.storage_records import SessionBuilder, db_setup

    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    (
        SessionBuilder(db_path, "conv-search")
        .provider("claude-code")
        .title("Search match")
        .add_message(
            "m1",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Investigating the same parser regression"},
                _tool_block(message_id="m1", session_id="conv-search", tool_name="Grep", semantic_type="search"),
            ],
        )
        .save()
    )

    (
        SessionBuilder(db_path, "conv-shell")
        .provider("claude-code")
        .title("Shell only")
        .add_message(
            "m2",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Investigating the same parser regression"},
                _tool_block(
                    message_id="m2",
                    session_id="conv-shell",
                    tool_name="Bash",
                    semantic_type="shell",
                    tool_input='{"command":"python -m pytest"}',
                ),
            ],
        )
        .save()
    )

    with ArchiveStore.open_existing(archive_root) as archive:
        hits = archive.search_summaries(
            "parser regression",
            action_terms=("search",),
            excluded_action_terms=("git",),
            limit=10,
        )
        assert [hit.session_id for hit in hits] == [native_session_id_for("claude-code", "conv-search")]


@pytest.mark.asyncio
async def test_filter_action_terms_reconcile_runtime_semantics_after_sql_candidate_fetch(
    workspace_env: dict[str, Path],
) -> None:
    """Native write-time tool classification supersedes any seeded semantic_type.

    The archive writer derives ``semantic_type`` from the tool name + input via
    ``classify_tool`` at ingest time, so a ``TaskCreate`` tool use is persisted
    as the ``agent`` category regardless of any seeded label. There is no stale
    ``other`` row to reconcile at query time.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_scenarios import native_session_id_for
    from tests.infra.storage_records import SessionBuilder, db_setup

    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    (
        SessionBuilder(db_path, "conv-agent")
        .provider("claude-code")
        .title("Agent tool use")
        .add_message(
            "m1",
            role="assistant",
            content_blocks=[
                {"type": "text", "text": "Create a task for the next review pass"},
                _tool_block(
                    message_id="m1",
                    session_id="conv-agent",
                    tool_name="TaskCreate",
                    semantic_type="other",
                ),
            ],
        )
        .save()
    )

    expected_id = native_session_id_for("claude-code", "conv-agent")
    with ArchiveStore.open_existing(archive_root) as archive:
        assert [s.session_id for s in archive.list_summaries(action_terms=("other",), limit=10)] == []
        assert [s.session_id for s in archive.list_summaries(action_terms=("agent",), limit=10)] == [expected_id]
        assert archive.count_sessions(action_terms=("agent",)) == 1


def test_store_records_roundtrip_contract(test_conn: sqlite3.Connection) -> None:
    """store_records() must insert, skip, update, and handle sparse payloads coherently."""
    initial = make_session("conv-create", content_hash="hash-create")
    created = store_records(
        session=initial,
        messages=[make_message("msg-create", "conv-create", text="Hello")],
        attachments=[],
        conn=test_conn,
    )
    assert created == {
        "sessions": 1,
        "messages": 1,
        "attachments": 0,
        "skipped_sessions": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    created_row = _session_row(test_conn, "conv-create")
    assert created_row is not None
    assert created_row["title"] == "Test Session"
    assert _message_count(test_conn, "conv-create") == 1

    duplicate = store_records(
        session=initial,
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert duplicate["sessions"] == 0
    assert duplicate["skipped_sessions"] == 1

    updated = store_records(
        session=make_session("conv-create", title="Updated Title", content_hash="hash-updated"),
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert updated["sessions"] == 1
    updated_row = _session_row(test_conn, "conv-create")
    assert updated_row is not None
    assert updated_row["title"] == "Updated Title"
    assert updated_row["content_hash"].hex() == hashlib.sha256(b"hash-updated").hexdigest()

    multi = store_records(
        session=make_session("conv-multi", title="Multi Message"),
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
        session=make_session("conv-empty", title="Empty Session"),
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
    assert sparse["sessions"] == 1
    assert sparse["messages"] == 0
    assert sparse["attachments"] == 0
    assert sparse["skipped_attachments"] == 1
    sparse_attachment = _attachment_row(test_conn, "att-empty")
    assert sparse_attachment is None


def test_prune_attachment_refs_contract(test_conn: sqlite3.Connection) -> None:
    """Pruning refs must keep requested refs, recalculate counts, and delete zero-ref attachments."""
    conv = make_session("conv-prune", title="Prune Test")
    msg1 = make_message("msg-prune-1", "conv-prune", text="First")
    msg2 = make_message("msg-prune-2", "conv-prune", text="Second")
    att1 = make_attachment("att-prune-1", "conv-prune", "msg-prune-1", mime_type="image/png")
    att2 = make_attachment("att-prune-2", "conv-prune", "msg-prune-2", mime_type="image/jpeg", size_bytes=2048)
    shared_att_1 = make_attachment("att-shared", "conv-prune", "msg-prune-1", mime_type="image/png")
    shared_att_2 = make_attachment("att-shared", "conv-prune", "msg-prune-2", mime_type="image/png")
    store_records(
        session=conv,
        messages=[msg1, msg2],
        attachments=[att1, att2, shared_att_1, shared_att_2],
        conn=test_conn,
    )

    current_session_id = "unknown-export:conv-prune"
    keep_rows = test_conn.execute(
        """
        SELECT ar.ref_id, ani.native_id
        FROM attachment_refs ar
        JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id AND ani.id_kind = 'attachment'
        WHERE ar.session_id = ? AND ar.message_id = ? AND ani.native_id IN ('att-prune-1', 'att-shared')
        ORDER BY ani.native_id
        """,
        (current_session_id, f"{current_session_id}:msg-prune-1"),
    ).fetchall()
    keep_refs = {str(row["ref_id"]) for row in keep_rows}
    assert len(keep_refs) == 2
    _prune_attachment_refs(test_conn, current_session_id, keep_refs)

    remaining_refs = test_conn.execute(
        "SELECT ref_id FROM attachment_refs WHERE session_id = ? ORDER BY ref_id",
        (current_session_id,),
    ).fetchall()
    assert [row["ref_id"] for row in remaining_refs] == sorted(keep_refs)
    pruned_attachment = _attachment_row(test_conn, "att-prune-1")
    shared_attachment = _attachment_row(test_conn, "att-shared")
    assert pruned_attachment is not None
    assert shared_attachment is not None
    assert pruned_attachment["ref_count"] == 1
    assert shared_attachment["ref_count"] == 1
    assert _attachment_row(test_conn, "att-prune-2") is None


def test_upsert_optional_and_attachment_contracts(test_conn: sqlite3.Connection) -> None:
    """Optional-field upserts and attachment metadata updates must round-trip cleanly."""
    session = _session_record(
        session_id="conv-optional",
        origin=origin_from_provider(Provider.from_string("test")).value,
        native_id="conv-optional",
        title=None,
        created_at=None,
        updated_at=None,
        content_hash="hash1",
    )
    assert upsert_session(test_conn, session) is True
    conv_row = _session_row(test_conn, "conv-optional")
    assert conv_row is not None
    assert conv_row["title"] is None
    assert conv_row["created_at_ms"] is None
    assert "provider_meta" not in conv_row.keys()  # noqa: SIM118 — sqlite3.Row membership is over values

    message = MessageRecord(
        message_id=_message_id("msg-optional"),
        session_id=_session_id("conv-optional"),
        provider_message_id=None,
        role=None,
        text=None,
        sort_key=None,
        content_hash=_content_hash("msg-optional-hash"),
        source_name="",
        word_count=0,
        has_tool_use=0,
        has_thinking=0,
    )
    assert upsert_message(test_conn, message) is True
    msg_row = test_conn.execute(
        "SELECT * FROM messages WHERE native_id = ?",
        ("msg-optional",),
    ).fetchone()
    assert msg_row is not None
    assert msg_row["message_id"] == "unknown-export:conv-optional:msg-optional"
    assert msg_row["role"] == "unknown"
    assert msg_row["native_id"] == "msg-optional"
    assert (
        test_conn.execute("SELECT COUNT(*) FROM blocks WHERE message_id = ?", (msg_row["message_id"],)).fetchone()[0]
        == 0
    )

    msg2 = make_message("msg-attachment-2", "conv-optional", text="Second")
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
    assert att_row["path"] == "path.jpg"
    source_native = test_conn.execute(
        """
        SELECT ani.native_id
        FROM attachment_refs ar
        JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id
        WHERE ar.attachment_id = ? AND ani.id_kind = 'source'
        ORDER BY ani.native_id DESC
        LIMIT 1
        """,
        (att_row["attachment_id"],),
    ).fetchone()
    assert source_native is not None
    assert source_native["native_id"] == "new/path.jpg"
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
    """Attachment ref IDs must be deterministic and sensitive to attachment, session, and message."""
    same_1 = _ref_id("att1", "conv1", "msg1")
    same_2 = _ref_id("att1", "conv1", "msg1")
    different_attachment = _ref_id("att2", "conv1", "msg1")
    different_session = _ref_id("att1", "conv2", "msg1")
    none_message_1 = _ref_id("att1", "conv1", None)
    none_message_2 = _ref_id("att1", "conv1", None)

    assert same_1 == same_2
    assert same_1 != different_attachment
    assert same_1 != different_session
    assert none_message_1 == none_message_2
    assert none_message_1 != same_1
    assert same_1.startswith("ref-")
    assert len(same_1) == len("ref-") + 16


@pytest.mark.slow
def test_write_lock_prevents_concurrent_writes(test_db: Path) -> None:
    """Threaded store_records() calls must complete without corrupting session or message counts."""
    results = []
    errors = []

    def write_session(conv_id: int) -> None:
        try:
            conv = make_session(f"conv{conv_id}", title=f"Session {conv_id}")
            messages = [make_message(f"msg{conv_id}-{i}", f"conv{conv_id}", text=f"Message {i}") for i in range(3)]
            with open_connection(test_db) as conn:
                results.append(store_records(session=conv, messages=messages, attachments=[], conn=conn))
        except Exception as exc:  # pragma: no cover - failure path assertion target
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_session, idx) for idx in range(10)]
        for future in as_completed(futures):
            future.result()

    assert errors == []
    assert len(results) == 10
    with open_connection(test_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 10
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
        session=make_session("conv-default", title="No Conn Test"),
        messages=[],
        attachments=[],
    )
    assert counts["sessions"] == 1

    with open_connection(default_path) as conn:
        assert _session_row(conn, "conv-default") is not None


@pytest.mark.slow
def test_concurrent_upsert_same_attachment_ref_count_correct(test_db: Path) -> None:
    """Concurrent upserts of the same attachment must keep ref_count equal to actual refs."""
    shared_attachment_id = "shared-attachment-race-test"

    def create_session(index: int) -> None:
        conv = make_session(
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
            store_records(session=conv, messages=[msg], attachments=[attachment], conn=conn)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(create_session, range(10)))

    with open_connection(test_db) as conn:
        attachment_row = conn.execute(
            """
            SELECT ar.attachment_id
            FROM attachment_refs ar
            JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id
            WHERE ani.id_kind = 'attachment' AND ani.native_id = ?
            LIMIT 1
            """,
            (shared_attachment_id,),
        ).fetchone()
        assert attachment_row is not None
        attachment_id = str(attachment_row["attachment_id"])
        stored_ref_count = conn.execute(
            "SELECT ref_count FROM attachments WHERE attachment_id = ?",
            (attachment_id,),
        ).fetchone()[0]
        actual_refs = conn.execute(
            "SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?",
            (attachment_id,),
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
            session_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=size_bytes,
        )
        assert record.size_bytes == size_bytes
    else:
        with pytest.raises(ValidationError):
            AttachmentRecord(
                attachment_id=_attachment_id("test"),
                session_id=_session_id("conv1"),
                message_id=_message_id("msg1"),
                mime_type="text/plain",
                size_bytes=size_bytes,
            )


@pytest.mark.parametrize("name", ["claude-ai", "claude-code", "Provider123"])
def test_origin_accepts_provider_tokens(name: str) -> None:
    """Representative parser provider tokens should normalize to origins."""
    record = _session_record(
        session_id="test",
        origin=origin_from_provider(Provider.from_string(name)).value,
        native_id="ext1",
        title="Test",
        content_hash="hash123",
    )
    assert record.origin == origin_from_provider(Provider.from_string(name))


# ============================================================================
# CRUD Laws (from test_crud_laws.py)
# ============================================================================


class TestCrudLaws:
    """Property-based CRUD round-trip laws."""

    @given(session_strategy(min_messages=1, max_messages=5))
    @settings(max_examples=30, deadline=None)
    async def test_save_retrieve_roundtrip(self, conv_data: dict[str, object]) -> None:
        """Saving a strategy-generated session and retrieving it preserves identity."""
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

            conv = make_session(
                session_id=conv_id,
                source_name=provider,
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
                    session_id=conv_id,
                    role=raw_role,
                    text=raw_text,
                )
                messages.append(msg)

            await backend.save_session_record(conv)
            if messages:
                await backend.save_messages(messages)

            retrieved = await backend.get_session(conv_id)
            assert retrieved is not None
            assert retrieved.session_id.endswith(f":{conv_id}")
            assert retrieved.origin == origin_from_provider(Provider.from_string(provider))

            retrieved_msgs = await backend.get_messages(conv_id)
            assert len(retrieved_msgs) == len(messages)

            await backend.close()

    @given(session_strategy(min_messages=1, max_messages=3))
    @settings(max_examples=20, deadline=None)
    async def test_save_is_idempotent(self, conv_data: dict[str, object]) -> None:
        """Saving the same session twice yields the same stored data."""
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
            conv = make_session(
                session_id=conv_id,
                source_name=raw_provider,
                title=raw_title,
            )

            # Save twice
            await backend.save_session_record(conv)
            await backend.save_session_record(conv)

            # Should still be exactly one session
            all_convs = await backend.queries.list_sessions(_record_query(limit=100))
            expected_session_id = f"{origin_from_provider(Provider.from_string(raw_provider)).value}:{conv_id}"
            matching = [c for c in all_convs if c.session_id == expected_session_id]
            assert len(matching) == 1

            await backend.close()


# ============================================================================
# Repository Laws (from test_repository_laws.py)
# ============================================================================


@st.composite
def simple_tag_spec(draw: st.DrawFn) -> SimpleTagSpec:
    """Generate a tag assignment spec: session ID + list of tags."""
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
    return {"session_id": f"tag-{conv_suffix}", "tags": tags}


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
        """Adding a tag to a session makes it appear in the tag read surface."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from tests.infra.archive_scenarios import archive_for_scenario_db, native_session_id_for
            from tests.infra.storage_records import SessionBuilder

            db_path = Path(tmp_dir) / "index.db"
            conv_id = spec["session_id"]

            (SessionBuilder(db_path, conv_id).provider("test").title("Tag Test").add_message("m1", text="Hello").save())

            repo = archive_for_scenario_db(db_path)
            try:
                tag = spec["tags"][0]
                await repo.add_tag(native_session_id_for("test", conv_id), tag)
                listed = await repo.list_tags()
                assert tag.strip().lower() in listed
            finally:
                await repo.close()

    @given(simple_tag_spec())
    @settings(max_examples=15, deadline=None)
    async def test_remove_tag_is_idempotent(self, spec: SimpleTagSpec) -> None:
        """Removing a tag that doesn't exist doesn't crash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from tests.infra.archive_scenarios import archive_for_scenario_db, native_session_id_for
            from tests.infra.storage_records import SessionBuilder

            db_path = Path(tmp_dir) / "index.db"
            conv_id = spec["session_id"]

            (
                SessionBuilder(db_path, conv_id)
                .provider("test")
                .title("Remove Tag Test")
                .add_message("m1", text="Hello")
                .save()
            )

            repo = archive_for_scenario_db(db_path)
            try:
                tag = spec["tags"][0]
                await repo.remove_tag(native_session_id_for("test", conv_id), tag)
                listed = await repo.list_tags()
                assert tag.strip().lower() not in listed
            finally:
                await repo.close()


class TestTitleSearchLaws:
    """Property-based tests for title-based search."""

    @given(simple_title_search_spec())
    @settings(max_examples=15, deadline=None)
    async def test_title_search_finds_matching(self, spec: SimpleTitleSearchSpec) -> None:
        """Searching by title substring finds the matching session."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
            from tests.infra.archive_scenarios import native_session_id_for
            from tests.infra.storage_records import SessionBuilder

            db_path = Path(tmp_dir) / "index.db"
            conv_id = "search-conv-1"

            (
                SessionBuilder(db_path, conv_id)
                .provider("test")
                .title(spec["title"])
                .add_message("m1", text="Search test content")
                .save()
            )

            with ArchiveStore.open_existing(db_path.parent) as archive:
                results = archive.list_summaries(title=spec["search_term"], limit=50)
            found_ids = [summary.session_id for summary in results]
            assert native_session_id_for("test", conv_id) in found_ids, (
                f"Expected to find '{conv_id}' when searching title='{spec['title']}' for term='{spec['search_term']}'"
            )

    @given(simple_title_search_spec())
    @settings(max_examples=15, deadline=None)
    async def test_title_search_excludes_non_matching(self, spec: SimpleTitleSearchSpec) -> None:
        """Title search doesn't return sessions with unrelated titles."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
            from tests.infra.archive_scenarios import native_session_id_for
            from tests.infra.storage_records import SessionBuilder

            db_path = Path(tmp_dir) / "index.db"

            (
                SessionBuilder(db_path, "match-conv")
                .provider("test")
                .title(spec["title"])
                .add_message("m1", text="Content")
                .save()
            )

            (
                SessionBuilder(db_path, "nomatch-conv")
                .provider("test")
                .title("Zzqxjk Wvpnrl Tmygbs")
                .add_message("m2", text="Other content")
                .save()
            )

            with ArchiveStore.open_existing(db_path.parent) as archive:
                results = archive.list_summaries(title=spec["search_term"], limit=50)
            found_ids = {summary.session_id for summary in results}
            assert native_session_id_for("test", "match-conv") in found_ids


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
            db_path=tmp_path / "test.db",
            limit=50,
            source="claude-ai",
            since="2024-01-01",
        )
        assert key.query == "test query"
        assert key.limit == 50
        assert key.source == "claude-ai"
        assert key.since == "2024-01-01"
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

        archive = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v2 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v3 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version

        assert archive < v2 < v3


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
    """SessionRepository CRUD operations."""

    async def test_repository_basic_operations(self, workspace_env: dict[str, Path]) -> None:
        """Native get/list returns the seeded session and its messages."""
        from tests.infra.archive_scenarios import archive_for_scenario_db, native_session_id_for
        from tests.infra.storage_records import DbFactory, db_setup

        db_path = db_setup(workspace_env)
        DbFactory(db_path).create_session(
            id="c1", provider="chatgpt", messages=[{"id": "m1", "role": "user", "text": "hello world"}]
        )
        native_id = native_session_id_for("chatgpt", "c1")

        repo = archive_for_scenario_db(db_path)
        try:
            conv = await repo.get_session(native_id)
            assert conv is not None
            assert conv.id == native_id
            messages = conv.messages.to_list()
            assert len(messages) == 1
            assert messages[0].text == "hello world"

            lst = await repo.list_sessions(limit=10)
            assert len(lst) == 1
            assert lst[0].id == native_id
        finally:
            await repo.close()

    async def test_get_eager_includes_attachment_session_id(self, workspace_env: dict[str, Path]) -> None:
        """Seeded attachments persist directly keyed to their session and message.

        The archive store keeps attachments in ``attachments`` and the per-message
        grouping in ``attachment_refs`` (carrying the original provider id and the
        native session/message ids); there is no domain-level ``get_eager`` that
        hydrates attachments onto messages.
        """
        from tests.infra.archive_scenarios import native_session_id_for, open_index_db
        from tests.infra.storage_records import DbFactory, db_setup

        db_path = db_setup(workspace_env)
        DbFactory(db_path).create_session(
            id="c-with-att",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "message with attachment",
                    "attachments": [
                        {"id": "att1", "mime_type": "image/png", "size_bytes": 2048, "path": "/path/to/image.png"}
                    ],
                }
            ],
        )

        session_id = native_session_id_for("test", "c-with-att")
        with open_index_db(db_path) as conn:
            rows = conn.execute(
                """
                SELECT ar.session_id, ani.native_id AS attachment_native_id, a.media_type
                FROM attachment_refs ar
                JOIN attachments a ON a.attachment_id = ar.attachment_id
                JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id AND ani.id_kind = 'attachment'
                WHERE ar.session_id = ?
                """,
                (session_id,),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["session_id"] == session_id
        assert rows[0]["attachment_native_id"] == "att1"
        assert rows[0]["media_type"] == "image/png"

    async def test_get_eager_multiple_attachments(self, workspace_env: dict[str, Path]) -> None:
        """Multiple attachments persist grouped by their owning message."""
        from tests.infra.archive_scenarios import native_session_id_for, open_index_db
        from tests.infra.storage_records import DbFactory, db_setup

        db_path = db_setup(workspace_env)
        DbFactory(db_path).create_session(
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
                    "attachments": [{"id": "att3", "mime_type": "application/pdf"}],
                },
            ],
        )

        session_id = native_session_id_for("test", "c-multi-att")
        with open_index_db(db_path) as conn:
            rows = conn.execute(
                """
                SELECT ar.message_id, ani.native_id AS attachment_native_id
                FROM attachment_refs ar
                JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id AND ani.id_kind = 'attachment'
                WHERE ar.session_id = ?
                """,
                (session_id,),
            ).fetchall()
        by_message: dict[str, set[str]] = {}
        for row in rows:
            by_message.setdefault(str(row["message_id"]), set()).add(str(row["attachment_native_id"]))

        assert by_message[f"{session_id}:m1"] == {"att1", "att2"}
        assert by_message[f"{session_id}:m2"] == {"att3"}

    async def test_get_eager_attachment_metadata_not_stored_in_index(self, workspace_env: dict[str, Path]) -> None:
        """Attachment metadata is not an index-tier escape hatch."""
        from tests.infra.archive_scenarios import native_session_id_for, open_index_db
        from tests.infra.storage_records import DbFactory, db_setup

        db_path = db_setup(workspace_env)
        meta = {"original_name": "photo.png", "source": "upload"}
        DbFactory(db_path).create_session(
            id="c-att-meta",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "with meta",
                    "attachments": [{"id": "att-meta", "mime_type": "image/png", "meta": meta}],
                }
            ],
        )

        session_id = native_session_id_for("test", "c-att-meta")
        with open_index_db(db_path) as conn:
            row = conn.execute(
                """
                SELECT a.display_name, a.media_type, ani.native_id AS attachment_native_id
                FROM attachment_refs ar
                JOIN attachments a ON a.attachment_id = ar.attachment_id
                JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id AND ani.id_kind = 'attachment'
                WHERE ar.session_id = ?
                """,
                (session_id,),
            ).fetchone()
        assert row is not None
        assert row["attachment_native_id"] == "att-meta"
        assert row["media_type"] == "image/png"


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

    def upsert(self, session_id: str, messages: list[MessageRecord]) -> None:
        self.upsert_calls.append((session_id, messages))


class TestRepositoryVectorAsyncBoundary:
    async def test_search_similar_offloads_vector_query(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "vectors.db")
        repo = SessionRepository(backend=backend)
        provider = _VectorSpy()
        monkeypatch.setattr(repo, "_get_message_session_mapping", AsyncMock(return_value={"msg-1": "conv-1"}))
        monkeypatch.setattr(repo, "get_many", AsyncMock(return_value=[_session_model("conv-1")]))

        to_thread_calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []

        async def fake_to_thread(func: Callable[..., object], /, *args: object, **kwargs: object) -> object:
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        result = await repo.search_similar("semantic query", limit=4, vector_provider=provider)

        assert [str(session.id) for session in result] == ["conv-1"]
        assert provider.query_calls == [("semantic query", 12)]
        assert len(to_thread_calls) == 1
        assert getattr(to_thread_calls[0][0], "__self__", None) is provider
        assert getattr(to_thread_calls[0][0], "__name__", "") == "query"

    async def test_embed_session_offloads_vector_upsert(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        messages = [make_message("msg-embed", "conv-embed", text="Message long enough to embed.")]
        backend = SQLiteBackend(db_path=tmp_path / "vectors.db")
        repo = SessionRepository(backend=backend)
        monkeypatch.setattr(repo.queries, "get_messages", AsyncMock(return_value=messages))
        provider = _VectorSpy()

        to_thread_calls: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []

        async def fake_to_thread(func: Callable[..., object], /, *args: object, **kwargs: object) -> object:
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        result = await repo.embed_session("conv-embed", vector_provider=provider)

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
        repo = SessionRepository(backend=backend)
        provider = _VectorSpy()
        monkeypatch.setattr(repo, "_get_message_session_mapping", AsyncMock(return_value={"msg-1": "conv-1"}))

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

    @given(infra_tag_assignment_strategy(min_sessions=2, max_sessions=4))
    @settings(max_examples=10, deadline=None)
    async def test_tag_assignment_roundtrip(self, spec: TagAssignmentSpec) -> None:
        """Tags assigned via strategy-generated specs are retrievable and consistent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from tests.infra.archive_scenarios import archive_for_scenario_db, native_session_id_for

            db_path = Path(tmp_dir) / "index.db"
            seed_session_graph(db_path, spec.sessions)

            repo = archive_for_scenario_db(db_path)
            try:
                for conv, tags in zip(spec.sessions, spec.tag_sequences, strict=True):
                    for tag in tags:
                        await repo.add_tag(native_session_id_for(conv.provider, conv.session_id), tag)

                listed = await repo.list_tags()
                for _conv, tags in zip(spec.sessions, spec.tag_sequences, strict=True):
                    for tag in set(tags):
                        assert tag in listed, f"Tag '{tag}' missing from tag read surface: {listed}"
            finally:
                await repo.close()

    @given(infra_tag_assignment_strategy(min_sessions=2, max_sessions=4))
    @settings(max_examples=10, deadline=None)
    async def test_tag_counts_match_expected(self, spec: TagAssignmentSpec) -> None:
        """Tag counts computed from strategy match actual stored tag counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from tests.infra.archive_scenarios import archive_for_scenario_db, native_session_id_for

            db_path = Path(tmp_dir) / "index.db"
            seed_session_graph(db_path, spec.sessions)

            repo = archive_for_scenario_db(db_path)
            try:
                for conv, tags in zip(spec.sessions, spec.tag_sequences, strict=True):
                    for tag in tags:
                        await repo.add_tag(native_session_id_for(conv.provider, conv.session_id), tag)

                assert await repo.list_tags() == expected_tag_counts(spec)
            finally:
                await repo.close()


class TestInfraTitleSearch:
    """Property-based tests using the full TitleSearchSpec strategy."""

    @given(infra_title_search_strategy())
    @settings(max_examples=15, deadline=None)
    async def test_literal_title_search_finds_matching_with_special_chars(self, spec: TitleSearchSpec) -> None:
        """Title search with wildcard-sensitive characters finds exact matches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
            from tests.infra.archive_scenarios import native_session_id_for
            from tests.infra.storage_records import SessionBuilder

            db_path = Path(tmp_dir) / "index.db"

            (
                SessionBuilder(db_path, "match-conv")
                .provider("test")
                .title(spec.matching_title)
                .add_message("m1", text="Content")
                .save()
            )

            (
                SessionBuilder(db_path, "decoy-conv")
                .provider("test")
                .title(spec.decoy_title)
                .add_message("m2", text="Other")
                .save()
            )

            with ArchiveStore.open_existing(db_path.parent) as archive:
                results = archive.list_summaries(title=spec.needle, limit=50)
            found_ids = {summary.session_id for summary in results}
            assert native_session_id_for("test", "match-conv") in found_ids, (
                f"Expected 'match-conv' for needle='{spec.needle}' in title='{spec.matching_title}'"
            )

    @given(infra_title_search_strategy())
    @settings(max_examples=15, deadline=None)
    async def test_literal_title_search_excludes_decoy(self, spec: TitleSearchSpec) -> None:
        """Title search with special characters does not match the decoy title."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
            from tests.infra.archive_scenarios import native_session_id_for
            from tests.infra.storage_records import SessionBuilder

            db_path = Path(tmp_dir) / "index.db"

            (
                SessionBuilder(db_path, "decoy-only")
                .provider("test")
                .title(spec.decoy_title)
                .add_message("m1", text="Content")
                .save()
            )

            with ArchiveStore.open_existing(db_path.parent) as archive:
                results = archive.list_summaries(title=spec.needle, limit=50)
            found_ids = {summary.session_id for summary in results}
            assert native_session_id_for("test", "decoy-only") not in found_ids, (
                f"Decoy '{spec.decoy_title}' should not match needle='{spec.needle}'"
            )
