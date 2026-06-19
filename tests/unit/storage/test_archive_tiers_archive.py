from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedPasteEvidence, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import assertion_id_for_session_tag, read_assertion_envelope
from polylogue.storage.sqlite.archive_tiers.write import read_session_tags


def test_active_archive_root_facade_writes_reads_and_searches_archive_db(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-archive-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="one searchable needle")],
            )
        ],
    )

    with ArchiveStore(tmp_path / "archive") as facade:
        assert (tmp_path / "archive" / "index.db").exists()
        session_id = facade.write_parsed(session)
        envelope = facade.read_session(session_id)
        matching_blocks = facade.search_blocks("needle")

    assert session_id == "codex-session:codex-archive-1"
    assert envelope.session_id == "codex-session:codex-archive-1"
    assert len(envelope.messages) == 1
    assert matching_blocks == ["codex-session:codex-archive-1:m1:0"]


def test_archive_tiers_archive_facade_sorts_search_matches(tmp_path: Path) -> None:
    short = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-search-short",
        updated_at="2026-01-02T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle short")],
            )
        ],
    )
    long = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-search-long",
        updated_at="2026-01-03T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle first")],
            ),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle second")],
            ),
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        short_id = facade.write_parsed(short)
        long_id = facade.write_parsed(long)

    with ArchiveStore.open_existing(root) as facade:
        by_messages = facade.search_summaries("needle", limit=5, sort="messages")
        by_messages_reversed = facade.search_summaries("needle", limit=5, sort="messages", reverse=True)
        short_only = facade.search_summaries("needle", limit=5, session_id=short_id)

    assert [hit.session_id for hit in by_messages[:2]] == [long_id, long_id]
    assert by_messages_reversed[0].session_id == short_id
    assert [hit.session_id for hit in short_only] == [short_id]


def test_archive_tiers_archive_facade_links_raw_and_parsed_rows(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-archive-raw-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="raw linked")],
            )
        ],
    )
    root = tmp_path / "archive"
    source_path = root / "source.db"
    index_path = root / "index.db"

    with ArchiveStore(root) as facade:
        raw_id, session_id = facade.write_raw_and_parsed(
            session,
            payload=b'{"provider":"codex","id":"codex-archive-raw-1"}',
            source_path="/tmp/codex-session.jsonl",
            acquired_at_ms=1_767_000_000_000,
        )

    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    try:
        session = conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        assert session["raw_id"] == raw_id
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
    source_conn = sqlite3.connect(source_path)
    source_conn.row_factory = sqlite3.Row
    try:
        raw = source_conn.execute(
            "SELECT native_id, source_path FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone()
        assert dict(raw) == {
            "native_id": "codex-archive-raw-1",
            "source_path": "/tmp/codex-session.jsonl",
        }
        assert source_conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        source_conn.close()


def test_archive_tiers_archive_facade_adds_user_tags(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-tag-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="taggable")],
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        session_id = facade.write_parsed(session)
        changed = facade.add_user_tags((session_id,), ("Review", "ready"))
        duplicate_changed = facade.add_user_tags((session_id,), ("review",))
        removed = facade.remove_user_tags((session_id,), ("READY",))
        summaries = facade.list_summaries(tags=("review",), limit=5)
        tag_counts = facade.list_user_tags()

    user_conn = sqlite3.connect(root / "user.db")
    user_conn.row_factory = sqlite3.Row
    try:
        tags = read_session_tags(user_conn, session_id=session_id, tag_source="user")
        review_assertion = read_assertion_envelope(
            user_conn,
            assertion_id_for_session_tag(session_id, "review", "user"),
        )
        ready_assertion = read_assertion_envelope(
            user_conn,
            assertion_id_for_session_tag(session_id, "ready", "user"),
        )
    finally:
        user_conn.close()

    assert changed == 2
    assert duplicate_changed == 0
    assert removed == 1
    assert sorted(tags) == ["review"]
    assert tag_counts == {"review": 1}
    assert review_assertion is not None
    assert review_assertion.status == "active"
    assert ready_assertion is not None
    assert ready_assertion.status == "deleted"
    assert [summary.session_id for summary in summaries] == [session_id]


def test_archive_tiers_archive_facade_deletes_archive_sessions_but_keeps_user_tags(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-delete-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="delete me")],
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        session_id = facade.write_parsed(session)
        facade.add_user_tags((session_id,), ("keep-user-state",))
        deleted = facade.delete_sessions((session_id,))
        remaining = facade.count_sessions()

    user_conn = sqlite3.connect(root / "user.db")
    user_conn.row_factory = sqlite3.Row
    try:
        tags = read_session_tags(user_conn, session_id=session_id, tag_source="user")
    finally:
        user_conn.close()

    assert deleted == 1
    assert remaining == 0
    assert sorted(tags) == ["keep-user-state"]


def test_archive_tiers_archive_facade_sets_user_metadata(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-meta-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="metadata target")],
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        session_id = facade.write_parsed(session)
        changed = facade.set_user_metadata((session_id,), (("priority", "high"), ("status", "reviewed")))
        unchanged = facade.set_user_metadata((session_id,), (("priority", "high"),))
        metadata = facade.read_user_metadata(session_id)
        deleted = facade.delete_user_metadata(session_id, "status")

    user_conn = sqlite3.connect(root / "user.db")
    try:
        rows = user_conn.execute(
            "SELECT key, value_json FROM session_metadata WHERE session_id = ? ORDER BY key",
            (session_id,),
        ).fetchall()
    finally:
        user_conn.close()

    assert changed == 2
    assert unchanged == 0
    assert metadata == {"priority": "high", "status": "reviewed"}
    assert deleted == 1
    assert rows == [("priority", '"high"')]


def test_archive_tiers_archive_facade_lists_and_searches_session_summaries(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-read-1",
        title="Read facade",
        created_at="2026-01-02T03:04:05Z",
        updated_at="2026-01-02T03:04:07Z",
        working_directories=["/realm/project/polylogue"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                message_type=MessageType.TOOL_USE,
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="alpha read token"),
                    ParsedContentBlock(type=BlockType.THINKING, text="plan"),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Read",
                        tool_id="tool-1",
                        tool_input={"file_path": "README.md"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id="tool-2",
                        tool_input={"command": "pytest tests/unit/storage/test_archive_tiers_archive.py"},
                    ),
                ],
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-read-1",
        title="Other origin",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                paste_spans=[ParsedPasteEvidence(position=0, start_offset=0, end_offset=4, boundary_state="exact")],
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="beta read token")],
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        first_id = facade.write_parsed(first)
        second_id = facade.write_parsed(second)
        conn = facade._conn
        facade.add_user_tags((first_id,), ("archive",))
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, workflow_shape, workflow_shape_method, workflow_shape_confidence,
                terminal_state, terminal_state_method, terminal_state_confidence, search_text
            ) VALUES (?, 'implementation', 'fixture', 1.0, 'complete', 'fixture', 1.0, '')
            """,
            (first_id,),
        )
        conn.commit()

    with ArchiveStore.open_existing(root) as facade:
        assert facade.count_sessions() == 2
        assert facade.count_sessions(origin="codex-session") == 1
        assert facade.count_sessions(origins=("codex-session", "chatgpt-export")) == 2
        assert facade.count_sessions(excluded_origins=("chatgpt-export",)) == 1
        assert facade.count_sessions(tags=("archive",)) == 1
        assert facade.count_sessions(excluded_tags=("archive",)) == 1
        assert facade.count_sessions(repo_names=("polylogue",)) == 1
        assert facade.count_sessions(has_types=("tool_use",), has_tool_use=True, has_thinking=True) == 1
        assert facade.count_sessions(tool_terms=("read",), excluded_tool_terms=("write",)) == 1
        assert facade.count_sessions(tool_terms=("none",)) == 1
        assert facade.count_sessions(action_terms=("file_read",), excluded_action_terms=("file_write",)) == 1
        assert facade.count_sessions(action_terms=("none",)) == 1
        assert facade.count_sessions(action_sequence=("file_read", "shell")) == 1
        assert facade.count_sessions(action_sequence=("shell", "file_read")) == 0
        assert facade.count_sessions(action_text_terms=("readme",)) == 1
        assert facade.count_sessions(referenced_paths=("README",), cwd_prefix="/realm/project", typed_only=True) == 1
        assert facade.count_sessions(has_paste=True) == 1
        assert facade.count_sessions(message_type="tool_use") == 1
        stats = facade.stats()
        codex_stats = facade.stats(origin="codex-session")
        search_stats = facade.stats(session_ids=(first_id,))
        origin_group_stats = facade.stats_by("origin")
        origin_stats = facade.stats_by("origin")
        day_stats = facade.stats_by("day")
        tag_stats = facade.stats_by("tag")
        repo_stats = facade.stats_by("repo")
        tool_stats = facade.stats_by("tool")
        action_stats = facade.stats_by("action")
        work_kind_stats = facade.stats_by("work-kind")
        summaries = facade.list_summaries(limit=5)
        codex_summaries = facade.list_summaries(limit=5, origin="codex-session")
        multi_origin_summaries = facade.list_summaries(limit=5, origins=("codex-session", "chatgpt-export"))
        excluded_origin_summaries = facade.list_summaries(limit=5, excluded_origins=("chatgpt-export",))
        tagged_summaries = facade.list_summaries(limit=5, tags=("archive",))
        excluded_tag_summaries = facade.list_summaries(limit=5, excluded_tags=("archive",))
        repo_summaries = facade.list_summaries(limit=5, repo_names=("polylogue",))
        shaped_summaries = facade.list_summaries(limit=5, has_types=("thinking",), has_thinking=True)
        tool_summaries = facade.list_summaries(limit=5, tool_terms=("read",))
        action_summaries = facade.list_summaries(limit=5, action_terms=("file_read",))
        action_sequence_summaries = facade.list_summaries(limit=5, action_sequence=("file_read", "shell"))
        action_text_summaries = facade.list_summaries(limit=5, action_text_terms=("readme",))
        path_summaries = facade.list_summaries(limit=5, referenced_paths=("README.md",), cwd_prefix="/realm/project")
        typed_summaries = facade.list_summaries(limit=5, typed_only=True, origin="codex-session")
        message_type_summaries = facade.list_summaries(limit=5, message_type="tool_use")
        titled_summaries = facade.list_summaries(limit=5, title="Read", min_messages=1, max_messages=1, min_words=0)
        dated_summaries = facade.list_summaries(limit=5, since_ms=1767323046000, until_ms=1767323048000)
        sampled_summaries = facade.list_summaries(limit=1, offset=1, sample=True)
        hits = facade.search_summaries("alpha", limit=5)
        offset_hits = facade.search_summaries("read", limit=1, offset=1)
        semantic_hits = facade.semantic_summaries([("codex-session:codex-read-1:m1", 0.2)], limit=5)
        tagged_hits = facade.search_summaries("alpha", limit=5, tags=("archive",))
        excluded_origin_hits = facade.search_summaries("alpha", limit=5, excluded_origins=("chatgpt-export",))
        multi_origin_hits = facade.search_summaries("alpha", limit=5, origins=("codex-session", "chatgpt-export"))
        excluded_origin_miss_hits = facade.search_summaries("alpha", limit=5, excluded_origins=("codex-session",))
        excluded_tag_hits = facade.search_summaries("alpha", limit=5, excluded_tags=("archive",))
        shaped_hits = facade.search_summaries("alpha", limit=5, has_types=("tool_use",), has_tool_use=True)
        shape_miss_hits = facade.search_summaries("alpha", limit=5, has_paste=True)
        tool_hits = facade.search_summaries("alpha", limit=5, tool_terms=("read",), excluded_tool_terms=("write",))
        tool_miss_hits = facade.search_summaries("alpha", limit=5, excluded_tool_terms=("read",))
        action_hits = facade.search_summaries(
            "alpha",
            limit=5,
            action_terms=("file_read",),
            excluded_action_terms=("file_write",),
        )
        action_miss_hits = facade.search_summaries("alpha", limit=5, excluded_action_terms=("file_read",))
        action_sequence_hits = facade.search_summaries("alpha", limit=5, action_sequence=("file_read", "shell"))
        action_sequence_miss_hits = facade.search_summaries("alpha", limit=5, action_sequence=("shell", "file_read"))
        action_text_hits = facade.search_summaries("alpha", limit=5, action_text_terms=("readme",))
        action_text_miss_hits = facade.search_summaries("alpha", limit=5, action_text_terms=("missing",))
        path_hits = facade.search_summaries(
            "alpha",
            limit=5,
            referenced_paths=("README.md",),
            cwd_prefix="/realm/project",
            typed_only=True,
        )
        path_miss_hits = facade.search_summaries("alpha", limit=5, referenced_paths=("missing.py",))
        message_type_hits = facade.search_summaries("alpha", limit=5, message_type="tool_use")
        message_type_miss_hits = facade.search_summaries("alpha", limit=5, message_type="summary")
        repo_hits = facade.search_summaries("alpha", limit=5, repo_names=("polylogue",))
        repo_miss_hits = facade.search_summaries("alpha", limit=5, repo_names=("other",))
        tag_miss_hits = facade.search_summaries("alpha", limit=5, tags=("other",))
        titled_hits = facade.search_summaries("alpha", limit=5, title="Read", min_messages=1)
        dated_hits = facade.search_summaries("alpha", limit=5, since_ms=1767323046000)

    assert {summary.session_id for summary in summaries} == {first_id, second_id}
    assert codex_summaries[0].session_id == first_id
    assert {summary.session_id for summary in multi_origin_summaries} == {first_id, second_id}
    assert [summary.session_id for summary in excluded_origin_summaries] == [first_id]
    assert [summary.session_id for summary in tagged_summaries] == [first_id]
    assert [summary.session_id for summary in excluded_tag_summaries] == [second_id]
    assert [summary.session_id for summary in repo_summaries] == [first_id]
    assert [summary.session_id for summary in shaped_summaries] == [first_id]
    assert [summary.session_id for summary in tool_summaries] == [first_id]
    assert [summary.session_id for summary in action_summaries] == [first_id]
    assert [summary.session_id for summary in action_sequence_summaries] == [first_id]
    assert [summary.session_id for summary in action_text_summaries] == [first_id]
    assert [summary.session_id for summary in path_summaries] == [first_id]
    assert [summary.session_id for summary in typed_summaries] == [first_id]
    assert [summary.session_id for summary in message_type_summaries] == [first_id]
    assert [summary.session_id for summary in titled_summaries] == [first_id]
    assert [summary.session_id for summary in dated_summaries] == [first_id]
    assert len(sampled_summaries) == 1
    assert sampled_summaries[0].session_id in {first_id, second_id}
    assert codex_summaries[0].provider is Provider.CODEX
    assert codex_summaries[0].created_at == "2026-01-02T03:04:05Z"
    assert codex_summaries[0].updated_at == "2026-01-02T03:04:07Z"
    assert codex_summaries[0].message_count == 1
    assert codex_summaries[0].tags == ("archive",)
    assert stats.total_sessions == 2
    assert stats.total_messages == 2
    assert stats.origins == {"codex-session": 1, "chatgpt-export": 1}
    assert codex_stats.origins == {"codex-session": 1}
    assert search_stats.total_sessions == 1
    assert origin_group_stats == {"chatgpt-export": 1, "codex-session": 1}
    assert origin_stats == {"chatgpt-export": 1, "codex-session": 1}
    assert day_stats == {"2026-01-02": 1}
    assert tag_stats == {"archive": 1}
    assert repo_stats == {"polylogue": 1}
    assert tool_stats == {"bash": 1, "read": 1}
    assert action_stats == {"file_read": 1, "shell": 1}
    assert work_kind_stats == {"implementation": 1}
    assert hits[0].session_id == first_id
    assert [hit.rank for hit in offset_hits] == [2]
    assert semantic_hits[0].session_id == first_id
    assert semantic_hits[0].message_id == "codex-session:codex-read-1:m1"
    assert tagged_hits[0].session_id == first_id
    assert excluded_origin_hits[0].session_id == first_id
    assert multi_origin_hits[0].session_id == first_id
    assert excluded_origin_miss_hits == []
    assert excluded_tag_hits == []
    assert shaped_hits[0].session_id == first_id
    assert shape_miss_hits == []
    assert tool_hits[0].session_id == first_id
    assert tool_miss_hits == []
    assert action_hits[0].session_id == first_id
    assert action_miss_hits == []
    assert action_sequence_hits[0].session_id == first_id
    assert action_sequence_miss_hits == []
    assert action_text_hits[0].session_id == first_id
    assert action_text_miss_hits == []
    assert path_hits[0].session_id == first_id
    assert path_miss_hits == []
    assert message_type_hits[0].session_id == first_id
    assert message_type_miss_hits == []
    assert repo_hits[0].session_id == first_id
    assert tag_miss_hits == []
    assert repo_miss_hits == []
    assert titled_hits[0].session_id == first_id
    assert dated_hits[0].session_id == first_id
    assert hits[0].block_id == "codex-session:codex-read-1:m1:0"
    assert hits[0].provider is Provider.CODEX
    assert "[alpha]" in hits[0].snippet


def test_archive_tiers_archive_facade_resolves_exact_and_prefix_session_ids(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-resolve-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="resolve me")],
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        session_id = facade.write_parsed(session)

    with ArchiveStore.open_existing(root) as facade:
        assert facade.resolve_session_id(session_id) == session_id
        assert facade.resolve_session_id("codex-session:codex-resolve") == session_id


def test_archive_tiers_archive_facade_filters_since_session_scope(tmp_path: Path) -> None:
    anchor = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="anchor",
        title="Anchor",
        updated_at="2026-01-02T10:00:00Z",
        working_directories=["/realm/project/polylogue"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="anchor token")],
            )
        ],
    )
    later_same_scope = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="later-same",
        title="Later same",
        updated_at="2026-01-02T10:05:00Z",
        working_directories=["/realm/project/polylogue/subdir"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle same scope")],
            )
        ],
    )
    later_other_scope = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="later-other",
        title="Later other",
        updated_at="2026-01-02T10:06:00Z",
        working_directories=["/realm/project/other"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle other scope")],
            )
        ],
    )
    earlier_same_scope = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="earlier-same",
        title="Earlier same",
        updated_at="2026-01-02T09:59:00Z",
        working_directories=["/realm/project/polylogue"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle earlier scope")],
            )
        ],
    )

    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        anchor_id = facade.write_parsed(anchor)
        later_same_id = facade.write_parsed(later_same_scope)
        facade.write_parsed(later_other_scope)
        facade.write_parsed(earlier_same_scope)

    with ArchiveStore.open_existing(root) as facade:
        assert facade.count_sessions(since_session_id=anchor_id) == 1
        summaries = facade.list_summaries(limit=10, since_session_id=anchor_id)
        hits = facade.search_summaries("needle", limit=10, since_session_id=anchor_id)
        stats = facade.stats(since_session_id=anchor_id)
        grouped = facade.stats_by("origin", since_session_id=anchor_id)
        missing = facade.list_summaries(limit=10, since_session_id="missing")

    assert [summary.session_id for summary in summaries] == [later_same_id]
    assert [hit.session_id for hit in hits] == [later_same_id]
    assert stats.total_sessions == 1
    assert grouped == {"codex-session": 1}
    assert missing == []
