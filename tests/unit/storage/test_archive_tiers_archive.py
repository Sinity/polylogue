from __future__ import annotations

import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest

from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_INGEST_FLAG
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedPasteEvidence,
    ParsedSession,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import (
    assertion_id_for_session_metadata,
    assertion_id_for_session_tag,
    read_assertion_envelope,
)


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


def test_open_existing_read_timeout_updates_busy_timeout(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        assert facade.index_db_path.exists()

    with ArchiveStore.open_existing(root, read_timeout=0.25) as facade:
        busy_timeout_ms = facade._conn.execute("PRAGMA busy_timeout").fetchone()[0]

    assert busy_timeout_ms == 250


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


def test_archive_tiers_archive_facade_queries_session_actions_by_session_index(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-session-actions",
        title="Session actions",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                timestamp="2026-06-30T08:00:00Z",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id="tool-1",
                        tool_input={"command": "pytest tests/unit/storage/test_archive_tiers_archive.py"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id="tool-1",
                        text="failed",
                        is_error=True,
                        exit_code=2,
                    ),
                ],
            )
        ],
    )
    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        session_id = facade.write_parsed(session)

    with ArchiveStore.open_existing(root) as facade:
        rows = facade.query_session_actions([session_id], limit=10)
        failed_source = parse_unit_source_expression("actions where is_error:true AND exit_code > 0")
        assert failed_source is not None
        failed_rows = facade.query_actions(failed_source.predicate, limit=10)
        error_counts = facade.query_unit_counts("action", failed_source.predicate, group_by="is_error")
        exit_counts = facade.query_unit_counts("action", failed_source.predicate, group_by="exit_code")

    assert [
        (row.session_id, row.tool_name, row.tool_command, row.output_text, row.is_error, row.exit_code) for row in rows
    ] == [
        (
            session_id,
            "Bash",
            "pytest tests/unit/storage/test_archive_tiers_archive.py",
            "failed",
            1,
            2,
        )
    ]
    assert [(row.session_id, row.is_error, row.exit_code) for row in failed_rows] == [(session_id, 1, 2)]
    assert [(row.group_key, row.count) for row in error_counts] == [("1", 1)]
    assert [(row.group_key, row.count) for row in exit_counts] == [("2", 1)]


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


def test_archive_tiers_archive_facade_skips_lower_precedence_dom_fallback(tmp_path: Path) -> None:
    native = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-capture-gap",
        messages=[
            ParsedMessage(provider_message_id="m1", role=Role.USER, text="native user"),
            ParsedMessage(provider_message_id="m2", role=Role.ASSISTANT, text="native assistant"),
        ],
    )
    dom_fallback = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-capture-gap",
        messages=[ParsedMessage(provider_message_id="dom-1", role=Role.USER, text="dom fallback")],
        ingest_flags=[DOM_FALLBACK_INGEST_FLAG],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        first = facade.write_raw_and_parsed_result(
            native,
            payload=b'{"native": true}',
            source_path="/tmp/native.json",
            acquired_at_ms=1_767_000_000_000,
        )
        second = facade.write_raw_and_parsed_result(
            dom_fallback,
            payload=b'{"dom": true}',
            source_path="/tmp/dom.json",
            acquired_at_ms=1_767_000_000_001,
        )

    conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        stored = conn.execute(
            "SELECT raw_id, message_count FROM sessions WHERE session_id = ?",
            (first.session_id,),
        ).fetchone()
        event = conn.execute(
            "SELECT event_type, summary FROM session_events WHERE session_id = ?",
            (first.session_id,),
        ).fetchone()
    finally:
        conn.close()

    assert first.counts["sessions"] == 1
    assert second.content_changed is False
    assert second.counts["sessions"] == 0
    assert second.counts["session_events"] == 1
    assert second.counts["skipped_sessions"] == 1
    assert second.counts["skipped_messages"] == 1
    assert stored["raw_id"] == first.raw_id
    assert stored["message_count"] == 2
    assert event["event_type"] == "capture_gap"
    assert "DOM browser-capture fallback" in event["summary"]


def test_archive_tiers_archive_facade_replaces_dom_fallback_with_native(tmp_path: Path) -> None:
    dom_fallback = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-capture-gap-reverse",
        updated_at="2026-07-04T09:55:00Z",
        messages=[ParsedMessage(provider_message_id="dom-1", role=Role.USER, text="dom fallback")],
        ingest_flags=[DOM_FALLBACK_INGEST_FLAG],
    )
    native = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="chatgpt-capture-gap-reverse",
        updated_at="2026-07-04T09:00:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role=Role.USER, text="native user"),
            ParsedMessage(provider_message_id="m2", role=Role.ASSISTANT, text="native assistant"),
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        first = facade.write_raw_and_parsed_result(
            dom_fallback,
            payload=b'{"dom": true}',
            source_path="/tmp/dom.json",
            acquired_at_ms=1_767_000_000_000,
        )
        second = facade.write_raw_and_parsed_result(
            native,
            payload=b'{"native": true}',
            source_path="/tmp/native.json",
            acquired_at_ms=1_767_000_000_001,
        )

    conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        stored = conn.execute(
            "SELECT raw_id, message_count FROM sessions WHERE session_id = ?",
            (first.session_id,),
        ).fetchone()
        event = conn.execute(
            "SELECT event_type, summary FROM session_events WHERE session_id = ?",
            (first.session_id,),
        ).fetchone()
    finally:
        conn.close()

    assert second.content_changed is True
    assert second.counts["sessions"] == 1
    assert second.counts["session_events"] == 1
    assert stored["raw_id"] == second.raw_id
    assert stored["message_count"] == 2
    assert event["event_type"] == "capture_gap"
    assert "DOM browser-capture fallback" in event["summary"]


@pytest.mark.parametrize(
    (
        "initial_kind",
        "initial_count",
        "incoming_kind",
        "incoming_count",
        "expected_title",
        "expected_count",
        "expected_owner",
        "incoming_is_older",
    ),
    [
        ("native", 2, "export", 2, "Native browser", 2, "initial", False),
        ("export", 2, "native", 2, "Native browser", 2, "incoming", False),
        ("native", 2, "export", 1, "Native browser", 2, "initial", False),
        ("export", 1, "native", 2, "Native browser", 2, "incoming", False),
        ("native", 2, "export", 3, "Fuller export", 3, "incoming", False),
        ("export", 3, "native", 2, "Fuller export", 3, "initial", False),
        ("export", 2, "native", 2, "Native browser", 2, "incoming", True),
    ],
    ids=(
        "native-before-equal",
        "native-after-equal",
        "native-before-weaker",
        "native-after-weaker",
        "fuller-export-advances",
        "fuller-export-resists-shorter-native",
        "older-native-after-equal",
    ),
)
def test_archive_tiers_archive_facade_native_browser_precedence_matrix(
    tmp_path: Path,
    initial_kind: str,
    initial_count: int,
    incoming_kind: str,
    incoming_count: int,
    expected_title: str,
    expected_count: int,
    expected_owner: str,
    incoming_is_older: bool,
) -> None:
    session_native_id = f"browser-precedence-{initial_kind}-{initial_count}-{incoming_kind}-{incoming_count}"

    def session(kind: str, message_count: int, *, updated_at: str) -> ParsedSession:
        native = kind == "native"
        title = "Native browser" if native else ("Fuller export" if message_count == 3 else "Ordinary export")
        return ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id=session_native_id,
            title=title,
            updated_at=updated_at,
            messages=[
                ParsedMessage(
                    provider_message_id=f"{kind}-{position}",
                    role=Role.USER if position == 0 else Role.ASSISTANT,
                    text=f"{kind} message {position}",
                )
                for position in range(message_count)
            ],
            ingest_flags=[NATIVE_BROWSER_CAPTURE_INGEST_FLAG] if native else [],
        )

    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        initial = facade.write_raw_and_parsed_result(
            session(initial_kind, initial_count, updated_at="2026-04-03T00:00:00Z"),
            payload=f"{initial_kind}-initial".encode(),
            source_path=f"/tmp/{initial_kind}-initial.json",
            acquired_at_ms=1_767_000_000_000,
        )
        incoming = facade.write_raw_and_parsed_result(
            session(
                incoming_kind,
                incoming_count,
                updated_at="2026-04-02T00:00:00Z" if incoming_is_older else "2026-04-03T00:00:00Z",
            ),
            payload=f"{incoming_kind}-incoming-{incoming_count}".encode(),
            source_path=f"/tmp/{incoming_kind}-incoming.json",
            acquired_at_ms=1_767_000_000_001,
        )

    conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        stored = conn.execute(
            "SELECT raw_id, title, message_count FROM sessions WHERE session_id = ?",
            (initial.session_id,),
        ).fetchone()
    finally:
        conn.close()

    expected_raw_id = initial.raw_id if expected_owner == "initial" else incoming.raw_id
    assert incoming.content_changed is (expected_owner == "incoming")
    assert incoming.counts["skipped_sessions"] == int(expected_owner == "initial")
    assert dict(stored) == {
        "raw_id": expected_raw_id,
        "title": expected_title,
        "message_count": expected_count,
    }


@pytest.mark.parametrize(
    ("arrivals", "expected_title", "expected_native_flag", "expected_capture_gap_count"),
    [
        (
            (
                ("native", 2, "Native browser", "2026-04-01T00:00:00Z"),
                ("export", 3, "Fuller export", "2026-04-02T00:00:00Z"),
                ("export", 3, "Newest export", "2026-04-03T00:00:00Z"),
            ),
            "Newest export",
            False,
            0,
        ),
        (
            (
                ("export", 2, "Newer export", "2026-04-03T00:00:00Z"),
                ("native", 2, "Older native", "2026-04-01T00:00:00Z"),
                ("native", 2, "Native update", "2026-04-02T00:00:00Z"),
            ),
            "Native update",
            True,
            0,
        ),
        (
            (
                ("dom", 1, "DOM fallback", "2026-04-03T00:00:00Z"),
                ("native", 2, "Older native", "2026-04-01T00:00:00Z"),
                ("export", 3, "Fuller export", "2026-04-02T00:00:00Z"),
            ),
            "Fuller export",
            False,
            1,
        ),
    ],
    ids=("owner-flag-is-replaced", "forced-owner-resets-freshness", "capture-gap-survives-later-owner"),
)
def test_archive_tiers_archive_facade_tracks_three_browser_arrivals(
    tmp_path: Path,
    arrivals: tuple[tuple[str, int, str, str], ...],
    expected_title: str,
    expected_native_flag: bool,
    expected_capture_gap_count: int,
) -> None:
    session_native_id = f"browser-three-arrivals-{expected_title.lower().replace(' ', '-')}"

    def session(kind: str, count: int, title: str, updated_at: str) -> ParsedSession:
        ingest_flags = {
            "dom": [DOM_FALLBACK_INGEST_FLAG],
            "native": [NATIVE_BROWSER_CAPTURE_INGEST_FLAG],
        }.get(kind, [])
        return ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id=session_native_id,
            title=title,
            updated_at=updated_at,
            messages=[
                ParsedMessage(
                    provider_message_id=f"{title}-{position}",
                    role=Role.USER if position == 0 else Role.ASSISTANT,
                    text=f"{title} message {position}",
                )
                for position in range(count)
            ],
            ingest_flags=ingest_flags,
        )

    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        outcomes = [
            facade.write_raw_and_parsed_result(
                session(kind, count, title, updated_at),
                payload=f"arrival-{index}-{title}".encode(),
                source_path=f"/tmp/arrival-{index}.json",
                acquired_at_ms=1_767_000_000_000 + index,
            )
            for index, (kind, count, title, updated_at) in enumerate(arrivals)
        ]

    conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        stored = conn.execute(
            "SELECT raw_id, title FROM sessions WHERE session_id = ?",
            (outcomes[0].session_id,),
        ).fetchone()
        native_flag = conn.execute(
            "SELECT 1 FROM session_tags WHERE session_id = ? AND tag = ?",
            (outcomes[0].session_id, NATIVE_BROWSER_CAPTURE_INGEST_FLAG),
        ).fetchone()
        capture_gap_count = conn.execute(
            "SELECT COUNT(*) FROM session_events WHERE session_id = ? AND event_type = 'capture_gap'",
            (outcomes[0].session_id,),
        ).fetchone()[0]
    finally:
        conn.close()

    assert [outcome.content_changed for outcome in outcomes] == [True, True, True]
    assert [outcome.counts["sessions"] for outcome in outcomes] == [1, 1, 1]
    assert dict(stored) == {"raw_id": outcomes[-1].raw_id, "title": expected_title}
    assert (native_flag is not None) is expected_native_flag
    assert capture_gap_count == expected_capture_gap_count


def test_archive_tiers_archive_facade_hash_skips_identical_content_and_refreshes_raw_link(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="content-identical-raw-refresh",
        title="Stable content",
        updated_at="2026-04-03T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="message-1",
                role=Role.USER,
                text="same parsed content",
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        first = facade.write_raw_and_parsed_result(
            session,
            payload=b'[{"capture":"first"}]',
            source_path="/tmp/first.json",
            acquired_at_ms=1_767_000_000_000,
        )
        second = facade.write_raw_and_parsed_result(
            session,
            payload=b'[{"capture":"second"}]',
            source_path="/tmp/second.json",
            acquired_at_ms=1_767_000_000_001,
        )

    conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    try:
        stored_raw_id = conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            (first.session_id,),
        ).fetchone()[0]
    finally:
        conn.close()

    assert first.content_changed is True
    assert second.content_changed is False
    assert second.counts["sessions"] == 0
    assert second.counts["skipped_sessions"] == 1
    assert second.counts["raw_links"] == 1
    assert stored_raw_id == second.raw_id


def test_archive_tiers_archive_facade_replaces_same_size_changed_attachment_bytes(tmp_path: Path) -> None:
    def session(payload: bytes) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="changed-inline-attachment",
            title="Stable content",
            updated_at="2026-04-03T00:00:00Z",
            messages=[
                ParsedMessage(
                    provider_message_id="message-1",
                    role=Role.USER,
                    text="same parsed message",
                )
            ],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="attachment-1",
                    message_provider_id="message-1",
                    name="payload.bin",
                    mime_type="application/octet-stream",
                    size_bytes=len(payload),
                    inline_bytes=payload,
                )
            ],
        )

    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        first = facade.write_raw_and_parsed_result(
            session(b"one"),
            payload=b"first raw",
            source_path="/tmp/first.json",
            acquired_at_ms=1_767_000_000_000,
        )
        second = facade.write_raw_and_parsed_result(
            session(b"two"),
            payload=b"second raw",
            source_path="/tmp/second.json",
            acquired_at_ms=1_767_000_000_001,
        )

    conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    try:
        stored_raw_id, stored_blob_hash = conn.execute(
            "SELECT s.raw_id, a.blob_hash FROM sessions AS s CROSS JOIN attachments AS a WHERE s.session_id = ?",
            (first.session_id,),
        ).fetchone()
    finally:
        conn.close()

    assert second.content_changed is True
    assert second.counts["attachments"] == 1
    assert stored_raw_id == second.raw_id
    assert bytes(stored_blob_hash) == sha256(b"two").digest()


def test_archive_tiers_archive_facade_acquires_empty_inline_attachment(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="empty-inline-attachment",
        messages=[
            ParsedMessage(
                provider_message_id="message-1",
                role=Role.USER,
                text="empty attachment",
            )
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="attachment-1",
                message_provider_id="message-1",
                size_bytes=0,
                inline_bytes=b"",
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as archive:
        archive.write_raw_and_parsed_result(
            session,
            payload=b"raw",
            source_path="/tmp/empty.json",
            acquired_at_ms=1_767_000_000_000,
        )

    with sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True) as conn:
        blob_hash, byte_count, acquisition_status = conn.execute(
            "SELECT blob_hash, byte_count, acquisition_status FROM attachments"
        ).fetchone()
    expected_hash = sha256(b"").digest()
    assert bytes(blob_hash) == expected_hash
    assert byte_count == 0
    assert acquisition_status == "acquired"
    assert (root / "blob" / expected_hash.hex()[:2] / expected_hash.hex()[2:]).read_bytes() == b""


def test_archive_tiers_archive_facade_repairs_missing_fts_on_identical_repeat(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="identical-repeat-fts-repair",
        title="Stable content",
        updated_at="2026-04-03T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="message-1",
                role=Role.USER,
                text="searchable needle",
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        first = facade.write_raw_and_parsed_result(
            session,
            payload=b"stable raw",
            source_path="/tmp/stable.json",
            acquired_at_ms=1_767_000_000_000,
        )

    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 1
        conn.execute("DELETE FROM messages_fts")
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0

    with ArchiveStore(root) as facade:
        repeated = facade.write_raw_and_parsed_result(
            session,
            payload=b"stable raw",
            source_path="/tmp/stable.json",
            acquired_at_ms=1_767_000_000_001,
        )

    with sqlite3.connect(root / "index.db") as conn:
        repaired_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]

    assert repeated.session_id == first.session_id
    assert repeated.content_changed is False
    assert repeated.counts["skipped_sessions"] == 1
    assert repeated.counts["_fts_repair"] == 1
    assert repaired_count == 1


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
        changed = facade.add_user_tags(
            (session_id,),
            ("Review", "ready"),
            author_ref="agent:codex-session:tagger",
            author_kind="agent",
        )
        duplicate_changed = facade.add_user_tags((session_id,), ("review",))
        removed = facade.remove_user_tags((session_id,), ("READY",))
        summaries = facade.list_summaries(tags=("review",), limit=5)
        tag_counts = facade.list_user_tags()

    user_conn = sqlite3.connect(root / "user.db")
    user_conn.row_factory = sqlite3.Row
    user_conn.row_factory = sqlite3.Row
    try:
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
    assert tag_counts == {"review": 1}
    assert review_assertion is not None
    assert review_assertion.status == "active"
    assert review_assertion.author_ref == "agent:codex-session:tagger"
    assert review_assertion.author_kind == "agent"
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
        assertion = read_assertion_envelope(
            user_conn,
            assertion_id_for_session_tag(session_id, "keep-user-state", "user"),
        )
    finally:
        user_conn.close()

    assert deleted == 1
    assert remaining == 0
    assert assertion is not None
    assert assertion.status == "active"


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
        priority_assertion = read_assertion_envelope(
            user_conn,
            assertion_id_for_session_metadata(session_id, "priority"),
        )
        status_assertion = read_assertion_envelope(
            user_conn,
            assertion_id_for_session_metadata(session_id, "status"),
        )
    finally:
        user_conn.close()

    assert changed == 2
    assert unchanged == 0
    assert metadata == {"priority": "high", "status": "reviewed"}
    assert deleted == 1
    assert priority_assertion is not None
    assert priority_assertion.kind == "metadata"
    assert priority_assertion.value == "high"
    assert priority_assertion.status == "active"
    assert status_assertion is not None
    assert status_assertion.status == "deleted"


def test_archive_tiers_archive_facade_preserves_metadata_json_type_changes(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-meta-type-change",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="metadata type target")],
            )
        ],
    )
    root = tmp_path / "archive"

    with ArchiveStore(root) as facade:
        session_id = facade.write_parsed(session)
        assert facade.set_user_metadata((session_id,), (("ambiguous", True),)) == 1
        assert facade.set_user_metadata((session_id,), (("ambiguous", 1),)) == 1
        assert facade.read_user_metadata(session_id) == {"ambiguous": 1}
        assert facade.set_user_metadata((session_id,), (("ambiguous", 1.0),)) == 1
        assert facade.read_user_metadata(session_id) == {"ambiguous": 1.0}


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
        # Suffix fallback: a bare native id (no origin prefix) — e.g. the UUID
        # that appears as a session's source filename — resolves uniquely.
        assert facade.resolve_session_id("codex-resolve-1") == session_id
        # Regression for polylogue-7q16: a *prefix* of the bare native id
        # (e.g. what `find` listings display as the short id) must also
        # resolve, not just the byte-for-byte full native id.
        assert facade.resolve_session_id("codex-resolve") == session_id
        with pytest.raises(KeyError):
            facade.resolve_session_id("codex-resolve-1-nonexistent-suffix")


def test_archive_tiers_archive_facade_exact_bare_native_id_not_shadowed_by_prefix_sibling(
    tmp_path: Path,
) -> None:
    """Regression for the #7q16 fix's own review (#2626).

    In an archive containing native ids ``dup`` and ``dup-extra`` (one a
    prefix of the other), resolving the *exact* bare native id ``dup`` must
    return only that session, not raise ambiguous just because the widened
    prefix pattern that supports truncated-prefix lookups also matches the
    sibling.
    """
    short = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="dup",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="short")],
            )
        ],
    )
    long = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="dup-extra",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="long")],
            )
        ],
    )
    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        short_id = facade.write_parsed(short)
        long_id = facade.write_parsed(long)

    with ArchiveStore.open_existing(root) as facade:
        assert facade.resolve_session_id("dup") == short_id
        assert facade.resolve_session_id("dup-extra") == long_id
        # A genuine shared prefix (of BOTH native ids, unlike "dup" itself,
        # which is the exact/full id of one and merely a prefix of the
        # other) is still ambiguous.
        with pytest.raises(ValueError, match="ambiguous"):
            facade.resolve_session_id("du")


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


def test_archive_coverage_averages_render_none_not_zero_over_empty_denominator(tmp_path: Path) -> None:
    """9e5.29: an average/percentage with zero backing rows is not-applicable, never a lying 0.0."""

    all_assistant_session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="all-assistant",
        created_at="2026-06-30T08:00:00Z",
        updated_at="2026-06-30T08:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                timestamp="2026-06-30T08:00:00Z",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="assistant only, no user words at all")],
            )
        ],
    )

    root = tmp_path / "archive"
    with ArchiveStore(root) as facade:
        facade.write_parsed(all_assistant_session)

    with ArchiveStore.open_existing(root) as facade:
        origin_rows = facade.list_archive_coverage_insights(group_by="origin")
        day_rows = facade.list_archive_coverage_insights(group_by="day")

    assert len(origin_rows) == 1
    origin_row = origin_rows[0]
    # session_count > 0 (a real group), so this average IS computable.
    assert origin_row.session_count == 1
    assert origin_row.avg_messages_per_session == 1.0
    # user_message_count == 0 for an all-assistant session -- the average
    # over zero user messages must be None, never a fabricated 0.0.
    assert origin_row.avg_user_words is None
    assert origin_row.avg_authored_user_words is None
    # assistant_message_count == 1 (nonzero denominator), so this field is a
    # real computed value -- possibly 0.0 -- never None.
    assert origin_row.avg_assistant_words is not None

    assert len(day_rows) == 1
    day_row = day_rows[0]
    # day/week grouping computes avg_messages_per_session from already-fetched
    # message/session counts, but does not fetch per-role message counts --
    # those fields must render None (uncovered), never a fabricated 0.0.
    assert day_row.avg_messages_per_session == 1.0
    assert day_row.avg_user_words is None
    assert day_row.avg_authored_user_words is None
    assert day_row.avg_assistant_words is None
    assert day_row.tool_use_percentage is None
    assert day_row.thinking_percentage is None
