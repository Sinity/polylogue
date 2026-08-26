"""Cross-path law for the denormalized session message counts.

The same message multiset must produce the same session counters through the
full-write, merge-append, and recount paths.  The reference fold below is
intentionally independent of the writer helpers so a mutation in one path is
observable.
"""

from __future__ import annotations

import sqlite3

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    _build_message_rows,
    _duplicate_message_native_ids,
    _refresh_session_counts,
    write_parsed_session_to_archive,
)

_COUNT_COLUMNS = (
    "message_count",
    "word_count",
    "tool_use_count",
    "thinking_count",
    "paste_count",
    "user_message_count",
    "authored_user_message_count",
    "assistant_message_count",
    "system_message_count",
    "tool_message_count",
    "user_word_count",
    "authored_user_word_count",
    "assistant_word_count",
)

_ROLES = (Role.USER, Role.ASSISTANT, Role.SYSTEM, Role.TOOL)
_MATERIAL_ORIGINS = tuple(MaterialOrigin)


@st.composite
def message_sets(draw: st.DrawFn) -> list[ParsedMessage]:
    """Generate bounded message sets, including duplicate native IDs."""
    size = draw(st.integers(min_value=0, max_value=7))
    messages: list[ParsedMessage] = []
    for index in range(size):
        provider_message_id = "seed" if index == 0 else draw(st.sampled_from(("duplicate-a", "duplicate-b")))
        role = draw(st.sampled_from(_ROLES))
        material_origin = draw(st.sampled_from(_MATERIAL_ORIGINS))
        text = draw(st.one_of(st.just(""), st.text(min_size=1, max_size=30)))
        block_types = draw(st.sets(st.sampled_from((BlockType.TOOL_USE, BlockType.THINKING)), max_size=2))
        blocks = [ParsedContentBlock(type=block_type, text="block") for block_type in sorted(block_types, key=str)]
        messages.append(
            ParsedMessage(
                provider_message_id=provider_message_id,
                role=role,
                text=text,
                material_origin=material_origin,
                blocks=blocks,
            )
        )
    return messages


def _reference_counts(messages: list[ParsedMessage]) -> dict[str, int]:
    counts = dict.fromkeys(_COUNT_COLUMNS, 0)
    for message in messages:
        text = message.text or ""
        words = len(text.split())
        counts["message_count"] += 1
        counts["word_count"] += words
        counts["tool_use_count"] += int(any(block.type is BlockType.TOOL_USE for block in message.blocks))
        counts["thinking_count"] += int(any(block.type is BlockType.THINKING for block in message.blocks))
        counts["paste_count"] += int(bool(message.paste_spans))
        if message.role is Role.USER:
            counts["user_message_count"] += 1
            counts["user_word_count"] += words
        elif message.role is Role.ASSISTANT:
            counts["assistant_message_count"] += 1
            counts["assistant_word_count"] += words
        elif message.role is Role.SYSTEM:
            counts["system_message_count"] += 1
        elif message.role is Role.TOOL:
            counts["tool_message_count"] += 1
        if message.material_origin is MaterialOrigin.HUMAN_AUTHORED:
            counts["authored_user_message_count"] += 1
            counts["authored_user_word_count"] += words
    return counts


def _session_counts(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    columns = ", ".join(_COUNT_COLUMNS)
    row = conn.execute(f"SELECT {columns} FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    assert row is not None
    return dict(zip(_COUNT_COLUMNS, row, strict=True))


def _new_session(messages: list[ParsedMessage]) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session-count-law",
        messages=messages,
    )


def _new_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=40)
@given(messages=message_sets())
def test_session_counts_agree_across_full_append_and_refresh_paths(messages: list[ParsedMessage]) -> None:
    """Anti-vacuity: mutating any count increment or recount predicate makes this red."""
    expected = _reference_counts(messages)
    connections = [_new_connection() for _ in range(3)]
    try:
        full_conn, append_conn, refresh_conn = connections

        full_id = write_parsed_session_to_archive(full_conn, _new_session(messages))

        split = 1 if messages else 0
        first, tail = messages[:split], messages[split:]
        append_id = write_parsed_session_to_archive(append_conn, _new_session(first))
        if tail:
            write_parsed_session_to_archive(append_conn, _new_session(tail), merge_append=True)

        refresh_id = write_parsed_session_to_archive(refresh_conn, _new_session([]))
        duplicate_ids = _duplicate_message_native_ids(messages)
        refresh_conn.executemany(
            archive_tier_write._messages_insert_sql(),
            _build_message_rows(refresh_id, messages, duplicate_native_ids=duplicate_ids),
        )
        _refresh_session_counts(refresh_conn, refresh_id)

        observed = [
            _session_counts(conn, session_id)
            for conn, session_id in (
                (full_conn, full_id),
                (append_conn, append_id),
                (refresh_conn, refresh_id),
            )
        ]
        assert observed == [expected, expected, expected]
    finally:
        for conn in connections:
            conn.close()
