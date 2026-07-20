"""Property test: the SQLite generated ``messages.message_id`` column and the
Python-side identity computation that feeds ``blocks.message_id`` FK values
must never diverge for any provider-native id shape.

This is the actual law the v42 index rebuild (operation ab5bad1f) violated:
``_write_messages`` stored a whitespace-only (or surrogate-bearing)
``native_id`` as-is, while ``_message_id`` (used to compute the ``message_id``
that ``_write_blocks`` inserts) independently re-derived identity via
``core.identity_law``, which strips/normalizes differently. The two
computations disagreed, so a ``blocks`` row referenced a ``message_id`` the
``messages`` table never stored -- a FOREIGN KEY constraint failure that took
down a 10-hour rebuild with zero context.

``_stored_message_native_id`` (archive_tiers/write.py) is now the single
source of truth both call sites route through; this sweep is the regression
guard against the two ever splitting again.
"""

from __future__ import annotations

import sqlite3

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import (
    _duplicate_message_native_ids,
    _message_id,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.schema import _ensure_schema

_SURROGATE_CHARS = st.characters(min_codepoint=0xD800, max_codepoint=0xDFFF)
_ASCII_ID_CHARS = st.characters(min_codepoint=0x21, max_codepoint=0x7E)

# Deliberately weighted toward pathological shapes (empty, whitespace-only,
# padded, lone-surrogate, mixed) over a small alphabet so hypothesis finds
# duplicate collisions (including post-normalization collisions) quickly,
# rather than drowning in unique-string noise.
_WEIRD_NATIVE_IDS = st.one_of(
    st.just(""),
    st.just(" "),
    st.just("\t \n"),
    st.just("  msg-padded  "),
    st.text(alphabet=_ASCII_ID_CHARS, min_size=1, max_size=6),
    st.text(alphabet=_SURROGATE_CHARS, min_size=1, max_size=3),
    st.builds(
        lambda prefix, surrogate: prefix + surrogate,
        st.text(alphabet=_ASCII_ID_CHARS, min_size=1, max_size=3),
        st.text(alphabet=_SURROGATE_CHARS, min_size=1, max_size=2),
    ),
)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(_WEIRD_NATIVE_IDS, min_size=1, max_size=6))
def test_db_generated_message_id_matches_python_identity_law(native_ids: list[str]) -> None:
    messages = [
        ParsedMessage(
            provider_message_id=native_id,
            role=Role.USER,
            text=f"body-{i}",
            timestamp="2024-01-01T00:00:00Z",
        )
        for i, native_id in enumerate(native_ids)
    ]
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="identity-law-sweep",
        title="sweep",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=messages,
        attachments=[],
    )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        # Real write path, FK enforcement on (write_parsed_session_to_archive
        # itself turns PRAGMA foreign_keys ON) -- a divergence here is exactly
        # the FK failure that killed operation ab5bad1f.
        session_id = write_parsed_session_to_archive(
            conn,
            session,
            content_hash=session_content_hash(session),
        )
        rows = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (session_id,),
        ).fetchall()
        assert len(rows) == len(messages)

        duplicate_native_ids = _duplicate_message_native_ids(messages)
        for fallback_position, message in enumerate(messages):
            expected = _message_id(
                session_id,
                message,
                fallback_position,
                duplicate_native_ids=duplicate_native_ids,
            )
            actual = rows[fallback_position]["message_id"]
            assert actual == expected, (
                f"DB message_id {actual!r} != Python _message_id {expected!r} "
                f"for provider_message_id={message.provider_message_id!r}"
            )
    finally:
        conn.close()
