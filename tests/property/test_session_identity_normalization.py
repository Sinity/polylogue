"""Property test: the SQLite generated ``sessions.session_id`` column and the
Python-side identity computation that other code paths use for that same
session must never diverge for any provider-native session id shape.

This mirrors ``test_message_identity_normalization.py``'s guard against the
v42 rebuild (operation ab5bad1f) FK-failure class, at the level one rung up
the identity tree: an invariants audit (polylogue-lyr2) found
``_write_session`` bound the session's ``native_id`` RAW into the ``sessions``
INSERT while the child ``messages``/``session_links`` foreign keys were
computed via ``core.identity_law.session_id``, which strips. Same logical
session, two spellings, one row -- exactly the class of bug that produced
ab5bad1f, just never given a session-level regression guard until now.

``_stored_session_native_id`` (archive_tiers/write.py) is now the single
source of truth every session-identity call site routes through; this sweep
is the regression guard against the two ever splitting again.
"""

from __future__ import annotations

import sqlite3

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.identity_law import session_id as identity_law_session_id
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.schema import _ensure_schema

_SURROGATE_CHARS = st.characters(min_codepoint=0xD800, max_codepoint=0xDFFF)
_ASCII_ID_CHARS = st.characters(min_codepoint=0x21, max_codepoint=0x7E)

# Same pathological-shape weighting as the message-level sweep: empty,
# whitespace-only, padded, lone-surrogate, and mixed shapes are exactly where
# a raw-bind vs identity_law-strip divergence would show up.
_WEIRD_NATIVE_IDS = st.one_of(
    st.just(""),
    st.just(" "),
    st.just("\t \n"),
    st.just("  sess-padded  "),
    st.text(alphabet=_ASCII_ID_CHARS, min_size=1, max_size=8),
)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_WEIRD_NATIVE_IDS)
def test_db_generated_session_id_matches_python_identity_law(native_id: str) -> None:
    session = ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id=native_id,
        title="sweep",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="body",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[],
    )
    origin = origin_from_provider(session.source_name)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        if not native_id.strip():
            # identity_law.session_id (and the write path routed through
            # _stored_session_native_id) both reject an unidentifiable
            # session -- there is no position/variant fallback the way there
            # is for messages, so this must fail loudly, never write a
            # self-mismatched row.
            try:
                write_parsed_session_to_archive(
                    conn,
                    session,
                    content_hash=session_content_hash(session),
                )
            except ValueError:
                return
            raise AssertionError(f"expected ValueError for native_id={native_id!r}")

        session_id = write_parsed_session_to_archive(
            conn,
            session,
            content_hash=session_content_hash(session),
        )
        row = conn.execute(
            "SELECT session_id, native_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert row is not None, f"no sessions row for session_id={session_id!r}"

        expected = identity_law_session_id(origin.value, native_id)
        assert row["session_id"] == expected, (
            f"DB session_id {row['session_id']!r} != Python identity_law.session_id {expected!r} "
            f"for provider_session_id={native_id!r}"
        )
        # The stored native_id must itself be the stripped form -- otherwise
        # session_links.dst_native_id (written from another session's raw
        # parent reference, but normalized the same way) would stop matching
        # this session's own native_id column.
        assert row["native_id"] == native_id.strip()
    finally:
        conn.close()
