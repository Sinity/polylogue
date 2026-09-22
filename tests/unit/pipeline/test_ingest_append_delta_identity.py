"""polylogue-3hfl7: the append path's digest must describe the rows it writes.

``_write_session``'s append branch hands the writer a DELTA (only the messages
this revision adds) while ``content_hash`` stays the FULL session's digest --
which is correct for ``sessions.content_hash``, because the next ingest of this
session compares its own full-session digest against that row to decide the
content is unchanged. It was NOT correct as the digest a prepared identity
carrier is admitted against: the two were never compared, so the writer would
accept a carrier covering the merged session and then refuse inside
``_validated_prepared_content_identities`` on a length mismatch -- a hard
failure on the ingest_batch route, not a slow path.

Two production edits establish the separation and these tests pin both:

  * ``_append_delta_payload`` no longer lets the delta inherit the full
    session's parse-bound ``content_hash``; every digest consumer prefers that
    carrier over recomputing, so the delta used to self-report a digest for a
    row set it does not cover.
  * ``write_parsed_session_to_archive`` takes ``pending_input_content_hash``,
    the digest of the rows THIS call publishes, and admits prepared carriers
    against it -- refusing, by name, a carrier that describes the merged
    session instead.

Anti-vacuity: ``test_delta_digest_covers_only_delta_rows`` is red if the
``_core`` edit is reverted (the delta reports the merged digest again);
``test_append_carrier_admitted_and_reused`` is red if the ``write.py`` edit is
reverted (the carrier is refused as stale, or silently ignored and the writer
recomputes identities); ``test_merged_carrier_refused_on_append`` is red under
either revert (the mismatch resurfaces as an opaque length complaint).
``test_stored_hash_stays_merged_digest`` pins the opposite direction: storing
the delta's digest instead would make every re-ingest look changed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.ids import bound_session_content_hash, session_content_hash
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionWriteRefusedError,
    prepare_session_rows,
    prepare_session_write,
)
from polylogue.storage.sqlite.connection import open_connection

_SESSION_ID = "codex-session:append-delta-identity"
_NATIVE_ID = _SESSION_ID.split(":", 1)[1]


def _message(native_id: str, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=Role.USER,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _payload(*texts: str, append_only: bool) -> SessionWritePayload:
    """Build a payload exactly as ``ingest_worker`` does: hash, then bind it."""
    parsed = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=_NATIVE_ID,
        title="Append delta identity",
        created_at="2026-04-02T00:00:00Z",
        updated_at="2026-04-02T00:00:00Z",
        messages=[_message(f"m{index}", text) for index, text in enumerate(texts)],
    )
    content_hash = str(session_content_hash(parsed))
    parsed = parsed.model_copy(update={"content_hash": content_hash})
    return SessionWritePayload(
        session_id=_SESSION_ID,
        content_hash=content_hash,
        parsed_session=parsed,
        message_count=len(parsed.messages),
        append_only=append_only,
    )


def _seeded(tmp_path: Path) -> sqlite3.Connection:
    conn = open_connection(tmp_path / "index.db").__enter__()
    ingest_batch_core._write_session(conn, _payload("first", append_only=False))
    conn.commit()
    return conn


def test_delta_digest_covers_only_delta_rows(tmp_path: Path) -> None:
    conn = _seeded(tmp_path)
    try:
        payload = _payload("first", "second", append_only=True)
        delta, skipped = ingest_batch_core._append_delta_payload(conn, payload)
        assert delta is not None
        assert skipped == 1
        assert [message.provider_message_id for message in delta.messages] == ["m1"]

        # The delta's parse-bound carrier is the digest OF THE DELTA ...
        assert bound_session_content_hash(delta) == session_content_hash(delta)
        # ... and therefore not the merged session's, which covers two messages.
        assert bound_session_content_hash(delta) != payload.content_hash

        # Anything preparing rows from the delta inherits that honesty: the
        # digest a carrier reports names the row set the carrier holds.
        prepared = prepare_session_rows(delta)
        assert len(prepared.content_identities) == 1
        assert prepared.session_content_hash == bytes.fromhex(str(session_content_hash(delta)))
        assert prepared.session_content_hash != bytes.fromhex(payload.content_hash)
    finally:
        conn.close()


def test_append_carrier_admitted_and_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A carrier prepared for the delta is admitted on the ingest_batch route.

    The writer is wrapped exactly where ``_write_session`` calls it, so the
    carrier is introduced on the production route rather than beside it. The
    writer's own identity computation is made fatal: reaching it means the
    carrier was not used.
    """
    conn = _seeded(tmp_path)
    try:
        payload = _payload("first", "second", append_only=True)
        # Prepared ahead of the writer hold, from the delta the append branch
        # will itself derive -- what polylogue-fdzb3 moves to the parse worker.
        delta, _skipped = ingest_batch_core._append_delta_payload(conn, payload)
        assert delta is not None
        carrier = prepare_session_write(conn, delta, merge_append=True)

        real_write = archive_tier_write.write_parsed_session_to_archive

        def _write_with_carrier(connection: sqlite3.Connection, session: ParsedSession, **kwargs: Any) -> str:
            if kwargs.get("merge_append"):
                kwargs["prepared_write"] = carrier
            return real_write(connection, session, **kwargs)

        monkeypatch.setattr(ingest_batch_core, "write_parsed_session_to_archive", _write_with_carrier)

        def _boom(*args: object, **kwargs: object) -> object:
            raise AssertionError("writer recomputed identities instead of using the prepared carrier")

        monkeypatch.setattr(archive_tier_write, "message_content_identities", _boom)

        changed, counts = ingest_batch_core._write_session(conn, payload)
        conn.commit()

        assert changed is True
        assert counts["skipped_messages"] == 1
        stored = conn.execute(
            "SELECT native_id FROM messages WHERE session_id = ? ORDER BY position", (_SESSION_ID,)
        ).fetchall()
        assert [row[0] for row in stored] == ["m0", "m1"]
    finally:
        conn.close()


def test_merged_carrier_refused_on_append(tmp_path: Path) -> None:
    """A carrier describing the merged session is refused, naming the cause."""
    conn = _seeded(tmp_path)
    try:
        payload = _payload("first", "second", append_only=True)
        delta, _skipped = ingest_batch_core._append_delta_payload(conn, payload)
        assert delta is not None
        merged_carrier = prepare_session_rows(payload.parsed_session)

        with pytest.raises(PreparedSessionWriteRefusedError, match="merged session"):
            archive_tier_write.write_parsed_session_to_archive(
                conn,
                delta,
                content_hash=payload.content_hash,
                pending_input_content_hash=bound_session_content_hash(delta),
                merge_append=True,
                prepared=merged_carrier,
            )
    finally:
        conn.close()


def test_stored_hash_stays_merged_digest(tmp_path: Path) -> None:
    """sessions.content_hash keeps describing the merged session after an append."""
    conn = _seeded(tmp_path)
    try:
        payload = _payload("first", "second", append_only=True)
        changed, _counts = ingest_batch_core._write_session(conn, payload)
        conn.commit()
        assert changed is True

        stored = conn.execute("SELECT content_hash FROM sessions WHERE session_id = ?", (_SESSION_ID,)).fetchone()
        assert bytes(stored[0]).hex() == payload.content_hash

        # Which is what makes the identical replay a no-op rather than a
        # second append: the delta's own digest would never match here.
        _changed_again, replay_counts = ingest_batch_core._write_session(
            conn, _payload("first", "second", append_only=True)
        )
        conn.commit()
        assert replay_counts["skipped_sessions"] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (_SESSION_ID,)).fetchone()[0] == 2
    finally:
        conn.close()
