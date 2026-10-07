"""A sidecar blob locator is stored beside its event, outside the bound session (polylogue-bgnxh).

The ingest batch publishes a matched tool-result sidecar's text as a
content-addressed blob after the session's identity was bound. Where it put
the bytes is publication metadata: the writer stores it on the event row, the
session object that was bound is written unchanged, and the hash partition
excludes it, so the stored representation hashes to the stored identity on
every write route.

Anti-vacuity: copying the locator into the session's events (the route
before this change) leaves the worker-prepared carrier's rows without it and,
without the partition exclusion, makes the stored payload hash to a different
identity than the one stored. Publishing an excised sidecar's bytes again, or
refusing the whole session, fails the refusal test.
"""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.source_write import record_excised_blob_hash
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.index_writer import write_fixture_retained_session

_FULL_TEXT = "full sidecar output line\n" * 300
_SESSION_ID = "claude-code-session:sidecar-bound"


def _bound_session() -> ParsedSession:
    """A parsed session carrying the identity retained publication binds."""
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="sidecar-bound",
        title="Session",
        created_at="2026-04-02T00:00:00Z",
        updated_at="2026-04-02T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role=Role.normalize("user"),
                text="tool output",
                occurred_at_ms=1_777_636_900_000,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        outcome_unknown_reason="not_reported",
                        tool_id="toolu_1",
                        text=_FULL_TEXT,
                    )
                ],
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="claude_tool_result_sidecar",
                payload={
                    "acquisition_status": "matched",
                    "tool_use_id": "toolu_1",
                    "filename": "toolu_1.txt",
                    "byte_size": len(_FULL_TEXT),
                    "content_replaced": True,
                },
            )
        ],
    )
    bound = str(session_content_hash(session))
    return session.model_copy(update={"content_hash": bound})


def _stored(conn: sqlite3.Connection) -> tuple[str, dict[str, object], str]:
    content_hash = conn.execute(
        "SELECT lower(hex(content_hash)) FROM sessions WHERE session_id = ?", (_SESSION_ID,)
    ).fetchone()[0]
    [event_row] = conn.execute(
        "SELECT payload_json FROM session_events WHERE session_id = ? AND event_type = 'claude_tool_result_sidecar'",
        (_SESSION_ID,),
    ).fetchall()
    [block_text] = conn.execute(
        "SELECT text FROM blocks WHERE session_id = ? AND block_type = 'tool_result'", (_SESSION_ID,)
    ).fetchone()
    return str(content_hash), json.loads(str(event_row[0])), str(block_text)


def _stored_representation_hash(session: ParsedSession, stored_event: dict[str, object]) -> str:
    """Hash the session as committed: its stored sidecar event in place of the parsed one."""
    [parsed_event] = session.session_events
    committed = session.model_copy(
        update={"session_events": [parsed_event.model_copy(update={"payload": stored_event})]}
    )
    return str(session_content_hash(committed))


def _publish(archive_root: Path, session: ParsedSession) -> bool:
    run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        changed, _counts = write_fixture_retained_session(conn, session)
        return changed
    finally:
        conn.close()


def test_sidecar_locator_is_committed_beside_the_bound_session(tmp_path: Path) -> None:
    bound_session = _bound_session()
    bound = str(bound_session.content_hash)
    archive_root = tmp_path / "archive"
    changed = _publish(archive_root, bound_session)
    with sqlite3.connect(archive_root / "index.db") as conn:
        stored_hash, stored_event, block_text = _stored(conn)

    assert changed is True
    expected_hash = sha256(_FULL_TEXT.encode("utf-8")).hexdigest()
    assert (archive_root / "blob" / expected_hash[:2] / expected_hash[2:]).read_text() == _FULL_TEXT
    [parsed_event] = bound_session.session_events
    assert "blob_hash" not in parsed_event.payload, "the bound session object was rewritten"
    assert stored_event == {**parsed_event.payload, "blob_hash": expected_hash}
    assert block_text == _FULL_TEXT
    assert stored_hash == bound
    assert _stored_representation_hash(bound_session, stored_event) == bound


def test_excised_sidecar_is_refused_alone_and_the_session_still_writes(tmp_path: Path) -> None:
    """A sidecar whose bytes were forgotten gets a typed refusal, not a republished blob.

    Excision marks a hash only when no live session references it, so a new
    session meeting that hash is refused per sidecar: its bytes are not put
    back on disk, its own block text stays, and nothing hashed is rewritten.
    """
    archive_root = tmp_path / "archive"
    run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
    excised_hash = sha256(_FULL_TEXT.encode("utf-8")).digest()
    with sqlite3.connect(archive_root / "source.db") as ledger:
        record_excised_blob_hash(
            ledger, blob_hash=excised_hash, reason="synthetic", actor="user:local", excised_at_ms=1
        )
    bound_session = _bound_session()
    changed = _publish(archive_root, bound_session)
    with sqlite3.connect(archive_root / "index.db") as conn:
        stored_hash, stored_event, block_text = _stored(conn)

    assert changed is True
    assert not (archive_root / "blob" / excised_hash.hex()[:2] / excised_hash.hex()[2:]).exists()
    [parsed_event] = bound_session.session_events
    assert stored_event == {**parsed_event.payload, "blob_refusal": "content_excised"}
    assert block_text == _FULL_TEXT
    assert stored_hash == bound_session.content_hash
    assert _stored_representation_hash(bound_session, stored_event) == bound_session.content_hash
