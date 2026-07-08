"""Block content-hash citation anchor tests (svfj)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.block_anchor import (
    BlockAnchor,
    InvalidBlockAnchorError,
    format_block_anchor,
    parse_block_anchor,
    resolve_block_anchor,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _anchor_for(conn: sqlite3.Connection, session_id: str, native_message_id: str, position: int) -> BlockAnchor:
    row = conn.execute(
        """
        SELECT m.message_id, b.content_hash
        FROM blocks b
        JOIN messages m ON m.message_id = b.message_id
        WHERE m.session_id = ? AND m.native_id = ? AND b.position = ?
        """,
        (session_id, native_message_id, position),
    ).fetchone()
    assert row is not None
    return BlockAnchor(
        session_id=session_id,
        message_id=str(row["message_id"]),
        content_hash_hex=bytes(row["content_hash"]).hex(),
    )


def test_format_and_parse_anchor_round_trip() -> None:
    anchor = BlockAnchor(session_id="codex-session:abc", message_id="codex-session:abc:m1:0", content_hash_hex="a" * 64)
    text = anchor.to_text()
    assert text == format_block_anchor(anchor.session_id, anchor.message_id, anchor.content_hash_hex)
    assert parse_block_anchor(text) == anchor


@pytest.mark.parametrize(
    "bad_anchor",
    [
        "only-one-part",
        "session::message-without-block-part",
        "session::message::not-a-block-prefix:deadbeef",
        "session::message::block@sha256:tooshort",
        "session::message::block@sha256:" + ("g" * 64),  # not hex
    ],
)
def test_parse_block_anchor_rejects_malformed_input(bad_anchor: str) -> None:
    with pytest.raises(InvalidBlockAnchorError):
        parse_block_anchor(bad_anchor)


def test_resolve_block_anchor_ok_when_unchanged(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="anchor-ok",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="stable evidence")],
                )
            ],
        )
        session_id = write_parsed_session_to_archive(conn, session)
        anchor = _anchor_for(conn, session_id, "m1", 0)

        resolution = resolve_block_anchor(conn, anchor, position_hint=0)
        assert resolution.state == "ok"
        assert resolution.resolved_message_id == anchor.message_id
        assert resolution.resolved_position == 0

        # No position hint at all is also ok -- the anchor itself carries no position.
        resolution_no_hint = resolve_block_anchor(conn, anchor)
        assert resolution_no_hint.state == "ok"
    finally:
        conn.close()


def test_resolve_block_anchor_drifted_position_after_reorder(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="anchor-drift-position",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[
                        ParsedContentBlock(type=BlockType.TEXT, text="first"),
                        ParsedContentBlock(type=BlockType.TEXT, text="second"),
                    ],
                )
            ],
        )
        session_id = write_parsed_session_to_archive(conn, session)
        anchor = _anchor_for(conn, session_id, "m1", 0)  # anchors "first" at position 0

        reordered = session.model_copy(
            update={
                "messages": [
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                        blocks=[
                            ParsedContentBlock(type=BlockType.TEXT, text="second"),
                            ParsedContentBlock(type=BlockType.TEXT, text="first"),
                        ],
                    )
                ]
            }
        )
        write_parsed_session_to_archive(conn, reordered)

        resolution = resolve_block_anchor(conn, anchor, position_hint=0)
        assert resolution.state == "drifted_position"
        assert resolution.resolved_position == 1

        # Without a position hint, a moved-but-findable block still resolves ok.
        resolution_no_hint = resolve_block_anchor(conn, anchor)
        assert resolution_no_hint.state == "ok"
    finally:
        conn.close()


def test_resolve_block_anchor_ambiguous_on_duplicate_evidence(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="anchor-ambiguous",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[
                        ParsedContentBlock(type=BlockType.TEXT, text="same text twice"),
                        ParsedContentBlock(type=BlockType.TEXT, text="same text twice"),
                    ],
                )
            ],
        )
        session_id = write_parsed_session_to_archive(conn, session)
        anchor = _anchor_for(conn, session_id, "m1", 0)

        resolution = resolve_block_anchor(conn, anchor)
        assert resolution.state == "ambiguous"
        assert len(resolution.candidates) == 2
        assert {position for _, position in resolution.candidates} == {0, 1}
    finally:
        conn.close()


def test_resolve_block_anchor_drifted_message_when_content_moves_within_session(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="anchor-drift-message",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="moved evidence")],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="unrelated")],
                ),
            ],
        )
        session_id = write_parsed_session_to_archive(conn, session)
        anchor = _anchor_for(conn, session_id, "m1", 0)

        # Re-ingest with the SAME evidence now attached to m2 instead of m1
        # (simulating a provider renumbering messages) -- m1 no longer
        # carries any block with this hash, but the session still does.
        moved = session.model_copy(
            update={
                "messages": [
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="replacement content")],
                    ),
                    ParsedMessage(
                        provider_message_id="m2",
                        role=Role.ASSISTANT,
                        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="moved evidence")],
                    ),
                ]
            }
        )
        write_parsed_session_to_archive(conn, moved)

        resolution = resolve_block_anchor(conn, anchor)
        assert resolution.state == "drifted_message"
        assert resolution.resolved_message_id != anchor.message_id
        assert resolution.resolved_position == 0
    finally:
        conn.close()


def test_resolve_block_anchor_hash_mismatch_never_guesses(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="anchor-hash-mismatch",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="original content")],
                )
            ],
        )
        session_id = write_parsed_session_to_archive(conn, session)
        anchor = _anchor_for(conn, session_id, "m1", 0)

        rewritten = session.model_copy(
            update={
                "messages": [
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="rewritten content, same position")],
                    )
                ]
            }
        )
        write_parsed_session_to_archive(conn, rewritten)

        resolution = resolve_block_anchor(conn, anchor, position_hint=0)
        assert resolution.state == "hash_mismatch"
        assert resolution.resolved_message_id == anchor.message_id
        assert resolution.resolved_position == 0
    finally:
        conn.close()


def test_resolve_block_anchor_missing_when_nothing_matches(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="anchor-missing",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="evidence")],
                )
            ],
        )
        session_id = write_parsed_session_to_archive(conn, session)
        anchor = _anchor_for(conn, session_id, "m1", 0)

        # Delete the session entirely -- the anchor now resolves nowhere.
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

        resolution = resolve_block_anchor(conn, anchor)
        assert resolution.state == "missing"
    finally:
        conn.close()
