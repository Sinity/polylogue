from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.cli.read_views.streaming_markdown import _row_to_renderable_block, stream_exact_session_markdown
from polylogue.core.enums import BlockType, Provider
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.identity import archive_block_id, archive_message_id

_SESSION_ID = "codex-session:abc"
_CHILD_SESSION_ID = "codex-session:child"

# The clock runs backwards against content position, so a read keyed on
# `occurred_at_ms` returns the exact reverse of the transcript. This is the
# ordinary shape, not a pathology: non-monotonic timestamps occur on every
# origin.
_STAMPS = ("2026-01-01T00:00:03Z", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z")


def _connect(root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(root / "index.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _message(
    native_id: str,
    role: Role,
    position: int,
    blocks: list[ParsedContentBlock],
    *,
    text: str | None = None,
) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        timestamp=_STAMPS[position],
        blocks=blocks,
    )


def _seed_index(root: Path) -> str:
    """Write the fixture session through the production index writer.

    The view under test reads `messages`/`blocks`/`session_links` from a real
    archive tier, so the fixture is built by the writer that produces them
    rather than by hand-rolled DDL that can drift from it.
    """
    conn = _connect(root)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="abc",
                title="Large export",
                messages=[
                    _message("m1", Role.USER, 0, [ParsedContentBlock(type=BlockType.TEXT, text="hello")], text="hello"),
                    _message(
                        "m2",
                        Role.ASSISTANT,
                        1,
                        [
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="shell",
                                tool_id="call-1",
                                tool_input={"command": "pytest"},
                            )
                        ],
                    ),
                    _message(
                        "m3",
                        Role.ASSISTANT,
                        2,
                        [
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                text="1 passed",
                                tool_id="call-1",
                                is_error=False,
                                exit_code=0,
                            )
                        ],
                    ),
                ],
            ),
        )
        conn.commit()
    finally:
        conn.close()
    assert session_id == _SESSION_ID
    return session_id


def _seed_prefix_sharing_child(root: Path) -> str:
    """Add a real fork of the fixture session, carrying a `prefix-sharing` edge."""
    conn = _connect(root)
    try:
        child_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="child",
                title="Child",
                parent_session_provider_id="abc",
                branch_type=BranchType.FORK,
                messages=[
                    _message("m1", Role.USER, 0, [ParsedContentBlock(type=BlockType.TEXT, text="hello")], text="hello"),
                    _message(
                        "m2",
                        Role.ASSISTANT,
                        1,
                        [ParsedContentBlock(type=BlockType.TEXT, text="parent reply")],
                        text="parent reply",
                    ),
                    _message(
                        "c1",
                        Role.USER,
                        2,
                        [ParsedContentBlock(type=BlockType.TEXT, text="child tail")],
                        text="child tail",
                    ),
                ],
            ),
        )
        conn.commit()
    finally:
        conn.close()
    assert child_id == _CHILD_SESSION_ID
    return child_id


def test_stream_exact_session_markdown_writes_full_file(tmp_path: Path) -> None:
    """The export follows content position, which this fixture's clock reverses.

    Anti-vacuity: ordering the stream by `occurred_at_ms` emits the three
    messages backwards and the final assertion fails.
    """
    _seed_index(tmp_path)
    out = tmp_path / "out.md"

    assert stream_exact_session_markdown(tmp_path, _SESSION_ID, out, prose_only=False)

    text = out.read_text(encoding="utf-8")
    assert "# Large export" in text
    assert "## user" in text
    assert "hello" in text
    assert "**Tool: shell**" in text
    assert "1 passed" in text
    assert text.index("hello") < text.index("**Tool: shell**") < text.index("1 passed")


def test_stream_exact_session_markdown_prose_only_omits_tools(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    out = tmp_path / "dialogue.md"

    assert stream_exact_session_markdown(tmp_path, "abc", out, prose_only=True)

    text = out.read_text(encoding="utf-8")
    assert "hello" in text
    assert "Tool: shell" not in text
    assert "1 passed" not in text


def test_stream_exact_session_markdown_defers_lineage_composition(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    child_id = _seed_prefix_sharing_child(tmp_path)

    assert not stream_exact_session_markdown(tmp_path, child_id, tmp_path / "out.md", prose_only=False)


def test_stream_exact_session_markdown_without_session_links_streams(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    child_id = _seed_prefix_sharing_child(tmp_path)
    conn = _connect(tmp_path)
    conn.execute("DROP TABLE session_links")
    conn.commit()
    conn.close()

    assert stream_exact_session_markdown(tmp_path, child_id, tmp_path / "out.md", prose_only=False)


def test_stream_exact_session_markdown_reports_unreadable_index(tmp_path: Path) -> None:
    (tmp_path / "index.db").write_bytes(b"not sqlite")

    with pytest.raises(ArchiveTierUnavailableError) as error:
        stream_exact_session_markdown(tmp_path, _SESSION_ID, tmp_path / "out.md", prose_only=False)

    assert error.value.code == "archive_tier_unavailable"
    assert error.value.tier == "index"
    assert error.value.path == str((tmp_path / "index.db").resolve())


def test_streaming_adapter_preserves_structural_outcome_and_exact_block_id(tmp_path: Path) -> None:
    _seed_index(tmp_path)
    expected_block_id = archive_block_id(archive_message_id(_SESSION_ID, "m3", position=2), position=0)
    conn = _connect(tmp_path)
    row = conn.execute("SELECT * FROM blocks WHERE block_id = ?", (expected_block_id,)).fetchone()
    assert row is not None

    block = _row_to_renderable_block(row)

    assert block.block_id == expected_block_id
    assert block.tool_result_is_error is False
    assert block.tool_result_exit_code == 0
    conn.close()
