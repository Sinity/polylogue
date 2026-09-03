"""The writer keeps derived session facts consistent with their evidence.

Three facts the writer derives rather than copies: a session's kind, which
must follow the branch type its resolved links project; a block's tool
verdict, whose canonical and legacy columns describe one outcome and must
therefore move together across a merge; and an attachment's provenance, which
names the turn that produced it even when the provider supplied no id for
that turn.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider, SessionKind
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    _coalesce_block_row,
    _refresh_session_projection,
    write_parsed_session_to_archive,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_session_kind_follows_a_later_resolved_subagent_link(tmp_path: Path) -> None:
    """A subagent edge discovered after admission updates the stored kind.

    Admission derives the kind from parser evidence only. Hook evidence
    resolved into ``session_links`` reaches the session through
    ``_write_session_link`` -> ``_refresh_session_projection``, which is what
    this exercises directly.

    Anti-vacuity: drop ``session_kind`` from the UPDATE in
    ``_refresh_session_projection`` and this fails -- ``branch_type`` still
    becomes ``subagent`` while the kind stays ``primary``, which is exactly
    the inconsistency being fixed.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        parent_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="spawning-parent",
                updated_at="2026-01-01T00:00:01+00:00",
                messages=[ParsedMessage(provider_message_id="p1", role=Role.USER, text="delegate this")],
            ),
        )
        child_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="spawned-child",
                updated_at="2026-01-01T00:00:02+00:00",
                messages=[ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="on it")],
            ),
        )
        # The parser saw no delegation, so the child was admitted as primary.
        assert (
            conn.execute("SELECT session_kind FROM sessions WHERE session_id = ?", (child_id,)).fetchone()[0]
            == SessionKind.PRIMARY.value
        )

        # Evidence arrives afterwards and names the child a subagent of the parent.
        conn.execute(
            """
            INSERT INTO session_links (
                src_session_id, dst_origin, dst_native_id, resolved_dst_session_id,
                link_type, method, confidence, observed_at_ms
            ) VALUES (?, 'claude-code-session', 'spawning-parent', ?, 'subagent', 'hook', 1.0, 1767225603000)
            """,
            (child_id, parent_id),
        )
        _refresh_session_projection(conn, child_id, seen=set())

        row = conn.execute(
            "SELECT branch_type, session_kind FROM sessions WHERE session_id = ?", (child_id,)
        ).fetchone()
        assert row["branch_type"] == "subagent"
        assert row["session_kind"] == SessionKind.SUBAGENT.value
    finally:
        conn.close()


def _block_columns() -> dict[str, int]:
    names = ("tool_outcome", "tool_result_is_error", "tool_result_exit_code", "tool_result_outcome_unknown_reason")
    base = ("message_id", "session_id", "position", "content_hash", "block_type", "text", "tool_name")
    extra = ("tool_input", "semantic_type", "media_type", "language")
    return {name: index for index, name in enumerate(base + extra + names)}


def _row(b_idx: dict[str, int], **values: object) -> tuple[object, ...]:
    row: list[object] = [None] * len(b_idx)
    row[b_idx["block_type"]] = "tool_result"
    for name, value in values.items():
        row[b_idx[name]] = value
    return tuple(row)


def test_a_merged_tool_verdict_comes_from_one_row() -> None:
    """A merge never pairs one row's outcome with another row's error flags.

    Anti-vacuity: restore the per-field ``_coalesce_scalar`` for the four
    verdict columns and this fails -- the fresh ``success`` outcome is kept
    while ``tool_result_is_error=1`` and ``exit_code=1`` fall back to the old
    row, persisting a success that records a failure.
    """
    b_idx = _block_columns()
    new_row = _row(b_idx, tool_outcome="success")
    old_row = _row(b_idx, tool_outcome="failure", tool_result_is_error=1, tool_result_exit_code=1)

    merged = _coalesce_block_row(new_row, old_row, b_idx, message_id="m", position=0)

    assert merged[b_idx["tool_outcome"]] == "success"
    assert merged[b_idx["tool_result_is_error"]] is None
    assert merged[b_idx["tool_result_exit_code"]] is None


def test_a_row_with_no_verdict_inherits_the_whole_prior_verdict() -> None:
    """A row that states no outcome does not partially erase the stored one.

    Anti-vacuity: make ``_apply_tool_verdict`` always read ``new_row`` and
    this fails -- the merged row loses the recorded failure entirely instead
    of carrying it forward.
    """
    b_idx = _block_columns()
    new_row = _row(b_idx, text="retried output")
    old_row = _row(b_idx, tool_outcome="failure", tool_result_is_error=1, tool_result_exit_code=2)

    merged = _coalesce_block_row(new_row, old_row, b_idx, message_id="m", position=0)

    assert merged[b_idx["tool_outcome"]] == "failure"
    assert merged[b_idx["tool_result_is_error"]] == 1
    assert merged[b_idx["tool_result_exit_code"]] == 2
