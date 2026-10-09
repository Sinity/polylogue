"""4ts.6: the lineage-completeness signal must reach the MCP payloads that
name get_messages/session envelope as consumers, not just live trapped on
the internal ArchiveSessionEnvelope (CodeRabbit finding on PR #2603)."""

from __future__ import annotations

from typing import cast

from polylogue.mcp.archive_support import archive_messages_payload
from polylogue.mcp.payloads import MCPArchiveMessagePayload, MCPArchiveSessionPayload
from polylogue.storage.runtime import LineageTruncationReason
from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope
from tests.infra.identity import archive_block_id, fixture_block_content_identity


def _envelope(
    *, lineage_complete: bool, lineage_truncation_reason: LineageTruncationReason | None
) -> ArchiveSessionEnvelope:
    return ArchiveSessionEnvelope(
        session_id="codex-session:native-1",
        native_id="native-1",
        origin="codex-session",
        title="Truncated session",
        active_leaf_message_id=None,
        messages=(),
        lineage_complete=lineage_complete,
        lineage_truncation_reason=lineage_truncation_reason,
    )


def test_archive_messages_payload_carries_lineage_completeness() -> None:
    truncated = _envelope(lineage_complete=False, lineage_truncation_reason="cycle")
    payload = archive_messages_payload(truncated, limit=50, offset=0)
    assert payload.lineage_complete is False
    assert payload.lineage_truncation_reason == "cycle"

    complete = _envelope(lineage_complete=True, lineage_truncation_reason=None)
    payload = archive_messages_payload(complete, limit=50, offset=0)
    assert payload.lineage_complete is True
    assert payload.lineage_truncation_reason is None


def test_mcp_archive_session_payload_carries_lineage_completeness() -> None:
    truncated = _envelope(lineage_complete=False, lineage_truncation_reason="dangling_branch_point")
    payload = MCPArchiveSessionPayload.from_session(truncated)
    assert payload.lineage_complete is False
    assert payload.lineage_truncation_reason == "dangling_branch_point"

    complete = _envelope(lineage_complete=True, lineage_truncation_reason=None)
    payload = MCPArchiveSessionPayload.from_session(complete)
    assert payload.lineage_complete is True
    assert payload.lineage_truncation_reason is None


def test_mcp_archive_message_payload_preserves_unknown_active_path() -> None:
    row = ArchiveMessageRow(
        message_id="m1",
        native_id="native-m1",
        role="assistant",
        position=3,
        variant_index=1,
        # A narrow adapter can be missing this optional signal even though
        # the production archive schema currently stores 0/1.
        is_active_path=cast("bool", None),
        is_active_leaf=False,
        blocks=(
            ArchiveBlockRow(
                block_id=archive_block_id("m1", content_identity=fixture_block_content_identity("text", "answer")),
                content_identity=fixture_block_content_identity("text", "answer"),
                content_occurrence=0,
                message_id="m1",
                block_type="text",
                text="answer",
            ),
        ),
    )

    payload = MCPArchiveMessagePayload.from_message(row)

    assert payload.position == 3
    assert payload.variant_index == 1
    assert payload.is_active_path is None
    assert payload.is_active_leaf is False


def test_archive_messages_payload_page_past_the_end_is_empty() -> None:
    """The page outcome counts the page's rows, not the session's messages.

    Anti-vacuity: deciding it from the filtered total (1) labels the empty
    page at offset 5 ``ok``.
    """
    from dataclasses import replace

    row = ArchiveMessageRow(
        message_id="m1",
        native_id="native-m1",
        role="assistant",
        position=0,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=True,
        blocks=(
            ArchiveBlockRow(
                block_id=archive_block_id("m1", content_identity=fixture_block_content_identity("text", "answer")),
                content_identity=fixture_block_content_identity("text", "answer"),
                content_occurrence=0,
                message_id="m1",
                block_type="text",
                text="answer",
            ),
        ),
    )
    session = replace(_envelope(lineage_complete=True, lineage_truncation_reason=None), messages=(row,))

    assert archive_messages_payload(session, limit=50, offset=5).outcome.state == "empty"
    assert archive_messages_payload(session, limit=50, offset=0).outcome.state == "ok"
