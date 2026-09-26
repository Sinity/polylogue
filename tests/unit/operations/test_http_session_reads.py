"""Pinned product contracts for the daemon's session detail and messages routes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from polylogue.operations.http_session_reads import (
    HttpSessionProjectionAdapters,
    execute_http_session_detail,
    execute_http_session_messages,
)


def _adapters() -> HttpSessionProjectionAdapters:
    return HttpSessionProjectionAdapters(
        attachment=lambda attachment, *, session_id, message_id=None: {
            "attachment_id": attachment,
            "session_id": session_id,
            "message_id": message_id,
        },
        paste_spans=lambda text, *, has_paste: [],
    )


def test_session_detail_reads_only_requested_composed_page_and_reports_true_total() -> None:
    archive = MagicMock()
    archive.resolve_session_id.return_value = "codex-session:child"
    archive.read_summary.return_value = SimpleNamespace(
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:01:00Z",
        word_count=500,
        tags=("review",),
    )
    envelope = SimpleNamespace(
        session_id="codex-session:child",
        title="Child",
        origin="codex-session",
        messages=(SimpleNamespace(message_id="m1"),),
        orphan_attachments=(),
        total_message_count=201,
        branch_type="fork",
        parent_session_id="codex-session:parent",
        git_repository_url=None,
        working_directories=(),
    )
    archive.read_session_page.return_value = envelope
    placement = MagicMock()
    placement.session_entries = ()
    placement.entries_for.return_value = []
    placement.cards_for.return_value = []
    placement.is_suppressed.return_value = False
    with (
        patch("polylogue.operations.http_session_reads._semantic_placement", return_value=placement),
        patch("polylogue.operations.http_session_reads._message_payload", return_value={"id": "m1", "attachments": []}),
    ):
        result = execute_http_session_detail(
            {"session_id": "child", "shape": "full", "limit": 1, "offset": 200},
            archive=archive,
            adapters=_adapters(),
        )
    archive.read_session_page.assert_called_once_with("codex-session:child", limit=1, offset=200)
    archive.read_session.assert_not_called()
    assert result is not None
    assert result["message_count"] == result["total"] == 201
    assert result["parent_id"] == "codex-session:parent"
    assert result["messages"] == [{"id": "m1", "attachments": []}]


def test_session_summary_shape_does_not_hydrate_messages() -> None:
    archive = MagicMock()
    archive.resolve_session_id.return_value = "codex-session:child"
    archive.read_summary.return_value = SimpleNamespace(
        branch_type="fork",
        parent_id="codex-session:parent",
        session_kind="standard",
        display_name="Child",
        title_source="provider",
        title_ref="title:1",
    )
    with patch(
        "polylogue.operations.http_session_reads._summary_payload",
        return_value={"title": "Child", "message_count": 201},
    ):
        result = execute_http_session_detail(
            {"session_id": "child", "shape": "summary"}, archive=archive, adapters=_adapters()
        )
    archive.read_session.assert_not_called()
    archive.read_session_page.assert_not_called()
    assert result is not None
    assert result["total"] == 201
    assert result["parent_id"] == "codex-session:parent"


def test_session_messages_preserves_giant_single_message_and_continuation() -> None:
    archive = MagicMock()
    archive.resolve_session_id.return_value = "codex-session:child"
    row = SimpleNamespace(
        message_id="m1",
        occurred_at="2026-01-01T00:00:00Z",
        source_session_id=None,
        word_count=1,
        attachments=(),
    )
    envelope = SimpleNamespace(
        session_id="codex-session:child",
        origin="codex-session",
        messages=(row,),
        total_message_count=2,
        lineage_complete=False,
        lineage_truncation_reason="missing_parent",
    )
    archive.read_session_page.return_value = envelope
    placement = MagicMock()
    placement.session_entries = ()
    placement.entries_for.return_value = []
    placement.cards_for.return_value = []
    placement.is_suppressed.return_value = False
    giant_text = "x" * (8 * 1024 * 1024 + 1)
    domain = SimpleNamespace(
        id="m1",
        identity_source="native",
        role="assistant",
        text=giant_text,
        message_type="text",
        material_origin="assistant_authored",
        duration_ms=None,
        stop_reason=None,
        has_tool_use=False,
        has_thinking=False,
        has_paste=False,
    )

    def _window(_archive: object, _request: object, *, read: Any, **_kwargs: object) -> SimpleNamespace:
        rows, total, completeness = read(1, 0)
        assert completeness.complete is False
        assert completeness.truncation_reason == "missing_parent"
        return SimpleNamespace(
            rows=rows,
            total=total,
            limit=1,
            offset=0,
            next_offset=1,
            continuation="snapshot-token",
        )

    with (
        patch("polylogue.operations.http_session_reads.read_transcript_window_sync", side_effect=_window),
        patch("polylogue.operations.http_session_reads._semantic_placement", return_value=placement),
        patch("polylogue.operations.http_session_reads.archive_message_to_domain", return_value=domain),
        patch("polylogue.operations.http_session_reads.message_topology_from_domain", return_value={}),
        patch("polylogue.operations.http_session_reads.authority_for_reader", return_value=object()),
        patch("polylogue.operations.http_session_reads.serialize_authority", return_value={"mode": "daemon"}),
    ):
        result = execute_http_session_messages(
            {"session_id": "child", "limit": 1, "offset": 0},
            archive=archive,
            adapters=_adapters(),
        )
    assert cast(list[dict[str, object]], result["messages"])[0]["text"] == giant_text
    assert result["continuation"] == "snapshot-token"
    assert result["next_offset"] == 1
    assert result["authority"] == {"mode": "daemon"}
    assert result["lineage_complete"] is False
    assert result["lineage_truncation_reason"] == "missing_parent"
