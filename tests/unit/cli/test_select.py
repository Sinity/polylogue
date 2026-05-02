"""Query-backed selector helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from polylogue.archive.conversation.models import ConversationSummary
from polylogue.cli.select import (
    SelectConversationRow,
    _parse_fzf_output,
    choose_select_row,
    render_select_row,
    select_row_from_result,
)
from polylogue.cli.shared.types import AppEnv
from polylogue.types import ConversationId, Provider


def _row(index: int = 1) -> SelectConversationRow:
    return SelectConversationRow(
        conversation_id=f"conv-{index}",
        provider="claude-code",
        title=f"Conversation {index}",
        date="2026-05-02",
    )


def test_select_row_from_summary_uses_query_result_display_contract() -> None:
    summary = ConversationSummary(
        id=ConversationId("conv-select"),
        provider=Provider.CODEX,
        title="Selector Contract",
        updated_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    row = select_row_from_result(summary)

    assert row == SelectConversationRow(
        conversation_id="conv-select",
        provider="codex",
        title="Selector Contract",
        date="2026-05-02",
    )


def test_render_select_row_outputs_requested_field() -> None:
    row = _row()

    assert render_select_row(row, "id") == "conv-1"
    assert render_select_row(row, "title") == "Conversation 1"
    assert render_select_row(row, "provider") == "claude-code"
    assert json.loads(render_select_row(row, "json")) == {
        "id": "conv-1",
        "provider": "claude-code",
        "title": "Conversation 1",
        "date": "2026-05-02",
    }


def test_choose_select_row_returns_first_when_stdout_is_not_tty() -> None:
    ui = MagicMock()
    env = cast(AppEnv, SimpleNamespace(ui=ui))

    with patch("sys.stdout.isatty", return_value=False):
        assert choose_select_row(env, [_row(1), _row(2)]) == _row(1)


def test_choose_select_row_prefers_fzf_then_falls_back_to_ui() -> None:
    ui = MagicMock()
    env = cast(AppEnv, SimpleNamespace(ui=ui))
    rows = [_row(1), _row(2)]

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("polylogue.cli.select._choose_with_fzf", return_value=rows[1]),
    ):
        assert choose_select_row(env, rows) == rows[1]
        ui.choose.assert_not_called()

    ui.choose.return_value = rows[0].label
    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("polylogue.cli.select._choose_with_fzf", return_value=None),
    ):
        assert choose_select_row(env, rows) == rows[0]
        ui.choose.assert_called_once_with("Select conversation", [row.label for row in rows])


def test_parse_fzf_output_returns_selected_id() -> None:
    assert _parse_fzf_output("conv-1\tclaude-code | title\n") == "conv-1"
    assert _parse_fzf_output("\n") is None
