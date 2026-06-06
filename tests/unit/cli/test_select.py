"""Query-backed selector helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.query.spec import QuerySpecError
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.select import (
    SelectSessionRow,
    _parse_fzf_output,
    async_run_select,
    choose_select_row,
    render_select_row,
    select_row_from_result,
)
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import Origin
from polylogue.types import SessionId


def _row(index: int = 1) -> SelectSessionRow:
    return SelectSessionRow(
        session_id=f"conv-{index}",
        origin="claude-code-session",
        title=f"Session {index}",
        date="2026-05-02",
    )


def test_select_row_from_summary_uses_query_result_display_contract() -> None:
    summary = SessionSummary(
        id=SessionId("conv-select"),
        origin=Origin.CODEX_SESSION,
        title="Selector Contract",
        updated_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    row = select_row_from_result(summary)

    assert row == SelectSessionRow(
        session_id="conv-select",
        origin="codex-session",
        title="Selector Contract",
        date="2026-05-02",
    )


def test_render_select_row_outputs_requested_field() -> None:
    row = _row()

    assert render_select_row(row, "id") == "conv-1"
    assert render_select_row(row, "title") == "Session 1"
    assert render_select_row(row, "origin") == "claude-code-session"
    assert json.loads(render_select_row(row, "json")) == {
        "id": "conv-1",
        "origin": "claude-code-session",
        "title": "Session 1",
        "date": "2026-05-02",
    }


@pytest.mark.parametrize(
    ("plain", "stdin_tty", "stdout_tty"),
    [
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_choose_select_row_returns_first_when_not_interactive(
    plain: bool,
    stdin_tty: bool,
    stdout_tty: bool,
) -> None:
    ui = MagicMock()
    ui.plain = plain
    env = cast(AppEnv, SimpleNamespace(ui=ui))

    with (
        patch("sys.stdin.isatty", return_value=stdin_tty),
        patch("sys.stdout.isatty", return_value=stdout_tty),
    ):
        assert choose_select_row(env, [_row(1), _row(2)]) == _row(1)
        ui.choose.assert_not_called()


def test_choose_select_row_prefers_fzf_then_falls_back_to_ui() -> None:
    ui = MagicMock()
    env = cast(AppEnv, SimpleNamespace(ui=ui))
    rows = [_row(1), _row(2)]
    ui.plain = False

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("polylogue.cli.select._choose_with_fzf", return_value=rows[1]),
    ):
        assert choose_select_row(env, rows) == rows[1]
        ui.choose.assert_not_called()

    ui.choose.return_value = rows[0].label
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("polylogue.cli.select._choose_with_fzf", return_value=None),
    ):
        assert choose_select_row(env, rows) == rows[0]
        ui.choose.assert_called_once_with("Select session", [row.label for row in rows])


def test_parse_fzf_output_returns_selected_id() -> None:
    assert _parse_fzf_output("conv-1\tclaude-code | title\n") == "conv-1"
    assert _parse_fzf_output("\n") is None


@pytest.mark.asyncio
async def test_async_run_select_distinguishes_empty_results_from_cancel(capsys: pytest.CaptureFixture[str]) -> None:
    env = cast(AppEnv, SimpleNamespace(ui=MagicMock()))
    request = RootModeRequest.from_params({})

    with (
        patch("polylogue.cli.select.select_session_rows", AsyncMock(return_value=[_row()])),
        patch("polylogue.cli.select.choose_select_row", return_value=None),
    ):
        with pytest.raises(SystemExit) as exc_info:
            await async_run_select(env, request, limit=10, print_field="id")
    assert exc_info.value.code == 1
    assert "Selection cancelled." in capsys.readouterr().err

    with patch("polylogue.cli.select.select_session_rows", AsyncMock(return_value=[])):
        with pytest.raises(SystemExit) as exc_info:
            await async_run_select(env, request, limit=10, print_field="id")
    assert exc_info.value.code == 2
    assert "No sessions matched." in capsys.readouterr().err


@pytest.mark.asyncio
async def test_async_run_select_formats_query_errors(capsys: pytest.CaptureFixture[str]) -> None:
    env = cast(AppEnv, SimpleNamespace(ui=MagicMock()))
    request = RootModeRequest.from_params({})

    with patch(
        "polylogue.cli.select.select_session_rows",
        AsyncMock(side_effect=QuerySpecError("since", "bogus")),
    ):
        with pytest.raises(SystemExit) as exc_info:
            await async_run_select(env, request, limit=10, print_field="id")

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Cannot parse date: 'bogus'" in err
    assert "Hint: use ISO format" in err
