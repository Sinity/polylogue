"""Query-backed selector helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.archive.query.spec import QuerySpecError
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.cli.operation_kernel import OperationRequest
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.select import (
    SelectSessionRow,
    _parse_fzf_output,
    choose_select_row,
    render_select_row,
    render_select_rows,
    run_select,
    select_row_from_result,
)
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from tests.infra.frozen_clock import FrozenClock


def _row(index: int = 1) -> SelectSessionRow:
    return SelectSessionRow(
        session_id=f"conv-{index}",
        origin="claude-code-session",
        title=f"Session {index}",
        date="2026-05-02",
    )


def test_select_row_carries_shared_informativeness_fields() -> None:
    summary = SessionSummary(
        id=SessionId("conv-context"),
        origin=Origin.CODEX_SESSION,
        title="Context",
        message_count=7,
        git_repository_url="https://example.test/org/polylogue",
        working_directories=("/workspace/polylogue",),
    )

    row = select_row_from_result(summary)

    assert row.message_count == 7
    assert row.repo == "polylogue"
    assert row.cwd_display == "polylogue"
    assert row.to_json()["message_count"] == 7
    assert row.to_json()["repo"] == "polylogue"
    assert row.to_json()["cwd_display"] == "polylogue"


@pytest.mark.frozen_clock_modules("polylogue.surfaces.query_rows")
def test_select_row_from_summary_uses_query_result_display_contract(frozen_clock: FrozenClock) -> None:
    """Anti-vacuity: without the frozen clock the expected label ages weekly."""
    frozen_clock.set_time(datetime(2026, 8, 22, tzinfo=timezone.utc).timestamp())
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
        relative_time="16w ago",
    )


def test_select_row_bounds_multiline_titles() -> None:
    summary = SessionSummary(
        id=SessionId("conv-select"),
        origin=Origin.CODEX_SESSION,
        title="needle\n" + "\n".join(f"/tmp/hermes-agent/path-{index}.py" for index in range(50)),
        updated_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    row = select_row_from_result(summary)

    assert "\n" not in row.title
    assert len(row.title) <= 96
    assert row.title.endswith("...")
    assert "/tmp/hermes-agent/path-20.py" not in row.title


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
        "message_count": 0,
        "repo": None,
        "cwd_display": None,
        "outcome": "unknown",
        "cost_usd": None,
        "relative_time": "unknown",
    }


@pytest.mark.parametrize(
    ("plain", "stdin_tty", "stdout_tty"),
    [
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_choose_select_row_rejects_ambiguous_non_interactive(
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
        assert choose_select_row(env, [_row(1), _row(2)]) is None
        ui.choose.assert_not_called()


def test_choose_select_row_returns_singleton_when_not_interactive() -> None:
    ui = MagicMock()
    ui.plain = True
    env = cast(AppEnv, SimpleNamespace(ui=ui))

    assert choose_select_row(env, [_row(1)]) == _row(1)
    ui.choose.assert_not_called()


def test_render_select_rows_emits_one_row_per_line() -> None:
    rows = [_row(1), _row(2)]

    assert render_select_rows(rows, "id") == "conv-1\nconv-2"


def test_render_select_rows_json_emits_single_array() -> None:
    rows = [_row(1), _row(2)]

    assert json.loads(render_select_rows(rows, "json")) == [
        {
            "id": "conv-1",
            "origin": "claude-code-session",
            "title": "Session 1",
            "date": "2026-05-02",
            "message_count": 0,
            "repo": None,
            "cwd_display": None,
            "outcome": "unknown",
            "cost_usd": None,
            "relative_time": "unknown",
        },
        {
            "id": "conv-2",
            "origin": "claude-code-session",
            "title": "Session 2",
            "date": "2026-05-02",
            "message_count": 0,
            "repo": None,
            "cwd_display": None,
            "outcome": "unknown",
            "cost_usd": None,
            "relative_time": "unknown",
        },
    ]


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


def test_select_reads_rows_from_one_declared_query_operation(tmp_path: Path) -> None:
    """Selection asks the declared read, and takes rows from either page shape.

    ``select`` used to build a ``SessionQuerySpec`` and run its own filter chain
    against the local archive. It now dispatches ``cli.query`` -- the same read
    ``find`` runs, so a daemon answers a selection exactly as it answers a query
    -- forwarding the operator's terms and the row limit, and projects the
    operation's rows into selector rows. A ranked selection reports ``hits``
    whose session is nested, so both shapes are read.

    Anti-vacuity: read only ``items`` and the ranked case yields no rows; stop
    forwarding ``limit`` and the payload assertion goes red.
    """
    from polylogue.cli.session_rows import query_session_rows

    config = Config(
        archive_root=tmp_path,
        db_path=tmp_path / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    request = RootModeRequest.from_params({"query": ("id:abc",)})
    captured: dict[str, object] = {}
    page: dict[str, object] = {}

    def _dispatch(_config: Config, operation_request: OperationRequest, **_kwargs: object) -> object:
        captured["operation"] = operation_request.operation
        captured["payload"] = operation_request.payload
        return SimpleNamespace(value=dict(page), authority={}, envelope=None)

    row = {"id": "conv-7", "origin": "claude-code-session", "title": "Seven", "message_count": 3}

    with patch("polylogue.cli.operation_kernel.dispatch", _dispatch):
        page.clear()
        page.update({"items": [row], "total": 1})
        listed = query_session_rows(config, request, limit=3)

        page.clear()
        page.update({"hits": [{"session": row, "match": {"rank": 1}}], "total": 1})
        ranked = query_session_rows(config, request, limit=3)

    assert captured["operation"] == "cli.query"
    params = cast("dict[str, object]", cast("dict[str, object]", captured["payload"])["params"])
    assert params["query"] == ["id:abc"]
    assert params["limit"] == 3
    assert [r.session_id for r in listed] == ["conv-7"]
    assert [r.session_id for r in ranked] == ["conv-7"]
    assert listed[0].title == "Seven"


def test_noninteractive_select_does_not_initialize_ui() -> None:
    class _Env:
        @property
        def ui(self) -> object:
            raise AssertionError("noninteractive selection should not touch UI")

    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("sys.stdout.isatty", return_value=True),
    ):
        assert choose_select_row(cast(AppEnv, _Env()), [_row(1), _row(2)]) is None


def test_run_select_prints_candidates_when_selection_is_ambiguous(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = cast(AppEnv, SimpleNamespace(ui=MagicMock(), config=MagicMock()))
    request = RootModeRequest.from_params({})

    with (
        patch("polylogue.cli.session_rows.query_session_rows", return_value=[_row(1), _row(2)]),
        patch("polylogue.cli.select.choose_select_row", return_value=None),
    ):
        run_select(env, request, limit=10, print_field="id")
    captured = capsys.readouterr()
    assert captured.out == "conv-1\nconv-2\n"
    assert captured.err == ""

    with patch("polylogue.cli.session_rows.query_session_rows", return_value=[]):
        with pytest.raises(SystemExit) as exc_info:
            run_select(env, request, limit=10, print_field="id")
    assert exc_info.value.code == 2
    assert "No sessions matched." in capsys.readouterr().err


def test_run_select_formats_query_errors(capsys: pytest.CaptureFixture[str]) -> None:
    env = cast(AppEnv, SimpleNamespace(ui=MagicMock(), config=MagicMock()))
    request = RootModeRequest.from_params({})

    with patch(
        "polylogue.cli.session_rows.query_session_rows",
        side_effect=QuerySpecError("since", "bogus"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            run_select(env, request, limit=10, print_field="id")

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Cannot parse date: 'bogus'" in err
    assert "Hint: use ISO format" in err


def test_selection_of_an_absent_session_is_empty_not_a_failure(tmp_path: Path) -> None:
    """``id:`` naming a session the archive lacks selects nothing, and says so.

    The read handler refuses an unknown session scope, because for a *read* of
    that session a refusal is the honest answer. A selection is a different
    question: zero rows. Without this translation ``find id:absent then select``
    exited 1 with no output instead of the documented "No sessions matched."
    and exit 2, and ``delete --dry-run`` lost its empty preview.

    Anti-vacuity: re-raise the refusal and both documented invocations go red
    in ``scripts/golden_find_bytes.py``.
    """
    from polylogue.cli.operation_kernel import OperationFailedError
    from polylogue.cli.session_rows import query_session_rows

    config = Config(
        archive_root=tmp_path,
        db_path=tmp_path / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    request = RootModeRequest.from_params({"query": ("id:absent",)})

    with patch(
        "polylogue.cli.operation_kernel.dispatch",
        side_effect=OperationFailedError("invalid_request", "session not found: absent"),
    ):
        assert query_session_rows(config, request, limit=5) == []

    with patch(
        "polylogue.cli.operation_kernel.dispatch",
        side_effect=OperationFailedError("invalid_request", "index is unreadable"),
    ):
        with pytest.raises(OperationFailedError):
            query_session_rows(config, request, limit=5)


def test_complete_selection_walks_every_page(tmp_path: Path) -> None:
    """A mutating verb's matched set is the whole selection, not one page.

    The operation bounds one response and clamps an over-large limit without
    saying so, so completeness is this loop over ``next_offset``. A one-page
    answer is what let ``delete --yes --all`` skip every match past the first
    page (#1873).

    Anti-vacuity: return after the first page and the assertion loses ``b``.
    """
    from polylogue.cli.session_rows import query_complete_session_ids

    config = Config(
        archive_root=tmp_path,
        db_path=tmp_path / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    pages = [
        {"items": [{"id": "a"}], "total": 2, "next_offset": 1},
        {"items": [{"id": "b"}], "total": 2, "next_offset": None},
    ]
    seen: list[int] = []

    def _dispatch(_config: Config, operation_request: OperationRequest, **_kwargs: object) -> object:
        offset = cast("dict[str, object]", operation_request.payload["params"])["offset"]
        seen.append(cast("int", offset))
        return SimpleNamespace(value=pages[len(seen) - 1], authority={}, envelope=None)

    with patch("polylogue.cli.operation_kernel.dispatch", _dispatch):
        assert query_complete_session_ids(config, RootModeRequest.from_params({})) == ["a", "b"]

    assert seen == [0, 1]
