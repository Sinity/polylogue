"""Regression coverage for the shared latest-resolver helper (#1626, #1642).

Verifies the resolution rules apply uniformly:
explicit conv_id wins, then ``--latest`` / any narrowing filter, then
``None``. The single-session surfaces (``read --view messages``/``raw``/
``neighbors``, ``export``, ``analyze turns``) all route through
this helper, so a single test here pins the contract for all of them.

The resolution itself is the declared ``cli.query`` operation asked for one
row. It used to open a ``Polylogue`` facade in the CLI process and list
summaries against the local archive, so ``--latest`` answered from a second
executor no daemon ever saw; the seam these tests control is therefore the
operation's rows, not a patched ``list_summaries``.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from polylogue.cli import select as select_module
from polylogue.cli.contextual_errors import AmbiguousSelectionError
from polylogue.cli.operation_kernel import OperationUnavailableError
from polylogue.cli.select import SelectSessionRow
from polylogue.cli.shared.helper_support import DaemonRequiredError
from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params


def _stub_ids(monkeypatch: pytest.MonkeyPatch, ids: Sequence[str], captured_limits: list[int] | None = None) -> None:
    def _query_session_ids(config: object, request: object, *, limit: int, **_kwargs: object) -> list[str]:
        if captured_limits is not None:
            captured_limits.append(limit)
        return list(ids)

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_ids", _query_session_ids)


def _stub_rows(monkeypatch: pytest.MonkeyPatch, ids: Sequence[str], captured_limits: list[int] | None = None) -> None:
    def _query_session_rows(
        config: object, request: object, *, limit: int, **_kwargs: object
    ) -> list[SelectSessionRow]:
        if captured_limits is not None:
            captured_limits.append(limit)
        return [SelectSessionRow(session_id=ref, origin="codex-session", title=ref, date=None) for ref in ids][:limit]

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_rows", _query_session_rows)


def _refuse_chooser(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the chooser ran without a terminal")

    monkeypatch.setattr(select_module, "choose_select_row", _explode)
    monkeypatch.setattr(select_module, "_choose_with_fzf", _explode)


def test_explicit_conv_id_wins_over_filters() -> None:
    """An explicit conv_id short-circuits — no query runs."""
    result = resolve_session_id_from_root_params({"conv_id": "claude-code:explicit", "latest": True})
    assert result == "claude-code:explicit"


def test_no_filters_returns_none() -> None:
    """Empty params returns None — caller surfaces its own missing-id error."""
    assert resolve_session_id_from_root_params({}) is None


def test_latest_runs_the_operation_with_limit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--latest`` asks the declared read for one row and returns the top match.

    Anti-vacuity: drop the ``limit=1`` and the captured-limit assertion is red;
    resolve locally again and the patched operation is never consulted.
    """
    captured_limits: list[int] = []
    _stub_ids(monkeypatch, ["claude-code:latest-conv"], captured_limits)

    result = resolve_session_id_from_root_params({"latest": True})

    assert result == "claude-code:latest-conv"
    assert captured_limits == [1]


def test_filter_alone_resolves_when_match_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """A narrowing filter (provider, since, etc.) also triggers resolution."""
    _stub_rows(monkeypatch, ["codex:filtered"])

    assert resolve_session_id_from_root_params({"origin": "codex-session"}) == "codex:filtered"


def test_a_filter_matching_several_sessions_refuses_instead_of_taking_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--latest`` a multi-row filter is ambiguous, not "the top row".

    The probe asks for one row past the candidate bound so a truncated list is
    marked as such, and the chooser is never consulted without a terminal.

    Anti-vacuity: restore ``limit=1`` plus ``session_ids[0]`` for a bare filter
    and this returns ``codex:a`` instead of raising.
    """
    captured_limits: list[int] = []
    _stub_rows(monkeypatch, ["codex:a", "codex:b", "codex:c"], captured_limits)
    _refuse_chooser(monkeypatch)

    with pytest.raises(AmbiguousSelectionError) as refusal:
        resolve_session_id_from_root_params({"origin": "codex-session"}, operation="analyze turns")

    assert refusal.value.candidates == ("codex:a", "codex:b", "codex:c")
    assert refusal.value.bounded is False
    assert captured_limits == [11]


def test_a_plain_terminal_session_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--plain`` opts out of the chooser even when a terminal is attached."""
    _stub_rows(monkeypatch, ["codex:a", "codex:b"])
    _refuse_chooser(monkeypatch)
    terminal = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(select_module, "sys", SimpleNamespace(stdin=terminal, stdout=terminal))

    with pytest.raises(AmbiguousSelectionError):
        resolve_session_id_from_root_params(
            {"origin": "codex-session"},
            env=SimpleNamespace(ui=SimpleNamespace(plain=True)),  # type: ignore[arg-type]
        )


def test_the_terminal_chooser_decides_an_ambiguous_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a terminal the operator's pick is the resolved session."""
    _stub_rows(monkeypatch, ["codex:a", "codex:b"])
    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    monkeypatch.setattr(select_module, "choose_select_row", lambda _env, rows: rows[1])

    resolved = resolve_session_id_from_root_params(
        {"origin": "codex-session"},
        env=SimpleNamespace(ui=SimpleNamespace(plain=False)),  # type: ignore[arg-type]
    )

    assert resolved == "codex:b"


def test_latest_returns_none_when_archive_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--latest`` against an empty archive returns None, not an error."""
    _stub_ids(monkeypatch, [])

    assert resolve_session_id_from_root_params({"latest": True}) is None


def test_latest_reports_typed_daemon_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable query keeps its operation in the CLI refusal."""

    def unavailable(*_args: object, **_kwargs: object) -> list[str]:
        raise OperationUnavailableError("daemon unavailable", operation="cli.query")

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_ids", unavailable)

    with pytest.raises(DaemonRequiredError) as exc_info:
        resolve_session_id_from_root_params({"latest": True})

    assert exc_info.value.code == "daemon_required"
    assert exc_info.value.operation == "cli.query"
    assert "polylogued run" in exc_info.value.format_message()
