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

import pytest

from polylogue.cli.operation_kernel import OperationUnavailableError
from polylogue.cli.shared.helper_support import DaemonRequiredError
from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params


def _stub_ids(monkeypatch: pytest.MonkeyPatch, ids: Sequence[str], captured_limits: list[int] | None = None) -> None:
    def _query_session_ids(config: object, request: object, *, limit: int, **_kwargs: object) -> list[str]:
        if captured_limits is not None:
            captured_limits.append(limit)
        return list(ids)

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_ids", _query_session_ids)


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
    _stub_ids(monkeypatch, ["codex:filtered"])

    assert resolve_session_id_from_root_params({"origin": "codex-session"}) == "codex:filtered"


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
