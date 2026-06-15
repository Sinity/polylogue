"""Regression coverage for the shared latest-resolver helper (#1626, #1642).

Verifies the resolution rules apply uniformly:
explicit conv_id wins, then ``--latest`` / any narrowing filter, then
``None``. The single-session surfaces (``read --view messages``/``raw``/
``neighbors``, ``export``, ``diagnostics turns``) all route through
this helper, so a single test here pins the contract for all of them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params


def test_explicit_conv_id_wins_over_filters() -> None:
    """An explicit conv_id short-circuits — no query runs."""
    result = resolve_session_id_from_root_params({"conv_id": "claude-code:explicit", "latest": True})
    assert result == "claude-code:explicit"


def test_no_filters_returns_none() -> None:
    """Empty params returns None — caller surfaces its own missing-id error."""
    assert resolve_session_id_from_root_params({}) is None


def test_latest_runs_spec_with_limit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--latest`` runs the spec with limit=1 and returns the top match."""
    captured_limits: list[int | None] = []

    async def fake_list_summaries(self: object, repo: object) -> list[SimpleNamespace]:
        captured_limits.append(getattr(self, "limit", None))
        return [SimpleNamespace(id="claude-code:latest-conv")]

    monkeypatch.setattr(
        "polylogue.archive.query.spec.SessionQuerySpec.list_summaries",
        fake_list_summaries,
    )

    class _API:
        config = SimpleNamespace()

        async def __aenter__(self) -> _API:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr("polylogue.api.Polylogue.open", lambda **_: _API())

    result = resolve_session_id_from_root_params({"latest": True})

    assert result == "claude-code:latest-conv"
    assert captured_limits == [1]


def test_filter_alone_resolves_when_match_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """A narrowing filter (provider, since, etc.) also triggers resolution."""

    async def fake_list_summaries(self: object, repo: object) -> list[SimpleNamespace]:
        return [SimpleNamespace(id="codex:filtered")]

    monkeypatch.setattr(
        "polylogue.archive.query.spec.SessionQuerySpec.list_summaries",
        fake_list_summaries,
    )

    class _API:
        config = SimpleNamespace()

        async def __aenter__(self) -> _API:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr("polylogue.api.Polylogue.open", lambda **_: _API())

    assert resolve_session_id_from_root_params({"origin": "codex-session"}) == "codex:filtered"


def test_latest_returns_none_when_archive_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--latest`` against an empty archive returns None, not an error."""

    async def fake_list_summaries(self: object, repo: object) -> list[SimpleNamespace]:
        return []

    monkeypatch.setattr(
        "polylogue.archive.query.spec.SessionQuerySpec.list_summaries",
        fake_list_summaries,
    )

    class _API:
        config = SimpleNamespace()

        async def __aenter__(self) -> _API:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr("polylogue.api.Polylogue.open", lambda **_: _API())

    assert resolve_session_id_from_root_params({"latest": True}) is None
