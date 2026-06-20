"""CLI tests for neighboring-session discovery via ``read --view neighbors``.

The standalone ``neighbors`` command was absorbed into the read-view surface
(#1842): ``find <seed> then read --view neighbors``. These tests exercise the
read-view handler directly (the cli-app path builds its own env, so the
neighbor backend is injected through a mock env here).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.archive.models import SessionSummary
from polylogue.archive.session.neighbor_candidates import NeighborReason, SessionNeighborCandidate
from polylogue.cli.read_view_handlers import ReadViewInvocation, ReadViewNeighborOptions, _run_read_neighbors
from polylogue.cli.root_request import RootModeRequest
from polylogue.core.enums import Origin
from polylogue.types import SessionId


def _candidate() -> SessionNeighborCandidate:
    return SessionNeighborCandidate(
        summary=SessionSummary(
            id=SessionId("candidate"),
            origin=Origin.CODEX_SESSION,
            title="Archive Lock Retries",
            updated_at=datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc),
            message_count=2,
        ),
        rank=1,
        score=3.25,
        reasons=(
            NeighborReason(
                kind="same_title",
                detail="same normalized title",
                weight=3.0,
                evidence="message-1",
            ),
        ),
        source_session_id="target",
    )


def test_read_view_neighbors_emits_json_payload(capsys: pytest.CaptureFixture[str]) -> None:
    env = MagicMock()
    env.polylogue.neighbor_candidates = AsyncMock(return_value=[_candidate()])
    request = RootModeRequest.from_params({"origin": "codex-session"})

    _run_read_neighbors(
        env,
        request,
        ReadViewInvocation(
            view="neighbors",
            session_id="target",
            output_format="json",
            destination="stdout",
            out_path=None,
            neighbors=ReadViewNeighborOptions(limit=10, window_hours=24),
        ),
    )

    payload = json.loads(capsys.readouterr().out)
    neighbor = payload["result"]["neighbors"][0]
    assert neighbor["session"]["id"] == "candidate"
    assert neighbor["reasons"][0]["kind"] == "same_title"
    env.polylogue.neighbor_candidates.assert_called_once_with(
        session_id="target",
        query=None,
        provider="codex",
        limit=10,
        window_hours=24,
    )


def test_read_view_neighbors_plain_renders_reasons(capsys: pytest.CaptureFixture[str]) -> None:
    env = MagicMock()
    env.polylogue.neighbor_candidates = AsyncMock(return_value=[_candidate()])
    request = RootModeRequest.from_params({})

    _run_read_neighbors(
        env,
        request,
        ReadViewInvocation(
            view="neighbors",
            session_id="target",
            output_format=None,
            destination="stdout",
            out_path=None,
            neighbors=ReadViewNeighborOptions(limit=10, window_hours=24),
        ),
    )

    out = capsys.readouterr().out
    assert "Neighbor candidates (1):" in out
    assert "1. candidate" in out
    # reason rendering includes the kind and the evidence in parentheses
    assert "same_title: same normalized title (message-1)" in out


def test_read_view_neighbors_surfaces_discovery_error(capsys: pytest.CaptureFixture[str]) -> None:
    from polylogue.archive.session.neighbor_candidates import NeighborDiscoveryError

    env = MagicMock()
    env.polylogue.neighbor_candidates = AsyncMock(side_effect=NeighborDiscoveryError("no candidates"))
    request = RootModeRequest.from_params({})

    with pytest.raises(SystemExit):
        _run_read_neighbors(
            env,
            request,
            ReadViewInvocation(
                view="neighbors",
                session_id="target",
                output_format=None,
                destination="stdout",
                out_path=None,
                neighbors=ReadViewNeighborOptions(limit=10, window_hours=24),
            ),
        )

    assert "no candidates" in capsys.readouterr().err


def test_read_view_neighbors_empty_renders_message(capsys: pytest.CaptureFixture[str]) -> None:
    env = MagicMock()
    env.polylogue.neighbor_candidates = AsyncMock(return_value=[])
    request = RootModeRequest.from_params({})

    _run_read_neighbors(
        env,
        request,
        ReadViewInvocation(
            view="neighbors",
            session_id="target",
            output_format=None,
            destination="stdout",
            out_path=None,
            neighbors=ReadViewNeighborOptions(limit=10, window_hours=24),
        ),
    )

    assert "No neighboring candidates found." in capsys.readouterr().out


def test_read_view_neighbors_requires_a_seed(capsys: pytest.CaptureFixture[str]) -> None:
    env = MagicMock()
    request = RootModeRequest.from_params({})

    with pytest.raises(SystemExit):
        _run_read_neighbors(
            env,
            request,
            ReadViewInvocation(
                view="neighbors",
                session_id=None,
                output_format=None,
                destination="stdout",
                out_path=None,
                neighbors=ReadViewNeighborOptions(limit=10, window_hours=24),
            ),
        )

    assert "requires a seed" in capsys.readouterr().err
    env.polylogue.neighbor_candidates.assert_not_called()
