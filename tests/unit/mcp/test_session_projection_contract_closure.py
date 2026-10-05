"""A session-list row binds every advertised read-view contract."""

from __future__ import annotations

import pytest

from polylogue.archive.session_projections import (
    SESSION_LIST_PROJECTIONS,
    SessionListProjection,
    bind_session_list_projection_contracts,
)
from tests.infra.session_projection_probe import run_projection_contract_probe


@pytest.mark.parametrize("retire_original", [False, True])
def test_projection_row_binds_profile_metadata_family_and_handler(retire_original: bool) -> None:
    """A new public name borrows real contracts; retired names leave all consumers."""
    result = run_projection_contract_probe(retire_original=retire_original)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "projection_contract_bound"


@pytest.mark.parametrize(
    "row,reason",
    [
        (SessionListProjection("unknown", "get_session_events", "events", "unknown"), "invalid session projection"),
        (SessionListProjection("summary", "get_session_events", "events", "events"), "collides with read view"),
    ],
)
def test_projection_contract_refuses_unknown_renderer_or_collision(
    monkeypatch: pytest.MonkeyPatch, row: SessionListProjection, reason: str
) -> None:
    monkeypatch.setitem(SESSION_LIST_PROJECTIONS, row.name, row)
    templates = {entry.cli_handler: entry.cli_handler for entry in SESSION_LIST_PROJECTIONS.values()}
    templates["summary"] = "summary"
    with pytest.raises(RuntimeError, match=reason):
        bind_session_list_projection_contracts(templates, lambda value, name: value)
