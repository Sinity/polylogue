"""Plain raw-authority frontier output labels its all-state count honestly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.maintenance._raw_identity import raw_authority_frontier_command


def test_raw_frontier_labels_all_plans_not_outstanding_obligations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails if the all-state plan count is labelled as outstanding obligations."""
    from polylogue.maintenance import raw_authority

    payload = {
        "pass_id": "w12-pass",
        "accepted_head_count": 1,
        "plan_count": 1,
        "state_counts": {"discharged": 1},
    }
    monkeypatch.setattr(raw_authority, "inspect_frontier", lambda _config: SimpleNamespace(to_dict=lambda: payload))
    result = CliRunner().invoke(raw_authority_frontier_command, [], obj=SimpleNamespace(config=object()))
    assert result.exit_code == 0, result.output
    assert "plans=1" in result.output
    assert "obligations=" not in result.output
