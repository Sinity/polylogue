from __future__ import annotations

import json

import pytest

from devtools import verify_schema_upgrade_lane


def test_schema_evolution_policy_lane_allows_durable_sql_migrations(capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_schema_upgrade_lane.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["upgrade_helpers"] == []
    assert payload["invalid_migration_resources"] == []
