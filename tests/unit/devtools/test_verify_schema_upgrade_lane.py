from __future__ import annotations

import json

import pytest

from devtools import verify_schema_upgrade_lane
from polylogue.storage.sqlite.archive_tiers.index_convergence import BenignDDLEntry


def test_schema_evolution_policy_lane_allows_durable_sql_migrations(capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_schema_upgrade_lane.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["upgrade_helpers"] == []
    assert payload["invalid_migration_resources"] == []
    assert payload["invalid_benign_ddl_entries"] == []


def test_live_index_benign_ddl_registry_entries_are_all_valid() -> None:
    """The real registry (polylogue-v2mg's two drops) passes the policy check."""
    assert verify_schema_upgrade_lane._invalid_benign_ddl_entries() == []


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            BenignDDLEntry("bad_alter", "ALTER TABLE sessions ADD COLUMN bogus TEXT", "not idempotent"),
            id="alter-table",
        ),
        pytest.param(
            BenignDDLEntry("bad_insert", "INSERT INTO price_catalogs (catalog_id) VALUES ('x')", "data-transforming"),
            id="insert-into",
        ),
        pytest.param(
            BenignDDLEntry("bad_delete", "DELETE FROM session_model_usage", "data-transforming"),
            id="delete-from",
        ),
        pytest.param(
            BenignDDLEntry("bad_drop_no_guard", "DROP TABLE some_table", "non-idempotent drop"),
            id="drop-without-if-exists",
        ),
        pytest.param(
            BenignDDLEntry("bad_create_no_guard", "CREATE TABLE some_table (x TEXT)", "non-idempotent create"),
            id="create-without-if-not-exists",
        ),
        pytest.param(
            BenignDDLEntry(
                "bad_smuggled_second_statement",
                "DROP TABLE IF EXISTS some_table; DELETE FROM sessions",
                "multi-statement smuggling",
            ),
            id="multi-statement",
        ),
    ],
)
def test_policy_check_rejects_non_idempotent_benign_ddl_entries(entry: BenignDDLEntry) -> None:
    """Anti-vacuity: the lint must actually flag a deliberately bad entry, not just pass the real registry."""
    violations = verify_schema_upgrade_lane._invalid_benign_ddl_entries(entries=(entry,))
    assert len(violations) == 1
    assert violations[0].entry_name == entry.name
