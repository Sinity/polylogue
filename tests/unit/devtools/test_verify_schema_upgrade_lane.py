from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_schema_upgrade_lane
from polylogue.storage.sqlite.archive_tiers.index_convergence import BenignDDLEntry


def test_schema_evolution_policy_lane_allows_durable_sql_migrations(capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_schema_upgrade_lane.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["invalid_migration_resources"] == []
    assert payload["invalid_benign_ddl_entries"] == []


def test_live_index_benign_ddl_registry_entries_are_all_valid() -> None:
    """The real registry (polylogue-v2mg's two drops + polylogue-resk's price_catalogs drop) passes the policy check."""
    assert verify_schema_upgrade_lane._invalid_benign_ddl_entries() == []


def test_ddl_lifecycle_gate_rejects_deleted_create_table_without_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: removing a source DDL object must make the gate red."""
    source_path = "polylogue/storage/sqlite/archive_tiers/source.py"
    current_source = (Path(__file__).parents[3] / source_path).read_text(encoding="utf-8")
    removed_line = "CREATE TABLE IF NOT EXISTS history_sidecars ("
    assert removed_line in current_source
    previous_source = current_source.replace(removed_line, "", 1)
    patch = f"diff --git a/{source_path} b/{source_path}\n@@ -1 +1 @@\n-{removed_line}\n+"

    monkeypatch.setattr(verify_schema_upgrade_lane, "_diff_base", lambda: "base")

    def fake_git_text(*args: str) -> str:
        if args[:2] == ("diff", "--no-ext-diff"):
            return patch
        if args[:2] == ("diff", "--name-status"):
            return ""
        if args[:1] == ("show",) and args[1].endswith(source_path):
            return previous_source
        if args[:1] == ("show",):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(verify_schema_upgrade_lane, "_git_text", fake_git_text)

    violations = verify_schema_upgrade_lane._ddl_lifecycle_report()
    assert [(item.tier, item.path) for item in violations] == [("source", source_path)]
    assert "schema-version bump" in violations[0].reason


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
