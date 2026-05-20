"""Tests for the in-place schema-upgrade verification lane (#1302).

These tests pin two behaviours of ``devtools verify-schema-upgrade-lane``:

* In the current steady state (no in-place upgrade helpers committed),
  the lane passes cleanly. This is the user-visible contract that the
  lint does not block normal work as long as fresh-first policy holds.

* When a migration-shaped helper exists in a scanned directory without a
  paired driving test that references it by name, the lint fails. That
  asymmetric direction is the one that catches the policy violation
  scenario the issue (#1302) was filed to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import verify_schema_upgrade_lane as lane


def test_steady_state_passes_cleanly() -> None:
    """No in-place upgrade helpers committed today → lint exits 0."""
    helpers = lane._collect_upgrade_helpers()
    test_text = lane._collect_migration_test_text()
    missing = [hit for hit in helpers if hit.name not in test_text]
    assert missing == [], (
        f"Unexpected in-place upgrade helpers without paired driving tests: {[hit.name for hit in missing]}"
    )


def test_lint_main_returns_zero_today() -> None:
    rc = lane.main([])
    assert rc == 0


def test_helper_name_pattern_matches_historical_shapes() -> None:
    """The pattern set must recognise every historically-used shape."""
    for name in (
        "build_v3_to_v4",
        "apply_version_upgrade_plan",
        "_apply_version_upgrade",
        "upgrade_v5_to_v6",
        "_upgrade_v5_to_v6",
        "migrate_v7",
        "migrate_v7_to_v8",
        "ensure_schema_upgrades_v12",
    ):
        assert lane._is_helper_name(name), f"expected {name!r} to be recognised"


def test_helper_name_pattern_ignores_unrelated_functions() -> None:
    for name in (
        "ensure_schema",
        "decide_schema_bootstrap",
        "open_archive",
        "rebuild_fts_index",
        "validate_schema_version",
    ):
        assert not lane._is_helper_name(name), f"expected {name!r} to be ignored"


def test_lint_flags_unpaired_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If a helper exists in the scanned dir but no test references it, lint fails."""
    scan_dir = tmp_path / "polylogue" / "storage" / "sqlite"
    scan_dir.mkdir(parents=True)
    (scan_dir / "schema_upgrades.py").write_text(
        "def build_v9_to_v10(conn):\n    pass\n",
        encoding="utf-8",
    )

    test_dir = tmp_path / "tests" / "unit" / "storage" / "migrations"
    test_dir.mkdir(parents=True)
    # README only — no driving test for the new helper.
    (test_dir / "README.md").write_text("policy", encoding="utf-8")

    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(lane, "STORAGE_SQLITE_DIR", scan_dir)
    monkeypatch.setattr(lane, "MIGRATIONS_TEST_DIR", test_dir)

    rc = lane.main([])
    assert rc == 1


def test_lint_accepts_helper_with_named_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A helper paired with a test that references it by name → lint passes."""
    scan_dir = tmp_path / "polylogue" / "storage" / "sqlite"
    scan_dir.mkdir(parents=True)
    (scan_dir / "schema_upgrades.py").write_text(
        "def build_v9_to_v10(conn):\n    pass\n",
        encoding="utf-8",
    )

    test_dir = tmp_path / "tests" / "unit" / "storage" / "migrations"
    test_dir.mkdir(parents=True)
    (test_dir / "test_build_v9_to_v10.py").write_text(
        "def test_build_v9_to_v10_round_trip():\n    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(lane, "STORAGE_SQLITE_DIR", scan_dir)
    monkeypatch.setattr(lane, "MIGRATIONS_TEST_DIR", test_dir)

    rc = lane.main([])
    assert rc == 0


def test_json_output_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    rc = lane.main(["--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert rc == 0
    assert "upgrade_helpers" in payload
    assert "missing_driving_tests" in payload
    assert "migrations_test_dir_present" in payload
