"""Tests for layering gate classification: blocking vs advisory.

These tests verify that:
  - verify layering blocks on import-boundary violations
  - verify layering passes on clean imports
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from devtools import verify_layering


def test_layering_no_violations_passes(tmp_path: Path) -> None:
    storage = tmp_path / "polylogue" / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "module.py").write_text("import os\nfrom polylogue.core import json\n", encoding="utf-8")

    imports = verify_layering._collect_imports(storage, repo_root=tmp_path)
    assert "polylogue.cli" not in imports.get("polylogue/storage/module.py", set())


def test_layering_disallow_violation_detected(tmp_path: Path) -> None:
    storage = tmp_path / "polylogue" / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "bad_importer.py").write_text("from polylogue.cli import click_app\n", encoding="utf-8")

    cli = tmp_path / "polylogue" / "cli"
    cli.mkdir(parents=True, exist_ok=True)
    (cli / "click_app.py").write_text("", encoding="utf-8")

    imports = verify_layering._collect_imports(storage, repo_root=tmp_path)
    # from polylogue.cli import click_app -> module = "polylogue.cli"
    assert "polylogue.cli" in imports.get("polylogue/storage/bad_importer.py", set()), "storage imports cli module"

    rules: list[dict[str, Any]] = [
        {
            "target": "polylogue/storage",
            "description": "Storage substrate.",
            "disallow": {
                "from": ["polylogue/cli", "polylogue/mcp", "polylogue/daemon", "polylogue/ui", "polylogue/rendering"]
            },
        }
    ]

    violations: list[dict[str, object]] = []
    for rule in rules:
        target = str(rule["target"])
        target_dir = tmp_path / target
        disallow_from = list(rule.get("disallow", {}).get("from", []))
        file_imports = verify_layering._collect_imports(target_dir, repo_root=tmp_path)
        for file_rel, file_imports_set in file_imports.items():
            for imp in file_imports_set:
                if not imp.startswith("polylogue"):
                    continue
                for disallowed in disallow_from:
                    if verify_layering._package_matches(str(disallowed), imp):
                        violations.append({"file": file_rel, "import": imp, "disallowed": disallowed})

    assert len(violations) >= 1, "storage importing cli should produce violation"


def test_load_baseline_missing_file_returns_empty(tmp_path: Path) -> None:
    assert verify_layering._load_baseline(tmp_path / "does-not-exist.json") == set()


def test_load_baseline_parses_valid_entries_and_skips_malformed(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            [
                {"target": "polylogue/cli", "file": "polylogue/cli/x.py", "import": "polylogue.storage.y"},
                {"target": "polylogue/mcp", "file": "polylogue/mcp/z.py"},  # missing "import" -- skipped
                "not-a-dict",  # skipped
            ]
        ),
        encoding="utf-8",
    )
    entries = verify_layering._load_baseline(baseline_path)
    assert entries == {("polylogue/cli", "polylogue/cli/x.py", "polylogue.storage.y")}


def _write_ratchet_fixture(tmp_path: Path, *, baseline_entries: list[dict[str, str]] | None) -> None:
    """Build a minimal repo with one pre-existing cli->storage import and a
    ratcheted disallow rule, optionally seeded with a baseline."""
    cli_dir = tmp_path / "polylogue" / "cli"
    cli_dir.mkdir(parents=True, exist_ok=True)
    (cli_dir / "commands.py").write_text("from polylogue.storage import archive_identity\n", encoding="utf-8")
    (tmp_path / "polylogue" / "storage").mkdir(parents=True, exist_ok=True)

    plans_dir = tmp_path / "docs" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    baseline_ref = "docs/plans/ratchet-baseline.json"
    if baseline_entries is not None:
        (tmp_path / baseline_ref).write_text(json.dumps(baseline_entries), encoding="utf-8")

    rules_yaml = f"""\
rules:
  - target: polylogue/cli
    description: test fixture
    disallow:
      from: [polylogue/storage]
      baseline: {baseline_ref}
"""
    (plans_dir / "layering.yaml").write_text(rules_yaml, encoding="utf-8")


def test_layering_ratchet_exempts_baselined_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_ratchet_fixture(
        tmp_path,
        baseline_entries=[
            {"target": "polylogue/cli", "file": "polylogue/cli/commands.py", "import": "polylogue.storage"},
        ],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    assert verify_layering.main([]) == 0


def test_layering_ratchet_fails_on_violation_not_in_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_ratchet_fixture(tmp_path, baseline_entries=[])
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    assert verify_layering.main([]) == 1


def test_layering_ratchet_reports_stale_baseline_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_ratchet_fixture(
        tmp_path,
        baseline_entries=[
            {"target": "polylogue/cli", "file": "polylogue/cli/commands.py", "import": "polylogue.storage"},
            # This entry no longer reproduces (no such file/import exists) --
            # it should be flagged as prunable without failing the gate.
            {"target": "polylogue/cli", "file": "polylogue/cli/gone.py", "import": "polylogue.storage.gone"},
        ],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    exit_code = verify_layering.main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "1 baseline entr" in out


def test_layering_cli_imports_storage_is_detected(tmp_path: Path) -> None:
    # polylogue-2ciy: cli->storage is no longer unconditionally "ok" -- the
    # production rule now disallows it too (behind a ratchet baseline). This
    # test only pins that `_collect_imports` itself surfaces the import; see
    # the baseline tests below for the ratchet's pass/fail behavior.
    cli_dir = tmp_path / "polylogue" / "cli"
    cli_dir.mkdir(parents=True, exist_ok=True)
    (cli_dir / "commands.py").write_text("from polylogue.storage import something\n", encoding="utf-8")

    imports = verify_layering._collect_imports(cli_dir, repo_root=tmp_path)
    # from polylogue.storage import something -> module = "polylogue.storage"
    assert "polylogue.storage" in imports.get("polylogue/cli/commands.py", set())


def test_package_matches_exact_and_prefix() -> None:
    assert verify_layering._package_matches("polylogue/cli", "polylogue.cli.click_app") is True
    assert verify_layering._package_matches("polylogue/cli", "polylogue.cli") is True
    assert verify_layering._package_matches("polylogue/cli", "polylogue.storage") is False
    assert verify_layering._package_matches("polylogue/cli", "polylogue.cliclone") is False


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARCHIVE_TIERS_RELATIVE = Path("polylogue/storage/sqlite/archive_tiers")


def _production_writer_policy() -> verify_layering.WriterModulePolicy:
    manifest = verify_layering._load_manifest(_REPO_ROOT / "docs/plans/layering.yaml")
    policy = verify_layering._writer_module_policy(manifest)
    assert policy is not None
    return policy


def _copy_production_writer_surface(tmp_path: Path) -> Path:
    destination = tmp_path / _ARCHIVE_TIERS_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_REPO_ROOT / _ARCHIVE_TIERS_RELATIVE, destination)
    return destination


def test_layering_production_writer_inventory_passes() -> None:
    policy = _production_writer_policy()

    assert verify_layering._collect_writer_module_violations(_REPO_ROOT, policy) == []


def test_layering_unmarked_production_writer_mutation_fails(tmp_path: Path) -> None:
    writer_root = _copy_production_writer_surface(tmp_path)
    source_writer = writer_root / "source_write.py"
    source_writer.write_text(
        source_writer.read_text(encoding="utf-8").replace("Writer module: source.\n", ""),
        encoding="utf-8",
    )

    violations = verify_layering._collect_writer_module_violations(tmp_path, _production_writer_policy())

    assert any(
        violation["file"] == "polylogue/storage/sqlite/archive_tiers/source_write.py"
        and violation["rule"] == "writer_module_unmarked_mutation"
        for violation in violations
    )


def test_layering_user_ops_mutation_fails_without_a_twin_write_contract(tmp_path: Path) -> None:
    writer_root = _copy_production_writer_surface(tmp_path)
    user_writer = writer_root / "user_write.py"
    user_writer.write_text(
        user_writer.read_text(encoding="utf-8")
        + "\n\ndef upsert_ops_control_plane(conn: sqlite3.Connection) -> None:\n"
        + '    conn.execute("INSERT INTO ingest_cursor (source_path, updated_at_ms) VALUES (?, ?)", ("test", 0))\n',
        encoding="utf-8",
    )

    violations = verify_layering._collect_writer_module_violations(tmp_path, _production_writer_policy())

    assert any(
        violation["file"] == "polylogue/storage/sqlite/archive_tiers/user_write.py"
        and violation["rule"] == "writer_module_observed_tier_mismatch"
        for violation in violations
    )


def test_layering_delegated_public_writer_is_inventoried(tmp_path: Path) -> None:
    writer_root = _copy_production_writer_surface(tmp_path)
    source_writer = writer_root / "source_write.py"
    source_writer.write_text(
        source_writer.read_text(encoding="utf-8")
        + "\n\ndef publish_raw_revision(conn: sqlite3.Connection) -> None:\n"
        + "    _publish_raw_revision(conn)\n\n"
        + "def _publish_raw_revision(conn: sqlite3.Connection) -> None:\n"
        + '    conn.execute("UPDATE raw_sessions SET parsed_at_ms = 0")\n',
        encoding="utf-8",
    )

    violations = verify_layering._collect_writer_module_violations(tmp_path, _production_writer_policy())

    mismatch = next(
        violation
        for violation in violations
        if violation["file"] == "polylogue/storage/sqlite/archive_tiers/source_write.py"
        and violation["rule"] == "writer_module_entrypoint_inventory_mismatch"
    )
    observed = mismatch.get("observed")
    assert isinstance(observed, list)
    assert "publish_raw_revision" in observed


def test_layering_imported_sql_cannot_hide_a_mutation(tmp_path: Path) -> None:
    writer_root = _copy_production_writer_surface(tmp_path)
    source_writer = writer_root / "source_write.py"
    source_writer.write_text(
        source_writer.read_text(encoding="utf-8")
        + "\nfrom tests.fixtures.sql import HIDDEN_MUTATION_SQL\n\n"
        + "def run_hidden_mutation(conn: sqlite3.Connection) -> None:\n"
        + "    conn.execute(HIDDEN_MUTATION_SQL)\n",
        encoding="utf-8",
    )

    violations = verify_layering._collect_writer_module_violations(tmp_path, _production_writer_policy())

    assert any(
        violation["file"] == "polylogue/storage/sqlite/archive_tiers/source_write.py"
        and violation["rule"] == "writer_module_imported_sql_opaque"
        for violation in violations
    )
