"""Tests for topology/layering gate classification: blocking vs advisory.

These tests verify that:
  - verify topology blocks on orphans, missing, conflicts, kernel_rule
  - verify topology does NOT block on TBD (warning-only)
  - verify topology blocks on TBD with --strict-tbd
  - verify layering blocks on import-boundary violations
  - verify layering passes on clean imports
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from devtools import verify_layering, verify_topology

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_yaml(rows: list[dict[str, str | int]], path: Path) -> None:
    lines = ["files:"]
    for row in rows:
        lines.append(f"  - path: {row['path']}")
        for key in ("loc", "target", "owner", "reason"):
            if key in row:
                val = row[key]
                if isinstance(val, str) and any(c in val for c in ":'\""):
                    val = f'"{val}"'
                lines.append(f"    {key}: {val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_tmp_tree(tmp_path: Path, files: list[str]) -> set[str]:
    polylogue = tmp_path / "polylogue"
    polylogue.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = polylogue / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# placeholder\n", encoding="utf-8")
    return {f"polylogue/{f}" for f in files}


# ---------------------------------------------------------------------------
# verify topology: blocking classification
# ---------------------------------------------------------------------------


def test_topology_no_findings_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py", "types.py", "storage/__init__.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
        {"path": "polylogue/types.py", "target": "polylogue/types.py", "owner": "kernel"},
        {"path": "polylogue/storage/__init__.py", "target": "polylogue/storage/__init__.py", "owner": "stable"},
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    assert verify_topology.main(["--yaml", str(yaml_path)]) == 0


def test_orphan_is_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py", "orphan_file.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path)])
    assert rc == 1, "orphan file should be blocking"


def test_missing_is_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
        {"path": "polylogue/gone.py", "target": "polylogue/gone.py", "owner": "stable"},
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path)])
    assert rc == 1, "missing file should be blocking"


def test_conflict_is_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "stable"},
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path)])
    assert rc == 1, "duplicate path should be blocking"


def test_kernel_rule_is_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py", "rogue.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
        {
            "path": "polylogue/rogue.py",
            "target": "polylogue/rogue.py",
            "owner": "archive-query",
            "reason": "should not stay at root with non-kernel owner",
        },
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path)])
    assert rc == 1, "kernel_rule violation should be blocking"


def test_stable_owner_does_not_bypass_root_kernel_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py", "product_service.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
        {
            "path": "polylogue/product_service.py",
            "target": "polylogue/product_service.py",
            "owner": "stable",
            "reason": "a stable owner is not a kernel primitive",
        },
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path)])
    assert rc == 1, "stable ownership must not allow a product service at package root"


def test_tbd_is_warning_not_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py", "uncertain.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
        {"path": "polylogue/uncertain.py", "target": "TBD", "owner": "", "reason": "root file, no rule yet"},
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path)])
    assert rc == 0, "TBD alone (without --strict-tbd) should NOT be blocking"


def test_tbd_is_blocking_with_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["__init__.py", "uncertain.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
        {"path": "polylogue/uncertain.py", "target": "TBD", "owner": "", "reason": "root file, no rule yet"},
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path), "--strict-tbd"])
    assert rc == 1, "TBD with --strict-tbd should be blocking"


def test_json_output_marks_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys as _sys

    _make_tmp_tree(tmp_path, ["__init__.py", "orphan.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {"path": "polylogue/__init__.py", "target": "polylogue/__init__.py", "owner": "kernel"},
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    buf = io.StringIO()
    monkeypatch.setattr(_sys, "stdout", buf)
    rc = verify_topology.main(["--yaml", str(yaml_path), "--json"])
    out = json.loads(buf.getvalue())
    assert rc == 1
    assert out["blocking"] is True
    assert out["counts"]["orphans"] == 1
    assert out["counts"]["tbd"] == 0


def test_blob_gc_not_tbd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_tmp_tree(tmp_path, ["storage/blob_gc.py"])
    yaml_rows: list[dict[str, str | int]] = [
        {
            "path": "polylogue/storage/blob_gc.py",
            "target": "polylogue/storage/blob_gc.py",
            "owner": "storage-root",
            "loc": 241,
            "reason": "storage-root cross-cutting helper",
        },
    ]
    yaml_path = tmp_path / "topology-target.yaml"
    _write_yaml(yaml_rows, yaml_path)

    monkeypatch.setattr(verify_topology, "ROOT", tmp_path)
    rc = verify_topology.main(["--yaml", str(yaml_path)])
    assert rc == 0, "blob_gc.py with storage-root owner and concrete target should pass"


# ---------------------------------------------------------------------------
# verify layering: blocking classification
# ---------------------------------------------------------------------------


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


def test_layering_cli_imports_storage_is_ok(tmp_path: Path) -> None:
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
