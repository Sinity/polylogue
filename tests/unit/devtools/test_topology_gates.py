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
