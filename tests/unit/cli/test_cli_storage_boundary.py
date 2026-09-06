"""The checked-in ratchet cannot be widened by adding a CLI module.

``tests/unit/devtools/test_layering_gates.py`` proves the ratchet mechanism on
a synthetic manifest.  These tests bind the same guarantee to the manifest and
baseline the repository actually ships: a new CLI module that imports a storage
writer is a violation because no baseline entry names it, and the entry that
does exempt a real importer exempts that exact triple only.

Anti-vacuity: replacing an exact baseline triple with any prefix or wildcard
match, or exempting ``polylogue/cli`` wholesale, makes
:func:`test_new_cli_module_importing_a_storage_writer_fails_the_gate` pass a
gate it must fail.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from devtools import repo_root, verify_layering

_SCRATCH_MODULE = "polylogue/cli/scratch_storage_writer_importer.py"
_STORAGE_WRITER = "polylogue.storage.sqlite.archive_tiers.user_write"


def _shipped_manifest_tree(tmp_path: Path, *, extra_modules: dict[str, str]) -> Path:
    """Mirror the shipped layering manifest and baseline over a scratch tree.

    Every package the manifest names has to exist or the gate fails closed on
    the missing root before it evaluates a single import.
    """
    import yaml

    root = repo_root()
    plans = tmp_path / "docs" / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "docs" / "plans" / "layering.yaml", plans / "layering.yaml")
    shutil.copy(
        root / "docs" / "plans" / "layering-surface-baseline.json",
        plans / "layering-surface-baseline.json",
    )
    manifest = yaml.safe_load((plans / "layering.yaml").read_text(encoding="utf-8"))
    for rule in manifest.get("rules", []):
        declared = [rule["target"], *rule.get("disallow", {}).get("from", [])]
        for package in declared:
            if package.startswith("polylogue"):
                (tmp_path / package).mkdir(parents=True, exist_ok=True)
    for relative, source in extra_modules.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return tmp_path


def _violations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> list[dict[str, object]]:
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    verify_layering.main(["--json"])
    return list(json.loads(capsys.readouterr().out)["violations"])


def test_new_cli_module_importing_a_storage_writer_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CLI module the baseline never named is a violation, not inherited debt."""
    _shipped_manifest_tree(tmp_path, extra_modules={_SCRATCH_MODULE: f"from {_STORAGE_WRITER} import upsert_setting\n"})
    reported = _violations(tmp_path, monkeypatch, capsys)
    assert any(entry.get("file") == _SCRATCH_MODULE and entry.get("import") == _STORAGE_WRITER for entry in reported), (
        reported
    )


def test_baselined_importer_is_exempt_at_its_exact_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shipped baseline still exempts the importer it was recorded for."""
    baseline = json.loads((repo_root() / "docs" / "plans" / "layering-surface-baseline.json").read_text())
    exempt = next(entry for entry in baseline if entry["target"] == "polylogue/cli")
    _shipped_manifest_tree(
        tmp_path,
        extra_modules={exempt["file"]: f"from {exempt['import']} import anything\n"},
    )
    reported = _violations(tmp_path, monkeypatch, capsys)
    assert not any(entry.get("file") == exempt["file"] for entry in reported), reported


def test_baselined_importer_is_not_exempt_at_a_different_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exemption is per (file, import) triple, not per file."""
    baseline = json.loads((repo_root() / "docs" / "plans" / "layering-surface-baseline.json").read_text())
    exempt = next(entry for entry in baseline if entry["target"] == "polylogue/cli")
    assert exempt["import"] != _STORAGE_WRITER
    _shipped_manifest_tree(
        tmp_path,
        extra_modules={exempt["file"]: f"from {_STORAGE_WRITER} import upsert_setting\n"},
    )
    reported = _violations(tmp_path, monkeypatch, capsys)
    assert any(entry.get("file") == exempt["file"] and entry.get("import") == _STORAGE_WRITER for entry in reported), (
        reported
    )
