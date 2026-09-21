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
_MUTATION_EXECUTOR = "polylogue.operations.mutation_transaction"
_ANNOTATION_IMPORT_WRITER = "polylogue.annotations.importer"


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
    for baseline in (
        "layering-surface-baseline.json",
        "layering-cli-mutation-authority-baseline.json",
    ):
        shutil.copy(root / "docs" / "plans" / baseline, plans / baseline)
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


def test_new_cli_module_driving_the_mutation_executor_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The daemon is the sole writer: a CLI module cannot hold a second one.

    Anti-vacuity: dropping the mutation-authority rule from ``layering.yaml``,
    or adding ``polylogue/cli`` wholesale to its baseline, makes this pass a
    gate it must fail.
    """
    _shipped_manifest_tree(
        tmp_path,
        extra_modules={_SCRATCH_MODULE: f"from {_MUTATION_EXECUTOR} import OperationExecutor\n"},
    )
    reported = _violations(tmp_path, monkeypatch, capsys)
    assert any(
        entry.get("file") == _SCRATCH_MODULE and entry.get("import") == _MUTATION_EXECUTOR for entry in reported
    ), reported


def test_query_mutation_route_is_absent_from_the_mutation_authority_baseline() -> None:
    """The root query's write route gave up its own executor; it may not return."""
    baseline = json.loads(
        (repo_root() / "docs" / "plans" / "layering-cli-mutation-authority-baseline.json").read_text()
    )
    assert all(entry["file"] != "polylogue/cli/archive_query.py" for entry in baseline), baseline


def test_new_cli_module_driving_the_executor_indirectly_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CLI write is a violation however it spells its way to the executor.

    ``polylogue.annotations.importer`` is not an executor module; it *drives*
    one, constructing ``OperationExecutor`` and running prepare/authorize/
    execute against ``user.db``. Until the mutation-authority rule named it,
    ``polylogue annotations import`` -- a live CLI in-process durable writer --
    produced zero violations and zero baseline entries, so an empty result read
    as an empty behaviour (polylogue-gjwto).

    Anti-vacuity: drop ``polylogue/annotations/importer`` from the CLI rule's
    ``disallow.from`` list in ``docs/plans/layering.yaml`` and this is red,
    because the gate goes back to reporting nothing for a module that writes a
    durable tier from the CLI process.
    """
    _shipped_manifest_tree(
        tmp_path,
        extra_modules={_SCRATCH_MODULE: f"from {_ANNOTATION_IMPORT_WRITER} import import_annotation_batch\n"},
    )
    reported = _violations(tmp_path, monkeypatch, capsys)
    assert any(
        entry.get("file") == _SCRATCH_MODULE and entry.get("import") == _ANNOTATION_IMPORT_WRITER for entry in reported
    ), reported


def test_every_mutation_authority_baseline_entry_carries_a_reviewed_reason() -> None:
    """Debt that survives is explained, not merely tolerated (polylogue-r29bv AC5).

    The loader keys on ``(target, file, import)`` and ignores everything else,
    so a ``reason`` cannot be enforced by the gate itself. It is enforced here:
    an entry added without one is a silent exemption, which is the exact shape
    the extension of this rule exists to remove.

    Anti-vacuity: delete any entry's ``reason``, or replace it with a blank
    string, and this is red.
    """
    baseline = json.loads(
        (repo_root() / "docs" / "plans" / "layering-cli-mutation-authority-baseline.json").read_text()
    )
    assert baseline, "an empty baseline would make this vacuous"
    for entry in baseline:
        reason = entry.get("reason")
        assert isinstance(reason, str) and reason.strip(), entry
        # A reason that does not name what unblocks it is a restatement, not a
        # review: every surviving entry here is blocked on a named predecessor.
        assert "Reviewed" in reason, entry
