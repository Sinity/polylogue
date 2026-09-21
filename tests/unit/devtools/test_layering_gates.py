"""Tests for layering gate classification: blocking vs advisory.

These tests verify that:
  - verify layering blocks on import-boundary violations
  - verify layering passes on clean imports
"""

from __future__ import annotations

import ast
import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from devtools import required_gate, verify_layering


def test_layering_no_violations_passes(tmp_path: Path) -> None:
    storage = tmp_path / "polylogue" / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "module.py").write_text("import os\nfrom polylogue.core import json\n", encoding="utf-8")

    imports, unreadable = verify_layering._collect_imports(storage, repo_root=tmp_path)
    assert unreadable == ()
    assert "polylogue.cli" not in imports.get("polylogue/storage/module.py", set())


def test_layering_disallow_violation_detected(tmp_path: Path) -> None:
    storage = tmp_path / "polylogue" / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "bad_importer.py").write_text("from polylogue.cli import click_app\n", encoding="utf-8")

    cli = tmp_path / "polylogue" / "cli"
    cli.mkdir(parents=True, exist_ok=True)
    (cli / "click_app.py").write_text("", encoding="utf-8")

    imports, unreadable = verify_layering._collect_imports(storage, repo_root=tmp_path)
    assert unreadable == ()
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
        file_imports, unreadable = verify_layering._collect_imports(target_dir, repo_root=tmp_path)
        assert unreadable == ()
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


def test_layering_ratchet_reports_semantic_violation_in_human_and_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_ratchet_fixture(tmp_path, baseline_entries=[])
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    assert verify_layering.main([]) == 1
    assert "imports polylogue.storage (disallow)" in capsys.readouterr().out
    assert verify_layering.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["required_gate"]["diagnosis"] == "gate_semantic_violation"
    assert payload["required_gate"]["semantic_violation_count"] == 1
    assert payload["required_gate"]["error_count"] == 0


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

    imports, unreadable = verify_layering._collect_imports(cli_dir, repo_root=tmp_path)
    assert unreadable == ()
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


def test_top_level_package_docstring_inventory_passes_for_production_tree() -> None:
    assert verify_layering._top_level_package_docstring_violations(_REPO_ROOT) == []


def test_top_level_package_docstring_inventory_rejects_missing_docstring(tmp_path: Path) -> None:
    package = tmp_path / "polylogue" / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from __future__ import annotations\n", encoding="utf-8")

    namespace_package = tmp_path / "polylogue" / "rendering"
    namespace_package.mkdir()
    (namespace_package / "formatter.py").write_text('"""Formatter."""\n', encoding="utf-8")

    assert verify_layering._top_level_package_docstring_violations(tmp_path) == [
        {
            "file": "polylogue/example/__init__.py",
            "rule": "package_docstring_missing",
        }
    ]


def test_layering_main_fails_closed_on_missing_package_docstring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "polylogue" / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    plans = tmp_path / "docs" / "plans"
    plans.mkdir(parents=True)
    (plans / "layering.yaml").write_text("rules: []\n", encoding="utf-8")

    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    assert verify_layering.main([]) == 1
    assert "polylogue/example/__init__.py: package_docstring_missing" in capsys.readouterr().out


def test_layering_main_fails_closed_on_missing_declared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plans = tmp_path / "docs" / "plans"
    plans.mkdir(parents=True)
    (plans / "layering.yaml").write_text(
        "rules:\n  - target: polylogue/renamed\n    description: renamed root\n    disallow:\n      from: [polylogue/cli]\n",
        encoding="utf-8",
    )
    (tmp_path / "polylogue").mkdir()
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    assert verify_layering.main([]) == 1
    output = capsys.readouterr().out
    assert "polylogue/renamed: declared_root_missing" in output
    assert '"required_gate"' not in output
    assert verify_layering.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["required_gate"]["diagnosis"] == "gate_missing_input"
    assert payload["required_gate"]["missing_count"] == 1


def test_layering_main_fails_closed_on_unreadable_declared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plans = tmp_path / "docs" / "plans"
    plans.mkdir(parents=True)
    (plans / "layering.yaml").write_text(
        "rules:\n  - target: polylogue/example\n    description: example root\n    disallow: {}\n",
        encoding="utf-8",
    )
    root = tmp_path / "polylogue" / "example"
    root.mkdir(parents=True)
    (root / "module.py").write_text('"""module"""\n', encoding="utf-8")
    original = Path.read_text

    def unreadable(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == root / "module.py":
            raise OSError("synthetic unreadable input")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    assert verify_layering.main([]) == 1
    assert "declared_root_unreadable" in capsys.readouterr().out
    assert verify_layering.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["required_gate"]["diagnosis"] == "gate_unreadable_input"
    assert payload["required_gate"]["unreadable_count"] == 1


def test_mutation_scanner_distinguishes_replace_projection_from_replace_into() -> None:
    tree = ast.parse(
        '''

def read_projection(conn):
    conn.execute("""
        SELECT
            REPLACE(path, '/', '') AS normalized_path
        FROM files
    """)


def write_row(conn):
    conn.execute("""
        REPLACE INTO files(path) VALUES (?)
    """)
'''
    )

    calls = verify_layering._mutation_calls(tree)

    assert len(calls) == 1
    assert verify_layering._mutation_table(verify_layering._mutation_sql(calls[0], values={}) or "") == "files"


def _census_policy(tmp_path: Path, baseline: list[dict[str, object]]) -> verify_layering.WriterModulePolicy:
    (tmp_path / "docs" / "plans").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "plans" / "census.json").write_text(json.dumps(baseline), encoding="utf-8")
    return dataclasses.replace(
        _production_writer_policy(),
        census_roots=("polylogue",),
        census_baseline="docs/plans/census.json",
    )


def _write_census_module(tmp_path: Path, rel: str, sql: str) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'import sqlite3\n\n\ndef write(conn: sqlite3.Connection) -> None:\n    conn.execute("{sql}")\n',
        encoding="utf-8",
    )


def test_layering_census_flags_a_new_out_of_inventory_write_path(tmp_path: Path) -> None:
    """A DML module outside the writer inventory must be censused or fail.

    Anti-vacuity: delete the census wiring (or add this file to the baseline)
    and the assertion goes green while the write path stays unpoliced.
    """
    _write_census_module(tmp_path, "polylogue/ops/rogue_writer.py", "INSERT INTO sessions (native_id) VALUES (?)")

    violations = verify_layering._collect_writer_module_census_violations(tmp_path, _census_policy(tmp_path, []))

    assert [
        violation["file"] for violation in violations if violation["rule"] == "writer_module_uncensused_mutation"
    ] == ["polylogue/ops/rogue_writer.py"]


def test_layering_census_baseline_entry_that_stopped_mutating_is_stale(tmp_path: Path) -> None:
    """The ratchet can only shrink: a retired entry must be removed from the file.

    Anti-vacuity: drop the stale-entry arm and a baseline keeps exempting a
    path forever, including one a later refactor re-adds DML to.
    """
    policy = _census_policy(tmp_path, [{"file": "polylogue/ops/retired_writer.py", "tiers": ["index"]}])

    violations = verify_layering._collect_writer_module_census_violations(tmp_path, policy)

    assert [
        violation["file"] for violation in violations if violation["rule"] == "writer_module_census_baseline_stale"
    ] == ["polylogue/ops/retired_writer.py"]


def test_layering_production_census_baseline_is_exact() -> None:
    """The checked-in census matches the tree, so the ratchet is real today."""
    assert verify_layering._collect_writer_module_census_violations(_REPO_ROOT, _production_writer_policy()) == []


def _break_grimp_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce a checkout synced without the ``audit`` dependency group.

    ``sys.modules[name] = None`` is exactly what CPython's import machinery
    consults first, so ``import grimp`` raises the same ``ImportError`` a
    pruned group produces -- the failure is not simulated by stubbing the
    gate's own helper.
    """
    monkeypatch.setitem(sys.modules, "grimp", None)


def test_a_pruned_audit_group_refuses_instead_of_reporting_a_layering_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """polylogue-mkucv: the environment problem must not arrive as a finding.

    Anti-vacuity: routing the missing checker back through ``violations`` --
    which is what the gate did before, as a ``grimp_unavailable`` entry --
    restores exit 1, a non-empty ``violations`` list and a
    ``gate_semantic_violation`` diagnosis, and every assertion here goes red.
    """
    # A tree that *would* produce a real finding, so a refusal that leaked
    # findings would be visible rather than trivially empty.
    _write_ratchet_fixture(tmp_path, baseline_entries=[])
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    _break_grimp_import(monkeypatch)

    assert verify_layering.main(["--json"]) == verify_layering.ENVIRONMENT_REFUSAL_EXIT
    payload = json.loads(capsys.readouterr().out)

    # The refusal is its own channel, not an entry in the findings channel.
    assert payload["violations"] == []
    assert payload["count"] == 0
    assert payload["inspected"] is False
    refusal = payload["environment_refusal"]
    assert refusal["dependency"] == "grimp"
    assert refusal["group"] == "audit"
    assert refusal["remedy"] == "uv sync --extra dev --group audit --frozen"

    gate = payload["required_gate"]
    assert gate["gate_passed"] is False
    assert gate["diagnosis"] == "gate_missing_analysis_dependency"
    # The distinguishing fact: nothing was inspected, so nothing was violated.
    assert gate["semantic_violation_count"] == 0
    assert gate["inspected_count"] == 0
    assert "uv sync --extra dev --group audit --frozen" in gate["details"][0]


def test_the_pruned_audit_group_banner_names_the_remedy_before_anything_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What a reader sees first must not read as a code problem.

    Anti-vacuity: printing the refusal through ``_format_violation`` alongside
    real findings (the prior behaviour) drops the banner and the remedy line,
    so both assertions fail.
    """
    _write_ratchet_fixture(tmp_path, baseline_entries=[])
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    _break_grimp_import(monkeypatch)

    assert verify_layering.main([]) == verify_layering.ENVIRONMENT_REFUSAL_EXIT
    captured = capsys.readouterr()
    first_line = captured.err.splitlines()[0]
    assert first_line == "ENVIRONMENT REFUSAL -- this is NOT a layering finding."
    assert "uv sync --extra dev --group audit --frozen" in captured.err
    # No finding was printed, so nobody is sent looking for an offending import.
    assert "imports polylogue.storage" not in captured.err
    assert "imports polylogue.storage" not in captured.out


def test_the_refusal_exit_code_is_distinct_from_the_finding_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit classification separates the two, on the same tree.

    Anti-vacuity: collapsing ``ENVIRONMENT_REFUSAL_EXIT`` back to 1 makes the
    two codes equal and the inequality assertion red.
    """
    _write_ratchet_fixture(tmp_path, baseline_entries=[])
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    with_grimp = verify_layering.main([])
    capsys.readouterr()
    _break_grimp_import(monkeypatch)
    without_grimp = verify_layering.main([])
    capsys.readouterr()

    assert with_grimp == 1, "the same tree carries a real layering finding"
    assert without_grimp == verify_layering.ENVIRONMENT_REFUSAL_EXIT
    assert without_grimp != with_grimp


def test_the_audit_sync_command_is_the_one_that_actually_provisions_the_group() -> None:
    """The remedy must not reproduce the trap it exists to prevent.

    ``uv sync --extra dev --frozen`` prunes the group and ``--extra audit``
    errors, so a remedy missing ``--group audit`` sends the reader back into
    the same failure. Anti-vacuity: dropping ``--group audit`` from the
    constant makes this red.
    """
    command = required_gate.AUDIT_GROUP_SYNC_COMMAND
    assert "--group audit" in command
    assert "--extra audit" not in command
    assert "--extra dev" in command
