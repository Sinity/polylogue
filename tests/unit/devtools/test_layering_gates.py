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
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from devtools import gate, required_gate, verify, verify_layering, verify_runs
from devtools.sqlite_degradation import census_sqlite_degradation_anchors
from devtools.testmon_provision import TestmonGraphState, TestmonGraphStatus
from devtools.toolchain import venv_bin


def _root_imports(repo_root: Path, target: str) -> tuple[dict[str, set[str]], tuple[str, ...]]:
    """The imports the gate's package pass reads for one declared root."""
    package_pass = verify_layering._package_pass(repo_root, import_roots=[target], writer_modules=None, manifest={})
    return package_pass.imports_by_root[target]


def test_layering_no_violations_passes(tmp_path: Path) -> None:
    storage = tmp_path / "polylogue" / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "module.py").write_text("import os\nfrom polylogue.core import json\n", encoding="utf-8")

    imports, unreadable = _root_imports(tmp_path, "polylogue/storage")
    assert unreadable == ()
    assert "polylogue.cli" not in imports.get("polylogue/storage/module.py", set())


def test_layering_disallow_violation_detected(tmp_path: Path) -> None:
    storage = tmp_path / "polylogue" / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "bad_importer.py").write_text("from polylogue.cli import click_app\n", encoding="utf-8")

    cli = tmp_path / "polylogue" / "cli"
    cli.mkdir(parents=True, exist_ok=True)
    (cli / "click_app.py").write_text("", encoding="utf-8")

    imports, unreadable = _root_imports(tmp_path, "polylogue/storage")
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
        disallow_from = list(rule.get("disallow", {}).get("from", []))
        file_imports, unreadable = _root_imports(tmp_path, target)
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

    # ``gate layering`` also runs the declaration censuses (#5377, #5380). This
    # fixture's package executes no DML, so an empty declaration is the truthful
    # one -- without it every ratchet test here fails on an unrelated missing
    # declaration instead of on the import property it is about.
    (plans_dir / "durable-write-census.yaml").write_text("package: polylogue\nwrites: []\n", encoding="utf-8")
    (plans_dir / "derived-sweep-census.yaml").write_text("package: polylogue\nsites: []\n", encoding="utf-8")


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
            # This entry no longer reproduces (no such file/import exists) and
            # must be removed so it cannot exempt a later reintroduction.
            {"target": "polylogue/cli", "file": "polylogue/cli/gone.py", "import": "polylogue.storage.gone"},
        ],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    baseline_path = tmp_path / "docs/plans/ratchet-baseline.json"
    committed = baseline_path.read_bytes()

    # The plain gate is read-only: the stale exemption is a blocking finding
    # that names the prune command, and the baseline file is untouched.
    exit_code = verify_layering.main([])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "layering_baseline_stale" in out
    assert "devtools gate layering --prune-baselines" in out
    assert baseline_path.read_bytes() == committed

    # The explicit prune removes it, after which the gate is clean.
    assert verify_layering.main(["--prune-baselines"]) == 0
    assert "pruned 1 stale layering baseline entr" in capsys.readouterr().out
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline == [{"target": "polylogue/cli", "file": "polylogue/cli/commands.py", "import": "polylogue.storage"}]
    assert verify_layering.main([]) == 0


def test_layering_gate_never_writes_the_checkout_without_prune_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate run must leave every baseline byte-identical (polylogue-hzyyv).

    Verify voids a run whose Git-visible content changed while it ran, so a gate
    that rewrote its own baseline voided every CI run on a branch that carried a
    stale entry.

    Anti-vacuity: prune unconditionally again (drop the ``args.prune_baselines``
    guard) and both baseline files change here.
    """
    _write_ratchet_fixture(
        tmp_path,
        baseline_entries=[
            {"target": "polylogue/cli", "file": "polylogue/cli/gone.py", "import": "polylogue.storage.gone"},
        ],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    ratchet = tmp_path / "docs/plans/ratchet-baseline.json"
    before = ratchet.read_bytes()
    assert verify_layering.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert "layering_baseline_stale" in {violation["rule"] for violation in payload["violations"]}
    assert ratchet.read_bytes() == before


@pytest.mark.uses_real_clock
def test_whole_quick_preserves_tracked_stale_baselines_and_reports_their_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unconditional pruning changes Git content and voids this actual quick receipt.

    This integration control runs the real quick plan and layering subprocess.
    Unrelated gate processes are stubbed; the real all-gate proof is recorded
    separately by the task's declared quick run.
    """
    _write_ratchet_fixture(
        tmp_path,
        baseline_entries=[
            {"target": "polylogue/cli", "file": "polylogue/cli/commands.py", "import": "polylogue.storage"},
            {"target": "polylogue/cli", "file": "polylogue/cli/gone.py", "import": "polylogue.storage.gone"},
        ],
    )
    plans = tmp_path / "docs/plans"
    sqlite_baseline = plans / "sqlite-degradation-baseline.json"
    sqlite_baseline.write_text(
        json.dumps({"anchors": [{"file": "polylogue/storage/gone.py", "digest": "1" * 40}]}), encoding="utf-8"
    )
    with (plans / "layering.yaml").open("a", encoding="utf-8") as handle:
        handle.write(
            "sqlite_degradation:\n  baseline: docs/plans/sqlite-degradation-baseline.json\n  roots: [polylogue]\n"
        )
    (tmp_path / ".gitignore").write_text(".cache/\n__pycache__/\n", encoding="utf-8")
    for args in (
        ["init", "--initial-branch=test/stale-baseline"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "Synthetic stale baseline",
        ],
    ):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True)
    tracked = (
        subprocess.run(["git", "ls-files", "-z"], cwd=tmp_path, capture_output=True, check=True)
        .stdout.decode()
        .split("\0")
    )
    before = {relative: (tmp_path / relative).read_bytes() for relative in tracked if relative}
    content = verify_runs.git_worktree_content_sha256(tmp_path)
    assert content is not None
    original_bin = venv_bin
    checkout = verify.ROOT
    monkeypatch.setattr(gate, "venv_bin", lambda name, *, root: original_bin(name, root=checkout))
    monkeypatch.setattr(gate, "venv_python", lambda *, root: sys.executable)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(verify_runs.VERIFY_HISTORY_PATH_ENV, str(tmp_path / ".cache/history.jsonl"))
    monkeypatch.setenv(verify_runs.VERIFY_EVIDENCE_PATH_ENV, str(tmp_path / ".cache/evidence.jsonl"))
    monkeypatch.setattr(verify, "refuse_verify_tier", lambda *_args: None)
    monkeypatch.setattr(verify, "_declared_agentctl_operation", lambda _args: None)
    monkeypatch.setattr(verify, "validate_authority_matrix", lambda: None)
    monkeypatch.setattr(verify, "sync_testmon_graph", lambda _root: False)
    monkeypatch.setattr(
        verify, "inspect_testmon_graph", lambda _root: TestmonGraphState(TestmonGraphStatus.ABSENT, "fixture")
    )
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    from polylogue.context import failure_seed

    monkeypatch.setattr(failure_seed, "write_failure_seed", lambda **_kwargs: None)
    run_process = verify._run_gate_process

    def gate_process(command: list[str], *, env: Any) -> subprocess.CompletedProcess[str]:
        if "devtools.verify_layering" in command:
            return run_process(command, env=env)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(verify, "_run_gate_process", gate_process)
    expected_labels = {item.label for item in gate.quick_gates()}
    assert verify._main(["--quick", "--json"]) == 1
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["diagnosis"] == "gate_semantic_violation"
    assert payload["status"] == "failed"
    assert {step["name"] for step in payload["steps"]} == expected_labels
    failed = [step for step in payload["steps"] if step["exit"]]
    assert [step["name"] for step in failed] == ["gate layering"]
    detail = (tmp_path / failed[0]["output_path"]).read_text(encoding="utf-8")
    assert "layering_baseline_stale" in detail
    assert "polylogue/cli/gone.py" in detail and "polylogue.storage.gone" in detail
    assert "sqlite_degradation_baseline_stale" in detail
    assert "polylogue/storage/gone.py:" + "1" * 40 in detail
    assert "devtools gate layering --prune-baselines" in detail
    assert "checkout_moved_during_run" not in output.out
    assert {relative: (tmp_path / relative).read_bytes() for relative in before} == before
    assert verify_runs.git_worktree_content_sha256(tmp_path) == content
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        == ""
    )
    recorded = json.loads((tmp_path / payload["artifact_dir"] / "run.json").read_text(encoding="utf-8"))
    assert recorded["diagnosis"] == payload["diagnosis"]
    assert recorded["git_head"] == recorded["final_git_head"]
    assert recorded["git_dirty"] is False and recorded["final_git_dirty"] is False


def test_fixed_layering_violation_is_not_exempt_when_reintroduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_ratchet_fixture(
        tmp_path,
        baseline_entries=[
            {"target": "polylogue/cli", "file": "polylogue/cli/commands.py", "import": "polylogue.storage"},
        ],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    commands = tmp_path / "polylogue/cli/commands.py"
    commands.write_text("# fixed\n", encoding="utf-8")

    assert verify_layering.main(["--prune-baselines"]) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "docs/plans/ratchet-baseline.json").read_text(encoding="utf-8")) == []

    commands.write_text("from polylogue.storage import archive_identity\n", encoding="utf-8")
    assert verify_layering.main([]) == 1
    assert "imports polylogue.storage (disallow)" in capsys.readouterr().out


_DEGRADED_HANDLER_MODULE = """\
import sqlite3


def read(connection):
    try:
        return connection.execute("SELECT 1").fetchone()
    except sqlite3.DatabaseError:
        return None
"""


def _write_sqlite_degradation_fixture(tmp_path: Path, *, anchors: list[dict[str, object]]) -> None:
    """Build a repo whose one degradation site is anchored by ``anchors``."""
    package = tmp_path / "polylogue" / "storage"
    package.mkdir(parents=True, exist_ok=True)
    (tmp_path / "polylogue" / "__init__.py").write_text('"""Fixture package."""\n', encoding="utf-8")
    (tmp_path / "polylogue" / "storage" / "__init__.py").write_text('"""Fixture package."""\n', encoding="utf-8")
    (package / "degraded.py").write_text(_DEGRADED_HANDLER_MODULE, encoding="utf-8")

    plans_dir = tmp_path / "docs" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    baseline_ref = "docs/plans/sqlite-degradation-baseline.json"
    (tmp_path / baseline_ref).write_text(
        json.dumps({"rule": "fixture", "anchors": anchors}),
        encoding="utf-8",
    )
    (plans_dir / "layering.yaml").write_text(
        "rules:\n"
        "  - target: polylogue/storage\n"
        "    description: sqlite degradation fixture\n"
        f"sqlite_degradation:\n  baseline: {baseline_ref}\n  roots: [polylogue]\n",
        encoding="utf-8",
    )
    (plans_dir / "durable-write-census.yaml").write_text("package: polylogue\nwrites: []\n", encoding="utf-8")
    (plans_dir / "derived-sweep-census.yaml").write_text("package: polylogue\nsites: []\n", encoding="utf-8")


def _fixture_anchor_digest(tmp_path: Path) -> str:
    anchors = census_sqlite_degradation_anchors(tmp_path, ("polylogue",))
    assert len(anchors) == 1, f"fixture should carry exactly one degradation site, got {sorted(anchors)}"
    return next(iter(anchors))[1]


def test_layering_plaintext_names_a_baseline_anchor_that_no_longer_reproduces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The human path must print the shrink finding, not raise on its own payload.

    ``_sqlite_degradation_findings`` returns entries keyed file/anchor/digest/
    removed. The plaintext printer read ``entry['observed']`` and
    ``entry['baseline']`` -- a per-file-count shape from before the baseline
    became content-anchored -- so every plaintext run with a non-reproducing
    anchor died with ``KeyError: 'observed'`` while the ``--json`` form the
    gate table uses stayed green (bd polylogue-0gcri).

    Anti-vacuity: restore either ``entry['observed']`` or ``entry['baseline']``
    in ``verify_layering.main`` and this raises ``KeyError`` instead of
    asserting. A ``--json``-only test cannot see it, so this one never passes
    ``--json``.
    """
    _write_sqlite_degradation_fixture(
        tmp_path,
        anchors=[
            {"file": "polylogue/storage/gone.py", "digest": "0" * 40},
        ],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    exit_code = verify_layering.main(["--prune-baselines"])

    out = capsys.readouterr().out
    assert exit_code == 1, "the fixture's unanchored live handler is a blocking finding"
    assert "polylogue/storage/gone.py:" + "0" * 40 in out
    assert "sqlite_degradation_anchor_no_longer_reproduces" in out
    assert "observed" not in out


def test_layering_plaintext_and_json_report_the_same_shrunk_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two surfaces name one set of findings, or the human path is a lie.

    Anti-vacuity: printing a different anchor (or dropping the loop) leaves the
    JSON anchor unnamed in the plaintext output and this fails.
    """
    _write_sqlite_degradation_fixture(tmp_path, anchors=[])
    live_digest = _fixture_anchor_digest(tmp_path)
    _write_sqlite_degradation_fixture(
        tmp_path,
        anchors=[
            {"file": "polylogue/storage/degraded.py", "digest": live_digest},
            {"file": "polylogue/storage/gone.py", "digest": "1" * 40, "count": 2},
        ],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    json_code = verify_layering.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert json_code == 1, "a stale anchor blocks until the shrink is committed"

    shrunk = payload["sqlite_degradation_shrunk"]
    assert [entry["digest"] for entry in shrunk] == ["1" * 40]
    for entry in shrunk:
        assert str(entry["removed"]) == "2"

    # The read-only runs agree, and the plaintext names the same anchor.
    plaintext_code = verify_layering.main([])
    plaintext = capsys.readouterr().out
    assert plaintext_code == json_code
    assert "sqlite_degradation_baseline_stale" in plaintext
    assert "1" * 40 in plaintext


def test_fixed_sqlite_degradation_anchor_is_not_exempt_when_reintroduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_sqlite_degradation_fixture(tmp_path, anchors=[])
    digest = _fixture_anchor_digest(tmp_path)
    _write_sqlite_degradation_fixture(
        tmp_path,
        anchors=[{"file": "polylogue/storage/degraded.py", "digest": digest}],
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)
    handler = tmp_path / "polylogue/storage/degraded.py"
    handler.write_text("def read(connection):\n    raise RuntimeError('closed')\n", encoding="utf-8")

    assert verify_layering.main(["--prune-baselines"]) == 0
    capsys.readouterr()
    baseline = json.loads((tmp_path / "docs/plans/sqlite-degradation-baseline.json").read_text(encoding="utf-8"))
    assert baseline["anchors"] == []

    handler.write_text(_DEGRADED_HANDLER_MODULE, encoding="utf-8")
    assert verify_layering.main([]) == 1
    assert "sqlite_degradation_site_added" in capsys.readouterr().out


def test_layering_cli_imports_storage_is_detected(tmp_path: Path) -> None:
    # polylogue-2ciy: cli->storage is no longer unconditionally "ok" -- the
    # production rule now disallows it too (behind a ratchet baseline). This
    # test only pins that the package pass itself surfaces the import; see
    # the baseline tests below for the ratchet's pass/fail behavior.
    cli_dir = tmp_path / "polylogue" / "cli"
    cli_dir.mkdir(parents=True, exist_ok=True)
    (cli_dir / "commands.py").write_text("from polylogue.storage import something\n", encoding="utf-8")

    imports, unreadable = _root_imports(tmp_path, "polylogue/cli")
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


def test_scratch_census_authority_cannot_cover_archive_tier_mutation(tmp_path: Path) -> None:
    """A scratch declaration admits scratch DML but rejects known archive-table writes.

    Anti-vacuity: if the census ignores ``authority`` or fails to classify a
    known archive table, changing this file from scratch storage to archive
    storage would remain green under the same baseline row.
    """
    rel = "polylogue/ops/scratch_writer.py"
    _write_census_module(tmp_path, rel, "INSERT INTO scratch_records (value) VALUES (?)")
    baseline: list[dict[str, object]] = [{"file": rel, "tiers": [], "authority": "scratch"}]

    assert verify_layering._collect_writer_module_census_violations(tmp_path, _census_policy(tmp_path, baseline)) == []

    _write_census_module(tmp_path, rel, "INSERT INTO sessions (native_id) VALUES (?)")

    violations = verify_layering._collect_writer_module_census_violations(tmp_path, _census_policy(tmp_path, baseline))
    assert [
        violation["rule"]
        for violation in violations
        if violation["rule"] == "writer_module_scratch_archive_tier_mutation"
    ] == ["writer_module_scratch_archive_tier_mutation"]

    invalid_policy = _census_policy(tmp_path, [{"file": rel, "tiers": ["index"], "authority": "scratch"}])
    invalid = verify_layering._collect_writer_module_census_violations(tmp_path, invalid_policy)
    assert [violation["rule"] for violation in invalid] == ["writer_module_census_declaration_invalid"]


def _archive_open_findings(tmp_path: Path, source: str) -> list[dict[str, object]]:
    module = tmp_path / "polylogue" / "ops" / "open_probe.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(source, encoding="utf-8")
    package_pass = verify_layering._package_pass(
        tmp_path,
        import_roots=[],
        writer_modules=None,
        manifest={"sqlite_degradation": {"baseline": "unused.json", "roots": ["polylogue"]}},
    )
    return package_pass.sqlite_archive_open_violations


def test_layering_flags_unadmitted_locally_resolved_archive_sqlite_opens(tmp_path: Path) -> None:
    """The parsed package pass rejects known tier paths through all raw openers.

    Anti-vacuity: changing any of the six raw opens to skip its finding, or
    failing to resolve one of the imported aliases, reduces the six findings.
    """
    findings = _archive_open_findings(
        tmp_path,
        '''\
import sqlite3 as sql
from pathlib import Path as P
from polylogue.storage.sqlite.managed_connection import sqlite_connection as managed_open
from polylogue.storage.io_phase_metrics import connect_measured as measured_open
from polylogue.storage.sqlite.population_admission import assert_population_admitted


def raw_archive_opens():
    database = P("/archive") / "user.db"
    sql.connect(database)
    managed_open(database)
    measured_open(database)
    rollback_uri = "file:/archive/ops.db?mode=rollback"
    sql.connect(rollback_uri, uri=True)


def asserted_population_is_not_an_admission(root):
    assert_population_admitted(root)
    sql.connect(root / "source.db")


def scratch_name_rebound_to_archive_tier(root):
    with P("/tmp") as ignored:
        pass
    ignored = root / "embeddings.db"
    sql.connect(ignored)
''',
    )

    assert [finding["rule"] for finding in findings] == ["sqlite_archive_open_without_factory"] * 6
    assert [finding["line"] for finding in findings] == [10, 11, 12, 14, 19, 25]


def test_layering_main_fails_for_new_raw_open_in_a_clean_package_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A previously clean arbitrary module cannot add a direct tier open."""
    _write_ratchet_fixture(tmp_path, baseline_entries=[])
    module = tmp_path / "polylogue" / "cli" / "unrelated.py"
    module.write_text(
        'import sqlite3\n\n\ndef probe(root):\n    return sqlite3.connect(root / "user.db")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    assert verify_layering.main([]) == 1
    report = capsys.readouterr().out
    assert "sqlite_archive_open_without_factory" in report
    assert "polylogue/cli/unrelated.py" in report


def test_layering_archive_open_gate_allows_admission_readonly_and_scratch(tmp_path: Path) -> None:
    """Read-only, temp-scratch, ordinary scratch, and admitted opens pass."""
    findings = _archive_open_findings(
        tmp_path,
        '''\
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory as TempDir
from polylogue.storage.sqlite.write_lease import require_write_lease
from polylogue.storage.sqlite.population_admission import require_population_admission


def legitimate_opens(root):
    require_write_lease("fixture", archive_root=root)
    sqlite3.connect(root / "audit.db")


def population_stage(root):
    require_population_admission(root)
    sqlite3.connect(root / "source.db")


def read_and_scratch(root):
    uri = "file:/archive/index.db?mode=ro"
    sqlite3.connect(uri, uri=True)
    sqlite3.connect(root / "scratch.db")
    sqlite3.connect(":memory:")
    with TempDir() as directory:
        sqlite3.connect(Path(directory) / "index.db")
''',
    )

    assert findings == []


def test_layering_archive_open_gate_visits_lambda_scopes(tmp_path: Path) -> None:
    findings = _archive_open_findings(
        tmp_path,
        '''\
import sqlite3

open_index = lambda root: sqlite3.connect(root / "index.db")
''',
    )

    assert [finding["rule"] for finding in findings] == ["sqlite_archive_open_without_factory"]


def test_layering_archive_open_readonly_uri_requires_uri_true(tmp_path: Path) -> None:
    findings = _archive_open_findings(
        tmp_path,
        '''\
import sqlite3

def uri_modes(root):
    sqlite3.connect(root / "user.db?mode=ro")
    sqlite3.connect(root / "audit.db?mode=ro", uri=True)
''',
    )

    assert len(findings) == 1
    assert findings[0]["line"] == 5


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


# ---------------------------------------------------------------------------
# Declaration-census violations must be printable (polylogue-1or21).
#
# ``_format_violation`` fell through to ``violation['file']`` / ``['import']``
# for any rule family without a branch. The ``durable_write_*`` family from
# #5377 had none and reports rows keyed on a census key that may carry no file
# at all, so rendering one raised ``KeyError`` and took down the whole
# plaintext report -- an armed census that structurally could not report a
# finding.
# ---------------------------------------------------------------------------

_DURABLE_REWRITE_MODULE = """\
def rewrite(conn):
    conn.execute("UPDATE raw_sessions SET parse_error = NULL")
"""


def _write_durable_census_fixture(tmp_path: Path, declaration: str | None) -> None:
    """Seed the ratchet fixture plus one real durable rewrite and a declaration."""
    _write_ratchet_fixture(tmp_path, baseline_entries=[])
    storage = tmp_path / "polylogue" / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "rewriter.py").write_text(_DURABLE_REWRITE_MODULE, encoding="utf-8")
    census = tmp_path / "docs" / "plans" / "durable-write-census.yaml"
    if declaration is None:
        census.unlink(missing_ok=True)
    else:
        census.write_text(declaration, encoding="utf-8")


def test_layering_prints_a_durable_write_finding_beside_the_import_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The forbidden-classification finding reaches the reader.

    Anti-vacuity: with the pre-fix ``_format_violation`` this raises
    ``KeyError: 'import'`` before printing anything, so both the census line
    and the unrelated import line are lost. The import assertion also refutes
    a fix that renders every violation through one generic dump.
    """
    _write_durable_census_fixture(
        tmp_path,
        "package: polylogue\n"
        "writes:\n"
        '  - file: "polylogue/storage/rewriter.py"\n'
        '    function: "rewrite"\n'
        '    table: "raw_sessions"\n'
        '    kind: "update"\n'
        '    tier: "source"\n'
        "    classification: masking_backfill\n"
        '    reason: "a synthetic mutant"\n',
    )
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    assert verify_layering.main([]) == 1
    out = capsys.readouterr().out

    census_lines = [line for line in out.splitlines() if "durable_write_masks_a_producer" in line]
    assert len(census_lines) == 1, out
    assert "polylogue/storage/rewriter.py:2" in census_lines[0]
    assert "fix the producer instead" in census_lines[0]
    assert "<no renderer for rule family>" not in out
    # The unrelated import finding still renders in its own form.
    assert "imports polylogue.storage (disallow)" in out


def test_layering_prints_a_census_violation_that_carries_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing declaration names the path it could not read.

    This is the exact shape nine gate tests hit at #5380's head: the rule
    carries only ``rule`` and ``key``. Anti-vacuity: the pre-fix fall-through
    raises ``KeyError: 'file'`` on it.
    """
    _write_durable_census_fixture(tmp_path, None)
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    assert verify_layering.main([]) == 1
    out = capsys.readouterr().out
    assert "docs/plans/durable-write-census.yaml: durable_write_census_declaration_missing" in out
    assert "<no renderer for rule family>" not in out


def test_a_rule_family_with_no_renderer_fails_loudly_instead_of_silently() -> None:
    """The fall-through is total, and says so when it fires.

    A future violation family nobody wrote a branch for must still reach the
    report, and must not read as if it were understood. Anti-vacuity: the
    pre-fix fall-through raises ``KeyError``; a fall-through that quietly
    rendered the payload as an ordinary finding would fail the marker
    assertion, and one that dropped the evidence would fail the field
    assertion.
    """
    rendered = verify_layering._format_violation(
        {"rule": "some_future_family_violation", "key": "k", "evidence_ref": "e"}
    )
    assert "<no renderer for rule family>" in rendered
    assert "some_future_family_violation" in rendered
    assert "_format_violation" in rendered, "the line must say where to add the branch"
    assert "evidence_ref='e'" in rendered, "the finding's own evidence must survive"


@pytest.mark.parametrize(
    "violation",
    [
        {"rule": "durable_write_undeclared", "key": "k", "file": "a.py", "line": 3, "tier": "source", "detail": "d"},
        {"rule": "durable_write_census_stale", "key": "k", "file": "a.py", "detail": "d"},
        {"rule": "durable_write_tier_drift", "key": "k", "file": "a.py", "declared": "user", "observed": "source"},
        {"rule": "durable_write_census_row_malformed", "key": "writes[0]"},
        {"rule": "caller_supplied_sql_undeclared", "key": "k", "file": "a.py", "line": 1, "detail": "d"},
        {"rule": "caller_supplied_sql_census_stale", "key": "k", "detail": "d"},
        {"rule": "derived_sweep_undeclared", "key": "k", "file": "b.py", "line": 2, "detail": "d"},
        {"rule": "derived_sweep_census_declaration_missing", "key": "p"},
        {"rule": "controlled_read_site_undeclared", "key": "k", "file": "c.py", "line": 4},
        {"rule": "rebuild_route_undeclared", "key": "k", "file": "d.py"},
    ],
)
def test_every_census_rule_shape_renders_without_a_keyerror(violation: dict[str, Any]) -> None:
    """One branch covers every census family, including the fileless shapes.

    Anti-vacuity: each of these raises ``KeyError`` on the pre-fix renderer,
    and none of them may land on the unrendered marker -- that would mean the
    family lost its branch.
    """
    rendered = verify_layering._format_violation(violation)
    assert str(violation["rule"]) in rendered
    assert "<no renderer for rule family>" not in rendered


def test_one_package_pass_feeds_every_whole_package_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each census the gate takes from its single package pass still fails.

    The gate parses each module once and hands it to the import, writer-module
    census, durable-write, derived-sweep and SQLite-degradation censuses. One
    planted defect per census must surface through ``main``.

    Anti-vacuity: stop feeding any one census from the pass (or feed it no
    modules) and its rule disappears from the reported set.
    """
    package = tmp_path / "polylogue"
    for relative, body in {
        "storage/bad_importer.py": "from polylogue.cli import click_app\n",
        "storage/degraded.py": _DEGRADED_HANDLER_MODULE,
        "ops/rogue_writer.py": (
            'def lock(conn):\n    conn.execute("UPDATE assertions SET updated_at_ms = updated_at_ms WHERE 1")\n'
        ),
        "ops/sweeper.py": (
            "def sweep(conn):\n"
            '    conn.execute("UPDATE session_profiles SET is_continuation = 1 WHERE parent_id IS NOT NULL")\n'
        ),
        # The manifest requires one inventoried writer module.
        "storage/sqlite/archive_tiers/facade.py": '"""Fixture facade.\n\nWriter module: index.\n"""\n',
    }.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    plans = tmp_path / "docs" / "plans"
    plans.mkdir(parents=True)
    (plans / "census.json").write_text("[]", encoding="utf-8")
    (plans / "sqlite-degradation-baseline.json").write_text(
        json.dumps({"rule": "fixture", "anchors": []}), encoding="utf-8"
    )
    (plans / "layering.yaml").write_text(
        "rules:\n"
        "  - target: polylogue/storage\n"
        "    description: fixture\n"
        "    disallow:\n"
        "      from: [polylogue/cli]\n"
        "sqlite_degradation:\n"
        "  baseline: docs/plans/sqlite-degradation-baseline.json\n"
        "  roots: [polylogue]\n"
        "writer_modules:\n"
        '  marker: "Writer module:"\n'
        "  mutation_roots: [polylogue/storage/sqlite/archive_tiers]\n"
        "  census_roots: [polylogue]\n"
        "  census_baseline: docs/plans/census.json\n"
        "  modules:\n"
        "    - path: polylogue/storage/sqlite/archive_tiers/facade.py\n"
        "      surfaces:\n"
        "        - tier: index\n"
        "          durability: rebuildable\n"
        "          interruption: replayable\n"
        "      entrypoints: [write]\n",
        encoding="utf-8",
    )
    (plans / "durable-write-census.yaml").write_text("package: polylogue\nwrites: []\n", encoding="utf-8")
    (plans / "derived-sweep-census.yaml").write_text("package: polylogue\nsites: []\n", encoding="utf-8")
    monkeypatch.setattr(verify_layering, "_get_root", lambda: tmp_path)

    assert verify_layering.main(["--json"]) == 1

    violations = json.loads(capsys.readouterr().out)["violations"]
    reported = {(violation["rule"], violation.get("file")) for violation in violations}
    assert {
        ("disallow", "polylogue/storage/bad_importer.py"),
        ("writer_module_uncensused_mutation", "polylogue/ops/rogue_writer.py"),
        ("durable_write_undeclared", "polylogue/ops/rogue_writer.py"),
        ("derived_sweep_undeclared", "polylogue/ops/sweeper.py"),
        ("sqlite_degradation_site_added", "polylogue/storage/degraded.py"),
    } <= reported


def test_daemon_collection_and_stage_adapters_have_no_substrate_exemptions() -> None:
    """F756: restoring either direct acquisition or its exemption makes this red."""
    root = Path(__file__).resolve().parents[3]
    adapters = {"polylogue/daemon/convergence_stages.py", "polylogue/daemon/metrics.py"}
    for adapter in adapters:
        imports = verify_layering._module_imports(ast.parse((root / adapter).read_text(encoding="utf-8")))
        assert not any(
            verify_layering._package_matches(package, imported)
            for imported in imports
            for package in ("polylogue/storage", "polylogue/sources")
        )
    baseline = verify_layering._load_baseline(root / "docs/plans/layering-surface-baseline.json")
    assert not any(file in adapters for _, file, _ in baseline)


def test_production_writer_inventory_resolves_nested_and_class_method_names_in_their_actual_scope(
    tmp_path: Path,
) -> None:
    writer_root = _copy_production_writer_surface(tmp_path)
    writer = writer_root / "write.py"
    writer.write_text(
        writer.read_text(encoding="utf-8")
        + """
class _CollisionWriter:
    def add(self, conn):
        self.flush(conn)
    def flush(self, conn):
        conn.execute("INSERT INTO sessions(session_id) VALUES ('neutral')")

class _CollisionReader:
    def flush(self):
        return "neutral"

def read_collision():
    visited = set()
    def depth():
        visited.add("neutral")
        return 1
    def flush():
        return depth()
    reader = _CollisionReader()
    reader.flush()
    return flush()

def nested_collision_writer(conn):
    def flush():
        _CollisionWriter().add(conn)
    flush()

def typed_collision_writer(conn, writer: _CollisionWriter):
    writer.add(conn)

def captured_collision_writer(conn):
    writer = _CollisionWriter()
    def flush():
        writer.add(conn)
    flush()

def _send_collision(writer, conn):
    writer.add(conn)

def argument_collision_writer(conn):
    _send_collision(_CollisionWriter(), conn)

class _FieldWriter:
    def __init__(self, /):
        self.writer = _CollisionWriter()
    def flush(self, conn, /):
        self.writer.add(conn)

def field_collision_writer(conn):
    _FieldWriter().flush(conn)

def _writer_factory() -> _CollisionWriter:
    return _CollisionWriter()

def factory_collision_writer(conn):
    _writer_factory().add(conn)

class _ConstructorWriter:
    def __init__(self, conn, /):
        conn.execute("INSERT INTO sessions(session_id) VALUES ('neutral')")

def constructor_collision_writer(conn):
    _ConstructorWriter(conn)

def sql_scope_writer(conn):
    sql = "INSERT INTO sessions(session_id) VALUES ('neutral')"
    def reader():
        sql = "SELECT session_id FROM sessions"
        return conn.execute(sql)
    conn.execute(sql)
    reader()

_CAPTURED_SQL = "INSERT INTO sessions(session_id) VALUES ('neutral')"

def inherited_sql_writer(conn):
    def flush():
        conn.execute(_CAPTURED_SQL)
    flush()

class _ClassSqlReader:
    sql = "INSERT INTO sessions(session_id) VALUES ('neutral')"
    @staticmethod
    def read(conn):
        sql = "SELECT session_id FROM sessions"
        return conn.execute(sql)

def class_sql_reader(conn):
    return _ClassSqlReader.read(conn)

class _StaticForwarder:
    @staticmethod
    def send(writer, conn):
        writer.add(conn)

def static_argument_writer(conn):
    _StaticForwarder.send(_CollisionWriter(), conn)

def _lexical_insert(conn):
    conn.execute("INSERT INTO sessions(session_id) VALUES ('neutral')")

def _lexical_read(conn):
    return conn.execute("SELECT session_id FROM sessions")

class _BareNameReader:
    def _lexical_insert(self, conn):
        return "neutral"
    def _lexical_read(self, conn):
        conn.execute("INSERT INTO sessions(session_id) VALUES ('neutral')")
    def run_writer(self, conn):
        _lexical_insert(conn)
    def run_reader(self, conn):
        return _lexical_read(conn)

def bare_module_writer(conn):
    _BareNameReader().run_writer(conn)

def bare_module_reader(conn):
    return _BareNameReader().run_reader(conn)

def sql_scope_reader(conn):
    sql = "SELECT session_id FROM sessions"
    def unused_writer():
        sql = "INSERT INTO sessions(session_id) VALUES ('neutral')"
        conn.execute(sql)
    return conn.execute(sql)
""",
        encoding="utf-8",
    )
    violations = verify_layering._collect_writer_module_violations(tmp_path, _production_writer_policy())
    mismatch = next(
        item
        for item in violations
        if item["file"] == "polylogue/storage/sqlite/archive_tiers/write.py"
        and item["rule"] == "writer_module_entrypoint_inventory_mismatch"
    )
    observed_names = mismatch["observed"]
    expected_names = mismatch["expected"]
    assert isinstance(observed_names, list) and isinstance(expected_names, list)
    observed = set(observed_names)
    assert observed == set(expected_names) | {
        "nested_collision_writer",
        "typed_collision_writer",
        "captured_collision_writer",
        "argument_collision_writer",
        "field_collision_writer",
        "factory_collision_writer",
        "constructor_collision_writer",
        "sql_scope_writer",
        "inherited_sql_writer",
        "static_argument_writer",
        "bare_module_writer",
    }
    assert "read_collision" not in observed
    assert "sql_scope_reader" not in observed
    assert "class_sql_reader" not in observed
    assert "bare_module_reader" not in observed
    assert not any("." in name for name in observed)
