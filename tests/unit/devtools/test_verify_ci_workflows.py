"""Tests for devtools/verify_ci_workflows.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from devtools.verify_ci_workflows import (
    WorkflowInventory,
    _devtools_command_names,
    check_workflow,
    inventory_workflows,
    main,
)


class TestDevtoolsCommandNames:
    def test_returns_nonempty_frozenset(self) -> None:
        names = _devtools_command_names()
        assert isinstance(names, frozenset)
        assert len(names) > 0

    def test_contains_known_commands(self) -> None:
        names = _devtools_command_names()
        assert "verify" in names
        assert "status" in names
        assert "render all" in names


class TestCheckWorkflow:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        path = tmp_path / "test.yml"
        path.write_text(textwrap.dedent(content))
        return path

    def test_valid_workflow_no_errors(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            """
            jobs:
              lint:
                steps:
                  - name: lint
                    run: uv run devtools render all --check
            """,
        )
        errors, warnings = check_workflow(path, tmp_path.parent, _devtools_command_names())
        assert errors == []

    def test_python_devtools_module_path_is_not_command(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            """
            jobs:
              lint:
                steps:
                  - name: script
                    run: uv run python devtools/some_script.py --flag
            """,
        )
        errors, _warnings = check_workflow(path, tmp_path.parent, _devtools_command_names())
        assert errors == []

    def test_unknown_devtools_command_is_error(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            """
            jobs:
              test:
                steps:
                  - name: broken
                    run: devtools nonexistent-command-xyz
            """,
        )
        errors, _ = check_workflow(path, tmp_path.parent, _devtools_command_names())
        assert any("nonexistent-command-xyz" in e for e in errors)

    def test_invalid_yaml_returns_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text(": bad yaml: [unclosed")
        errors, _ = check_workflow(path, tmp_path.parent, _devtools_command_names())
        assert errors

    def test_no_run_steps_no_errors(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            """
            jobs:
              test:
                steps:
                  - name: checkout
                    uses: actions/checkout@v4
            """,
        )
        errors, warnings = check_workflow(path, tmp_path.parent, _devtools_command_names())
        assert errors == []
        assert warnings == []

    def test_existing_path_no_warning(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (repo_root / "polylogue").mkdir()
        path = tmp_path / "repo" / "workflow.yml"
        path.write_text(
            textwrap.dedent(
                """
                jobs:
                  lint:
                    steps:
                      - run: ruff check polylogue/
                """
            )
        )
        errors, warnings = check_workflow(path, repo_root, _devtools_command_names())
        assert errors == []
        assert warnings == []

    def test_nonexistent_path_is_warning(self, tmp_path: Path) -> None:
        path = self._write_yaml(
            tmp_path,
            """
            jobs:
              lint:
                steps:
                  - run: ruff check totally_nonexistent_dir/
            """,
        )
        _, warnings = check_workflow(path, tmp_path, _devtools_command_names())
        assert any("totally_nonexistent_dir" in w for w in warnings)


class TestInventoryWorkflows:
    def _write(self, dir_: Path, name: str, content: str) -> None:
        (dir_ / name).write_text(textwrap.dedent(content))

    def test_extracts_workflow_name_jobs_runs_and_uploads(self, tmp_path: Path) -> None:
        wf_dir = tmp_path / "workflows"
        wf_dir.mkdir()
        self._write(
            wf_dir,
            "ci.yml",
            """
            name: CI
            on:
              workflow_dispatch:
              pull_request:
            jobs:
              lint:
                steps:
                  - run: uv run ruff check polylogue/
              test:
                steps:
                  - run: uv run devtools verify coverage
                  - uses: actions/upload-artifact@v7
                    with:
                      name: coverage-report
                      path: coverage.xml
            """,
        )
        inv = inventory_workflows(wf_dir)
        assert isinstance(inv, WorkflowInventory)
        assert inv.workflow_names == ("CI",)
        assert set(inv.all_job_names) == {"lint", "test"}
        runs = inv.all_run_commands
        assert any("ruff check polylogue/" in r for r in runs)
        assert any("devtools verify coverage" in r for r in runs)
        assert inv.all_artifact_uploads == ("coverage-report",)
        assert set(inv.workflows[0].triggers) == {"workflow_dispatch", "pull_request"}

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        inv = inventory_workflows(tmp_path / "nonexistent")
        assert inv == WorkflowInventory()

    def test_malformed_yaml_is_skipped(self, tmp_path: Path) -> None:
        wf_dir = tmp_path / "workflows"
        wf_dir.mkdir()
        (wf_dir / "bad.yml").write_text(": bad yaml: [unclosed")
        inv = inventory_workflows(wf_dir)
        assert inv.workflows == ()


class TestMain:
    def test_passes_on_real_workflows(self) -> None:
        result = main([])
        assert result == 0

    def test_json_output_structure(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        main(["--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "blocking" in data
        assert "errors" in data
        assert "warnings" in data
        assert "files_checked" in data
        assert data["blocking"] is False
        assert data["files_checked"] >= 1
