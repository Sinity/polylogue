from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_catalog_bypasses as lint


def test_real_control_surfaces_have_only_sanctioned_direct_invocations() -> None:
    assert lint.collect_violations() == []


def test_scope_excludes_generated_provenance_and_argparse_help() -> None:
    root = Path(__file__).resolve().parents[3]
    paths = (Path("devtools/render_semantic_card_registry.py"), Path("devtools/resume_ranking_eval.py"))

    assert lint.scan_control_surfaces(root, paths=paths) == []


def test_mutation_direct_module_invocation_fails_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    control = tmp_path / "devtools" / "control.py"
    control.parent.mkdir()
    control.write_text(
        "import subprocess\nsubprocess.run(['python', '-m', 'devtools.fourth_bypass'], check=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(lint, "ROOT", tmp_path)

    assert lint.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["violations"] == [
        {"path": "devtools/control.py", "lineno": 2, "invocation": "python -m devtools.fourth_bypass"}
    ]


def test_scan_detects_direct_script_execution_in_workflow_run_block(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n  check:\n    steps:\n      - run: python devtools/fourth_bypass.py\n", encoding="utf-8"
    )

    violations = lint.collect_violations(tmp_path, paths=(Path(".github/workflows/ci.yml"),))

    assert [(item.path, item.invocation) for item in violations] == [
        (".github/workflows/ci.yml", "python devtools/fourth_bypass.py")
    ]
