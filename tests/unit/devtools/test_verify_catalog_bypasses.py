from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_catalog_bypasses as lint


def test_real_control_surfaces_have_only_sanctioned_direct_invocations() -> None:
    assert lint.collect_violations() == []


def test_control_surface_paths_include_ci_owned_npm_manifests() -> None:
    paths = {path.relative_to(lint.ROOT).as_posix() for path in lint.control_surface_paths()}

    assert {"webui/package.json", "browser-extension/package.json"} <= paths


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
        {
            "path": "devtools/control.py",
            "lineno": 2,
            "invocation": "python -m devtools.fourth_bypass",
            "reason": "undeclared-direct-invocation",
        }
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


def test_scan_detects_compact_module_execution_in_workflow_run_block(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n  check:\n    steps:\n      - run: python3 -mdevtools.fourth_bypass\n", encoding="utf-8"
    )

    violations = lint.collect_violations(tmp_path, paths=(Path(".github/workflows/ci.yml"),))

    assert [(item.path, item.invocation) for item in violations] == [
        (".github/workflows/ci.yml", "python -m devtools.fourth_bypass")
    ]


def test_scan_detects_direct_module_execution_in_ci_owned_npm_script(tmp_path: Path) -> None:
    manifest = tmp_path / "webui" / "package.json"
    manifest.parent.mkdir()
    manifest.write_text(
        '{\n  "scripts": {\n    "generate": "python3 -m devtools.fourth_bypass"\n  }\n}\n', encoding="utf-8"
    )

    violations = lint.collect_violations(tmp_path, paths=(Path("webui/package.json"),))

    assert [(item.path, item.lineno, item.invocation, item.reason) for item in violations] == [
        ("webui/package.json", 3, "python -m devtools.fourth_bypass", "undeclared-direct-invocation")
    ]


def test_scan_detects_compact_module_execution_in_ci_owned_npm_script(tmp_path: Path) -> None:
    manifest = tmp_path / "webui" / "package.json"
    manifest.parent.mkdir()
    manifest.write_text(
        '{\n  "scripts": {\n    "generate": "python3 -mdevtools.fourth_bypass"\n  }\n}\n', encoding="utf-8"
    )

    violations = lint.collect_violations(tmp_path, paths=(Path("webui/package.json"),))

    assert [(item.path, item.invocation) for item in violations] == [
        ("webui/package.json", "python -m devtools.fourth_bypass")
    ]


def test_scan_detects_direct_module_execution_in_keyword_args(tmp_path: Path) -> None:
    control = tmp_path / "devtools" / "control.py"
    control.parent.mkdir()
    control.write_text(
        "import subprocess\nimport sys\noption = '--json'\nsubprocess.run(args=[sys.executable, '-m', 'devtools.fourth_bypass', option], check=True)\n",
        encoding="utf-8",
    )

    violations = lint.collect_violations(tmp_path, paths=(Path("devtools/control.py"),))

    assert [(item.path, item.lineno, item.invocation, item.reason) for item in violations] == [
        ("devtools/control.py", 4, "python -m devtools.fourth_bypass", "undeclared-direct-invocation")
    ]


def test_scan_detects_compact_module_execution_in_subprocess_call_keyword_args(tmp_path: Path) -> None:
    control = tmp_path / "devtools" / "control.py"
    control.parent.mkdir()
    control.write_text(
        "import subprocess\nsubprocess.call(args=['python', '-mdevtools.fourth_bypass'])\n", encoding="utf-8"
    )

    violations = lint.collect_violations(tmp_path, paths=(Path("devtools/control.py"),))

    assert [(item.path, item.lineno, item.invocation, item.reason) for item in violations] == [
        ("devtools/control.py", 2, "python -m devtools.fourth_bypass", "undeclared-direct-invocation")
    ]


def test_scan_detects_compact_module_execution_in_hook(tmp_path: Path) -> None:
    hook = tmp_path / ".githooks" / "custom-pre-push"
    hook.parent.mkdir()
    hook.write_text("python3 -mdevtools.fourth_bypass\n", encoding="utf-8")

    violations = lint.collect_violations(tmp_path, paths=(Path(".githooks/custom-pre-push"),))

    assert [(item.path, item.invocation) for item in violations] == [
        (".githooks/custom-pre-push", "python -m devtools.fourth_bypass")
    ]


def test_duplicate_hook_invocation_exceeds_the_sanctioned_cardinality(tmp_path: Path) -> None:
    hook = tmp_path / ".githooks" / "pre-push"
    hook.parent.mkdir()
    hook.write_text(
        "\n" * 20
        + 'python -m devtools.pre_push_gate "$UPDATES_FILE"\npython -m devtools.pre_push_gate "$UPDATES_FILE"\n',
        encoding="utf-8",
    )

    violations = lint.collect_violations(tmp_path, paths=(Path(".githooks/pre-push"),))

    assert [(item.lineno, item.reason) for item in violations] == [
        (22, "undeclared-direct-invocation"),
        (21, "sanctioned-occurrence-count:2!=1"),
    ]
