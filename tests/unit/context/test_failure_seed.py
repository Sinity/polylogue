from __future__ import annotations

import json
from pathlib import Path

from polylogue.context.failure_seed import compile_failure_seed


def test_compile_failure_seed_joins_failed_test_and_implicated_files(tmp_path: Path) -> None:
    run_dir = tmp_path / ".cache" / "verify" / "runs" / "run-1"
    step_dir = run_dir / "steps" / "01-pytest"
    step_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "failed",
                "diagnosis": "pytest_failed",
                "steps": [{"artifact_dir": ".cache/verify/runs/run-1/steps/01-pytest"}],
            }
        ),
        encoding="utf-8",
    )
    (step_dir / "pytest-report.json").write_text(
        json.dumps(
            {"tests": [{"nodeid": "tests/unit/context/test_failure_seed.py::test_target", "outcome": "failed"}]}
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "failure-context.json"
    envelope.write_text(
        json.dumps(
            {
                "failure_id": "tests/unit/context/test_failure_seed.py::test_target",
                "testmon_dependencies": ["polylogue/context/failure_seed.py", "polylogue/cli/commands/context.py"],
            }
        ),
        encoding="utf-8",
    )

    result = compile_failure_seed(root=tmp_path, envelope_path=envelope)

    assert result["seed"]["failure_tests"] == ["tests/unit/context/test_failure_seed.py::test_target"]
    assert result["seed"]["implicated_files"] == [
        "polylogue/cli/commands/context.py",
        "polylogue/context/failure_seed.py",
    ]
    assert result["seed"]["next_command"].endswith("::test_target")


def test_compile_failure_seed_requires_workspace_envelope(tmp_path: Path) -> None:
    run_dir = tmp_path / ".cache" / "verify" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    try:
        compile_failure_seed(root=tmp_path)
    except FileNotFoundError as exc:
        assert "failure-context envelope" in str(exc)
    else:
        raise AssertionError("missing failure-context envelope must refuse seed compilation")
