"""Synthetic managed-verification source and graph fixtures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from devtools import pytest_slot, verify
from devtools.testmon_provision import testmon_datafile


def source_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text(".cache/\n.venv\n__pycache__/\n.pytest_cache/\n.hypothesis/\n.benchmarks/\n")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\ncache_dir = '.cache/pytest'\n")
    (root / "tests/nested").mkdir(parents=True)
    (root / "tests/conftest.py").write_text(
        "import sys\nfrom pathlib import Path\nfrom hypothesis import settings\n"
        "sys.path.insert(0, str(Path(__file__).parents[1]))\n"
        "settings.register_profile('verify', max_examples=10)\n"
    )
    (root / "helper.py").write_text("def value():\n    return True\n")
    (root / "tests/nested/test_one.py").write_text("from helper import value\ndef test_one():\n    assert value()\n")
    (root / "neutral.py").write_text("def value():\n    return True\n")
    (root / "tests/test_other.py").write_text("from neutral import value\ndef test_other():\n    assert value()\n")
    checkout = Path(verify.__file__).parents[1]
    (root / "devtools").mkdir(exist_ok=True)
    (root / "devtools/execution_custody.py").write_bytes((checkout / "devtools/execution_custody.py").read_bytes())
    (root / ".venv").symlink_to(checkout / ".venv", target_is_directory=True)
    for arguments in (
        ["init", "-b", "feature"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"],
    ):
        subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "branch", "master"], cwd=root, check=True)
    testmon_datafile(root).parent.mkdir(parents=True)
    monkeypatch.setattr(verify, "ROOT", root)
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: None)
    monkeypatch.setattr(pytest_slot, "admit_width", lambda argv, **_kwargs: (list(argv), None))
    return root


def record_graph(root: Path, *, profile: str = "default", subset: str | None = None) -> tuple[int, dict[str, Any]]:
    command = verify._pytest_command(selection="all", worker_args=(), hypothesis_profile=profile, explicit_tests=())
    command = [argument for argument in command if not argument.startswith(("--junitxml=", "--polylogue-report-file="))]
    from devtools.pytest_stream_report import REPORT_FILE_OPTION

    command = [argument for argument in command if not argument.startswith(REPORT_FILE_OPTION + "=")]
    if subset is not None:
        command.remove("tests")
        command.append(subset)
    env = dict(os.environ)
    verify._normalize_managed_pytest_environment(env, command)
    for name in tuple(env):
        if name.startswith("POLYLOGUE_PYTEST_"):
            env.pop(name)
    for name in tuple(env):
        if name.startswith("PYTEST_XDIST") or name == "PYTEST_CURRENT_TEST":
            env.pop(name)
    env.pop("POLYLOGUE_SUITE_COST_DIR", None)
    env.update(
        {
            "TESTMON_DATAFILE": str(testmon_datafile(root)),
            "COVERAGE_CORE": "ctrace",
            "POLYLOGUE_PYTEST_RUN_ID": "source-authority-fixture",
            "POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE": "1",
            "POLYLOGUE_TESTMON_COMPLETE": "1",
        }
    )
    from devtools.pytest_stream_report import report_file_argument
    from devtools.verify_runs import VerifyRun, env_for_pytest_step

    run = VerifyRun(tier="all", argv=[], git_head=None, root=root)
    artifacts = run.start_step(label="pytest (all)", cmd=command)
    command.append(report_file_argument(artifacts.step_dir / "pytest-report.json"))
    env = env_for_pytest_step(env, run=run, artifacts=artifacts)
    return pytest_slot._run_held(command, cwd=str(root), env=env, stdout=None, on_exit=lambda: None)
