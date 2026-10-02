"""A tiny traced corpus for actual devtools selector admission controls."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from devtools.testmon_provision import TESTMON_COVERAGE_CORE, testmon_datafile, testmon_environment
from devtools.verify_test_collection import collection_command


def seed_admission_graph(root: Path, *, checkout: Path) -> Path:
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (root / "neutral.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    testfile = root / "tests/test_nodes.py"
    testfile.write_text(
        "import pytest\nfrom neutral import value\ndef test_old():\n    assert value() == 1\n", encoding="utf-8"
    )
    (root / ".gitignore").write_text(".cache/\n__pycache__/\n.pytest_cache/\n.hypothesis/\n.benchmarks/\n")
    for arguments in (
        ["init", "-b", "feature"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"],
    ):
        subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=True)
    datafile = testmon_datafile(root)
    datafile.parent.mkdir(parents=True)
    command = collection_command(root=checkout, paths=["tests"], testmon=True)
    command.remove("--collect-only")
    command.extend(("--testmon", "--testmon-env=" + testmon_environment(root), "--testmon-noselect"))
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(root), str(checkout))),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "COVERAGE_CORE": TESTMON_COVERAGE_CORE,
            "TESTMON_DATAFILE": str(datafile),
        }
    )
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST"):
        env.pop(key, None)
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return testfile


def make_focused_checkout(root: Path) -> Path:
    """An owned path-selection root whose fake runs can publish real artifacts."""
    selected = root / "tests/unit/core/test_identity_law.py"
    selected.parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    selected.write_text("def test_session_id_is_origin_native_id():\n    pass\n", encoding="utf-8")
    (root / "tests/unit/pipeline").mkdir()
    return root
