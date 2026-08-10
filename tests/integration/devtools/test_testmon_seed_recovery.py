from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from devtools import testmon_bootstrap, testmon_state, verify
from devtools.testmon_state import file_fingerprint, inspect_testmon_database


def test_real_testmon_graph_copies_and_rebinds_in_a_temporary_lane(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname = "polylogue"\n', encoding="utf-8")
    (source / "test_sample.py").write_text(
        "def test_passed():\n    assert 1 == 1\n\ndef test_failed():\n    assert 1 == 2\n",
        encoding="utf-8",
    )
    data = source / ".cache" / "testmon" / "testmondata"
    data.parent.mkdir(parents=True)
    env = os.environ.copy()
    env["TESTMON_DATAFILE"] = str(data)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--testmon", "--testmon-noselect"],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    expected = ("test_sample.py::test_passed", "test_sample.py::test_failed")
    assert inspect_testmon_database(data, expected).usable_for_selection
    runtime_identity = testmon_state.testmon_runtime_identity(source)
    assert runtime_identity is not None
    dependency_environment, pytest_harness = runtime_identity
    attempt = {
        "protocol_version": verify.TESTMON_SEED_PROTOCOL_VERSION,
        "status": "reusable",
        "outcome": "red-baseline",
        "identity": {
            "git_head": "head",
            "worktree_fingerprint": "source-tree",
            "python": sys.version,
            "skip_slow": False,
            "lab": False,
            "dependency_environment": dependency_environment,
            "pytest_harness": pytest_harness,
        },
        "selection": {"selected_count": 2, "selected_nodeids_omitted": 0},
        "expected_nodeids": list(expected),
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
        "node_outcomes": [
            {"nodeid": expected[0], "outcome": "passed"},
            {"nodeid": expected[1], "outcome": "failed"},
        ],
        "exit_code": 1,
        "run_id": "real-testmon",
        "artifact_dir": ".cache/verify/runs/real-testmon",
        "testmon_data": file_fingerprint(data),
    }
    artifact_dir = source / ".cache" / "verify" / "runs" / "real-testmon"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "real-testmon",
                "checkout_root": str(source.resolve()),
                "artifact_dir": ".cache/verify/runs/real-testmon",
            }
        ),
        encoding="utf-8",
    )
    source_attempt = source / ".cache" / "testmon" / "seed-attempt.json"
    source_attempt.parent.mkdir(parents=True, exist_ok=True)
    source_attempt.write_text(json.dumps(attempt), encoding="utf-8")

    lane = tmp_path / "lane"
    lane.mkdir()
    (lane / "test_sample.py").write_text(
        "def test_passed():\n    assert 1 == 1\n\ndef test_failed():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(testmon_bootstrap, "_git_worktree_info", lambda _root: (True, source))
    copy_calls: list[tuple[Path, Path]] = []
    original_copy = testmon_bootstrap._atomic_copy_sqlite_db

    def counted_copy(src: Path, dst: Path) -> None:
        copy_calls.append((src, dst))
        original_copy(src, dst)

    monkeypatch.setattr(testmon_bootstrap, "_atomic_copy_sqlite_db", counted_copy)
    (lane / "pyproject.toml").write_text('[project]\nname = "polylogue"\n', encoding="utf-8")
    (lane / "polylogue" / "cli").mkdir(parents=True)
    (lane / "polylogue" / "__init__.py").write_text("", encoding="utf-8")
    (lane / "polylogue" / "cli" / "click_app.py").write_text("", encoding="utf-8")

    local_data = lane / ".cache" / "testmon" / "testmondata"
    local_stamp = lane / ".cache" / "testmon" / "seed.json"
    local_attempt = lane / ".cache" / "testmon" / "seed-attempt.json"

    monkeypatch.chdir(lane)
    monkeypatch.setattr(verify, "ROOT", lane)
    monkeypatch.setattr("devtools.checkout_guard.resolved_polylogue_path", lambda: lane / "polylogue" / "__init__.py")
    monkeypatch.setattr("devtools.checkout_guard._is_linked_worktree", lambda _root: True)
    monkeypatch.setattr("devtools.checkout_guard._python_environment_root", lambda _executable: lane)
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [("pytest testmon", ["pytest"])])
    run_count = 0

    def fake_run(*_args: object, **_kwargs: object) -> tuple[int, float, dict[str, object]]:
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            with sqlite3.connect(local_data) as connection:
                connection.execute("update test_execution set failed = 0 where test_name = ?", (expected[1],))
            return 1, 0.01, {"selected_count": 1}
        return 0, 0.01, {"selected_count": 1}

    monkeypatch.setattr(verify, "_run", fake_run)
    monkeypatch.setattr(verify, "_changed_executable_paths", lambda: ())
    monkeypatch.setattr(verify, "_stamp_head", lambda: None)

    assert verify.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert local_data.is_file()
    assert local_attempt.is_file()
    assert not local_stamp.exists()
    assert result["steps"][0]["selected_count"] == 1
    assert result["release_baseline_allowed"] is False
    assert verify._testmon_release_baseline_permission() is False
    assert not local_stamp.exists()
    assert len(copy_calls) == 1
    assert (lane / ".cache" / "verify" / "current-run.json").is_file()
    refreshed_attempt = json.loads(local_attempt.read_text())
    assert refreshed_attempt["testmon_data"] == file_fingerprint(local_data)
    current_run = json.loads((lane / ".cache" / "verify" / "current-run.json").read_text())
    assert refreshed_attempt["run_id"] == current_run["run_id"]

    assert verify.main([]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["steps"][0]["selected_count"] == 1
    assert len(copy_calls) == 1

    assert verify.main([]) == 0
    third = json.loads(capsys.readouterr().out)
    assert third["steps"][0]["selected_count"] == 1
    assert len(copy_calls) == 1

    with sqlite3.connect(local_data) as connection:
        connection.execute("delete from test_execution_file_fp")
    assert verify.main([]) == 2
