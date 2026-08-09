from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from devtools import testmon_bootstrap, verify
from devtools.testmon_bootstrap import maybe_bootstrap_testmon_seed
from devtools.testmon_state import file_fingerprint, inspect_testmon_database


def test_real_testmon_graph_copies_and_rebinds_in_a_temporary_lane(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
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
    attempt = {
        "protocol_version": 4,
        "status": "incomplete",
        "identity": {
            "git_head": "head",
            "worktree_fingerprint": "source-tree",
            "python": sys.version,
            "skip_slow": False,
            "lab": False,
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
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(testmon_bootstrap, "_git_worktree_info", lambda _root: (True, source))
        message = maybe_bootstrap_testmon_seed(lane, protocol_version=4)
        assert message is not None and "selection-only attempt receipt" in message

        local_data = lane / ".cache" / "testmon" / "testmondata"
        local_stamp = lane / ".cache" / "testmon" / "seed.json"
        local_attempt = lane / ".cache" / "testmon" / "seed-attempt.json"
        assert local_data.is_file()
        assert local_attempt.is_file()
        assert not local_stamp.exists()

        monkeypatch.chdir(lane)
        monkeypatch.setattr(verify, "ROOT", lane)
        assert verify._testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False) is None
        selected = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--testmon"],
            cwd=lane,
            env={**env, "TESTMON_DATAFILE": str(local_data)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert selected.returncode == 0, selected.stdout + selected.stderr
        assert "1 passed" in selected.stdout
        assert verify._testmon_release_baseline_permission() is False
        with sqlite3.connect(local_data) as connection:
            connection.execute("delete from test_execution_file_fp")
        assert verify._testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False) is not None
    finally:
        monkeypatch.undo()
