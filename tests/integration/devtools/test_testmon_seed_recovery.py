from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from devtools.testmon_bootstrap import BootstrapDecision, bootstrap_testmon_seed_files
from devtools.testmon_state import (
    BaselineStatus,
    file_fingerprint,
    inspect_testmon_database,
    stamp_from_attempt,
    validate_stamp,
)


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
    stamp = stamp_from_attempt(attempt, data, checkout_root=source, protocol_version=4)
    assert stamp is not None and stamp.baseline_status is BaselineStatus.RED
    source_stamp = source / ".cache" / "testmon" / "seed.json"
    source_stamp.parent.mkdir(parents=True, exist_ok=True)
    source_stamp.write_text(json.dumps(stamp.as_dict()), encoding="utf-8")

    lane = tmp_path / "lane"
    local_data = lane / ".cache" / "testmon" / "testmondata"
    local_stamp = lane / ".cache" / "testmon" / "seed.json"
    decision = BootstrapDecision(
        True,
        "real graph",
        main_testmon_data=data,
        main_seed_stamp=source_stamp,
        protocol_version=4,
    )
    assert bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        checkout_root=lane,
        inherited_from=source,
    )
    rebound = validate_stamp(local_stamp, local_data, checkout_root=lane, protocol_version=4)
    assert rebound is not None
    assert rebound.baseline_status is BaselineStatus.RED
    assert rebound.binding.checkout_root == str(lane.resolve())
    assert rebound.affected_selection_allowed

    with sqlite3.connect(local_data) as connection:
        connection.execute("delete from test_execution_file_fp where test_execution_id = 1")
    assert validate_stamp(local_stamp, local_data, checkout_root=lane, protocol_version=4) is None
