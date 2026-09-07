"""The suite-cost receipt must name real archive construction and real write bytes.

Anti-vacuity: a receipt that reported a constant, or that lost the tier-init
tally across an xdist fan-out, would leave a suite-cost change unmeasurable.
Each test below drives the recorder over work whose cost is known and asserts
the receipt moves with it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from devtools import pytest_suite_cost_plugin as suite_cost
from devtools.pytest_invocation import CLEAR_CONFIGURED_ADDOPTS, SUITE_COST_PLUGIN_NAME


def test_receipt_records_tier_construction_and_write_bytes(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    receipts = tmp_path / "receipts"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recorder = suite_cost.SuiteCostRecorder(receipts, "gw0", scratch)

    initialize_active_archive_root(scratch / "archive")
    recorder.note_test()

    payload = json.loads(recorder.write().read_text())
    assert payload["worker_id"] == "gw0"
    assert payload["tests"] == 1
    # Six tiers were materialised through the production route; whichever way
    # each resolved, the tally has to see them.
    assert sum(payload["tier_init"].values()) >= 6
    assert payload["io"]["wchar"] > 0
    assert payload["peak_scratch_apparent_bytes"] > 0


def test_aggregate_sums_workers_and_takes_wall_clock_as_the_longest(tmp_path: Path) -> None:
    for index, (tests, write_bytes, duration) in enumerate(((10, 400, 5.0), (30, 600, 9.0))):
        (tmp_path / f"gw{index}.json").write_text(
            json.dumps(
                {
                    "worker_id": f"gw{index}",
                    "tests": tests,
                    "duration_s": duration,
                    "io": {"write_bytes": write_bytes},
                    "tier_init": {"ops.prototype_hit": tests, "ops.ddl_fresh": 1},
                    "peak_scratch_apparent_bytes": 100,
                    "peak_scratch_allocated_bytes": 200,
                }
            )
        )

    aggregate = suite_cost.aggregate_suite_cost(tmp_path)

    assert aggregate["workers"] == 2
    assert aggregate["tests"] == 40
    assert aggregate["io"]["write_bytes"] == 1000
    assert aggregate["write_bytes_per_test"] == 25.0
    # Workers run concurrently: summing their durations would misreport the run.
    assert aggregate["wall_clock_s"] == 9.0
    assert aggregate["tier_init"] == {"ops.ddl_fresh": 2, "ops.prototype_hit": 40}
    assert aggregate["archive_tier_initializations"] == 42
    assert aggregate["peak_scratch_apparent_bytes"] == 200


@pytest.mark.parametrize(
    ("configured", "expect_receipt"),
    [(True, True), (False, False)],
    ids=["configured", "unset"],
)
def test_receipt_appears_only_when_the_directory_is_configured(
    tmp_path: Path,
    configured: bool,
    expect_receipt: bool,
) -> None:
    """The same child run, with and without the switch: nothing is written by default."""
    test_path = tmp_path / "test_inert.py"
    test_path.write_text("def test_ok() -> None:\n    assert True\n")
    receipts = tmp_path / "receipts"

    result = _run_child(
        test_path,
        env_overrides={suite_cost.SUITE_COST_DIR_ENV: str(receipts)} if configured else {},
        cwd=Path(__file__).resolve().parents[3],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert receipts.exists() is expect_receipt
    if expect_receipt:
        assert [path.name for path in receipts.glob("*.json")] == ["master.json"]


def test_plugin_writes_one_receipt_per_worker_under_xdist(tmp_path: Path) -> None:
    test_path = tmp_path / "test_fanout.py"
    test_path.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('index', range(4))\n"
        "def test_ok(index: int) -> None:\n    assert index >= 0\n"
    )
    receipts = tmp_path / "receipts"

    result = _run_child(
        test_path,
        env_overrides={suite_cost.SUITE_COST_DIR_ENV: str(receipts)},
        cwd=Path(__file__).resolve().parents[3],
        extra=["-p", "xdist", "-n", "2"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # The controller writes nothing: counting it would report three workers
    # sharing the run's write bytes when only two built anything.
    names = sorted(path.name for path in receipts.glob("*.json"))
    assert names == ["gw0.json", "gw1.json"]
    aggregate = suite_cost.aggregate_suite_cost(receipts)
    assert aggregate["workers"] == 2
    assert aggregate["tests"] == 4


def test_run_receipt_sums_workers_and_is_not_counted_by_a_later_aggregate(tmp_path: Path) -> None:
    """The written run total must not become a worker of the next aggregation.

    Anti-vacuity: dropping the run-receipt exclusion from the worker glob
    doubles ``tests`` on the second call below.
    """
    for index in range(2):
        (tmp_path / f"gw{index}.json").write_text(
            json.dumps({"worker_id": f"gw{index}", "tests": 5, "io": {"write_bytes": 100}, "tier_init": {}})
        )

    written = suite_cost.write_run_receipt(tmp_path)

    assert written == tmp_path / suite_cost.RUN_RECEIPT_NAME
    assert json.loads(written.read_text())["tests"] == 10
    assert suite_cost.aggregate_suite_cost(tmp_path)["tests"] == 10


def test_run_receipt_is_absent_without_a_directory_or_workers(tmp_path: Path) -> None:
    """No switch and no receipts mean no file, never an empty zero-valued total."""
    assert suite_cost.write_run_receipt("") is None
    assert suite_cost.write_run_receipt(tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_managed_pytest_step_sets_the_receipt_directory_itself(tmp_path: Path) -> None:
    """A managed run measures itself without an ambient switch.

    Anti-vacuity: the queue reduces the submitting client's environment, so a
    step that only read ``POLYLOGUE_SUITE_COST_DIR`` from the ambient
    environment produced no receipt on any managed route. Dropping the
    assignment leaves the key absent below.
    """
    from devtools.verify_runs import VerifyRun, env_for_pytest_step

    run = VerifyRun(tier="focused-test", argv=["tests"], git_head="0" * 40, root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])

    resolved = env_for_pytest_step({}, run=run, artifacts=artifacts)
    assert resolved[suite_cost.SUITE_COST_DIR_ENV] == str(artifacts.step_dir / "suite-cost")

    # An explicit directory still wins, so a comparison run can collect
    # several steps' receipts in one place.
    override = env_for_pytest_step(
        {suite_cost.SUITE_COST_DIR_ENV: str(tmp_path / "elsewhere")},
        run=run,
        artifacts=artifacts,
    )
    assert override[suite_cost.SUITE_COST_DIR_ENV] == str(tmp_path / "elsewhere")


def _run_child(
    test_path: Path,
    *,
    env_overrides: dict[str, str],
    cwd: Path,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop(suite_cost.SUITE_COST_DIR_ENV, None)
    env.update(env_overrides)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            CLEAR_CONFIGURED_ADDOPTS,
            "-p",
            SUITE_COST_PLUGIN_NAME,
            *(extra or []),
            str(test_path),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
