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
    recorder = suite_cost.SuiteCostRecorder(receipts, "gw0", scratch, sample_scratch=True)

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


def test_scratch_tree_walk_is_off_unless_asked_for(tmp_path: Path) -> None:
    """The always-on receipt stays O(1) per test.

    Anti-vacuity: restoring an unconditional ``sample_storage`` -- the walk that
    costs O(tests^2/cadence) because pytest keeps a directory per test under the
    basetemp -- puts the peak keys back into the default payload and reddens
    this. The tier tally and the io counters must survive that, or the receipt
    has lost the two numbers the storage budget is stated in.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recorder = suite_cost.SuiteCostRecorder(tmp_path / "receipts", "gw0", scratch)
    # After the recorder, so the write lands inside the window it measures.
    (scratch / "payload.bin").write_bytes(b"x" * 4096)
    recorder.note_test()
    recorder.sample_storage()

    payload = recorder.payload()
    assert "peak_scratch_apparent_bytes" not in payload
    assert "peak_scratch_allocated_bytes" not in payload
    assert "tier_init" in payload
    assert payload["io"]["wchar"] > 0


def test_tree_walk_stops_at_its_entry_budget(tmp_path: Path) -> None:
    """One walk is bounded, so an opted-in sample cannot dominate the run.

    Anti-vacuity: dropping the budget check makes the walk visit both files and
    report ``truncated`` as False, reddening the first two assertions.
    """
    for index in range(6):
        (tmp_path / f"file-{index}.bin").write_bytes(b"x" * 512)

    apparent, _allocated, truncated = suite_cost._tree_bytes(tmp_path, budget=2)
    assert truncated is True
    assert apparent < 6 * 512

    whole, _allocated, complete = suite_cost._tree_bytes(tmp_path)
    assert complete is False
    assert whole == 6 * 512


def test_aggregate_uses_controller_elapsed_without_summing_parallel_peaks(tmp_path: Path) -> None:
    for index, (tests, write_bytes, duration) in enumerate(((10, 400, 5.0), (30, 600, 9.0))):
        (tmp_path / f"gw{index}.json").write_text(
            json.dumps(
                {
                    "worker_id": f"gw{index}",
                    "tests": tests,
                    "duration_s": duration,
                    "io": {"write_bytes": write_bytes},
                    "tier_init": {"ops.prototype_hit": tests, "ops.ddl_fresh": 1},
                    # Distinct per worker: with both at one value, max and sum
                    # are indistinguishable and the assertion below cannot tell
                    # the reported peak from the summed peak this test forbids.
                    "peak_scratch_apparent_bytes": 100 * (index + 1),
                    "peak_scratch_allocated_bytes": 200 * (index + 1),
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
    assert aggregate["controller_duration_s"] is None
    assert aggregate["worker_active_s"] == 14.0
    assert aggregate["tier_init"] == {"ops.ddl_fresh": 2, "ops.prototype_hit": 40}
    assert aggregate["archive_tier_initializations"] == 42
    assert aggregate["peak_scratch_apparent_bytes"] == 200
    assert aggregate["peak_scratch_allocated_bytes"] == 400


def test_controller_receipt_covers_collection_and_worker_warmup(tmp_path: Path) -> None:
    (tmp_path / "master.json").write_text(
        json.dumps(
            {"worker_id": "master", "role": "controller", "tests": 0, "duration_s": 12.0, "io": {}, "tier_init": {}}
        )
    )
    (tmp_path / "gw0.json").write_text(
        json.dumps({"worker_id": "gw0", "role": "worker", "tests": 2, "duration_s": 8.0, "io": {}, "tier_init": {}})
    )

    aggregate = suite_cost.aggregate_suite_cost(tmp_path)

    assert aggregate["workers"] == 1
    assert aggregate["wall_clock_s"] == 12.0
    assert aggregate["controller_duration_s"] == 12.0
    assert aggregate["worker_active_s"] == 8.0


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
    # The controller writes an elapsed receipt but is never counted as a worker.
    names = sorted(path.name for path in receipts.glob("*.json"))
    assert names == ["gw0.json", "gw1.json", "master.json"]
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


def test_rss_trajectory_localizes_growth_to_the_tests_that_caused_it(tmp_path: Path) -> None:
    """A worker's peak grows with tests executed; the trajectory names where.

    Anti-vacuity: dropping the nodeid label, or sampling once instead of on a
    cadence, leaves a single peak number that cannot distinguish an import
    floor from growth, and this reddens on the flat-then-rising shape.
    """
    recorder = suite_cost.SuiteCostRecorder(tmp_path / "receipts", "gw0", None, sample_rss=True)
    retained: list[bytearray] = []
    for index in range(3 * suite_cost._SAMPLE_EVERY):
        recorder.note_test(f"tests/unit/{'grows' if index >= suite_cost._SAMPLE_EVERY else 'flat'}/test_{index}.py::t")
        if index >= suite_cost._SAMPLE_EVERY:
            retained.append(bytearray(512_000))

    payload = recorder.payload()
    trajectory = payload["rss_trajectory"]
    assert [point["tests"] for point in trajectory] == [suite_cost._SAMPLE_EVERY * step for step in (1, 2, 3)]
    assert trajectory[0]["nodeid"].startswith("tests/unit/flat/")
    assert trajectory[-1]["nodeid"].startswith("tests/unit/grows/")
    # The flat segment allocates nothing; the rest retains 500 x 512 kB.
    # RSS is the resident subset, not the allocated byte count, so the floor
    # is a clearly-rising hundred-mebibyte delta rather than the full 244 MiB.
    assert trajectory[-1]["rss_kib"] - trajectory[0]["rss_kib"] > 100_000
    assert payload["rss_growth_kib"] >= trajectory[-1]["rss_kib"] - payload["rss_start_kib"]
    assert not payload["rss_trajectory_truncated"]
    assert len(retained) == 2 * suite_cost._SAMPLE_EVERY


def test_rss_trajectory_is_off_unless_asked_for(tmp_path: Path) -> None:
    """An unsampled run must not read as a measured flat trajectory.

    Anti-vacuity: emitting the keys unconditionally (as zeroes or otherwise)
    reddens this, which is what would let a run nobody asked to sample be
    reported as evidence that a worker's memory does not grow.
    """
    recorder = suite_cost.SuiteCostRecorder(tmp_path / "receipts", "gw0", None)
    recorder.note_test("tests/unit/x/test_a.py::t")
    recorder.sample_memory()

    payload = recorder.payload()
    assert "rss_trajectory" not in payload
    assert "peak_rss_kib" not in payload
    assert "rss_growth_kib" not in payload
    assert "tier_init" in payload
