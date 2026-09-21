"""Retention contracts for managed verification receipts."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

import devtools.verify_runs as verify_runs
from devtools.verify_runs import VerifyRun, append_verify_history, prune_successful_verify_runs

_TEST_NOW = datetime.fromisoformat("2026-08-24T00:00:30+00:00").timestamp()


def _finished_run(root: Path, *, index: int, exit_code: int) -> dict[str, object]:
    run = VerifyRun(tier="focused-test", argv=[], git_head="git:test", root=root, mirror_current=False)
    payload = run.finish(exit_code=exit_code, duration_s=0.1, final_git_head="git:test")
    payload["finished_at"] = f"2026-08-24T00:00:{index:02d}+00:00"
    run._payload["finished_at"] = payload["finished_at"]
    run.write()
    return payload


def test_successful_detail_retention_requires_durable_history_and_preserves_failures(tmp_path: Path) -> None:
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    successful = [_finished_run(tmp_path, index=index, exit_code=0) for index in range(10)]
    failed = _finished_run(tmp_path, index=20, exit_code=1)
    cancelled = _finished_run(tmp_path, index=21, exit_code=143)

    before_history = prune_successful_verify_runs(root=tmp_path, history_path=history, max_successful=2, now=_TEST_NOW)
    assert before_history["history_durable"] is False
    assert all((tmp_path / str(payload["artifact_dir"])).exists() for payload in successful)

    for payload in (*successful, failed, cancelled):
        append_verify_history(payload, path=history)
    receipt = prune_successful_verify_runs(root=tmp_path, history_path=history, max_successful=2, now=_TEST_NOW)
    retained = receipt["retained_run_ids"]
    pruned = receipt["pruned_run_ids"]
    assert isinstance(retained, list)
    assert isinstance(pruned, list)

    assert len(retained) == 2
    assert len(pruned) == 8
    assert (tmp_path / str(failed["artifact_dir"])).exists()
    assert (tmp_path / str(cancelled["artifact_dir"])).exists()
    assert all((tmp_path / str(payload["artifact_dir"])).exists() for payload in successful[-2:])


def test_successful_retention_keeps_malformed_detail_as_manual_evidence(tmp_path: Path) -> None:
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    payload = _finished_run(tmp_path, index=0, exit_code=0)
    run_dir = tmp_path / str(payload["artifact_dir"])
    (run_dir / "run.json").write_text("not-json\n", encoding="utf-8")
    append_verify_history(payload, path=history)

    prune_successful_verify_runs(root=tmp_path, history_path=history, max_successful=0)

    assert run_dir.exists()


def test_pruning_refuses_symlinked_runs_ancestor_without_touching_outside(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside_runs = outside / "runs"
    outside_runs.mkdir(parents=True)
    sentinel = outside_runs / "keep.txt"
    sentinel.write_text("outside evidence", encoding="utf-8")

    payload = _finished_run(repo, index=0, exit_code=0)
    history = repo / ".cache" / "verify" / "history.jsonl"
    append_verify_history(payload, path=history)
    real_runs = repo / ".cache" / "verify" / "runs"
    parked_runs = repo / ".cache" / "verify" / "runs-real"
    real_runs.rename(parked_runs)
    real_runs.symlink_to(outside_runs, target_is_directory=True)

    result = prune_successful_verify_runs(root=repo, history_path=history, max_successful=0)

    assert result["refused"] is True
    assert sentinel.read_text(encoding="utf-8") == "outside evidence"
    assert parked_runs.exists()


def test_pruning_refuses_symlinked_verify_ancestor(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / ".cache").symlink_to(outside, target_is_directory=True)

    result = prune_successful_verify_runs(root=repo)

    assert result["refused"] is True
    assert not (outside / "verify").exists()


def test_history_append_fsyncs_file_before_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_fsync = os.fsync

    def observe_fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        calls.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    append_verify_history({"run_id": "run-1", "status": "failed"}, path=history)

    assert calls[-2:] == ["file", "directory"]


def test_failed_detail_retention_is_bounded_but_history_keeps_every_summary(tmp_path: Path) -> None:
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    failures = [_finished_run(tmp_path, index=index, exit_code=1) for index in range(20)]
    for payload in failures:
        run_dir = tmp_path / str(payload["artifact_dir"])
        (run_dir / "diagnostic.log").write_bytes(b"diagnostic detail\n" * 8)
        append_verify_history(payload, path=history)

    receipt = prune_successful_verify_runs(
        root=tmp_path,
        history_path=history,
        max_successful=0,
        max_failed=3,
        max_failed_age_s=3600,
        max_failed_bytes=20,
        now=_TEST_NOW,
    )

    retained_failure_ids = cast(list[str], receipt["retained_failure_run_ids"])
    pruned_ids = cast(list[str], receipt["pruned_run_ids"])
    assert retained_failure_ids == [failures[-1]["run_id"]]
    assert len(pruned_ids) == 19
    assert (tmp_path / str(failures[-1]["artifact_dir"])).exists()
    assert len(history.read_text(encoding="utf-8").splitlines()) == 20


def test_recent_failure_is_not_erased_before_age_or_count_policy(tmp_path: Path) -> None:
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    recent = _finished_run(tmp_path, index=1, exit_code=1)
    older = _finished_run(tmp_path, index=2, exit_code=1)
    append_verify_history(recent, path=history)
    append_verify_history(older, path=history)

    receipt = prune_successful_verify_runs(
        root=tmp_path,
        history_path=history,
        max_failed=2,
        max_failed_age_s=3600,
        max_failed_bytes=10_000,
        now=_TEST_NOW,
    )

    retained_failure_ids = cast(list[str], receipt["retained_failure_run_ids"])
    pruned_ids = cast(list[str], receipt["pruned_run_ids"])
    assert set(retained_failure_ids) == {recent["run_id"], older["run_id"]}
    assert not pruned_ids
    assert (tmp_path / str(recent["artifact_dir"])).exists()
    assert (tmp_path / str(older["artifact_dir"])).exists()


def test_pruning_retains_corrupt_detail_and_skips_active_retention_lock(tmp_path: Path) -> None:
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    payload = _finished_run(tmp_path, index=0, exit_code=1)
    detail = tmp_path / str(payload["artifact_dir"])
    detail.joinpath("external").symlink_to(tmp_path, target_is_directory=True)
    append_verify_history(payload, path=history)

    first = prune_successful_verify_runs(root=tmp_path, history_path=history, max_failed=0)
    assert first["pruned_run_ids"] == []
    assert detail.exists()

    lock_path = tmp_path / ".cache" / "verify" / verify_runs._RETENTION_LOCK_NAME
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = prune_successful_verify_runs(root=tmp_path, history_path=history, max_failed=0)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert locked["retention_locked"] is True


def test_pruning_refuses_detail_tree_over_global_node_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    payload = _finished_run(tmp_path, index=0, exit_code=1)
    detail = tmp_path / str(payload["artifact_dir"])
    for index in range(5):
        (detail / f"diagnostic-{index}.log").write_text("evidence", encoding="utf-8")
    append_verify_history(payload, path=history)
    monkeypatch.setattr(verify_runs, "_DETAIL_NODE_BUDGET", 2)

    result = prune_successful_verify_runs(root=tmp_path, history_path=history, max_failed=0)

    assert result["pruned_run_ids"] == []
    assert detail.exists()


def test_coverage_pins_are_bounded_by_the_newest_skips(tmp_path: Path) -> None:
    """Anti-vacuity: pinning every historical ``covered_by_run`` keeps the
    oldest full run's detail forever."""
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    full = [_finished_run(tmp_path, index=index, exit_code=0) for index in range(4)]
    skips = []
    for index, covered in enumerate(full[:3]):
        skip = _finished_run(tmp_path, index=10 + index, exit_code=0)
        skip["diagnosis"] = "corpus_already_verified"
        skip["pytest_aggregate"] = {"covered_by_run": covered["run_id"]}
        skips.append(skip)
    for payload in (*full, *skips):
        append_verify_history(payload, path=history)

    receipt = prune_successful_verify_runs(root=tmp_path, history_path=history, max_successful=1)

    retained = set(cast(list[str], receipt["retained_run_ids"]))
    assert full[3]["run_id"] in retained, "the newest full run is retained on its own"
    assert full[2]["run_id"] in retained, "the newest skip's coverage is pinned"
    assert full[0]["run_id"] not in retained
    assert full[1]["run_id"] not in retained


def test_run_receipt_carries_the_suite_cost_beside_pytest_aggregate(tmp_path: Path) -> None:
    """The run's archive-construction cost is read from the run receipt itself.

    Anti-vacuity: dropping the ``summarize_step_receipts`` call in ``finish``
    leaves the counters reachable only by knowing the per-step suite-cost
    directory by hand, and the ``suite_cost`` assertions below go red.
    """
    run = VerifyRun(tier="focused-test", argv=[], git_head="git:test", root=tmp_path, mirror_current=False)
    artifacts = run.start_step(label="pytest-focused", cmd=["pytest"])
    suite_cost_dir = artifacts.step_dir / "suite-cost"
    suite_cost_dir.mkdir(parents=True, exist_ok=True)
    (suite_cost_dir / "gw0.json").write_text(
        '{"worker_id": "gw0", "role": "worker", "tests": 4, "duration_s": 2.0, '
        '"io": {"write_bytes": 8192}, "tier_init": {"index.ddl_reapply": 3, "index.prototype_hit": 9}}\n',
        encoding="utf-8",
    )
    from devtools.pytest_suite_cost_plugin import write_run_receipt

    write_run_receipt(suite_cost_dir)
    run.finish_step(step_id=artifacts.step_id, result={"duration_s": 2.0, "exit": 0})

    payload = run.finish(exit_code=0, duration_s=2.0, final_git_head="git:test")
    suite_cost = cast(dict[str, object], payload["suite_cost"])
    assert suite_cost["tier_init"] == {"index.ddl_reapply": 3, "index.prototype_hit": 9}
    assert suite_cost["archive_tier_initializations"] == 12
    assert suite_cost["write_bytes"] == 8192
    assert suite_cost["tests"] == 4


def test_detail_without_a_durable_history_row_is_reported_not_silently_retained(tmp_path: Path) -> None:
    """A run whose summary was never appended is outside the bound, and says so.

    Anti-vacuity: drop the orphan accounting and a checkout holding hundreds of
    pre-history detail trees looks like a bound that stopped running, which is
    what the retention report was read as.
    """
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    orphan = _finished_run(tmp_path, index=0, exit_code=0)
    recorded = _finished_run(tmp_path, index=1, exit_code=0)
    append_verify_history(recorded, path=history)

    receipt = prune_successful_verify_runs(root=tmp_path, history_path=history, max_successful=1, now=_TEST_NOW)

    assert receipt["orphaned_detail_run_ids"] == [orphan["run_id"]]
    assert (tmp_path / str(orphan["artifact_dir"])).exists()


def _running_run(root: Path, *, pid: int, job_id: str | None = None) -> Path:
    """A ``running`` receipt whose run id names ``pid`` as its owner.

    Built through the production id shape rather than a hand-written string:
    the reconciler reads the owning pid out of the run id, so a fixture that
    invented its own format would prove nothing about the real one.
    """
    run = VerifyRun(tier="all", argv=["--all"], git_head="git:test", root=root, mirror_current=False)
    stamp, tier, _pid, suffix = str(run.run_id).rsplit("-", 3)
    run_id = f"{stamp}-{tier}-{pid}-{suffix}"
    moved = root / verify_runs.VERIFY_RUNS_DIR / run_id
    run.run_dir.rename(moved)
    payload = dict(verify_runs._read_json(moved / "run.json") or {})
    payload["run_id"] = run_id
    payload["artifact_dir"] = str(verify_runs.VERIFY_RUNS_DIR / run_id)
    payload["steps"] = [{"step_id": "01-pytest", "name": "pytest", "status": "running"}]
    if job_id is not None:
        payload["agentctl_job_id"] = job_id
    verify_runs._write_json(moved / "run.json", payload)
    return moved / "run.json"


def _dead_pid() -> int:
    """A pid with no process behind it, established rather than assumed."""
    spawned = os.fork()
    if spawned == 0:  # pragma: no cover - the child never returns to pytest
        os._exit(0)
    os.waitpid(spawned, 0)
    return spawned


def test_a_killed_run_becomes_terminal_on_the_next_receipt_read(tmp_path: Path) -> None:
    """A SIGKILLed verification cannot close its own receipt; the next read must.

    ``VerifyRun.finish`` is the only in-process exit from ``running``, and
    systemd-oomd sends SIGKILL, so four consecutive scheduled runs sat
    ``running`` forever with nothing that would ever correct them.

    Anti-vacuity: remove the reconciler call (or the status rewrite inside it)
    and the receipt stays ``running`` and the history row is never appended --
    both assertions below go red.
    """
    runs_root = tmp_path / verify_runs.VERIFY_RUNS_DIR
    path = _running_run(tmp_path, pid=_dead_pid())

    reconciled = verify_runs.reconcile_and_record_abandoned_verify_runs(runs_root=runs_root)

    assert [entry["run_id"] for entry in reconciled] == [path.parent.name]
    payload = cast(dict[str, object], verify_runs._read_json(path))
    assert payload["status"] == "failed"
    assert payload["diagnosis"] == verify_runs.ABANDONED_DIAGNOSIS
    assert payload["finished_at"]
    assert payload["steps"][0]["status"] == "failed"  # type: ignore[index]
    # Terminal for every downstream reader, not merely on disk.
    assert verify_runs._terminal_status(payload) == "failed"
    assert verify_runs.canonical_verification_receipt(payload)["status"] != "running"
    history = verify_runs._read_history_pinned(tmp_path / verify_runs.VERIFY_HISTORY_PATH)
    assert [row["run_id"] for row in history] == [path.parent.name]


def test_a_running_run_with_a_live_owner_is_left_alone(tmp_path: Path) -> None:
    """A verification still running owns its own receipt.

    Anti-vacuity: reconcile on status alone, without the liveness check, and
    this run's receipt is overwritten as abandoned while its process is still
    writing steps into it.
    """
    runs_root = tmp_path / verify_runs.VERIFY_RUNS_DIR
    path = _running_run(tmp_path, pid=os.getpid())

    assert verify_runs.reconcile_and_record_abandoned_verify_runs(runs_root=runs_root) == []

    payload = cast(dict[str, object], verify_runs._read_json(path))
    assert payload["status"] == "running"
    assert "diagnosis" not in payload
    assert not (tmp_path / verify_runs.VERIFY_HISTORY_PATH).exists()


def test_an_abandoned_run_adopts_the_agentctl_outcome_when_one_exists(tmp_path: Path) -> None:
    """The authoritative ending was recorded next door the whole time.

    Anti-vacuity: drop the adoption and the run reports only that it stopped,
    with no exit code and no cancellation -- the ``cancelled`` assertion and
    the exit code both go red.
    """
    runs_root = tmp_path / verify_runs.VERIFY_RUNS_DIR
    state_root = tmp_path / "agentctl-jobs"
    state_root.mkdir()
    (state_root / "polylogue-verify_all-6e84077f.outcome").write_text(
        '{"exit_code": 130, "outcome": "cancelled", "pool": "pytest-heavy"}', encoding="utf-8"
    )
    path = _running_run(tmp_path, pid=_dead_pid(), job_id="polylogue-verify_all-6e84077f")

    verify_runs.reconcile_and_record_abandoned_verify_runs(runs_root=runs_root, state_root=state_root)

    payload = cast(dict[str, object], verify_runs._read_json(path))
    assert payload["exit_code"] == 130
    assert payload["diagnosis"] == verify_runs.ABANDONED_DIAGNOSIS
    assert payload["agentctl_outcome_adopted"] is True
    assert verify_runs._terminal_status(payload) == "cancelled"


def test_a_run_id_without_an_owning_pid_is_not_reconciled(tmp_path: Path) -> None:
    """An id this reconciler cannot read an owner from proves nothing.

    Anti-vacuity: treat an unparseable id as abandoned and any receipt written
    by a future id shape is closed out while its run is still going.
    """
    runs_root = tmp_path / verify_runs.VERIFY_RUNS_DIR
    run_dir = runs_root / "handwritten-run"
    run_dir.mkdir(parents=True)
    verify_runs._write_json(run_dir / "run.json", {"run_id": "handwritten-run", "status": "running", "steps": []})

    assert verify_runs.reconcile_abandoned_verify_runs(runs_root=runs_root) == []
    assert cast(dict[str, object], verify_runs._read_json(run_dir / "run.json"))["status"] == "running"


def test_three_differently_killed_runs_do_not_collapse_into_one_ending(tmp_path: Path) -> None:
    """polylogue-yk0zz: the killer is on disk, so the receipt must keep it.

    AgentCTL's own ``outcome`` bucket is coarse -- an oom-kill and an ordinary
    non-zero exit are both ``"failed"`` -- while ``systemd_result`` names the
    killer. These three payloads are the shapes the 2026-09-17, 09-19 and
    09-20 scheduled corpus runs actually left behind (exit 124 timeout, exit
    137 oom-kill, exit 130 cancelled).

    Anti-vacuity: drop ``systemd_result`` from the adoption and the oom-killed
    run is indistinguishable from any other ``failed`` -- the ``oom-kill``
    assertion goes red and the campaign is back to reading journals by hand.
    """
    runs_root = tmp_path / verify_runs.VERIFY_RUNS_DIR
    state_root = tmp_path / "agentctl-jobs"
    state_root.mkdir()
    endings = {
        "df109b43": (124, "timeout", "timeout"),
        "b0ccb32f": (137, "failed", "oom-kill"),
        "6e84077f": (130, "cancelled", None),
    }
    paths: dict[str, Path] = {}
    for suffix, (exit_code, outcome, systemd_result) in endings.items():
        job_id = f"polylogue-verify_all-{suffix}"
        (state_root / f"{job_id}.outcome").write_text(
            json.dumps(
                {
                    "exit_code": exit_code,
                    "outcome": outcome,
                    "systemd_result": systemd_result,
                    "unit": f"agentctl-pytest-heavy-{job_id}.service",
                }
            ),
            encoding="utf-8",
        )
        paths[suffix] = _running_run(tmp_path, pid=_dead_pid(), job_id=job_id)

    verify_runs.reconcile_and_record_abandoned_verify_runs(runs_root=runs_root, state_root=state_root)

    adopted = {suffix: cast(dict[str, object], verify_runs._read_json(path)) for suffix, path in paths.items()}
    assert adopted["df109b43"]["termination_killer"] == "timeout"
    assert adopted["b0ccb32f"]["termination_killer"] == "oom-kill"
    # The coarse bucket alone would have made this one look ordinary.
    assert adopted["b0ccb32f"]["termination_reason"] == "failed"
    # A clean cancel records no killer rather than inventing one.
    assert "termination_killer" not in adopted["6e84077f"]
    assert adopted["6e84077f"]["termination_reason"] == "cancelled"
    # The journal query a reader would need next.
    assert adopted["b0ccb32f"]["termination_unit"] == ("agentctl-pytest-heavy-polylogue-verify_all-b0ccb32f.service")
