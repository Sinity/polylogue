"""Retention contracts for managed verification receipts."""

from __future__ import annotations

import fcntl
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
