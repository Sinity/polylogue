"""Retention contracts for managed verification receipts."""

from __future__ import annotations

from pathlib import Path

from devtools.verify_runs import VerifyRun, append_verify_history, prune_successful_verify_runs


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

    before_history = prune_successful_verify_runs(root=tmp_path, history_path=history, max_successful=2)
    assert before_history["history_durable"] is False
    assert all((tmp_path / str(payload["artifact_dir"])).exists() for payload in successful)

    for payload in (*successful, failed, cancelled):
        append_verify_history(payload, path=history)
    receipt = prune_successful_verify_runs(root=tmp_path, history_path=history, max_successful=2)
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
