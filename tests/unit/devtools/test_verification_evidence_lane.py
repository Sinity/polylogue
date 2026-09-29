"""Canonical verification receipt and evidence-lane contracts."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from devtools.verify_runs import (
    VerifyRun,
    append_verification_evidence,
    append_verify_history,
    read_verification_evidence,
    verification_evidence_path,
)


def _payload(tmp_path: Path) -> dict[str, object]:
    run = VerifyRun(tier="focused-test", argv=["--secret-prompt"], git_head="sha:abc", root=tmp_path)
    step = run.start_step(label="pytest focused", cmd=["pytest", "private-test.py"])
    run.finish_step(step_id=step.step_id, result={"exit": 0, "duration_s": 0.25})
    return run.finish(exit_code=0, duration_s=0.3, final_git_head="sha:def")


def test_canonical_receipt_is_bounded_and_foreground_has_no_agentctl_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for variable in ("AGENTCTL_JOB_ID", "AGENTCTL_CORRELATION_ID", "SINNIXD_JOB_ID", "SINNIXD_CORRELATION_ID"):
        monkeypatch.delenv(variable, raising=False)
    payload = _payload(tmp_path)
    history = tmp_path / "history.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    append_verify_history(payload, path=history)
    append_verification_evidence(payload, path=evidence)

    history_row = json.loads(history.read_text(encoding="utf-8"))
    receipt = read_verification_evidence(evidence)[0]
    assert receipt["run_id"] == payload["run_id"]
    assert receipt["source_revision"] is None
    assert receipt["git_dirty"] is True
    assert receipt["status"] == "passed"
    assert "agentctl" not in receipt
    assert "argv" not in history_row
    assert "cmd" not in json.dumps(history_row)
    assert receipt["artifact_ref"].startswith("polylogue://verification/")


def test_evidence_receipt_drops_the_slot_log_path(tmp_path: Path) -> None:
    """The durable row keeps the slot outcome but not its checkout-local log path.

    Anti-vacuity: copy the slot receipt through unchanged and ``log_path``
    reappears in the row.
    """
    run = VerifyRun(tier="focused-test", argv=[], git_head="sha:abc", root=tmp_path)
    step = run.start_step(label="pytest focused", cmd=["pytest"])
    slot = {"status": "passed", "exit_code": 0, "log_path": str(tmp_path / ".cache" / "slot.log")}
    run.finish_step(step_id=step.step_id, result={"exit": 0, "duration_s": 0.1, "pytest_slot_receipt": slot})
    payload = run.finish(exit_code=0, duration_s=0.2, final_git_head="sha:abc")
    evidence = tmp_path / "evidence.jsonl"
    append_verification_evidence(payload, path=evidence)

    [row] = read_verification_evidence(evidence)
    assert row["steps"][0]["pytest_slot_receipt"] == {"status": "passed", "exit_code": 0}
    assert str(tmp_path) not in json.dumps(row)


def test_history_exposes_declared_agentctl_join_identity_without_lifecycle_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENTCTL_JOB_ID", "job-join")
    monkeypatch.setenv("AGENTCTL_CORRELATION_ID", "corr-join")
    run = VerifyRun(
        tier="quick", argv=["--quick"], git_head="sha:abc", root=tmp_path, agentctl_operation="verify_quick"
    )
    payload = run.finish(exit_code=0, duration_s=0.1, final_git_head="sha:def")
    history = tmp_path / "history.jsonl"
    append_verify_history(payload, path=history)

    row = json.loads(history.read_text(encoding="utf-8"))
    assert row["agentctl"] == {"job_id": "job-join", "correlation_id": "corr-join"}
    assert "status" not in row["agentctl"]
    assert "phase" not in row["agentctl"]


@pytest.mark.parametrize("prefix", ["AGENTCTL_", "SINNIXD_"])
def test_declared_identity_and_interruption_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefix: str
) -> None:
    for variable in ("AGENTCTL_JOB_ID", "AGENTCTL_CORRELATION_ID", "SINNIXD_JOB_ID", "SINNIXD_CORRELATION_ID"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(f"{prefix}JOB_ID", "job-17")
    monkeypatch.setenv(f"{prefix}CORRELATION_ID", "corr-17")
    run = VerifyRun(tier="quick", argv=[], git_head="sha:abc", root=tmp_path, agentctl_operation="verify_quick")
    payload = run.finish(
        exit_code=143,
        duration_s=1.0,
        diagnosis="verification_interrupted",
        final_git_head="sha:abc",
        pytest_aggregate={"termination_reason": "sigterm"},
    )
    evidence = tmp_path / "evidence.jsonl"
    append_verification_evidence(payload, path=evidence)
    receipt = read_verification_evidence(evidence)[0]
    assert receipt["agentctl"] == {"job_id": "job-17", "correlation_id": "corr-17"}
    assert receipt["status"] == "interrupted"
    assert receipt["semantic_status"] == "failed"


def test_concurrent_evidence_appends_keep_complete_json_rows(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    payloads = []
    for index in range(12):
        payload = _payload(tmp_path)
        payload["run_id"] = f"run-{index}"
        payloads.append(payload)
    errors: list[BaseException] = []

    def append(payload: dict[str, object]) -> None:
        try:
            append_verification_evidence(payload, path=evidence)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=append, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    rows = read_verification_evidence(evidence)
    assert {row["run_id"] for row in rows} == {f"run-{index}" for index in range(12)}
    assert len(evidence.read_text(encoding="utf-8").splitlines()) == 12


def test_evidence_lane_outlives_the_checkout_that_ran_the_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default lane is user state, so removing a worktree keeps its runs.

    Anti-vacuity: point the default back at the checkout's ``.cache/verify``
    and the row disappears with the removed checkout.
    """
    import shutil

    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.delenv("POLYLOGUE_VERIFICATION_EVIDENCE_PATH", raising=False)
    checkout = tmp_path / "worktree"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    payload = _payload(checkout)

    append_verification_evidence(payload)
    shutil.rmtree(checkout)

    lane = verification_evidence_path()
    assert lane == state / "polylogue" / "verification" / "evidence.jsonl"
    assert [row["run_id"] for row in read_verification_evidence(lane)] == [payload["run_id"]]


def test_configured_evidence_path_overrides_the_state_default(tmp_path: Path) -> None:
    configured = tmp_path / "elsewhere" / "evidence.jsonl"
    env = {"POLYLOGUE_VERIFICATION_EVIDENCE_PATH": str(configured), "XDG_STATE_HOME": str(tmp_path / "state")}
    assert verification_evidence_path(env) == configured
    assert verification_evidence_path({"HOME": str(tmp_path)}) == (
        tmp_path / ".local" / "state" / "polylogue" / "verification" / "evidence.jsonl"
    )


def test_reconciled_abandoned_run_joins_the_durable_lane(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A run stranded as ``running`` is recorded in the durable lane once reconciled.

    Anti-vacuity: route the checkout's reconciled append back to
    ``runs_root.parent`` (as #5720 did) and the durable lane stays empty.
    """
    from devtools import verify_runs
    from devtools.verify_runs import reconcile_and_record_abandoned_verify_runs

    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.delenv("POLYLOGUE_VERIFICATION_EVIDENCE_PATH", raising=False)
    checkout = tmp_path / "worktree"
    # This cache is the running checkout's own, as ``devtools verify`` passes it.
    monkeypatch.setattr(verify_runs, "_checkout_root", lambda: checkout.resolve())
    # A pid above any pid_max: no process owns it, so the run is abandoned.
    run_id = f"20260101T000000Z-focused-test-{2**31 - 2}-deadbeef"
    runs_root = checkout / ".cache" / "verify" / "runs"
    (runs_root / run_id).mkdir(parents=True)
    stranded = {
        "run_id": run_id,
        "tier": "focused-test",
        "status": "running",
        "started_at": "2026-01-01T00:00:00Z",
        "steps": [],
    }
    (runs_root / run_id / "run.json").write_text(json.dumps(stranded), encoding="utf-8")

    reconciled = reconcile_and_record_abandoned_verify_runs(runs_root=runs_root, state_root=tmp_path / "jobs")

    assert [entry["run_id"] for entry in reconciled] == [run_id]
    rows = read_verification_evidence(verification_evidence_path())
    assert [row["run_id"] for row in rows] == [run_id]
    assert not (checkout / ".cache" / "verify" / "evidence.jsonl").exists()
    # The checkout's history is the shared XDG history too, not a cache file.
    history = [row["run_id"] for row in verify_runs._iter_history_pinned(verify_runs.verify_history_path())]
    assert history == [run_id]
    assert not (checkout / ".cache" / "verify" / "history.jsonl").exists()


def test_relocated_cache_reconciles_into_its_own_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cache outside the running checkout keeps its reconciled evidence local.

    Anti-vacuity: send every reconciled run to the durable lane and this
    foreign cache's run appears in the operator's lane.
    """
    from devtools.verify_runs import reconcile_and_record_abandoned_verify_runs

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("POLYLOGUE_VERIFICATION_EVIDENCE_PATH", raising=False)
    run_id = f"20260101T000000Z-focused-test-{2**31 - 2}-deadbeef"
    runs_root = tmp_path / "elsewhere" / ".cache" / "verify" / "runs"
    (runs_root / run_id).mkdir(parents=True)
    (runs_root / run_id / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tier": "focused-test",
                "status": "running",
                "started_at": "2026-01-01T00:00:00Z",
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    reconcile_and_record_abandoned_verify_runs(runs_root=runs_root, state_root=tmp_path / "jobs")

    assert read_verification_evidence(verification_evidence_path()) == []
    local = read_verification_evidence(runs_root.parent / "evidence.jsonl")
    assert [row["run_id"] for row in local] == [run_id]
