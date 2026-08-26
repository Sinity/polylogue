"""Canonical verification receipt and evidence-lane contracts."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from devtools.verify_runs import (
    VerifyRun,
    append_verification_evidence,
    append_verify_history,
    read_verification_evidence,
)


def _payload(tmp_path: Path) -> dict[str, object]:
    run = VerifyRun(tier="focused-test", argv=["--secret-prompt"], git_head="sha:abc", root=tmp_path)
    step = run.start_step(label="pytest focused", cmd=["pytest", "private-test.py"])
    run.finish_step(step_id=step.step_id, result={"exit": 0, "duration_s": 0.25})
    return run.finish(exit_code=0, duration_s=0.3, final_git_head="sha:def")


def test_canonical_receipt_is_bounded_and_foreground_has_no_agentctl_ids(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SINNIXD_JOB_ID", raising=False)
    monkeypatch.delenv("SINNIXD_CORRELATION_ID", raising=False)
    payload = _payload(tmp_path)
    history = tmp_path / "history.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    append_verify_history(payload, path=history)
    append_verification_evidence(payload, path=evidence)

    history_row = json.loads(history.read_text(encoding="utf-8"))
    receipt = read_verification_evidence(evidence)[0]
    assert receipt["run_id"] == payload["run_id"]
    assert receipt["source_revision"] == "sha:def"
    assert receipt["status"] == "passed"
    assert "agentctl" not in receipt
    assert "argv" not in history_row
    assert "cmd" not in json.dumps(history_row)
    assert receipt["artifact_ref"].startswith("polylogue://verification/")


def test_declared_identity_and_interruption_are_explicit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SINNIXD_JOB_ID", "job-17")
    monkeypatch.setenv("SINNIXD_CORRELATION_ID", "corr-17")
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
