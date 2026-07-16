#!/usr/bin/env python3
"""Test double for the Sinnix attested launcher contract."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path


def _contract(prompt: str) -> dict[str, object]:
    start = prompt.index("BEGIN JOB CONTRACT\n") + len("BEGIN JOB CONTRACT\n")
    end = prompt.index("\nEND JOB CONTRACT", start)
    value = json.loads(prompt[start:end])
    assert isinstance(value, dict)
    return value


def _event(kind: str, job_id: str) -> None:
    raw = os.environ.get("TESTSUITE_DIET_FAKE_EVENTS")
    if not raw:
        return
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps({"kind": kind, "job_id": job_id, "time": time.monotonic()}) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--last-file", type=Path, required=True)
    parser.add_argument("--schema-file", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-state-dir", type=Path, required=True)
    parser.add_argument("--job-role")
    parser.add_argument("--work-item")
    args = parser.parse_args()

    prompt_bytes = args.prompt_file.read_bytes()
    prompt = prompt_bytes.decode()
    contract = _contract(prompt)
    _event("start", args.job_id)
    time.sleep(float(os.environ.get("TESTSUITE_DIET_FAKE_DELAY", "0.05")))

    incomplete = "INCOMPLETE_PACKET" in str(contract["mission"])
    prose_only = "PROSE_ONLY" in str(contract["mission"])
    unassigned_edit = "UNASSIGNED_EDIT" in str(contract["mission"])
    control_plane_edit = "CONTROL_PLANE_EDIT" in str(contract["mission"])
    changed: list[str] = []
    if not incomplete and not prose_only:
        for rel in contract["write_files"]:
            path = args.workdir / str(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            prior = path.read_text(encoding="utf-8") if path.is_file() else ""
            path.write_text(prior + f"\n# fake worker {contract['id']}\n", encoding="utf-8")
            changed.append(str(rel))
    if unassigned_edit:
        outside = args.workdir / "outside-assignment.py"
        outside.write_text("changed outside assignment\n", encoding="utf-8")
    if control_plane_edit:
        control = args.workdir / ".agent/scratch/testsuite_diet/control.md"
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_text("worker changed ignored control input\n", encoding="utf-8")

    checks = [
        {"command": command, "exit_code": 0, "output": "fake focused check passed"}
        for command in contract["focused_tests"]
    ]
    receipt = {
        "changed_files": changed,
        "behavioral_result": "blocked on missing design decision" if incomplete else "fake implementation completed",
        "production_dependencies": ["production.route"],
        "checks": [] if incomplete else checks,
        "deleted_tests_helpers": [],
        "proposed_deletions": [],
        "sensitivity": {
            "executed": False,
            "witness": "historical seed",
            "mutation": "representative mutation",
            "result": "coordinator certification required",
            "artifact": "",
        },
        "residual_risks": [],
        "recommended_integration_checks": ["devtools verify --quick"],
        "blocker": {
            "blocked": incomplete,
            "reason": "packet intentionally incomplete" if incomplete else "",
            "decision_needed": "coordinator must define the oracle" if incomplete else "",
        },
    }
    args.last_file.parent.mkdir(parents=True, exist_ok=True)
    args.last_file.write_text(json.dumps(receipt), encoding="utf-8")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text("fake launcher completed\n", encoding="utf-8")
    args.job_state_dir.mkdir(parents=True, exist_ok=True)
    attestation = {
        "schema_version": 1,
        "job_id": args.job_id,
        "lifecycle": "completed",
        "model": args.model,
        "effort": args.reasoning_effort,
        "worktree": str(args.workdir.resolve()),
        "prompt": {"path": str(args.prompt_file), "sha256": hashlib.sha256(prompt_bytes).hexdigest()},
        "exit_status": 0,
    }
    (args.job_state_dir / f"{args.job_id}.json").write_text(json.dumps(attestation), encoding="utf-8")
    _event("end", args.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
