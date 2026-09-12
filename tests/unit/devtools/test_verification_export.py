"""Tests for the stable, read-only verification receipt export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.verification_export import SCHEMA, normalized_export


def _write_receipt(root: Path, run_id: str) -> None:
    path = root / ".cache" / "verify" / "runs" / run_id / "run.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "git_head": "start-sha",
                "final_git_head": "tested-sha",
                "git_dirty": False,
                "final_git_dirty": False,
                "status": "success",
                "exit_code": 0,
                "pytest_aggregate": {"outcomes": {"passed": 1}},
                "steps": [
                    {
                        "name": "pytest focused",
                        "cmd": ["python", "-m", "pytest", "tests/unit/example.py"],
                        "duration_s": 1.25,
                        "exit": 0,
                        "runner": "managed",
                        "hypothesis_profile": "verify",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_normalized_export_is_versioned_and_receipt_only(tmp_path: Path) -> None:
    run_id = "20260911T000000Z-focused-test-1234-deadbeef"
    _write_receipt(tmp_path, run_id)

    exported = normalized_export(tmp_path, run_id)

    assert exported == {
        "schema": SCHEMA,
        "checkout": str(tmp_path),
        "run_ref": f".cache/verify/runs/{run_id}/run.json",
        "run_id": run_id,
        "initial_sha": "start-sha",
        "final_sha": "tested-sha",
        "git_dirty_before": False,
        "git_dirty_after": False,
        "tested_sha": None,
        "tested_sha_reason": "checkout HEAD changed during verification",
        "status": "success",
        "exit_code": 0,
        "hypothesis_profile": "verify",
        "commands": [["python", "-m", "pytest", "tests/unit/example.py"]],
        "phases": [{"name": "pytest focused", "duration_s": 1.25, "exit": 0, "runner": "managed"}],
        "coverage": {"outcomes": {"passed": 1}},
    }


def test_normalized_export_refuses_to_attest_a_dirty_checkout(tmp_path: Path) -> None:
    run_id = "20260911T000000Z-focused-test-1234-dirty"
    _write_receipt(tmp_path, run_id)
    receipt = tmp_path / ".cache" / "verify" / "runs" / run_id / "run.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload.update({"git_head": "same-sha", "final_git_head": "same-sha", "git_dirty": True})
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    exported = normalized_export(tmp_path, run_id)

    assert exported["tested_sha"] is None
    assert exported["tested_sha_reason"] == "checkout was dirty or final cleanliness was not recorded"


@pytest.mark.parametrize("run_ref", ["../outside", "/tmp/receipt.json", "nested/run"])
def test_normalized_export_rejects_non_run_ids(tmp_path: Path, run_ref: str) -> None:
    with pytest.raises(ValueError, match="one managed run id"):
        normalized_export(tmp_path, run_ref)
