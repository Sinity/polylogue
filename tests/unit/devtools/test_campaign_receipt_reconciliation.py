"""The external campaign index is an auditable projection, never a second ledger."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGNS = REPO_ROOT / ".agent" / "handoffs" / "external-agent-campaigns"
WAVE = CAMPAIGNS / "2026-07-16-gpt-pro-wave"
RECONCILE = CAMPAIGNS / "reconcile_results.py"
TRIAGE = WAVE / "triage-package.py"


def run_reconcile(wave: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RECONCILE), str(wave), *flags],
        capture_output=True,
        text=True,
        check=False,
    )


def test_live_campaign_projection_matches_immutable_receipts() -> None:
    result = run_reconcile(WAVE, "--check")
    assert result.returncode == 0, result.stderr


def test_materializer_repairs_projection_without_mutating_receipt(tmp_path: Path) -> None:
    wave = tmp_path / "wave"
    shutil.copytree(WAVE, wave)
    receipt = wave / "analysis" / "results" / "analysis-01" / "a01" / "result.json"
    before = receipt.read_bytes()
    index_path = wave / "analysis" / "results" / "index.json"
    index = json.loads(index_path.read_text())
    index["attempts"][0]["state"] = "acquired"
    index["attempts"][0]["status"] = "also_acquired"
    index_path.write_text(json.dumps(index) + "\n")

    stale = run_reconcile(wave, "--check")
    assert stale.returncode == 2
    assert "does not equal receipt state" in stale.stderr
    assert "forbidden legacy status field" in stale.stderr

    materialized = run_reconcile(wave, "--write")
    assert materialized.returncode == 0, materialized.stderr
    assert receipt.read_bytes() == before
    assert "status" not in json.loads(index_path.read_text())["attempts"][0]
    assert run_reconcile(wave, "--check").returncode == 0


def test_receipt_mutation_and_ambiguous_mapping_fail_closed(tmp_path: Path) -> None:
    wave = tmp_path / "wave"
    shutil.copytree(WAVE, wave)
    receipt_path = wave / "analysis" / "results" / "analysis-01" / "a01" / "result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["state"] = "tampered"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    changed_receipt = run_reconcile(wave, "--check")
    assert changed_receipt.returncode == 2
    assert "does not equal receipt state" in changed_receipt.stderr

    index_path = wave / "analysis" / "results" / "index.json"
    index = json.loads(index_path.read_text())
    index["attempts"].append(dict(index["attempts"][0]))
    index_path.write_text(json.dumps(index) + "\n")
    ambiguous = run_reconcile(wave, "--write")
    assert ambiguous.returncode == 2
    assert "duplicate projection identity" in ambiguous.stderr


def test_triage_publishes_immutable_receipt_before_reprojecting(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns"
    wave = campaign_root / "wave"
    wave.mkdir(parents=True)
    shutil.copy2(RECONCILE, campaign_root / "reconcile_results.py")
    triage = wave / "triage-package.py"
    shutil.copy2(TRIAGE, triage)
    workload = wave / "sample"
    (workload / "results").mkdir(parents=True)
    (workload / "campaign.json").write_text(json.dumps({"campaign_id": "sample-wave", "workload_id": "sample"}))
    (workload / "results" / "index.json").write_text(
        json.dumps({"schema_version": 1, "campaign_id": "sample-wave", "workload_id": "sample", "attempts": []})
    )
    package = tmp_path / "sample.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("HANDOFF.md", "handoff")
        archive.writestr("TESTS.md", "tests")
        archive.writestr("EVIDENCE.md", "evidence")
        archive.writestr("PATCH.diff", "--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+new\n")
    result = subprocess.run(
        [
            sys.executable,
            str(triage),
            str(package),
            "--workload",
            "sample",
            "--job",
            "sample-01",
            "--attempt",
            "a01",
            "--package-revision",
            "r01",
            "--prompt-sha256",
            "a" * 64,
            "--write",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2  # no snapshot is an explicitly recorded validation gap
    receipt = workload / "results" / "sample-01" / "a01" / "result.json"
    assert receipt.exists()
    assert json.loads(receipt.read_text())["state"] == "triaged"
    projection = json.loads((workload / "results" / "index.json").read_text())
    assert projection["attempts"] == [
        {
            "job": "sample-01",
            "workload": "sample",
            "attempt_id": "a01",
            "package_revision": "r01",
            "state": "triaged",
            "artifact": "sample.zip",
            "sha256": json.loads(receipt.read_text())["artifacts"][0]["sha256"],
            "bytes": package.stat().st_size,
        }
    ]
