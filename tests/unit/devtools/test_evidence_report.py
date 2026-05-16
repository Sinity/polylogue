"""Tests for ``devtools evidence-report``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import evidence_report


def test_runs_without_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """evidence-report returns 0 even when no cache data exists."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    rc = evidence_report.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Evidence Report" in out
    assert "blocking=False" in out


def test_json_output_structure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON output has required top-level keys."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "timestamp" in payload
    assert "verify_history" in payload
    assert "contract_evidence" in payload
    assert "suppressions" in payload
    assert "witnesses" in payload
    assert "benchmark_campaigns" in payload


def test_reads_verify_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify history from .cache/verify-history.jsonl is included in output."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    history = cache / "verify-history.jsonl"
    history.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T04:00:00+00:00",
                "tier": "quick",
                "exit_code": 0,
                "total_duration_s": 20.5,
                "steps": [],
            }
        )
        + "\n"
    )
    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    runs = payload["verify_history"]["last_5_runs"]
    assert len(runs) == 1
    assert runs[0]["tier"] == "quick"
    assert runs[0]["exit_code"] == 0


def test_reads_contract_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract evidence artifacts are counted and grouped."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    evidence_dir = tmp_path / ".cache" / "verification" / "evidence"
    evidence_dir.mkdir(parents=True)
    artifact = {
        "contract": "cli.json_envelope",
        "command": "list",
        "dirty": False,
        "git_sha": "abc123",
        "facts": [],
    }
    (evidence_dir / "cli.json_envelope-abc123.json").write_text(json.dumps(artifact))

    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    evidence = payload["contract_evidence"]
    assert evidence["total_artifacts"] == 1
    assert evidence["stale_artifacts"] == 0
    assert "cli" in evidence["by_prefix"]


def test_reads_witnesses(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Witness lifecycle data is included."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    witnesses_dir = tmp_path / "tests" / "witnesses"
    witnesses_dir.mkdir(parents=True)
    witness = {
        "witness_id": "test-witness",
        "lifecycle": {"last_exercised_at": "2026-05-16T00:00:00+00:00"},
    }
    (witnesses_dir / "test-witness.witness.json").write_text(json.dumps(witness))

    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["witnesses"]["total"] == 1
    assert payload["witnesses"]["stale"] == 0


def test_blocking_false_when_no_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No verify history means blocking=False."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] is False


def test_blocking_false_when_only_one_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single failure is not enough to trigger blocking (could be transient)."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    history = cache / "verify-history.jsonl"
    entries = [
        {"timestamp": "2026-05-16T03:00:00+00:00", "tier": "quick", "exit_code": 0, "total_duration_s": 10.0},
        {"timestamp": "2026-05-16T04:00:00+00:00", "tier": "quick", "exit_code": 1, "total_duration_s": 10.0},
    ]
    history.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # One pass then one fail: last two are NOT both failures
    assert payload["blocking"] is False


def test_blocking_true_when_last_two_both_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two consecutive failures in a row triggers blocking."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    history = cache / "verify-history.jsonl"
    entries = [
        {"timestamp": "2026-05-16T02:00:00+00:00", "tier": "quick", "exit_code": 0, "total_duration_s": 10.0},
        {"timestamp": "2026-05-16T03:00:00+00:00", "tier": "quick", "exit_code": 1, "total_duration_s": 10.0},
        {"timestamp": "2026-05-16T04:00:00+00:00", "tier": "quick", "exit_code": 1, "total_duration_s": 10.0},
    ]
    history.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] is True


def test_blocking_true_when_stale_evidence_and_recent_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale contract evidence combined with a recent verify failure triggers blocking."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    history = cache / "verify-history.jsonl"
    entry = {"timestamp": "2026-05-16T04:00:00+00:00", "tier": "quick", "exit_code": 1, "total_duration_s": 10.0}
    history.write_text(json.dumps(entry) + "\n")
    # Create a stale artifact (dirty=True means it's stale)
    evidence_dir = cache / "verification" / "evidence"
    evidence_dir.mkdir(parents=True)
    artifact = {"contract": "cli.json_envelope", "dirty": True, "git_sha": None}
    (evidence_dir / "stale.json").write_text(json.dumps(artifact))
    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] is True


def test_blocking_false_when_stale_evidence_but_no_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale evidence without any verify failure is not blocking."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    history = cache / "verify-history.jsonl"
    entry = {"timestamp": "2026-05-16T04:00:00+00:00", "tier": "quick", "exit_code": 0, "total_duration_s": 10.0}
    history.write_text(json.dumps(entry) + "\n")
    evidence_dir = cache / "verification" / "evidence"
    evidence_dir.mkdir(parents=True)
    artifact = {"contract": "cli.json_envelope", "dirty": True, "git_sha": None}
    (evidence_dir / "stale.json").write_text(json.dumps(artifact))
    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] is False


def test_blocking_human_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Human-readable output reflects the computed blocking status."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    history = cache / "verify-history.jsonl"
    entries = [
        {"timestamp": "2026-05-16T03:00:00+00:00", "tier": "quick", "exit_code": 1, "total_duration_s": 10.0},
        {"timestamp": "2026-05-16T04:00:00+00:00", "tier": "quick", "exit_code": 1, "total_duration_s": 10.0},
    ]
    history.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    rc = evidence_report.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "blocking=True" in out


def test_stale_witness_detection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Witnesses not exercised in 30+ days are flagged stale."""
    monkeypatch.setattr(evidence_report, "ROOT", tmp_path)
    witnesses_dir = tmp_path / "tests" / "witnesses"
    witnesses_dir.mkdir(parents=True)
    witness = {
        "witness_id": "old-witness",
        "lifecycle": {"last_exercised_at": "2026-01-01T00:00:00+00:00"},
    }
    (witnesses_dir / "old.witness.json").write_text(json.dumps(witness))

    rc = evidence_report.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["witnesses"]["stale"] == 1
