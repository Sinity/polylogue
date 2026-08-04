"""Focused tests for the production corpus-fidelity devtools command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import corpus_fidelity
from polylogue.maintenance.archive_verification import (
    CORPUS_FIDELITY_CHECKS,
    verify_archive,
)
from tests.infra.workload_artifacts import SeededArchiveArtifact


def test_command_runs_registered_gate_against_real_seeded_archive(
    corpus_fidelity_archive: SeededArchiveArtifact,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = corpus_fidelity.main(["--archive-root", str(corpus_fidelity_archive.root)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Corpus fidelity:" in output
    assert "[OK] corpus-absences:" in output
    assert "[OK] corpus-attachment-fidelity:" in output
    assert "[OK] corpus-revision-fidelity:" in output
    assert "clear" in output


def test_command_binds_exact_registry_selection_and_json_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_verify_archive(
        archive_root: Path,
        *,
        checks: tuple[str, ...],
        sample_limit: int,
    ) -> object:
        captured["archive_root"] = archive_root
        captured["checks"] = checks
        captured["sample_limit"] = sample_limit
        return verify_archive(tmp_path / "missing", checks=checks, sample_limit=sample_limit)

    monkeypatch.setattr(corpus_fidelity, "verify_archive", fake_verify_archive)
    exit_code = corpus_fidelity.main(["--archive-root", str(tmp_path / "requested"), "--sample-limit", "3", "--json"])

    assert exit_code == 1
    assert captured == {
        "archive_root": tmp_path / "requested",
        "checks": CORPUS_FIDELITY_CHECKS,
        "sample_limit": 3,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] is True
    assert [check["name"] for check in payload["checks"]] == list(CORPUS_FIDELITY_CHECKS)
    assert all(check["status"] == "skip" for check in payload["checks"])
