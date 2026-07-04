from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.degraded_archive_proof import main, run_degraded_archive_proof


def test_degraded_archive_proof_heals_rebuildable_state(tmp_path: Path) -> None:
    result = run_degraded_archive_proof(tmp_path / "proof")

    assert result.ok is True
    assert result.archive_preserved is False
    assert result.demo_verified is True
    assert result.fts_ready_clean is True
    assert result.fts_ready_degraded is False
    assert result.fts_ready_after is True
    assert result.wal_degraded_observed is True
    assert result.fts_repair_success is True
    assert result.optimize_ran >= 1
    assert result.checkpoint_errors == ()
    assert result.optimize_errors == ()
    assert Path(result.artifact_json).exists()
    assert Path(result.artifact_markdown).exists()
    assert not Path(result.archive_root).exists()


def test_degraded_archive_proof_json_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--out-dir", str(tmp_path / "proof"), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["archive_preserved"] is False
    assert payload["fts_ready_clean"] is True
    assert payload["fts_ready_degraded"] is False
    assert payload["fts_ready_after"] is True
    assert payload["wal_degraded_observed"] is True
