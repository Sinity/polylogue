from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID


def test_demo_seed_and_verify_json_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()

    seed = runner.invoke(cli, ["demo", "seed", "--with-overlays", "--format", "json"])
    assert seed.exit_code == 0, seed.output
    seed_payload = json.loads(seed.output)
    assert seed_payload["session_count"] == 3
    assert seed_payload["message_count"] == 19
    assert seed_payload["overlays_seeded"] is True

    verify = runner.invoke(cli, ["demo", "verify", "--require-overlays", "--format", "json"])
    assert verify.exit_code == 0, verify.output
    verify_payload = json.loads(verify.output)
    assert verify_payload["ok"] is True
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify_payload["query_hits"]
    assert verify_payload["absolute_path_leaks"] == []


def test_demo_script_prints_copy_pastable_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "script", "--root", str(tmp_path / "archive")])

    assert result.exit_code == 0, result.output
    assert "POLYLOGUE_ARCHIVE_ROOT" in result.output
    assert "polylogue demo seed" in result.output
    assert "polylogue demo verify" in result.output
    assert "--with-overlays --format json" in result.output
    assert "--require-overlays --format json" in result.output
    assert str(tmp_path / "archive") in result.output
