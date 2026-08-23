from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.maintenance.campaign_genesis import CAMPAIGN_GENESIS_SCHEMA, verify_campaign_genesis


def test_shipped_campaign_genesis_verifies_pinned_historical_blobs() -> None:
    root = Path(__file__).parents[3]

    verified = verify_campaign_genesis(root / "docs" / "campaign-genesis" / "reindex-2026.json", cwd=root)

    assert verified.campaign_id == "reindex-2026"
    assert set(verified.snapshots) == {"input_snapshot", "migration_snapshot", "formula_snapshot"}


def test_maintenance_command_verifies_a_campaign_genesis_record() -> None:
    root = Path(__file__).parents[3]

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "verify-campaign-genesis",
            "--genesis",
            str(root / "docs" / "campaign-genesis" / "reindex-2026.json"),
            "--repository",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output


def test_campaign_genesis_rejects_a_digest_that_does_not_match_its_historical_blob(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Polylogue Tests")
    historic = repo / "historical.json"
    historic.write_text('{"record":"immutable"}\n', encoding="utf-8")
    _git(repo, "add", "historical.json")
    _git(repo, "commit", "-qm", "historical evidence")
    revision = _git(repo, "rev-parse", "HEAD").strip()
    digest = hashlib.sha256(historic.read_bytes()).hexdigest()
    input_snapshot = {"revision": revision, "path": "historical.json", "sha256": digest}
    formula_snapshot = {"revision": revision, "path": "historical.json", "sha256": digest}
    genesis = {
        "schema": CAMPAIGN_GENESIS_SCHEMA,
        "campaign_id": "reindex-2026",
        "input_snapshot": input_snapshot,
        "migration_snapshot": input_snapshot,
        "formula_snapshot": formula_snapshot,
    }
    path = tmp_path / "genesis.json"
    path.write_text(json.dumps(genesis), encoding="utf-8")

    assert verify_campaign_genesis(path, cwd=repo).snapshots["input_snapshot"][0] == revision

    formula_snapshot["sha256"] = "0" * 64
    path.write_text(json.dumps(genesis), encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest does not match"):
        verify_campaign_genesis(path, cwd=repo)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout
