"""End-to-end proof of the continuity replay artifact.

Runs the full wiring: t8t's real continuity scenario catalog over real MCP
stdio JSON-RPC against a freshly seeded synthetic archive, the real
``polylogue.insights.work_effects`` adapters against a genuine (fixture) git
repository and Beads ledger, and the real query-discovery catalog -- combined
into one JSON artifact.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from devtools.mandate_continuity_replay import main, run_mandate_continuity_replay


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "agent@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Agent"], check=True)


def _commit(path: Path, *, filename: str, message: str) -> None:
    (path / filename).write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", filename], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", message], check=True)


@pytest.fixture(scope="module")
def repo_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    repo = tmp_path_factory.mktemp("mandate-replay-repo") / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="feat: land the mandate replay wiring (Ref polylogue-z9gh.7)")
    ledger = repo.parent / "interactions.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "id": "int-mandate-close",
                "kind": "field_change",
                "created_at": "2026-07-20T00:00:00Z",
                "actor": "Sinity",
                "issue_id": "polylogue-z9gh.7",
                "extra": {"field": "status", "old_value": "open", "new_value": "closed"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return repo, ledger


@pytest.mark.asyncio
async def test_mandate_continuity_replay_end_to_end_synthetic_lane(repo_fixture: tuple[Path, Path]) -> None:
    repo_path, ledger_path = repo_fixture

    report = await run_mandate_continuity_replay(
        repo_path=repo_path,
        beads_ledger_path=ledger_path,
        redact=False,
    )

    assert report["schema_version"] == 2
    assert report["live_archive"] is False

    continuity = report["continuity"]
    assert isinstance(continuity, dict)
    assert continuity["scenario_count"] == 8
    assert continuity["status"] == "pass"

    discovery = report["discovery_coverage"]
    assert isinstance(discovery, dict)
    assert discovery["status"] == "pass"
    assert discovery["gaps"] == []

    effect_proof = report["work_evidence_effect_proof"]
    assert isinstance(effect_proof, dict)
    assert effect_proof["claims_total"] == 1
    assert effect_proof["claims_evaluated"] == 1
    assert effect_proof["status"] == "pass"

    assert report["status"] == "pass"

    # Full report round-trips through JSON (it must be a valid standalone artifact).
    json.dumps(report)


def test_main_cli_writes_json_output_and_returns_pass_exit_code(
    tmp_path: Path,
    repo_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_path, ledger_path = repo_fixture
    output_path = tmp_path / "mandate-report.json"

    async def fake_replay(**_kwargs: object) -> dict[str, object]:
        return {"schema_version": 2, "status": "pass"}

    monkeypatch.setattr("devtools.mandate_continuity_replay.run_mandate_continuity_replay", fake_replay)

    exit_code = main(
        [
            "--repo-path",
            str(repo_path),
            "--beads-ledger",
            str(ledger_path),
            "--no-redact",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["status"] == "pass"
