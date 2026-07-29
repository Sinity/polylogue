"""CLI smoke tests for ``polylogue ops reconcile-work-effects``.

Exercises the real command against a real seeded archive (``workspace_env``)
and a real temp git repository -- not a stubbed operation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli import cli
from polylogue.core.refs import ObjectRef
from polylogue.insights.work_evidence import WorkEvidenceGraph, WorkEvidenceNode
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.repository import SessionRepository

_EVIDENCE = ObjectRef(kind="artifact", object_id="raw:test-evidence")
_SNAPSHOT = ObjectRef(kind="context-snapshot", object_id="snapshot:cli-test")


def _init_git_repo(path: Path, *, message: str) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "agent@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Agent"], check=True)
    (path / "a.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", message], check=True)


def _seed_graph() -> WorkEvidenceGraph:
    claim = WorkEvidenceNode(
        ref=ObjectRef(kind="work-claim", object_id="claim:cli"),
        kind="claim",
        label="claim:cli",
        claim_text="Claude Workflow finalResult: closed polylogue-1vpm.6.2",
        evidence_refs=(_EVIDENCE,),
        corpus_snapshot_ref=_SNAPSHOT,
        authority="provider",
        confidence=1.0,
    )
    return WorkEvidenceGraph(
        graph_id="claude-workflow:cli-test-run",
        corpus_snapshot_ref=_SNAPSHOT,
        nodes=(claim,),
        edges=(),
    )


@pytest.fixture
def _seeded_graph(workspace_env: dict[str, Path]) -> WorkEvidenceGraph:
    graph = _seed_graph()

    async def _seed() -> None:
        async with SessionRepository(db_path=resolve_active_index_path(archive_root())) as repository:
            await repository.replace_work_evidence_graph(graph)

    run_coroutine_sync(_seed())
    return graph


def test_dry_run_reports_json_summary_without_persisting(
    tmp_path: Path,
    _seeded_graph: WorkEvidenceGraph,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo, message="fix: land it (Ref polylogue-1vpm.6.2)")

    result = CliRunner().invoke(
        cli,
        [
            "ops",
            "reconcile-work-effects",
            "--graph-id",
            _seeded_graph.graph_id,
            "--repo",
            str(repo),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mutates"] is False
    assert payload["applied"] is False
    assert payload["claims_total"] == 1
    assert payload["claims_evaluated"] == 1
    assert payload["effect_count_by_authority"] == {"git": 1}
    assert payload["judgment_count_by_evaluation"] == {"supported": 1}


def test_yes_flag_persists_reconciled_graph(
    tmp_path: Path,
    _seeded_graph: WorkEvidenceGraph,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo, message="fix: land it (Ref polylogue-1vpm.6.2)")

    result = CliRunner().invoke(
        cli,
        [
            "ops",
            "reconcile-work-effects",
            "--graph-id",
            _seeded_graph.graph_id,
            "--repo",
            str(repo),
            "--yes",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["applied"] is True

    async def _read() -> WorkEvidenceGraph | None:
        async with SessionRepository(db_path=resolve_active_index_path(archive_root())) as repository:
            return await repository.get_work_evidence_graph(_seeded_graph.graph_id)

    stored = run_coroutine_sync(_read())
    assert stored is not None
    assert any(node.kind == "effect" for node in stored.nodes)
    assert any(edge.kind == "claimed" for edge in stored.edges)


def test_github_repo_flag_wires_in_pr_effects(
    tmp_path: Path,
    _seeded_graph: WorkEvidenceGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--github-repo reaches the real GitHubPullRequestEffectAdapter.collect
    (only the OS-level `gh` subprocess call is faked -- no live GitHub
    network/auth access in this test suite)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo, message="unrelated plain commit")

    pr_records = [
        {
            "number": 7,
            "title": "fix: land it (Ref polylogue-1vpm.6.2)",
            "body": "",
            "state": "MERGED",
            "url": "https://github.com/Sinity/polylogue/pull/7",
            "createdAt": "2026-07-10T09:00:00Z",
            "updatedAt": "2026-07-10T10:00:00Z",
            "closedAt": "2026-07-10T10:00:00Z",
            "mergedAt": "2026-07-10T10:00:00Z",
            "mergeCommit": {"oid": "deadbeef"},
        }
    ]

    real_run = subprocess.run

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str] | MagicMock:
        if cmd[:1] == ["gh"]:
            return MagicMock(returncode=0, stdout=json.dumps(pr_records), stderr="")
        return real_run(cmd, **kwargs)  # type: ignore[call-overload, no-any-return]

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(
        cli,
        [
            "ops",
            "reconcile-work-effects",
            "--graph-id",
            _seeded_graph.graph_id,
            "--repo",
            str(repo),
            "--github-repo",
            "Sinity/polylogue",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["effect_count_by_authority"] == {"git": 1, "github": 1}
    assert payload["claims_evaluated"] == 1
    assert payload["adapter_failures"] == []


def test_unknown_graph_id_is_a_usage_error(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo, message="irrelevant")

    result = CliRunner().invoke(
        cli,
        ["ops", "reconcile-work-effects", "--graph-id", "claude-workflow:does-not-exist", "--repo", str(repo)],
    )

    assert result.exit_code != 0
    assert "no work-evidence graph stored" in str(result.output)
