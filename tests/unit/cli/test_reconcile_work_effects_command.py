"""CLI smoke tests for ``polylogue ops reconcile-work-effects``.

Exercises the real command against a real seeded archive (``workspace_env``)
and a real temp git repository -- not a stubbed operation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.commands.reconcile_work_effects import reconcile_work_effects_command
from polylogue.core.refs import ObjectRef
from polylogue.insights.work_evidence import WorkEvidenceGraph, WorkEvidenceNode
from polylogue.paths import active_index_db_path
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
        async with SessionRepository(db_path=active_index_db_path()) as repository:
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
        reconcile_work_effects_command,
        ["--graph-id", _seeded_graph.graph_id, "--repo", str(repo), "--output-format", "json"],
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
        reconcile_work_effects_command,
        ["--graph-id", _seeded_graph.graph_id, "--repo", str(repo), "--yes", "--output-format", "json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["applied"] is True

    async def _read() -> WorkEvidenceGraph | None:
        async with SessionRepository(db_path=active_index_db_path()) as repository:
            return await repository.get_work_evidence_graph(_seeded_graph.graph_id)

    stored = run_coroutine_sync(_read())
    assert stored is not None
    assert any(node.kind == "effect" for node in stored.nodes)
    assert any(edge.kind == "claimed" for edge in stored.edges)


def test_unknown_graph_id_is_a_usage_error(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo, message="irrelevant")

    result = CliRunner().invoke(
        reconcile_work_effects_command,
        ["--graph-id", "claude-workflow:does-not-exist", "--repo", str(repo)],
    )

    assert result.exit_code != 0
    assert "no work-evidence graph stored" in str(result.output)
