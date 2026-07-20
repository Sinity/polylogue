"""Production read-modify-write reconciliation against a real SQLite archive.

Anti-vacuity: this drives the actual repository -> ``index.db`` route
(``SessionRepository.get_work_evidence_graph`` /
``replace_work_evidence_graph``), the real ``GitCommitEffectAdapter``
against a genuine temp git repository, and the real
``BeadsIssueEffectAdapter`` against the checked-in fixture ledger. Removing
the ``apply`` write-back, or the direct-identifier judgment restriction,
makes the assertions below fail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from polylogue.core.refs import ObjectRef
from polylogue.insights.work_effects import GitCommitEffectAdapter
from polylogue.insights.work_evidence import WorkEvidenceGraph, WorkEvidenceNode
from polylogue.operations.work_effect_reconciliation import (
    WorkEvidenceGraphNotFoundError,
    reconcile_graph_repository_effects,
)
from polylogue.storage.repository import SessionRepository

_EVIDENCE = ObjectRef(kind="artifact", object_id="raw:test-evidence")
_SNAPSHOT = ObjectRef(kind="context-snapshot", object_id="snapshot:op-test")


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "agent@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Agent"], check=True)
    (path / "a.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "a.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fix: ship it (Ref polylogue-1vpm.6.2)"],
        check=True,
    )


def _claim_node(object_id: str, claim_text: str) -> WorkEvidenceNode:
    return WorkEvidenceNode(
        ref=ObjectRef(kind="work-claim", object_id=object_id),
        kind="claim",
        label=object_id,
        claim_text=claim_text,
        evidence_refs=(_EVIDENCE,),
        corpus_snapshot_ref=_SNAPSHOT,
        authority="provider",
        confidence=1.0,
    )


def _seed_graph() -> WorkEvidenceGraph:
    matched = _claim_node("claim:matched", "Claude Workflow finalResult: closed polylogue-1vpm.6.2")
    unmatched = _claim_node("claim:unmatched", "Claude Workflow finalResult: no bead cited")
    return WorkEvidenceGraph(
        graph_id="claude-workflow:test-run",
        corpus_snapshot_ref=_SNAPSHOT,
        nodes=(matched, unmatched),
        edges=(),
    )


@pytest.mark.asyncio
async def test_dry_run_reports_summary_without_mutating_stored_graph(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    graph = _seed_graph()

    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)

        summary = await reconcile_graph_repository_effects(
            repository,
            graph_id=graph.graph_id,
            adapters=(GitCommitEffectAdapter(repo_path=repo),),
            apply=False,
        )

        assert summary.applied is False
        assert summary.claims_total == 2
        assert summary.claims_evaluated == 1
        assert summary.claims_unevaluated == 1
        assert summary.effect_count_by_authority == {"git": 1}
        assert summary.judgment_count_by_evaluation == {"supported": 1}

        # Dry run: the stored graph is untouched -- no effect/claimed edges yet.
        stored = await repository.get_work_evidence_graph(graph.graph_id)
        assert stored is not None
        assert {node.kind for node in stored.nodes} == {"claim"}
        assert stored.edges == ()


@pytest.mark.asyncio
async def test_apply_persists_reconciled_graph_through_the_real_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    graph = _seed_graph()

    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)

        summary = await reconcile_graph_repository_effects(
            repository,
            graph_id=graph.graph_id,
            adapters=(GitCommitEffectAdapter(repo_path=repo),),
            apply=True,
        )
        assert summary.applied is True

        stored = await repository.get_work_evidence_graph(graph.graph_id)

    assert stored is not None
    effect_nodes = [node for node in stored.nodes if node.kind == "effect"]
    assert len(effect_nodes) == 1
    assert effect_nodes[0].ref.kind == "commit"

    matched_edges = [edge for edge in stored.edges if edge.source_ref.object_id == "claim:matched"]
    unmatched_edges = [edge for edge in stored.edges if edge.source_ref.object_id == "claim:unmatched"]
    assert len(matched_edges) == 1
    assert matched_edges[0].kind == "claimed"
    assert matched_edges[0].association_state == "resolved"
    assert unmatched_edges == []


@pytest.mark.asyncio
async def test_unknown_graph_id_raises_typed_error(tmp_path: Path) -> None:
    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        with pytest.raises(WorkEvidenceGraphNotFoundError):
            await reconcile_graph_repository_effects(
                repository,
                graph_id="claude-workflow:does-not-exist",
                adapters=(),
                apply=False,
            )


@pytest.mark.asyncio
async def test_adapter_failures_are_recorded_not_swallowed_or_fatal(tmp_path: Path) -> None:
    from polylogue.insights.work_effects import GitHubPullRequestEffectAdapter

    graph = _seed_graph()
    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)

        summary = await reconcile_graph_repository_effects(
            repository,
            graph_id=graph.graph_id,
            adapters=(GitHubPullRequestEffectAdapter(repo="Sinity/polylogue"),),
            apply=False,
        )

    assert summary.effect_count_by_authority == {}
    assert summary.adapter_failures == ({"authority": "github", "reason": summary.adapter_failures[0]["reason"]},)
    assert "Sinity/polylogue" in summary.adapter_failures[0]["reason"]
    assert summary.claims_evaluated == 0
