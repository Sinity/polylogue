"""Reconciliation reads a stored graph; the checked replace persists the result.

Anti-vacuity: this drives the actual repository -> ``index.db`` route
(``SessionRepository.get_work_evidence_graph`` /
``replace_work_evidence_graph_checked``) and the real
``GitCommitEffectAdapter`` against a genuine temp git repository. Removing the
direct-identifier judgment restriction, or the base-digest check in the
replace, makes the assertions below fail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from polylogue.analysis.work_effects import GitCommitEffectAdapter
from polylogue.analysis.work_evidence import WorkEvidenceGraph, WorkEvidenceNode
from polylogue.core.refs import ObjectRef
from polylogue.operations.work_effect_reconciliation import (
    WorkEvidenceGraphNotFoundError,
    reconcile_graph_repository_effects,
)
from polylogue.operations.work_evidence_writes import (
    WorkEvidenceGraphConflictError,
    replace_work_evidence_graph_checked,
    work_evidence_graph_digest,
)
from polylogue.storage.repository import SessionRepository
from tests.infra.archive_templates import bootstrapped_tier_path

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

        reconciliation = await reconcile_graph_repository_effects(
            repository,
            graph_id=graph.graph_id,
            adapters=(GitCommitEffectAdapter(repo_path=repo),),
        )
        summary = reconciliation.summary

        assert reconciliation.base_digest == work_evidence_graph_digest(graph)
        assert summary.claims_total == 2
        assert summary.claims_evaluated == 1
        assert summary.claims_unevaluated == 1
        assert summary.effect_count_by_authority == {"git": 1}
        assert summary.judgment_count_by_evaluation == {"supported": 1}

        # Reconciliation only reads: no effect/claimed edges are stored yet.
        stored = await repository.get_work_evidence_graph(graph.graph_id)
        assert stored is not None
        assert {node.kind for node in stored.nodes} == {"claim"}
        assert stored.edges == ()


@pytest.mark.asyncio
async def test_checked_replace_persists_the_reconciled_graph(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    graph = _seed_graph()

    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)

        reconciliation = await reconcile_graph_repository_effects(
            repository,
            graph_id=graph.graph_id,
            adapters=(GitCommitEffectAdapter(repo_path=repo),),
        )
        replacement = await replace_work_evidence_graph_checked(
            repository, reconciliation.graph, expected_base_digest=reconciliation.base_digest
        )
        assert replacement.changed is True

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
async def test_checked_replace_refuses_a_base_that_moved(tmp_path: Path) -> None:
    """A graph replaced after reconciliation read it is not overwritten.

    Anti-vacuity: skip the digest comparison in
    ``replace_work_evidence_graph_checked`` and the stale reconciliation
    overwrites the concurrent graph without raising.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    graph = _seed_graph()
    concurrent = graph.model_copy(update={"nodes": graph.nodes[:1]})

    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)
        reconciliation = await reconcile_graph_repository_effects(
            repository,
            graph_id=graph.graph_id,
            adapters=(GitCommitEffectAdapter(repo_path=repo),),
        )
        await repository.replace_work_evidence_graph(concurrent)

        with pytest.raises(WorkEvidenceGraphConflictError) as refused:
            await replace_work_evidence_graph_checked(
                repository, reconciliation.graph, expected_base_digest=reconciliation.base_digest
            )
        stored = await repository.get_work_evidence_graph(graph.graph_id)

    assert refused.value.code == "work_evidence_graph_conflict"
    assert stored == concurrent


@pytest.mark.asyncio
async def test_checked_replace_replay_after_commit_is_no_effect(tmp_path: Path) -> None:
    """A retried replace whose first attempt committed is not a conflict.

    Anti-vacuity: drop the ``digest != previous_digest`` guard in
    ``replace_work_evidence_graph_checked`` and the replay raises
    ``WorkEvidenceGraphConflictError`` although the stored graph is exactly
    the requested one.
    """
    graph = _seed_graph()
    replacement = graph.model_copy(update={"nodes": graph.nodes[:1]})

    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)
        base = work_evidence_graph_digest(graph)
        first = await replace_work_evidence_graph_checked(repository, replacement, expected_base_digest=base)
        replay = await replace_work_evidence_graph_checked(repository, replacement, expected_base_digest=base)

    assert first.changed is True
    assert replay.changed is False
    assert replay.digest == first.digest


@pytest.mark.asyncio
async def test_unknown_graph_id_raises_typed_error(tmp_path: Path) -> None:
    # A read never initializes the archive it reads; the lookup runs against
    # an empty, bootstrapped archive so the typed "not found" is what answers.
    async with SessionRepository(db_path=bootstrapped_tier_path(tmp_path / "index.db")) as repository:
        with pytest.raises(WorkEvidenceGraphNotFoundError):
            await reconcile_graph_repository_effects(
                repository,
                graph_id="claude-workflow:does-not-exist",
                adapters=(),
            )


@pytest.mark.asyncio
async def test_adapter_failures_are_recorded_not_swallowed_or_fatal(tmp_path: Path) -> None:
    from polylogue.analysis.work_effects import GitHubPullRequestEffectAdapter

    graph = _seed_graph()
    async with SessionRepository(db_path=tmp_path / "index.db") as repository:
        await repository.replace_work_evidence_graph(graph)

        reconciliation = await reconcile_graph_repository_effects(
            repository,
            graph_id=graph.graph_id,
            # A deterministically-missing `gh_path`, not the real "gh"
            # binary: the real one succeeds on any machine authenticated
            # against Sinity/polylogue (this devbox included), which would
            # make the "adapter fails" assertion below environment-dependent
            # rather than a property of the code.
            adapters=(
                GitHubPullRequestEffectAdapter(repo="Sinity/polylogue", gh_path="polylogue-test-missing-gh-binary"),
            ),
        )
    summary = reconciliation.summary

    assert summary.effect_count_by_authority == {}
    assert summary.adapter_failures == ({"authority": "github", "reason": summary.adapter_failures[0]["reason"]},)
    assert "polylogue-test-missing-gh-binary" in summary.adapter_failures[0]["reason"]
    assert summary.claims_evaluated == 0
