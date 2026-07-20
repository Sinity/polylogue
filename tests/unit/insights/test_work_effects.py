"""Fixture-driven tests for repository-effect observation adapters.

Anti-vacuity: every adapter here runs its real production mechanism -- a
subprocess ``git log`` against a genuine (fixture) git repository, and a real
JSON-lines parse of a Beads interaction ledger -- not a stand-in double.
Removing the git/Beads I/O, or the direct-identifier restriction in
``derive_direct_identifier_judgments``, makes the assertions below fail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from polylogue.core.refs import ObjectRef
from polylogue.insights.work_effects import (
    BeadsIssueEffectAdapter,
    EffectAdapterUnavailableError,
    GitCommitEffectAdapter,
    GitHubPullRequestEffectAdapter,
    collect_repository_effects,
    derive_direct_identifier_judgments,
    reconcile_repository_effects,
    referenced_work_item_ids,
)
from polylogue.insights.work_evidence import WorkEvidenceGraph, WorkEvidenceNode

_EVIDENCE = ObjectRef(kind="artifact", object_id="raw:test-evidence")
_SNAPSHOT = ObjectRef(kind="context-snapshot", object_id="snapshot:work-effects-test")


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "agent@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Agent"], check=True)


def _commit(path: Path, *, filename: str, message: str) -> str:
    (path / filename).write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", filename], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", message], check=True)
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


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


# ── referenced_work_item_ids ─────────────────────────────────────────


def test_referenced_work_item_ids_extracts_exact_tokens_only() -> None:
    text = "Ref polylogue-1vpm.6.2 landed; unrelated polylogueish-thing and POLYLOGUE-UPPER ignored"
    assert referenced_work_item_ids(text) == frozenset({"polylogue-1vpm.6.2"})


def test_referenced_work_item_ids_returns_empty_for_no_match() -> None:
    assert referenced_work_item_ids("nothing to see here") == frozenset()


# ── GitCommitEffectAdapter ───────────────────────────────────────────


def test_git_commit_adapter_reads_real_commit_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    sha = _commit(repo, filename="a.txt", message="fix: land the effect (Ref polylogue-1vpm.6.2)")

    adapter = GitCommitEffectAdapter(repo_path=repo)
    effects = adapter.collect()

    assert len(effects) == 1
    [effect] = effects
    assert effect.ref == ObjectRef(kind="commit", object_id=sha)
    assert effect.authority == "git"
    assert "polylogue-1vpm.6.2" in effect.label
    assert isinstance(effect.evidence_ref, ObjectRef)
    assert effect.evidence_ref.kind == "artifact"
    assert effect.occurred_at_ms is not None
    assert effect.repository_snapshot_ref.kind == "context-snapshot"


def test_git_commit_adapter_time_bounds_exclude_out_of_window_commits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="first")

    adapter = GitCommitEffectAdapter(repo_path=repo)
    all_effects = adapter.collect()
    assert len(all_effects) == 1
    occurred_at_ms = all_effects[0].occurred_at_ms
    assert occurred_at_ms is not None
    far_future_ms = occurred_at_ms + 1000 * 60 * 60 * 24 * 365
    windowed = adapter.collect(since_ms=far_future_ms)
    assert windowed == ()


def test_git_commit_adapter_raises_explicitly_for_non_repository(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain-dir"
    not_a_repo.mkdir()

    with pytest.raises(EffectAdapterUnavailableError):
        GitCommitEffectAdapter(repo_path=not_a_repo).collect()


# ── BeadsIssueEffectAdapter ───────────────────────────────────────────

_BEADS_FIXTURE = Path(__file__).parents[2] / "fixtures" / "beads" / "issue-interactions.jsonl"


def test_beads_issue_adapter_reads_real_interaction_ledger() -> None:
    adapter = BeadsIssueEffectAdapter(jsonl_path=_BEADS_FIXTURE)

    effects = adapter.collect()

    assert len(effects) == 2
    assert {effect.authority for effect in effects} == {"beads"}
    labels = {effect.label for effect in effects}
    assert any("polylogue-7fj" in label and "status" in label for label in labels)
    assert all(effect.ref.kind == "beads-issue" for effect in effects)
    assert all(effect.ref.object_id.startswith("polylogue-7fj:") for effect in effects)


def test_beads_issue_adapter_raises_explicitly_for_missing_ledger(tmp_path: Path) -> None:
    missing = tmp_path / "interactions.jsonl"

    with pytest.raises(EffectAdapterUnavailableError):
        BeadsIssueEffectAdapter(jsonl_path=missing).collect()


def test_beads_issue_adapter_ignores_non_interaction_lines(tmp_path: Path) -> None:
    ledger = tmp_path / "interactions.jsonl"
    ledger.write_text(
        '{"not": "an interaction"}\n'
        "not even json\n"
        '{"id":"int-x","kind":"field_change","created_at":"2026-07-01T00:00:00Z",'
        '"actor":"a","issue_id":"proj-1","extra":{"field":"status","old_value":"open","new_value":"closed"}}\n',
        encoding="utf-8",
    )

    effects = BeadsIssueEffectAdapter(jsonl_path=ledger).collect()

    assert len(effects) == 1
    assert effects[0].ref.object_id == "proj-1:int-x"


# ── GitHubPullRequestEffectAdapter (honest stub) ─────────────────────


def test_github_adapter_fails_explicitly_instead_of_returning_empty() -> None:
    adapter = GitHubPullRequestEffectAdapter(repo="Sinity/polylogue")

    with pytest.raises(EffectAdapterUnavailableError, match="Sinity/polylogue"):
        adapter.collect()


def test_collect_repository_effects_records_adapter_failures_without_losing_others(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="plain commit, no bead ref")

    result = collect_repository_effects(
        (
            GitCommitEffectAdapter(repo_path=repo),
            GitHubPullRequestEffectAdapter(repo="Sinity/polylogue"),
        )
    )

    assert len(result.effects) == 1
    assert result.effects[0].authority == "git"
    assert len(result.unavailable) == 1
    assert result.unavailable[0].authority == "github"
    assert "Sinity/polylogue" in result.unavailable[0].reason


# ── derive_direct_identifier_judgments ────────────────────────────────


def test_direct_identifier_judgments_link_only_shared_ids(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="feat: ship it (Ref polylogue-7fj)")
    effects = GitCommitEffectAdapter(repo_path=repo).collect()

    matching_claim = _claim_node("claim:matches", "Claude Workflow finalResult: closed polylogue-7fj")
    unrelated_claim = _claim_node("claim:unrelated", "Claude Workflow finalResult: investigated but no bead cited")
    different_id_claim = _claim_node("claim:different", "Claude Workflow finalResult: closed polylogue-zzzz")
    graph = WorkEvidenceGraph(
        graph_id="g",
        corpus_snapshot_ref=_SNAPSHOT,
        nodes=(matching_claim, unrelated_claim, different_id_claim),
        edges=(),
    )

    judgments = derive_direct_identifier_judgments(graph, effects)

    assert len(judgments) == 1
    [judgment] = judgments
    assert judgment.claim_ref == matching_claim.ref
    assert judgment.effect_ref == effects[0].ref
    assert judgment.evaluation == "supported"


def test_direct_identifier_judgments_never_infer_from_time_or_session_presence(tmp_path: Path) -> None:
    """A claim with no cited id gets no judgment even though an effect exists
    in the same graph/time window -- presence alone is never "done"."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="fix: totally unrelated change")
    effects = GitCommitEffectAdapter(repo_path=repo).collect()

    silent_claim = _claim_node("claim:silent", "Claude Workflow finalResult: task completed")
    graph = WorkEvidenceGraph(graph_id="g", corpus_snapshot_ref=_SNAPSHOT, nodes=(silent_claim,), edges=())

    judgments = derive_direct_identifier_judgments(graph, effects)

    assert judgments == ()


# ── reconcile_repository_effects (adapters -> reconciled graph) ──────


def test_reconcile_repository_effects_leaves_unmatched_claims_unevaluated(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="fix: land it (Ref polylogue-7fj)")

    matched_claim = _claim_node("claim:matched", "Claude Workflow finalResult: closed polylogue-7fj")
    unmatched_claim = _claim_node("claim:unmatched", "Claude Workflow finalResult: no bead cited here")
    graph = WorkEvidenceGraph(
        graph_id="g",
        corpus_snapshot_ref=_SNAPSHOT,
        nodes=(matched_claim, unmatched_claim),
        edges=(),
    )

    reconciled, collection = reconcile_repository_effects(
        graph,
        adapters=(
            GitCommitEffectAdapter(repo_path=repo),
            BeadsIssueEffectAdapter(jsonl_path=_BEADS_FIXTURE),
        ),
    )

    assert collection.unavailable == ()
    effect_nodes = [node for node in reconciled.nodes if node.kind == "effect"]
    # one git commit + two beads interactions from the fixture ledger
    assert len(effect_nodes) == 3

    edges_from_matched = [edge for edge in reconciled.edges if edge.source_ref == matched_claim.ref]
    edges_from_unmatched = [edge for edge in reconciled.edges if edge.source_ref == unmatched_claim.ref]
    # The git commit and both beads interactions in the fixture ledger all
    # cite polylogue-7fj, so all three independently corroborate the claim --
    # multiplicity is expected, not collapsed to a single edge.
    assert {edge.target_ref.kind for edge in edges_from_matched} == {"commit", "beads-issue"}
    assert len(edges_from_matched) == 3
    assert all(edge.association_state == "resolved" and edge.kind == "claimed" for edge in edges_from_matched)
    assert edges_from_unmatched == []
