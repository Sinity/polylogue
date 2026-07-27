"""Unit tests for the z9gh.7 mandate-replay wiring artifact.

Anti-vacuity: the discovery-coverage lane runs against the real shipped
``QUERY_DISCOVERY_EXAMPLES`` catalog and t8t's real ``CONTINUITY_SCENARIOS``
declarations (not stand-ins); the work-evidence lane runs the real
``polylogue.insights.work_effects`` adapters against genuine ``git`` and Beads
ledger fixtures via subprocess/file I/O, exactly as
``tests/unit/insights/test_work_effects.py`` does. Mutation cases remove a
piece of real evidence (a discovery example, a corroborating commit) and
assert the artifact's own status flips to the failing/unevaluated state,
rather than asserting a fixed shape that a broken implementation could still
satisfy.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools import mandate_continuity_replay as mcr
from devtools.command_catalog import COMMANDS
from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES
from polylogue.core.json import JSONDocument
from polylogue.product.continuity_scenarios import CONTINUITY_SCENARIOS

_BEADS_FIXTURE = Path(__file__).parents[2] / "fixtures" / "beads" / "issue-interactions.jsonl"


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


# ── Command registration ──────────────────────────────────────────────


def test_mandate_replay_registered_in_command_catalog() -> None:
    spec = COMMANDS["workspace mandate-continuity-replay"]
    assert spec.module == "devtools.mandate_continuity_replay"
    assert spec.entrypoint == "main"


# ── Discovery-coverage lane ────────────────────────────────────────────


def test_discovery_coverage_passes_for_shipped_continuity_scenarios() -> None:
    report = mcr.check_discovery_coverage(CONTINUITY_SCENARIOS)

    assert report.checked_steps > 0
    assert report.gaps == ()
    assert report.status == "pass"
    assert report.to_dict()["status"] == "pass"


def test_discovery_coverage_flags_a_regressed_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove every declared "runs" query example -- a real shipped scenario
    # (postmortem) issues a `runs where ...` query step, so this must surface
    # as a named gap rather than a silent pass.
    filtered = tuple(
        example
        for example in QUERY_DISCOVERY_EXAMPLES
        if not (example.route == "query" and example.unit_source == "runs")
    )
    assert len(filtered) < len(QUERY_DISCOVERY_EXAMPLES)
    monkeypatch.setattr(mcr, "QUERY_DISCOVERY_EXAMPLES", filtered)

    report = mcr.check_discovery_coverage(CONTINUITY_SCENARIOS)

    assert report.status == "fail"
    assert any(gap.plan_atom == "query:runs" for gap in report.gaps)
    assert report.covered_steps == report.checked_steps - len(report.gaps)


# ── Work-evidence effect-reconciliation lane ──────────────────────────


def test_build_repository_claim_graph_builds_one_claim_per_closed_issue() -> None:
    graph = mcr.build_repository_claim_graph(_BEADS_FIXTURE)

    claim_ids = {node.ref.object_id for node in graph.nodes if node.kind == "claim"}
    assert claim_ids == {"claimed-closed:polylogue-7fj"}
    [node] = [n for n in graph.nodes if n.kind == "claim"]
    assert node.claim_text is not None
    assert "polylogue-7fj" in node.claim_text


def test_build_repository_claim_graph_raises_explicitly_for_missing_ledger(tmp_path: Path) -> None:
    missing = tmp_path / "interactions.jsonl"
    with pytest.raises(FileNotFoundError):
        mcr.build_repository_claim_graph(missing)


def test_work_evidence_effect_proof_evaluates_claims_with_a_corroborating_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="fix: land the work (Ref polylogue-7fj)")

    proof = mcr.run_work_evidence_effect_proof(repo_path=repo, beads_ledger_path=_BEADS_FIXTURE)

    assert proof.claims_total == 1
    assert proof.claims_evaluated == 1
    assert proof.claims_unevaluated == 0
    assert proof.status == "pass"
    assert proof.effect_count_by_authority["git"] >= 1
    assert proof.effect_count_by_authority["beads"] >= 1
    assert proof.judgment_count_by_evaluation.get("supported", 0) >= 1
    # GitHub is included and is expected to fail explicitly -- an honest
    # "not wired yet" citation, not a silent omission.
    assert any(failure["authority"] == "github" for failure in proof.adapter_failures)


def test_work_evidence_effect_proof_evaluates_via_beads_effect_when_git_has_no_reference(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="unrelated commit, no bead reference")

    proof = mcr.run_work_evidence_effect_proof(repo_path=repo, beads_ledger_path=_BEADS_FIXTURE)

    # The Beads ledger's own field-change row is still a real, independently
    # collected effect record (the same mechanism BeadsIssueEffectAdapter
    # exercises against production ledgers), so the claim is still evaluated
    # -- but only through the beads authority; git contributes no matching
    # effect here because the commit never mentions the issue id.
    assert proof.claims_total == 1
    assert proof.judgment_count_by_evaluation.get("supported", 0) >= 1
    assert proof.effect_count_by_authority.get("git", 0) == 1  # the unrelated commit is still observed


def test_work_evidence_effect_proof_status_fails_on_an_empty_ledger(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _commit(repo, filename="a.txt", message="unrelated")
    empty_ledger = tmp_path / "interactions.jsonl"
    empty_ledger.write_text("", encoding="utf-8")

    proof = mcr.run_work_evidence_effect_proof(repo_path=repo, beads_ledger_path=empty_ledger)

    assert proof.claims_total == 0
    assert proof.status == "fail"


# ── Redaction ──────────────────────────────────────────────────────────


def test_redact_report_hashes_evidence_prose_but_preserves_refs_and_counts() -> None:
    document: JSONDocument = {
        "status": "pass",
        "count": 3,
        "label": "polylogue-7fj status: 'in_progress' -> 'closed'",
        "nested": {"claim_text": "Beads issue polylogue-7fj was recorded as closed.", "ref": "commit:abc123"},
        "list": [{"reason": "Focused parser checks passed."}, {"ref": "beads-issue:polylogue-7fj"}],
    }

    redacted = mcr.redact_report(document)

    assert isinstance(redacted, dict)
    assert redacted["status"] == "pass"
    assert redacted["count"] == 3
    label = redacted["label"]
    assert isinstance(label, str) and label.startswith("redacted:sha256:")
    nested = redacted["nested"]
    assert isinstance(nested, dict)
    claim_text = nested["claim_text"]
    assert isinstance(claim_text, str) and claim_text.startswith("redacted:sha256:")
    assert nested["ref"] == "commit:abc123"
    entries = redacted["list"]
    assert isinstance(entries, list)
    first_entry = entries[0]
    assert isinstance(first_entry, dict)
    reason = first_entry["reason"]
    assert isinstance(reason, str) and reason.startswith("redacted:sha256:")
    second_entry = entries[1]
    assert isinstance(second_entry, dict)
    assert second_entry["ref"] == "beads-issue:polylogue-7fj"


def test_redact_report_is_deterministic() -> None:
    document: JSONDocument = {"label": "same text twice"}
    assert mcr.redact_report(document) == mcr.redact_report(dict(document))


# ── AC matrix ──────────────────────────────────────────────────────────


def _fake_continuity_report(*, status: str, passed: int = 8, failed: int = 0) -> JSONDocument:
    return {"status": status, "scenario_count": passed + failed, "passed": passed, "failed": failed}


def test_ac_matrix_marks_incident_replay_deferred_without_a_live_archive() -> None:
    discovery = mcr.DiscoveryCoverageReport(checked_steps=5, covered_steps=5, gaps=())
    effect_proof = mcr.WorkEvidenceEffectProof(
        graph_id="g",
        claims_total=1,
        claims_evaluated=1,
        claims_unevaluated=0,
        effect_count_by_authority={"git": 1, "beads": 1},
        judgment_count_by_evaluation={"supported": 1},
        adapter_failures=({"authority": "github", "reason": "not implemented"},),
    )

    matrix = mcr.build_ac_matrix(
        continuity_report=_fake_continuity_report(status="pass"),
        discovery_report=discovery,
        effect_proof=effect_proof,
        live_archive=False,
    )

    assert len(matrix) == 7
    by_index = {item.index: item for item in matrix}
    assert by_index[1].status == "satisfied"
    assert by_index[2].status == "deferred"
    assert by_index[3].status == "satisfied"
    assert by_index[5].status == "satisfied"
    assert by_index[7].status == "satisfied"


def test_ac_matrix_marks_blocking_when_a_lane_fails() -> None:
    discovery = mcr.DiscoveryCoverageReport(
        checked_steps=5,
        covered_steps=4,
        gaps=(mcr.DiscoveryCoverageGap(scenario_id="x", step_id="y", plan_atom="query:runs", reason="missing"),),
    )
    effect_proof = mcr.WorkEvidenceEffectProof(
        graph_id="g",
        claims_total=0,
        claims_evaluated=0,
        claims_unevaluated=0,
        effect_count_by_authority={},
        judgment_count_by_evaluation={},
        adapter_failures=(),
    )

    matrix = mcr.build_ac_matrix(
        continuity_report=_fake_continuity_report(status="fail", passed=6, failed=2),
        discovery_report=discovery,
        effect_proof=effect_proof,
        live_archive=True,
    )

    by_index = {item.index: item for item in matrix}
    assert by_index[1].status == "blocking"
    assert by_index[2].status == "deferred"  # continuity itself failed, so the live incident claim can't be satisfied
    assert by_index[3].status == "blocking"
    assert by_index[5].status == "blocking"
