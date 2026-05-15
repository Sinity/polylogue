from __future__ import annotations

from pathlib import Path

import pytest

from devtools import proof_pack
from devtools.proof_pack import build_proof_pack, evaluate_check_policy, render_markdown


def test_proof_pack_reports_diff_shaped_fields() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )

    assert report["changed_paths"] == ["docs/plans/layering.yaml"]
    assert report["affected_domains"]
    assert report["required_gates"]
    assert report["gate_groups"]["focused"]
    assert report["gate_groups"]["required"]
    assert report["gate_groups"]["optional_confidence"]
    assert "oracle_mix" in report
    assert "cost_tier" in report
    assert "agent_judgment_cells" in report
    assert "known_gaps" in report
    assert "additional_known_gaps" in report
    assert "stable_affected_obligations" in report
    assert "catalog_quality_checks" in report
    assert "taxonomy" in report
    assert "clean_tree" in report
    assert "_context" in report


def test_proof_pack_markdown_is_pr_comment_ready() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )

    rendered = render_markdown(report)

    assert "## Polylogue Verification Impact" in rendered
    assert "### Required Gates (run now)" in rendered
    assert "### Always-On PR Gates" in rendered
    assert "### Confidence Gates (optional)" in rendered
    assert "### Affected Evidence by Taxonomy" in rendered
    assert "stable affected obligations" in rendered


def test_proof_pack_surfaces_manifest_known_gaps_for_affected_domains() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["README.md"],
    )

    gap_domains = {item["domain"] for item in report["known_gaps"]}
    additional_gap_domains = {item["domain"] for item in report["additional_known_gaps"]}

    assert "docs_media" in gap_domains | additional_gap_domains


def test_proof_pack_markdown_has_taxonomy_section() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["polylogue/proof/catalog.py"],
    )

    rendered = render_markdown(report)

    assert "### Affected Evidence by Taxonomy" in rendered
    assert "executable_behavior" in rendered
    assert "architectural_static" in rendered
    assert "metadata_spec" in rendered


def test_proof_pack_markdown_lists_agent_judgment_cells() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["polylogue/proof/catalog.py"],
    )

    rendered = render_markdown(report)

    assert "### Agent Judgment Cells" in rendered
    assert "operation.effect.privacy_safe_evidence" in rendered
    assert "artifact `missing`" in rendered


def test_proof_pack_check_policy_blocks_catalog_quality_errors() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )
    report["catalog_quality_checks"] = [
        {
            "name": "catalog.serious_claim_oracle_independence",
            "status": "error",
            "summary": "serious claims rely on weak oracle independence: 1",
            "count": 1,
            "details": ["claim.id: self_attesting"],
            "breakdown": {},
        }
    ]

    result = evaluate_check_policy(report)

    assert result["status"] == "error"
    assert "catalog.serious_claim_oracle_independence" in result["errors"][0]


def test_proof_pack_check_policy_blocks_serious_judgment_cells() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )
    report["agent_judgment_cells"] = [
        {
            "claim_id": "claim.needs.review",
            "oracle": "manual_review",
            "independence_level": "independent",
            "severity": "serious",
            "tracked_exception": None,
            "artifact": None,
            "reviewer": None,
            "produced_at": None,
            "freshness": None,
            "result": "missing",
        }
    ]

    result = evaluate_check_policy(report)

    assert result["status"] == "error"
    assert "claim.needs.review" in result["errors"][0]


def test_proof_pack_check_policy_allows_tracked_judgment_cells() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )
    report["agent_judgment_cells"] = [
        {
            "claim_id": "claim.tracked.review",
            "oracle": "manual_review",
            "independence_level": "independent",
            "severity": "serious",
            "tracked_exception": "tracked by #594",
            "artifact": None,
            "reviewer": None,
            "produced_at": None,
            "freshness": None,
            "result": "tracked_exception",
        }
    ]

    result = evaluate_check_policy(report)

    assert result["status"] == "ok"


def test_proof_pack_check_policy_allows_completed_judgment_artifact() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )
    report["agent_judgment_cells"] = [
        {
            "claim_id": "claim.reviewed",
            "oracle": "manual_review",
            "independence_level": "independent",
            "severity": "serious",
            "tracked_exception": None,
            "artifact": "docs/reviews/proof-pack-claim-reviewed.md",
            "reviewer": "codex",
            "produced_at": "2026-05-02T00:00:00+00:00",
            "freshness": "current PR",
            "result": "accepted",
        }
    ]

    result = evaluate_check_policy(report)

    assert result["status"] == "ok"


def test_proof_pack_check_flag_returns_nonzero_on_policy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )
    report["catalog_quality_checks"] = [
        {
            "name": "catalog.runner_trust_metadata",
            "status": "error",
            "summary": "runner trust metadata is stale or incomplete: 1",
            "count": 1,
            "details": ["runner.id: expired"],
            "breakdown": {},
        }
    ]
    monkeypatch.setattr(proof_pack, "build_proof_pack", lambda *args, **kwargs: report)

    assert proof_pack.main(["--path", "docs/plans/layering.yaml", "--check"]) == 1


def test_clean_tree_report_distinguishes_baseline() -> None:
    """A clean-tree report should mark clean_tree=True with a distinct context message."""
    report = build_proof_pack(
        Path.cwd(),
        base_ref="HEAD",
        head_ref="HEAD",
    )

    assert report["clean_tree"] is True
    assert report["changed_paths"] == []
    assert "No changed paths" in str(report.get("_context", ""))


def test_proof_pack_taxonomy_groups_are_present_in_report() -> None:
    """The report should include taxonomy groups with correct structure."""
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )

    taxonomy = report["taxonomy"]
    assert isinstance(taxonomy, dict)
    for taxonomy_name in (
        "executable_behavior",
        "observability",
        "architectural_static",
        "metadata_spec",
        "manual_review",
        "advisory",
    ):
        assert taxonomy_name in taxonomy
        group = taxonomy[taxonomy_name]
        assert "description" in group
        assert "actionable" in group
        assert "obligation_count" in group
        assert "obligations" in group

    assert taxonomy["architectural_static"]["obligation_count"] > 0
    assert taxonomy["architectural_static"]["actionable"] is True


def test_proof_pack_markdown_shows_taxonomy_labels() -> None:
    """Markdown output should label actionable vs non-blocking taxonomies."""
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )

    rendered = render_markdown(report)

    assert "architectural_static" in rendered
    assert "(actionable)" in rendered
    assert "### Known Gaps (tracking, not blocking)" in rendered
