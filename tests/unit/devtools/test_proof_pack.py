from __future__ import annotations

from pathlib import Path

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


def test_proof_pack_markdown_is_pr_comment_ready() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )

    rendered = render_markdown(report)

    assert "## Polylogue Proof Pack" in rendered
    assert "### Focused Gates" in rendered
    assert "### Required PR Gates" in rendered
    assert "### Optional Confidence Gates" in rendered
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


def test_proof_pack_markdown_collapses_zero_claim_domains() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["polylogue/proof/catalog.py"],
    )

    rendered = render_markdown(report)

    assert "Additional routed domains with zero affected claims" in rendered
    assert "### Optional Confidence Gates" in rendered
    assert "stale evidence" not in rendered


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
        }
    ]

    result = evaluate_check_policy(report)

    assert result["status"] == "ok"


def test_proof_pack_check_flag_returns_nonzero_on_policy_failure(monkeypatch) -> None:
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
