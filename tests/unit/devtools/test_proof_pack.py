from __future__ import annotations

from pathlib import Path

from devtools.proof_pack import build_proof_pack, render_markdown


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
    assert "oracle_mix" in report
    assert "cost_tier" in report
    assert "manual_review_cells" in report
    assert "known_gaps" in report


def test_proof_pack_markdown_is_pr_comment_ready() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["docs/plans/layering.yaml"],
    )

    rendered = render_markdown(report)

    assert "## Polylogue Proof Pack" in rendered
    assert "### Required Gates" in rendered


def test_proof_pack_surfaces_manifest_known_gaps_for_affected_domains() -> None:
    report = build_proof_pack(
        Path.cwd(),
        base_ref="origin/master",
        head_ref="HEAD",
        changed_paths=["README.md"],
    )

    gap_domains = {item["domain"] for item in report["known_gaps"]}

    assert "docs_media" in gap_domains
