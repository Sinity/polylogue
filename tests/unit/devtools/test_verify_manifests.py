from __future__ import annotations

from pathlib import Path

from devtools import verify_manifests


def test_coverage_gap_manifest_records_require_closure_path(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - id: docs-media.generated-proof
    domain: docs_media
    gap: Missing generated media proof
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: 590
    next_evidence: devtools render-docs-surface --check
""",
        encoding="utf-8",
    )

    assert verify_manifests.check_coverage_gaps(plans) == []


def test_coverage_gap_manifest_rejects_gap_without_closure_path(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - domain: docs_media
    gap: Missing generated media proof
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_gaps(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing id",
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing owner",
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing or invalid severity",
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing or invalid declared_at",
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing or invalid review_after",
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing issue or suppression",
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing next_evidence",
    ]


def test_coverage_gap_manifest_rejects_non_command_next_evidence(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - id: docs-media.generated-proof
    domain: docs_media
    gap: Missing generated media proof
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: "#590"
    next_evidence: Add generated media proof gate
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_gaps(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] 'docs-media.generated-proof' "
        "next_evidence does not resolve to a known command"
    ]
