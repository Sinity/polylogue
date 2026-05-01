from __future__ import annotations

from pathlib import Path

from devtools import verify_manifests


def test_coverage_gap_manifest_records_require_closure_path(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - domain: docs_media
    gap: Missing generated media proof
    owner: docs-media
    next_evidence: Add generated media proof gate
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
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing owner",
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] missing next_evidence",
    ]
