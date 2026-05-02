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


def test_coverage_gap_manifest_allows_pytest_filter_next_evidence(tmp_path: Path) -> None:
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
    next_evidence: pytest -k "docs_media and not slow"
""",
        encoding="utf-8",
    )

    assert verify_manifests.check_coverage_gaps(plans) == []


def test_coverage_gap_manifest_rejects_proof_subject_slug_collision(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - id: docs_media.generated_proof
    domain: docs_media
    gap: Missing generated media proof
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: "#590"
    next_evidence: devtools render-docs-surface --check
  - id: docs-media.generated-proof
    domain: docs_media
    gap: Missing generated media proof
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: "#590"
    next_evidence: devtools render-docs-surface --check
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_gaps(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[1] 'docs-media.generated-proof' "
        "duplicate proof subject slug 'docs-media-generated-proof'"
    ]


def test_coverage_reference_manifest_accepts_existing_commands_and_paths(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "tests").mkdir()
    (plans / "tests" / "test_example.py").write_text("def test_example(): pass\n", encoding="utf-8")
    (plans / "example-coverage.yaml").write_text(
        """items:
  example:
    location: tests/test_example.py (unit coverage)
    verified_by: devtools verify-manifests
    tests:
      - tests/test_example.py
""",
        encoding="utf-8",
    )

    assert verify_manifests.check_coverage_references(plans) == []


def test_coverage_reference_manifest_rejects_missing_command_and_path(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """items:
  example:
    location: tests/missing.py
    verified_by: devtools missing-command
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_references(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: items.example.location path does not exist: 'tests/missing.py'",
        f"{plans / 'example-coverage.yaml'}: items.example.verified_by command does not resolve: "
        "'devtools missing-command'",
    ]
