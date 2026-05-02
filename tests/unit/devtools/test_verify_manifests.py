from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

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


def test_coverage_status_claims_accept_realized_implemented_item(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """areas:
  path_traversal:
    implemented: true
    controls:
      - sanitize_path: blocks traversal
    test_coverage:
      location: tests/unit/security/test_path_sanitization.py
""",
        encoding="utf-8",
    )

    assert verify_manifests.check_coverage_status_claims(plans) == []


def test_coverage_status_claims_reject_implemented_item_without_evidence(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """areas:
  path_traversal:
    implemented: true
    controls: []
    test_coverage:
      location: null
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_status_claims(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: areas.path_traversal implemented=true but controls are missing or empty",
        f"{plans / 'example-coverage.yaml'}: areas.path_traversal implemented=true but test_coverage.location is missing",
    ]


def test_coverage_status_claims_reject_missing_item_with_realized_evidence(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """areas:
  dependency_audit:
    implemented: false
    controls:
      - pip_audit: configured
    test_coverage:
      location: tests/unit/security/test_dependency_audit.py
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_status_claims(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: areas.dependency_audit implemented=false but controls are declared",
        f"{plans / 'example-coverage.yaml'}: areas.dependency_audit implemented=false "
        "but test_coverage.location is declared",
    ]


def test_campaign_coverage_catalog_accepts_authored_catalog(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    plans = tmp_path
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(
                SimpleNamespace(
                    name="filters",
                    description="Filter campaign",
                    paths_to_mutate=("polylogue/archive/filter/filters.py",),
                    tests=("tests/unit/core/test_filters_props.py",),
                ),
            ),
            benchmark_campaigns=(
                SimpleNamespace(
                    name="storage",
                    description="Storage benchmarks",
                    tests=("tests/benchmarks/test_storage.py",),
                ),
            ),
        ),
    )
    (plans / "campaign-coverage.yaml").write_text(
        """mutation_campaigns:
  - name: filters
    description: Filter campaign
    paths_to_mutate:
      - polylogue/archive/filter/filters.py
    tests:
      - tests/unit/core/test_filters_props.py
    status: active

benchmark_campaigns:
  - name: storage
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: active
""",
        encoding="utf-8",
    )

    assert verify_manifests.check_campaign_coverage_catalog(plans) == []


def test_campaign_coverage_catalog_rejects_stale_manifest(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    plans = tmp_path
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(
                SimpleNamespace(
                    name="filters",
                    description="Filter campaign",
                    paths_to_mutate=("polylogue/archive/filter/filters.py",),
                    tests=("tests/unit/core/test_filters_props.py",),
                ),
            ),
            benchmark_campaigns=(
                SimpleNamespace(
                    name="storage",
                    description="Storage benchmarks",
                    tests=("tests/benchmarks/test_storage.py",),
                ),
            ),
        ),
    )
    (plans / "campaign-coverage.yaml").write_text(
        """mutation_campaigns:
  - name: filters
    description: Old filter campaign
    paths_to_mutate:
      - polylogue/archive/filter/old_filters.py
    tests:
      - tests/unit/core/test_filters_props.py
    status: active

benchmark_campaigns:
  - name: storage-scale
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: active
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_campaign_coverage_catalog(plans)

    assert errors == [
        f"{plans / 'campaign-coverage.yaml'}: mutation_campaigns['filters'] paths_to_mutate does not match "
        "authored catalog (expected ('polylogue/archive/filter/filters.py',), "
        "got ('polylogue/archive/filter/old_filters.py',))",
        f"{plans / 'campaign-coverage.yaml'}: benchmark_campaigns missing catalog campaign 'storage'",
        f"{plans / 'campaign-coverage.yaml'}: benchmark_campaigns declares unknown campaign 'storage-scale'",
    ]
