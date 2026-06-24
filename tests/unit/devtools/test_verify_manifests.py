from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from devtools import verify_manifests
from devtools.verify_ci_workflows import WorkflowFacts, WorkflowInventory


def test_coverage_gap_manifest_records_require_closure_path(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - id: docs-media.generated-evidence
    domain: docs_media
    gap: Missing generated media evidence
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: 590
    next_evidence: devtools render docs-surface --check
""",
        encoding="utf-8",
    )

    assert verify_manifests.check_coverage_gaps(plans) == []


def test_coverage_gap_manifest_rejects_gap_without_closure_path(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - domain: docs_media
    gap: Missing generated media evidence
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
  - id: docs-media.generated-evidence
    domain: docs_media
    gap: Missing generated media evidence
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: "#590"
    next_evidence: Add generated media evidence gate
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_gaps(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[0] 'docs-media.generated-evidence' "
        "next_evidence does not resolve to a known command"
    ]


def test_coverage_gap_manifest_allows_pytest_filter_next_evidence(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - id: docs-media.generated-evidence
    domain: docs_media
    gap: Missing generated media evidence
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


def test_coverage_gap_manifest_rejects_coverage_subject_slug_collision(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "example-coverage.yaml").write_text(
        """coverage_gaps:
  - id: docs_media.generated_evidence
    domain: docs_media
    gap: Missing generated media evidence
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: "#590"
    next_evidence: devtools render docs-surface --check
  - id: docs-media.generated-evidence
    domain: docs_media
    gap: Missing generated media evidence
    owner: docs-media
    severity: major
    declared_at: "2026-05-02"
    review_after: "2026-08-01"
    issue: "#590"
    next_evidence: devtools render docs-surface --check
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_coverage_gaps(plans)

    assert errors == [
        f"{plans / 'example-coverage.yaml'}: coverage_gaps[1] 'docs-media.generated-evidence' "
        "duplicate coverage subject slug 'docs-media-generated-evidence'"
    ]


def test_coverage_reference_manifest_accepts_existing_commands_and_paths(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "tests").mkdir()
    (plans / "tests" / "test_example.py").write_text("def test_example(): pass\n", encoding="utf-8")
    (plans / "example-coverage.yaml").write_text(
        """items:
  example:
    location: tests/test_example.py (unit coverage)
    verified_by: devtools verify manifests
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


def _seed_paths(plans: Path, *relative_paths: str) -> None:
    """Create empty stand-ins for paths the manifest references.

    ``check_campaign_coverage_catalog`` now requires every active campaign's
    declared tests (and mutation paths_to_mutate) to exist on disk.
    """
    for rel in relative_paths:
        target = plans / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)


def test_campaign_coverage_catalog_accepts_authored_catalog(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    plans = tmp_path
    _seed_paths(
        plans,
        "polylogue/archive/filter/filters.py",
        "tests/unit/core/test_filters_props.py",
        "tests/benchmarks/test_storage.py",
    )
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
    # Seed the manifest-declared (old) paths so this test exercises only the
    # mismatch-vs-authored-catalog assertions; path-existence checks are
    # covered separately below.
    _seed_paths(
        plans,
        "polylogue/archive/filter/old_filters.py",
        "tests/unit/core/test_filters_props.py",
        "tests/benchmarks/test_storage.py",
    )
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


def _stub_inventory(
    *runs: str,
    jobs: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
) -> WorkflowInventory:
    facts = WorkflowFacts(
        path=Path("/fake/wf.yml"),
        workflow_name="Test",
        job_names=jobs,
        run_commands=tuple(runs),
        artifact_uploads=artifacts,
        triggers=("workflow_dispatch",),
    )
    return WorkflowInventory(workflows=(facts,))


def test_distribution_ci_claims_accept_real_workflow(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "distribution-coverage.yaml").write_text(
        """artifacts:
  nix_package:
    build_command: nix build
    ci_build: true
    ci_test: true
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_distribution_ci_claims(
        plans,
        inventory=_stub_inventory("nix build .#polylogue", "nix flake check"),
    )
    assert errors == []


def test_distribution_ci_claims_rejects_unbacked_ci_build(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "distribution-coverage.yaml").write_text(
        """artifacts:
  wheel:
    build_command: uv build --wheel .
    ci_build: true
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_distribution_ci_claims(
        plans,
        inventory=_stub_inventory("uv run pytest"),
    )
    assert errors == [
        f"{plans / 'distribution-coverage.yaml'}: artifacts.wheel ci_build=true but "
        "build_command 'uv build --wheel .' does not appear in any workflow run step"
    ]


def test_distribution_ci_claims_rejects_ci_build_without_build_command(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "distribution-coverage.yaml").write_text(
        """artifacts:
  wheel:
    ci_build: true
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_distribution_ci_claims(
        plans,
        inventory=_stub_inventory("uv run pytest"),
    )
    assert errors == [
        f"{plans / 'distribution-coverage.yaml'}: artifacts.wheel ci_build=true but build_command is missing"
    ]


def test_test_quality_ci_gate_accepts_real_workflow(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "test-quality-coverage.yaml").write_text(
        """dimensions:
  direct_coverage:
    tool: pytest-cov
    ci_gate: true
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_test_quality_ci_claims(
        plans,
        inventory=_stub_inventory("uv run devtools verify coverage"),
    )
    assert errors == []


def test_test_quality_ci_gate_rejects_unbacked_claim(tmp_path: Path) -> None:
    plans = tmp_path
    (plans / "test-quality-coverage.yaml").write_text(
        """dimensions:
  direct_coverage:
    tool: pytest-cov
    ci_gate: true
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_test_quality_ci_claims(
        plans,
        inventory=_stub_inventory("uv run ruff check"),
    )
    assert len(errors) == 1
    assert "ci_gate=true but no workflow run step invokes" in errors[0]


# ---------------------------------------------------------------------------
# Pack C: campaign test-path and freshness/artifact existence enforcement
# ---------------------------------------------------------------------------


def test_campaign_coverage_rejects_missing_test_path(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A campaign that points at a nonexistent test path must fail."""
    plans = tmp_path
    # Seed the mutation source but not the test path.
    _seed_paths(plans, "polylogue/archive/filter/filters.py")
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(),
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
        """mutation_campaigns: []
benchmark_campaigns:
  - name: storage
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: active
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_campaign_coverage_catalog(plans)
    assert errors == [
        f"{plans / 'campaign-coverage.yaml'}: benchmark_campaigns campaign 'storage' "
        "tests[0] path does not exist: 'tests/benchmarks/test_storage.py'"
    ]


def test_campaign_coverage_rejects_empty_tests_list(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """An active campaign with an empty tests list cannot pass."""
    plans = tmp_path
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(),
            benchmark_campaigns=(SimpleNamespace(name="storage", description="d", tests=()),),
        ),
    )
    (plans / "campaign-coverage.yaml").write_text(
        """mutation_campaigns: []
benchmark_campaigns:
  - name: storage
    description: d
    tests: []
    status: active
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_campaign_coverage_catalog(plans)
    assert any("declares no tests" in err for err in errors), errors


def test_campaign_coverage_freshness_passes_for_recent_artifact(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A campaign with freshness_days passes when a recent artifact exists."""
    plans = tmp_path
    _seed_paths(plans, "tests/benchmarks/test_storage.py")
    artifact_dir = plans / ".local" / "benchmark-campaigns"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "2026-05-16-storage.json"
    artifact_path.write_text('{"benchmarks": []}', encoding="utf-8")  # non-empty
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(),
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
        """mutation_campaigns: []
benchmark_campaigns:
  - name: storage
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: active
    freshness_days: 30
""",
        encoding="utf-8",
    )

    assert verify_manifests.check_campaign_coverage_catalog(plans) == []


def test_campaign_coverage_freshness_fails_when_artifact_missing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """freshness_days without a matching artifact must fail loudly."""
    plans = tmp_path
    _seed_paths(plans, "tests/benchmarks/test_storage.py")
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(),
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
        """mutation_campaigns: []
benchmark_campaigns:
  - name: storage
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: active
    freshness_days: 7
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_campaign_coverage_catalog(plans)
    assert any("no matching artifacts exist" in err for err in errors), errors


def test_campaign_coverage_freshness_fails_when_artifact_is_stale(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """An artifact older than freshness_days must trip the staleness check."""
    import os
    import time

    plans = tmp_path
    _seed_paths(plans, "tests/benchmarks/test_storage.py")
    artifact_dir = plans / ".local" / "benchmark-campaigns"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "2020-01-01-storage.json"
    artifact_path.write_text('{"benchmarks": []}', encoding="utf-8")
    old_ts = time.time() - 365 * 86400
    os.utime(artifact_path, (old_ts, old_ts))

    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(),
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
        """mutation_campaigns: []
benchmark_campaigns:
  - name: storage
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: active
    freshness_days: 30
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_campaign_coverage_catalog(plans)
    assert any("exceeding freshness_days=30" in err for err in errors), errors


def test_campaign_coverage_artifact_glob_must_resolve(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """An explicit artifact_glob must match at least one non-empty artifact."""
    plans = tmp_path
    _seed_paths(plans, "tests/benchmarks/test_storage.py")
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(),
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
        """mutation_campaigns: []
benchmark_campaigns:
  - name: storage
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: active
    artifact_glob: ".local/never-existed/*.json"
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_campaign_coverage_catalog(plans)
    assert any("no matching artifacts exist" in err for err in errors), errors


def test_campaign_coverage_inactive_status_skips_path_check(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Inactive campaigns are intentionally exempt from path enforcement."""
    plans = tmp_path
    # Note: no seeded paths — they don't exist.
    monkeypatch.setattr(
        verify_manifests,
        "get_authored_scenario_catalog",
        lambda: SimpleNamespace(
            mutation_campaigns=(),
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
        """mutation_campaigns: []
benchmark_campaigns:
  - name: storage
    description: Storage benchmarks
    tests:
      - tests/benchmarks/test_storage.py
    status: archived
""",
        encoding="utf-8",
    )

    errors = verify_manifests.check_campaign_coverage_catalog(plans)
    path_errors = [e for e in errors if "path does not exist" in e]
    assert path_errors == []
