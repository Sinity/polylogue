from __future__ import annotations

from pathlib import Path

from devtools import render_quality_reference
from devtools.authored_scenario_catalog import AuthoredScenarioCatalog
from devtools.benchmark_catalog import BenchmarkCampaignEntry
from devtools.lane_models import LaneEntry
from devtools.mutation_catalog import MutationCampaignEntry
from devtools.quality_registry import QualityRegistry
from devtools.scenario_coverage import RuntimePathCoverage, RuntimeScenarioCoverage, ScenarioCoverageRef
from polylogue.scenarios import (
    AssertionSpec,
    CorpusScenario,
    CorpusSpec,
    ScenarioProjectionEntry,
    ScenarioProjectionSourceKind,
    composite_execution,
    devtools_execution,
    pytest_execution,
)


def test_build_document_includes_live_registry_sections() -> None:
    registry = QualityRegistry(
        catalog=AuthoredScenarioCatalog(
            validation_lanes=(
                LaneEntry(
                    name="machine-contract",
                    description="Machine-readable CLI surface.",
                    timeout_s=120,
                    category="contract",
                    execution=pytest_execution("-m", "machine_contract"),
                ),
                LaneEntry(
                    name="live-archive-smoke",
                    description="Read-only live archive smoke lane.",
                    timeout_s=300,
                    category="live",
                    execution=devtools_execution("lab scenario", "run", "archive-smoke", "--live", "--tier", "0"),
                ),
                LaneEntry(
                    name="runtime-substrate-hardening",
                    description="Runtime substrate hardening lane.",
                    timeout_s=900,
                    category="composite",
                    execution=composite_execution("machine-contract", "live-archive-smoke"),
                    family="runtime-substrate",
                ),
                LaneEntry(
                    name="frontier-local",
                    description="Composite local frontier lane.",
                    timeout_s=900,
                    category="composite",
                    execution=composite_execution("machine-contract"),
                ),
            ),
            mutation_campaigns=(
                MutationCampaignEntry(
                    name="filters",
                    description="Filter semantics.",
                    paths_to_mutate=("polylogue/archive/filter/filters.py",),
                    tests=("tests/unit/core/test_filters.py",),
                ),
            ),
            benchmark_campaigns=(
                BenchmarkCampaignEntry(
                    name="search-filters",
                    description="Search latency domain.",
                    execution=pytest_execution("tests/benchmarks/test_search_filters.py"),
                    assertion=AssertionSpec(benchmark_warn_pct=10.0, benchmark_fail_pct=20.0),
                ),
            ),
            synthetic_benchmark_campaigns=(
                BenchmarkCampaignEntry(
                    name="startup-readiness",
                    description="Synthetic startup-readiness benchmark.",
                ),
            ),
            inferred_corpus_scenarios=(
                CorpusScenario(
                    provider="chatgpt",
                    package_version="v1",
                    corpus_specs=(
                        CorpusSpec.for_provider(
                            "chatgpt",
                            package_version="v1",
                            count=3,
                            messages_min=4,
                            messages_max=16,
                            origin="inferred.schema",
                            tags=("inferred", "schema", "synthetic"),
                        ),
                    ),
                    origin="compiled.inferred-corpus-scenario",
                    tags=("inferred", "schema", "synthetic", "scenario"),
                ),
            ),
        ),
        scenario_projections=(
            ScenarioProjectionEntry(
                source_kind=ScenarioProjectionSourceKind.VALIDATION_LANE,
                name="runtime-substrate-hardening",
                description="Runtime substrate hardening lane.",
                origin="authored",
                source_payload={"family": "runtime-substrate"},
            ),
            ScenarioProjectionEntry(
                source_kind=ScenarioProjectionSourceKind.VALIDATION_LANE,
                name="session-insight-repair",
                description="Session-insight repair lane.",
                origin="authored.validation-lane",
                path_targets=("session-insight-repair-loop",),
                artifact_targets=("session_insight_rows",),
                operation_targets=("project-session-insight-readiness",),
                maintenance_targets=("session_insights",),
                tags=("generated", "json-contract"),
            ),
            ScenarioProjectionEntry(
                source_kind=ScenarioProjectionSourceKind.BENCHMARK_CAMPAIGN,
                name="search-filters",
                description="Search latency domain.",
                origin="authored.benchmark-domain",
                artifact_targets=("message_fts",),
                operation_targets=("benchmark.query.search-filters",),
                tags=("benchmark", "search"),
            ),
            ScenarioProjectionEntry(
                source_kind=ScenarioProjectionSourceKind.INFERRED_CORPUS_SCENARIO,
                name="chatgpt:v1",
                description=("Compiled inferred corpus scenario for chatgpt archive across 1 corpus variant(s)."),
                origin="compiled.inferred-corpus-scenario",
                tags=("inferred", "schema", "synthetic", "scenario"),
                source_payload=CorpusSpec.for_provider(
                    "chatgpt",
                    package_version="v1",
                    count=3,
                    messages_min=4,
                    messages_max=16,
                    origin="inferred.schema",
                    tags=("inferred", "schema", "synthetic"),
                )
                .to_projection_entry()
                .source_payload
                | {"variant_count": 1, "target_labels": ["default"]},
            ),
        ),
    )

    rendered = render_quality_reference.build_document(registry)

    assert "Generated by `devtools render quality-reference`" in rendered
    assert "## Validation Lane Catalog" in rendered
    assert "| `machine-contract`" in rendered
    assert "| `runtime-substrate-hardening`" in rendered
    assert "| `frontier-local`" in rendered
    assert "`filters`" in rendered
    assert "`search-filters`" in rendered
    assert "`startup-readiness`" in rendered
    assert "## Inferred Corpus Catalog" in rendered
    assert "`chatgpt`" in rendered
    assert "## Synthetic Benchmark Campaign Catalog" in rendered
    assert "- inferred corpus scenarios: `1`" in rendered
    assert "- scenario projections: `4`" in rendered
    assert "  - benchmark-campaign: `1`" in rendered
    assert "  - inferred-corpus-scenario: `1`" in rendered
    assert "  - validation-lane: `2`" in rendered
    assert "## Scenario Projection Catalog" in rendered
    assert "| `validation-lane` | `session-insight-repair` |" in rendered
    assert "| `validation-lane` | `runtime-substrate-hardening` |" in rendered
    assert "| `inferred-corpus-scenario` | `chatgpt:v1` |" in rendered
    assert "`session-insight-repair-loop`" in rendered


def test_build_document_includes_runtime_coverage_section() -> None:
    registry = QualityRegistry(
        catalog=AuthoredScenarioCatalog(
            validation_lanes=(),
            mutation_campaigns=(),
            benchmark_campaigns=(),
            synthetic_benchmark_campaigns=(),
            inferred_corpus_scenarios=(),
        ),
        scenario_projections=(),
    )
    coverage = RuntimeScenarioCoverage(
        artifacts={
            "session_insight_rows": (
                ScenarioCoverageRef(
                    source="validation-lane",
                    name="session-insight-repair",
                    origin="authored.validation-lane",
                ),
            ),
        },
        operations={},
        maintenance_targets={},
        declared_operations={},
        paths={
            "session-insight-repair-loop": RuntimePathCoverage(
                name="session-insight-repair-loop",
                refs=(
                    ScenarioCoverageRef(
                        source="validation-lane",
                        name="session-insight-repair",
                        origin="authored.validation-lane",
                    ),
                ),
                uncovered_artifacts=("session_insight_fts",),
                uncovered_operations=("materialize-session-insights",),
            ),
        },
        uncovered_artifacts=("session_insight_fts",),
        uncovered_operations=("materialize-session-insights",),
        uncovered_maintenance_targets=("wal_checkpoint",),
        uncovered_declared_operations=("benchmark.pipeline.index-and-helpers",),
    )

    rendered = render_quality_reference.build_document(registry, runtime_coverage=coverage)

    assert "## Runtime Coverage" in rendered
    assert "- covered runtime paths: `0`" in rendered
    assert "- covered runtime artifacts: `1`" in rendered
    assert "- covered maintenance targets: `0`" in rendered
    assert "- covered declared operation targets: `0`" in rendered
    assert "- uncovered runtime paths: `session-insight-repair-loop`" in rendered
    assert "- uncovered runtime artifacts: `session_insight_fts`" in rendered
    assert "- uncovered runtime operations: `materialize-session-insights`" in rendered
    assert "- uncovered maintenance targets: `wal_checkpoint`" in rendered
    assert "- uncovered declared operation targets: `benchmark.pipeline.index-and-helpers`" in rendered
    assert "devtools lab graph" in rendered


def test_write_if_changed_reuses_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "test-quality-workflows.md"
    content = "hello\n"
    render_quality_reference.write_if_changed(output_path, content)
    original_mtime = output_path.stat().st_mtime_ns

    render_quality_reference.write_if_changed(output_path, content)

    assert output_path.read_text(encoding="utf-8") == content
    assert output_path.stat().st_mtime_ns == original_mtime
