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
    CorpusScenario,
    CorpusSpec,
    ScenarioProjectionEntry,
    ScenarioProjectionSourceKind,
    command_execution,
    composite_execution,
    pytest_execution,
)


def test_build_document_includes_live_registry_sections() -> None:
    registry = QualityRegistry(
        catalog=AuthoredScenarioCatalog(
            exercise_scenarios=(),
            qa_extra_scenarios=(),
            validation_lanes=(
                LaneEntry(
                    name="machine-contract",
                    description="Machine-readable CLI surface.",
                    timeout_s=120,
                    category="contract",
                    execution=pytest_execution("-m", "machine_contract"),
                ),
                LaneEntry(
                    name="live-exercises",
                    description="Read-only live archive exercises.",
                    timeout_s=300,
                    category="live",
                    execution=command_execution("polylogue", "--plain", "audit", "--only", "exercises"),
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
                    paths_to_mutate=("polylogue/lib/filters.py",),
                    tests=("tests/unit/core/test_filters.py",),
                ),
            ),
            benchmark_campaigns=(
                BenchmarkCampaignEntry(
                    name="search-filters",
                    description="Search latency domain.",
                    execution=pytest_execution("tests/benchmarks/test_search_filters.py"),
                    warn_pct=10.0,
                    fail_pct=20.0,
                ),
            ),
            synthetic_benchmark_campaigns=(
                BenchmarkCampaignEntry(
                    name="startup-health",
                    description="Synthetic startup-health benchmark.",
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
                source_kind=ScenarioProjectionSourceKind.EXERCISE,
                name="json-doctor-action-event-preview",
                description="Action-event doctor preview.",
                origin="generated.json-contract",
                path_targets=("action-event-repair-loop",),
                artifact_targets=("action_event_rows",),
                operation_targets=("project-action-event-health",),
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
                description="Compiled inferred corpus scenario for chatgpt v1 across 1 corpus variant(s).",
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
                ).to_projection_entry().source_payload | {"variant_count": 1, "target_labels": ["default"]},
            ),
        ),
    )

    rendered = render_quality_reference.build_document(registry)

    assert "Generated by `devtools render-quality-reference`" in rendered
    assert "## Validation Lane Catalog" in rendered
    assert "`machine-contract`" in rendered
    assert "`frontier-local`" in rendered
    assert "`filters`" in rendered
    assert "`search-filters`" in rendered
    assert "`startup-health`" in rendered
    assert "## Inferred Corpus Catalog" in rendered
    assert "`chatgpt`" in rendered
    assert "## Synthetic Benchmark Campaign Catalog" in rendered
    assert "- inferred corpus scenarios: `1`" in rendered
    assert "- scenario projections: `3`" in rendered
    assert "  - benchmark-campaign: `1`" in rendered
    assert "  - exercise: `1`" in rendered
    assert "  - inferred-corpus-scenario: `1`" in rendered
    assert "## Scenario Projection Catalog" in rendered
    assert "| `exercise` | `json-doctor-action-event-preview` |" in rendered
    assert "| `inferred-corpus-scenario` | `chatgpt:v1` |" in rendered
    assert "`action-event-repair-loop`" in rendered


def test_build_document_includes_runtime_coverage_section() -> None:
    registry = QualityRegistry(
        catalog=AuthoredScenarioCatalog(
            exercise_scenarios=(),
            qa_extra_scenarios=(),
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
            "action_event_rows": (
                ScenarioCoverageRef(
                    source="exercise",
                    name="json-doctor-action-event-preview",
                    origin="generated.json-contract",
                ),
            ),
        },
        operations={},
        declared_operations={},
        paths={
            "action-event-repair-loop": RuntimePathCoverage(
                name="action-event-repair-loop",
                refs=(
                    ScenarioCoverageRef(
                        source="exercise",
                        name="json-doctor-action-event-preview",
                        origin="generated.json-contract",
                    ),
                ),
                uncovered_artifacts=("tool_use_source_blocks",),
                uncovered_operations=("materialize-action-events",),
            ),
        },
        uncovered_artifacts=("tool_use_source_blocks",),
        uncovered_operations=("materialize-action-events",),
        uncovered_declared_operations=("benchmark.repair.action-events",),
    )

    rendered = render_quality_reference.build_document(registry, runtime_coverage=coverage)

    assert "## Runtime Coverage" in rendered
    assert "- covered runtime paths: `0`" in rendered
    assert "- covered runtime artifacts: `1`" in rendered
    assert "- covered declared operation targets: `0`" in rendered
    assert "- uncovered runtime paths: `action-event-repair-loop`" in rendered
    assert "- uncovered runtime artifacts: `tool_use_source_blocks`" in rendered
    assert "- uncovered runtime operations: `materialize-action-events`" in rendered
    assert "- uncovered declared operation targets: `benchmark.repair.action-events`" in rendered
    assert "devtools artifact-graph" in rendered


def test_write_if_changed_reuses_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "test-quality-workflows.md"
    content = "hello\n"
    render_quality_reference.write_if_changed(output_path, content)
    original_mtime = output_path.stat().st_mtime_ns

    render_quality_reference.write_if_changed(output_path, content)

    assert output_path.read_text(encoding="utf-8") == content
    assert output_path.stat().st_mtime_ns == original_mtime
