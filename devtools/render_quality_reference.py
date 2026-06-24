"""Render the quality workflow reference from live registries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from devtools.benchmark_catalog import BenchmarkCampaignEntry
from devtools.command_catalog import control_plane_command
from devtools.lane_models import LaneEntry
from devtools.mutation_catalog import MutationCampaignEntry
from devtools.quality_registry import QualityRegistry, build_quality_registry
from devtools.render_support import write_if_changed
from devtools.scenario_coverage import RuntimeScenarioCoverage, build_runtime_scenario_coverage
from polylogue.scenarios import CorpusScenario, ScenarioProjectionEntry


def _format_code_list(items: tuple[str, ...]) -> str:
    if not items:
        return "—"
    return "<br>".join(f"`{item}`" for item in items)


def _render_lane_table(entries: tuple[LaneEntry, ...]) -> list[str]:
    lines = [
        "| Lane | Timeout (s) | Description |",
        "| --- | ---: | --- |",
    ]
    for entry in entries:
        lines.append(f"| `{entry.name}` | {entry.timeout_s} | {entry.description} |")
    return lines


def _render_composite_lane_table(entries: tuple[LaneEntry, ...]) -> list[str]:
    lines = [
        "| Lane | Timeout (s) | Includes | Description |",
        "| --- | ---: | --- | --- |",
    ]
    for entry in entries:
        includes = _format_code_list(entry.sub_lanes)
        lines.append(f"| `{entry.name}` | {entry.timeout_s} | {includes} | {entry.description} |")
    return lines


def _render_mutation_table(entries: tuple[MutationCampaignEntry, ...]) -> list[str]:
    lines = [
        "| Campaign | Mutates | Tests | Description |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry.name}` | {_format_code_list(entry.paths_to_mutate)} | "
            f"{_format_code_list(entry.tests)} | {entry.description} |"
        )
    return lines


def _render_benchmark_table(entries: tuple[BenchmarkCampaignEntry, ...]) -> list[str]:
    lines = [
        "| Campaign | Tests | Warn | Fail | Description |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry.name}` | {_format_code_list(entry.tests)} | "
            f"{entry.warn_pct:.1f}% | {entry.fail_pct:.1f}% | {entry.description} |"
        )
    return lines


def _render_inferred_corpus_table(entries: tuple[CorpusScenario, ...]) -> list[str]:
    lines = [
        "| Provider | Package | Variants | Targets | Tags |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry.provider}` | `{entry.package_version}` | `{len(entry.corpus_specs)}` | "
            f"{_format_code_list(entry.target_labels)} | "
            f"{_format_code_list(entry.tags)} |"
        )
    return lines


def _render_scenario_projection_table(entries: tuple[ScenarioProjectionEntry, ...]) -> list[str]:
    lines = [
        "| Source | Projection | Path Targets | Artifact Targets | Operation Targets | Maintenance Targets | Tags | Description |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry.source_kind.value}` | `{entry.name}` | "
            f"{_format_code_list(entry.runtime_path_targets())} | "
            f"{_format_code_list(entry.artifact_targets)} | "
            f"{_format_code_list(entry.operation_targets)} | "
            f"{_format_code_list(entry.maintenance_targets)} | "
            f"{_format_code_list(entry.tags)} | {entry.description} |"
        )
    return lines


def _render_runtime_coverage_section(coverage: RuntimeScenarioCoverage) -> list[str]:
    uncovered_paths = tuple(sorted(name for name, path in coverage.paths.items() if not path.complete))
    return [
        "## Runtime Coverage",
        "",
        f"- covered runtime paths: `{sum(1 for path in coverage.paths.values() if path.complete)}`",
        f"- covered runtime artifacts: `{len(coverage.artifacts)}`",
        f"- covered runtime operations: `{len(coverage.operations)}`",
        f"- covered maintenance targets: `{len(coverage.maintenance_targets)}`",
        f"- covered declared operation targets: `{len(coverage.declared_operations)}`",
        "- uncovered runtime paths: "
        + ("—" if not uncovered_paths else ", ".join(f"`{name}`" for name in uncovered_paths)),
        "- uncovered runtime artifacts: "
        + (
            "—" if not coverage.uncovered_artifacts else ", ".join(f"`{name}`" for name in coverage.uncovered_artifacts)
        ),
        "- uncovered runtime operations: "
        + (
            "—"
            if not coverage.uncovered_operations
            else ", ".join(f"`{name}`" for name in coverage.uncovered_operations)
        ),
        "- uncovered maintenance targets: "
        + (
            "—"
            if not coverage.uncovered_maintenance_targets
            else ", ".join(f"`{name}`" for name in coverage.uncovered_maintenance_targets)
        ),
        "- uncovered declared operation targets: "
        + (
            "—"
            if not coverage.uncovered_declared_operations
            else ", ".join(f"`{name}`" for name in coverage.uncovered_declared_operations)
        ),
        "",
        "Inspect the full authored map with:",
        "",
        "```bash",
        control_plane_command("lab graph"),
        control_plane_command("lab graph", "--json"),
        "```",
        "",
    ]


def _render_test_infrastructure_section() -> list[str]:
    return [
        "## Test Infrastructure Contracts",
        "",
        "Shared helpers under `tests/infra/` are verification substrate. Prefer these contracts over",
        "per-suite JSON parsing, surface invocation, archive seeding, or cross-surface oracle helpers.",
        "",
        "| Helper | Contract | Primary consumers |",
        "| --- | --- | --- |",
        "| `tests/infra/json_contracts.py` | Typed JSON object/envelope/result narrowing for machine surfaces | CLI, MCP, insight, and devtools JSON tests |",
        "| `tests/infra/mcp.py` | MCP surface registration, invocation, and mock archive seams | MCP server and tool-contract tests |",
        "| `tests/infra/storage_records.py` | Durable archive row builders and DB factories | Storage, CLI, insight, and health tests |",
        "| `tests/infra/surfaces.py` | Cross-surface archive adapters over SQLite, repository, and facade projections | Scenario/oracle tests |",
        "",
    ]


def _render_scenario_projection_snapshot(registry: QualityRegistry) -> list[str]:
    projection_counts: dict[str, int] = {}
    for entry in registry.scenario_projections:
        source_kind = entry.source_kind.value
        projection_counts[source_kind] = projection_counts.get(source_kind, 0) + 1
    return [
        f"- scenario projections: `{len(registry.scenario_projections)}`",
        f"- inferred corpus scenarios: `{len(registry.inferred_corpus_scenarios)}`",
        *(f"  - {source_kind}: `{count}`" for source_kind, count in sorted(projection_counts.items())),
    ]


def build_document(registry: QualityRegistry, *, runtime_coverage: RuntimeScenarioCoverage | None = None) -> str:
    coverage = runtime_coverage or build_runtime_scenario_coverage(projections=registry.scenario_projections)
    parts = [
        "[← Back to README](../README.md)",
        "",
        f"<!-- Generated by `{control_plane_command('render quality-reference')}`. Edit devtools/render_quality_reference.py and devtools/quality_registry.py instead. -->",
        "",
        "# Test Quality Workflows",
        "",
        "This reference is generated from the executable validation-lane, mutation-campaign, and benchmark-campaign registries. It is navigation over concrete commands, not a separate proof ledger.",
        "",
        "Current registry snapshot:",
        "",
        f"- contract lanes: `{len(registry.contract_lanes)}`",
        f"- live lanes: `{len(registry.live_lanes)}`",
        f"- composite lanes: `{len(registry.composite_lanes)}`",
        f"- mutation campaigns: `{len(registry.mutation_campaigns)}`",
        f"- benchmark campaigns: `{len(registry.benchmark_campaigns)}`",
        f"- synthetic benchmark campaigns: `{len(registry.synthetic_benchmark_campaigns)}`",
        *_render_scenario_projection_snapshot(registry),
        "",
        *_render_runtime_coverage_section(coverage),
        "## Common Commands",
        "",
        "Commands below assume the project devshell is already active. If not, prefix them with `nix develop -c`.",
        "",
        "### Full correctness run",
        "",
        "```bash",
        "pytest -q -n 0",
        "```",
        "",
        "### Fast local run",
        "",
        "Use this when iterating locally and skipping slow checks and benchmarks.",
        "",
        "```bash",
        'pytest -q -n 0 -m "not slow and not benchmark"',
        "```",
        "",
        *_render_test_infrastructure_section(),
        "### Validation lanes",
        "",
        "Validation lanes are optional wrappers over executable checks. Prefer the focused pytest or `devtools test` command for a narrow edit; use a lane when you need the composed command set it declares.",
        "",
        "```bash",
        control_plane_command("lab lanes", "--list"),
        control_plane_command("lab lanes", "--lane", "frontier-local"),
        control_plane_command("lab lanes", "--lane", "live-archive-smoke", "--dry-run"),
        "```",
        "",
        "### Schema upgrade lane (#1302)",
        "",
        "Polylogue intentionally has no in-place storage schema upgrade chain:",
        "version mismatch is rejected and the operator rebuilds from source (see",
        "[Schema Versioning Model](internals.md#schema-versioning-model)).",
        f"`{control_plane_command('lab policy schema-versioning')}` enforces that policy boundary:",
        "",
        "- It scans `polylogue/storage/sqlite/` for upgrade-shaped helpers",
        "  (`build_vN_to_vM`, `_apply_version_upgrade_plan`, `upgrade_vN_to_vM`,",
        "  `migrate_vN_*`, `ensure_schema_upgrades_vN`).",
        "- If any are found, the lint fails; archive-shape changes edit the",
        "  canonical DDL and require a fresh rebuild from source.",
        "- In the steady state (no helpers committed), the lint passes cleanly.",
        "",
        "The lint runs as part of `devtools verify --lab`, not the fast default path:",
        "the policy boundary is an architectural concern, not a per-edit gate.",
        "",
        "```bash",
        control_plane_command("lab policy schema-versioning"),
        control_plane_command("lab policy schema-versioning", "--json"),
        "```",
        "",
        "### Mutation campaigns",
        "",
        "```bash",
        control_plane_command("bench mutation", "list"),
        control_plane_command("bench mutation", "run", "<campaign>"),
        control_plane_command("bench mutation", "index"),
        "```",
        "",
        "### Benchmark campaigns",
        "",
        "```bash",
        control_plane_command("bench campaign", "list"),
        control_plane_command("bench campaign", "run", "<campaign>"),
        control_plane_command("bench campaign", "compare") + " \\",
        "  .local/benchmark-campaigns/<baseline>.json \\",
        "  .local/benchmark-campaigns/<candidate>.json",
        control_plane_command("bench campaign", "index"),
        control_plane_command("bench synthetic", "--list"),
        control_plane_command("bench synthetic", "--scale", "medium", "--campaign", "<campaign>"),
        "```",
        "",
        "### Fast pipeline probes",
        "",
        "```bash",
        control_plane_command(
            "lab probe pipeline",
            "--provider",
            "chatgpt",
            "--count",
            "5",
            "--stage",
            "parse",
            "--workdir",
            "/tmp/polylogue-probe",
        ),
        control_plane_command(
            "lab probe pipeline",
            "--input-mode",
            "archive-subset",
            "--source-db",
            '"$XDG_DATA_HOME/polylogue/source.db"',
            "--sample-per-provider",
            "50",
            "--stage",
            "parse",
            "--workdir",
            "/tmp/polylogue-probe-real",
            "--manifest-out",
            "/tmp/polylogue-probe-real.json",
        ),
        control_plane_command(
            "lab probe pipeline",
            "--input-mode",
            "archive-subset",
            "--manifest-in",
            "/tmp/polylogue-probe-real.json",
            "--stage",
            "parse",
            "--workdir",
            "/tmp/polylogue-probe-replay",
        ),
        "```",
        "",
        "### Demo and visual behavior checks",
        "",
        "```bash",
        "devtools test tests/unit/cli/test_demo_command.py tests/unit/demo/test_demo_seed_verify.py tests/visual",
        "```",
        "",
        "## Validation Lane Catalog",
        "",
        "Use the named lanes through the runner.",
        "",
        "### Contract Lanes",
        "",
        *_render_lane_table(registry.contract_lanes),
        "",
        "### Live Lanes",
        "",
        *_render_lane_table(registry.live_lanes),
        "",
        "### Composite Lanes",
        "",
        *_render_composite_lane_table(registry.composite_lanes),
        "",
        "## Mutation Campaign Catalog",
        "",
        "Durable mutation ledgers live under `.local/mutation-campaigns/`; workflow policy lives in [../TESTING.md](../TESTING.md).",
        "",
        *_render_mutation_table(registry.mutation_campaigns),
        "",
        "## Benchmark Campaign Catalog",
        "",
        "Benchmark comparisons are manual.",
        "",
        *_render_benchmark_table(registry.benchmark_campaigns),
        "",
        "## Synthetic Benchmark Campaign Catalog",
        "",
        "These campaigns generate synthetic archives and run long-haul benchmark workloads through `devtools bench synthetic`.",
        "",
        *_render_benchmark_table(registry.synthetic_benchmark_campaigns),
        "",
        "## Inferred Corpus Catalog",
        "",
        "These inferred corpus specs come from the live schema registry and participate in the shared scenario projection map.",
        "",
        *_render_inferred_corpus_table(registry.inferred_corpus_scenarios),
        "",
        "## Scenario Projection Catalog",
        "",
        "These projections explain which executable lanes, inferred fixture scenarios, or benchmark campaigns feed runtime coverage maps.",
        "",
        *_render_scenario_projection_table(registry.scenario_projections),
        "",
        "## Artifact Locations",
        "",
        "- mutation campaigns: `.local/mutation-campaigns/`",
        "- benchmark campaigns: `.local/benchmark-campaigns/`",
        "- pipeline probe manifests and replay bundles: user-selected paths outside the repo by default",
        "",
        "## Scale Tier Model",
        "",
        "Scale-sensitive tests opt into one of three tiers (issue #1183). The tier",
        "determines which verification gate runs the test and what corpus size the",
        "fixture seeds.",
        "",
        "| Tier | Marker | Approx. size | Gate |",
        "| --- | --- | --- | --- |",
        "| small | `@pytest.mark.scale_small` | ~100 convs / ~1k msgs | default `devtools verify` |",
        "| medium | `@pytest.mark.scale_medium` | ~1k convs / ~10k msgs | `devtools verify --lab` |",
        "| large | `@pytest.mark.scale_large` | ~10k convs / ~100k msgs | nightly CI (`.github/workflows/nightly-scale.yml`) or explicit `devtools bench campaign` |",
        "",
        "Fixtures live in `tests/infra/scale_fixtures.py` as `tier_small_db`,",
        "`tier_medium_db`, and `tier_large_db`. They are session-scoped and seed",
        "a SQLite archive via the same realistic-distribution helpers used by",
        "`tests/benchmarks/conftest.py`.",
        "",
        "Adding a new scale benchmark:",
        "",
        "1. Decide the smallest tier that exposes the regression you want to catch.",
        "   Prefer `scale_small` so the test runs in the default gate.",
        "2. Mark the test with the matching `@pytest.mark.scale_*` marker and",
        "   request the corresponding `tier_*_db` fixture.",
        "3. Use ratio-based assertions across tiers (large/medium, medium/small)",
        "   instead of absolute milliseconds — the latter bake in host-machine",
        "   assumptions and break portability.",
        "4. If the test needs measured timings, also add `@pytest.mark.benchmark`",
        "   and surface it through a benchmark campaign.",
        "",
        "## Slow-Test Policy",
        "",
        "- keep the default correctness lane representative of real archive/storage invariants",
        "- mark a test `slow` only when it is an optional heavy check, not a core correctness contract",
        "- review `pytest --durations` output when a new heavy test appears",
        "",
        "## Closure Matrix",
        "",
        "The per-domain closure matrix at `docs/plans/test-closure-matrix.yaml` maps each",
        "production domain to its representative tests, the verification gate that runs them,",
        "and any known gaps. `devtools verify closure-matrix` (wired into `devtools verify`)",
        "fails when a declared target or representative test path disappears, when an",
        "`absent` row is missing a `known_gaps` bullet, or when a `required`/`optional`",
        "row has no representative tests.",
        "",
        "When adding a new domain, parser, or surface, add or update its row in the matrix",
        "alongside the test that exercises it. `docs/plans/test-coverage-domains.yaml` is",
        "retained only as audited qualitative background with path-existence validation;",
        "new coverage information should land in the closure matrix first.",
        "",
        "## Workflow Guidance",
        "",
        "When changing code in a narrow domain:",
        "",
        "1. run the targeted `pytest -q -n 0` slice for that domain",
        "2. run the corresponding mutation campaign if that domain has one",
        "3. run the corresponding benchmark campaign only if the change touches a hot path already represented in `tests/benchmarks`",
        "4. rerun the full correctness lane before closing the work",
        "5. update `docs/plans/test-closure-matrix.yaml` if the domain's representative tests change",
        "",
        "When a new validation lane, mutation campaign, or benchmark campaign is added, make sure it dispatches concrete commands and regenerate this document.",
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render docs/test-quality-workflows.md from live quality registries.")
    parser.add_argument(
        "--output",
        default="docs/test-quality-workflows.md",
        help="Output file path or '-' for stdout (default: docs/test-quality-workflows.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the output file is out of sync with the rendered content.",
    )
    args = parser.parse_args(argv)

    output_path = None if args.output == "-" else Path(args.output).expanduser()
    rendered = build_document(build_quality_registry())
    if not rendered.endswith("\n"):
        rendered += "\n"

    if args.output == "-":
        if args.check:
            print("render quality-reference: --check does not support --output -", file=sys.stderr)
            return 2
        sys.stdout.write(rendered)
        return 0

    assert output_path is not None
    if args.check:
        try:
            current = output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = ""
        if current != rendered:
            print(f"render quality-reference: out of sync: {output_path}", file=sys.stderr)
            print(
                f"render quality-reference: run: {control_plane_command('render quality-reference', '--output', str(output_path))}",
                file=sys.stderr,
            )
            return 1
        print(f"render quality-reference: sync OK: {output_path}")
        return 0

    write_if_changed(output_path, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_document", "main", "write_if_changed"]
