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
        "| Source | Projection | Path Targets | Artifact Targets | Operation Targets | Tags | Description |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry.source_kind.value}` | `{entry.name}` | "
            f"{_format_code_list(entry.runtime_path_targets())} | "
            f"{_format_code_list(entry.artifact_targets)} | "
            f"{_format_code_list(entry.operation_targets)} | "
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
        f"- covered declared operation targets: `{len(coverage.declared_operations)}`",
        "- uncovered runtime paths: " + ("—" if not uncovered_paths else ", ".join(f"`{name}`" for name in uncovered_paths)),
        "- uncovered runtime artifacts: "
        + ("—" if not coverage.uncovered_artifacts else ", ".join(f"`{name}`" for name in coverage.uncovered_artifacts)),
        "- uncovered runtime operations: "
        + (
            "—"
            if not coverage.uncovered_operations
            else ", ".join(f"`{name}`" for name in coverage.uncovered_operations)
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
        control_plane_command("artifact-graph"),
        control_plane_command("artifact-graph", "--json"),
        "```",
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
        *(
            f"  - {source_kind}: `{count}`"
            for source_kind, count in sorted(projection_counts.items())
        ),
    ]


def build_document(registry: QualityRegistry, *, runtime_coverage: RuntimeScenarioCoverage | None = None) -> str:
    coverage = runtime_coverage or build_runtime_scenario_coverage(projections=registry.scenario_projections)
    parts = [
        "[← Back to README](../README.md)",
        "",
        f"<!-- Generated by `{control_plane_command('render-quality-reference')}`. Edit devtools/render_quality_reference.py and devtools/quality_registry.py instead. -->",
        "",
        "# Test Quality Workflows",
        "",
        "This reference is generated from the live validation-lane, mutation-campaign, and benchmark-campaign registries.",
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
        "### Validation lanes",
        "",
        "```bash",
        control_plane_command("run-validation-lanes", "--list"),
        control_plane_command("run-validation-lanes", "--lane", "frontier-local"),
        control_plane_command("run-validation-lanes", "--lane", "live-exercises", "--dry-run"),
        "```",
        "",
        "### Mutation campaigns",
        "",
        "```bash",
        control_plane_command("mutmut-campaign", "list"),
        control_plane_command("mutmut-campaign", "run", "<campaign>"),
        control_plane_command("mutmut-campaign", "index"),
        "```",
        "",
        "### Benchmark campaigns",
        "",
        "```bash",
        control_plane_command("benchmark-campaign", "list"),
        control_plane_command("benchmark-campaign", "run", "<campaign>"),
        control_plane_command("benchmark-campaign", "compare") + " \\",
        "  .local/benchmark-campaigns/<baseline>.json \\",
        "  .local/benchmark-campaigns/<candidate>.json",
        control_plane_command("benchmark-campaign", "index"),
        control_plane_command("run-benchmark-campaigns", "--list"),
        control_plane_command("run-benchmark-campaigns", "--scale", "medium", "--campaign", "<campaign>"),
        "```",
        "",
        "### Fast pipeline probes",
        "",
        "```bash",
        control_plane_command(
            "pipeline-probe",
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
            "pipeline-probe",
            "--input-mode",
            "archive-subset",
            "--source-db",
            '"$XDG_DATA_HOME/polylogue/polylogue.db"',
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
            "pipeline-probe",
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
        "### Showcase baseline drift",
        "",
        "```bash",
        control_plane_command("verify-showcase"),
        control_plane_command("verify-showcase", "--update"),
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
        "These campaigns generate synthetic archives and run long-haul benchmark workloads through `devtools run-benchmark-campaigns`.",
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
        "These are the authored scenario-bearing projections currently feeding runtime coverage and related control-plane maps.",
        "",
        *_render_scenario_projection_table(registry.scenario_projections),
        "",
        "## Artifact Locations",
        "",
        "- mutation campaigns: `.local/mutation-campaigns/`",
        "- benchmark campaigns: `.local/benchmark-campaigns/`",
        "- pipeline probe manifests and replay bundles: user-selected paths outside the repo by default",
        "",
        "## Slow-Test Policy",
        "",
        "- keep the default correctness lane representative of real archive/storage invariants",
        "- mark a test `slow` only when it is an optional heavy check, not a core correctness contract",
        "- review `pytest --durations` output when a new heavy test appears",
        "",
        "## Workflow Guidance",
        "",
        "When changing code in a narrow domain:",
        "",
        "1. run the targeted `pytest -q -n 0` slice for that domain",
        "2. run the corresponding mutation campaign if that domain has one",
        "3. run the corresponding benchmark campaign only if the change touches a hot path already represented in `tests/benchmarks`",
        "4. rerun the full correctness lane before closing the work",
        "",
        "When a new validation lane, mutation campaign, or benchmark campaign is added, regenerate this document.",
        "",
    ]
    return "\n".join(parts)


def write_if_changed(output_path: Path, content: str) -> None:
    try:
        current = output_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        current = None
    if current == content:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(output_path)


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
            print("render-quality-reference: --check does not support --output -", file=sys.stderr)
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
            print(f"render-quality-reference: out of sync: {output_path}", file=sys.stderr)
            print(
                f"render-quality-reference: run: {control_plane_command('render-quality-reference', '--output', str(output_path))}",
                file=sys.stderr,
            )
            return 1
        print(f"render-quality-reference: sync OK: {output_path}")
        return 0

    write_if_changed(output_path, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
