"""Shared command catalog for repository developer tools."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

CommandMain = Callable[[list[str] | None], int]
CONTROL_PLANE = "devtools"
VERIFICATION_LAB_COMMAND_NAMES: tuple[str, ...] = (
    "render-verification-catalog",
    "affected-obligations",
    "semantic-axis-evidence",
    "lab-corpus",
    "lab-scenario",
    "schema-generate",
    "schema-promote",
    "schema-audit",
)

CATEGORY_ORDER: tuple[str, ...] = (
    "core",
    "generated surfaces",
    "verification",
    "campaigns",
    "maintenance",
)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    category: str
    description: str
    module: str
    entrypoint: str = "main"
    use_when: str | None = None
    examples: tuple[str, ...] = ()
    featured: bool = False

    @property
    def invocation(self) -> str:
        return control_plane_command(self.name)

    @property
    def argv(self) -> tuple[str, ...]:
        return control_plane_argv(self.name)

    def resolve_main(self) -> CommandMain:
        module = importlib.import_module(self.module)
        entrypoint = getattr(module, self.entrypoint)
        if not callable(entrypoint):
            raise TypeError(f"{self.module}.{self.entrypoint} is not callable")

        def _main(argv: list[str] | None = None) -> int:
            result = entrypoint(argv)
            if not isinstance(result, int):
                raise TypeError(f"{self.module}.{self.entrypoint} returned {type(result).__name__}, expected int")
            return result

        return _main

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["invocation"] = self.invocation
        data["argv"] = list(self.argv)
        return data


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "status",
        "core",
        "Render the devshell status view.",
        "devtools.project_motd",
        use_when="Check repo state, generated-surface drift, and the next default verification steps.",
        examples=("devtools status", "devtools status --json", "devtools status --verify-generated"),
        featured=True,
    ),
    CommandSpec("motd", "core", "Alias for `status`.", "devtools.project_motd"),
    CommandSpec(
        "render-all",
        "generated surfaces",
        "Refresh or verify generated docs and agent files.",
        "devtools.render_all",
        use_when="Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.",
        examples=("devtools render-all", "devtools render-all --check"),
        featured=True,
    ),
    CommandSpec(
        "render-agents",
        "generated surfaces",
        "Render AGENTS.md from CLAUDE.md and its included files.",
        "devtools.render_agents",
    ),
    CommandSpec(
        "render-cli-reference",
        "generated surfaces",
        "Render docs/cli-reference.md from live CLI help.",
        "devtools.render_cli_reference",
    ),
    CommandSpec(
        "render-devtools-reference",
        "generated surfaces",
        "Render the command catalog inside docs/devtools.md.",
        "devtools.render_devtools_reference",
    ),
    CommandSpec(
        "render-docs-surface",
        "generated surfaces",
        "Render docs/README.md and the README documentation table.",
        "devtools.render_docs_surface",
    ),
    CommandSpec(
        "render-quality-reference",
        "generated surfaces",
        "Render docs/test-quality-workflows.md from live validation, mutation, and benchmark registries.",
        "devtools.render_quality_reference",
    ),
    CommandSpec(
        "render-verification-catalog",
        "generated surfaces",
        "Render the verification-lab proof catalog from obligation registries.",
        "devtools.render_verification_catalog",
        use_when=(
            "Refresh or verify the proof-obligation catalog that anchors the verification-lab surface after "
            "changing proof subjects, claims, runners, or catalog rendering."
        ),
        examples=(
            "devtools render-verification-catalog",
            "devtools render-verification-catalog --check",
            "devtools render-verification-catalog --json",
        ),
    ),
    CommandSpec(
        "verify",
        "verification",
        "Run the local verification baseline before pushing or creating a PR.",
        "devtools.verify",
        use_when="Run format, lint, mypy, render-all, and test checks locally before pushing.",
        examples=("devtools verify", "devtools verify --quick", "devtools verify --lab"),
        featured=True,
    ),
    CommandSpec(
        "coverage-gate",
        "verification",
        "Run pytest with the repository coverage floor from pyproject.toml.",
        "devtools.coverage_gate",
        use_when="Enforce the committed coverage ratchet locally or in CI without duplicating threshold values.",
        examples=(
            "devtools coverage-gate",
            "devtools coverage-gate --ignore-integration --term-missing",
            "devtools coverage-gate -- --maxfail=1",
        ),
    ),
    CommandSpec(
        "affected-obligations",
        "verification",
        "Route changed paths or refs to affected verification-lab proof obligations and focused checks.",
        "devtools.affected_obligations",
        use_when="Find the proof obligations and inner-loop checks affected by local changes before escalating to full PR gates.",
        examples=(
            "devtools affected-obligations --base-ref master --head-ref HEAD",
            "devtools affected-obligations --path polylogue/sources/parsers/codex.py",
            "devtools affected-obligations --json --path docs/verification-catalog.md",
        ),
    ),
    CommandSpec("run-validation-lanes", "verification", "Run named validation lanes.", "devtools.run_validation_lanes"),
    CommandSpec(
        "artifact-graph",
        "verification",
        "Render the runtime artifact, operation, and scenario-coverage map.",
        "devtools.artifact_graph",
        use_when="Inspect the authored runtime graph and see which scenarios currently cover declared artifacts and operations.",
        examples=("devtools artifact-graph", "devtools artifact-graph --json"),
    ),
    CommandSpec(
        "scenario-projections",
        "verification",
        "Render the authored scenario-bearing verification projections.",
        "devtools.scenario_projections",
        use_when="Inspect the unified projection inventory that feeds runtime coverage, generated docs, and control-plane maps.",
        examples=(
            "devtools scenario-projections",
            "devtools scenario-projections --source-kind exercise --artifact-target action_event_rows",
            "devtools scenario-projections --json",
        ),
    ),
    CommandSpec(
        "verify-showcase",
        "verification",
        "Verify committed showcase/demo surfaces.",
        "devtools.verify_showcase",
        use_when="Check the committed showcase and demo surfaces after changing rendering or publication behavior.",
        examples=("devtools verify-showcase",),
    ),
    CommandSpec(
        "verify-topology",
        "verification",
        "Verify the realized polylogue tree against the topology projection.",
        "devtools.verify_topology",
        use_when=(
            "Detect orphans, conflicts, kernel-rule violations, or stale TBD cells against "
            "docs/plans/topology-target.yaml after moving files between packages."
        ),
        examples=(
            "devtools verify-topology",
            "devtools verify-topology --json",
            "devtools verify-topology --strict-tbd",
        ),
    ),
    CommandSpec(
        "verify-cluster-cohesion",
        "verification",
        "Validate proposed clusters from the topology projection using the import graph.",
        "devtools.verify_cluster_cohesion",
        use_when=(
            "Check whether a proposed subpackage split would be cohesive — flags cross-cluster "
            "imports through internals and cycles between clusters before any file is moved."
        ),
        examples=(
            "devtools verify-cluster-cohesion",
            "devtools verify-cluster-cohesion --cluster lib/query",
        ),
    ),
    CommandSpec(
        "build-topology-projection",
        "generated surfaces",
        "Generate docs/plans/topology-target.yaml from the current tree using placement rules.",
        "devtools.build_topology_projection",
        use_when=(
            "Refresh the topology projection after editing placement rules in this script "
            "or after a topology refactor lands."
        ),
        examples=("devtools build-topology-projection",),
    ),
    CommandSpec(
        "pipeline-probe",
        "verification",
        "Run typed pipeline probes against synthetic, staged, or archive-subset inputs.",
        "devtools.pipeline_probe",
        use_when="Exercise real pipeline stages and optionally capture emitted summaries as regression cases.",
        examples=(
            "devtools pipeline-probe --provider chatgpt --stage parse",
            "devtools pipeline-probe --input-mode archive-subset --capture-regression live-parse-drift",
        ),
    ),
    CommandSpec(
        "query-memory-budget",
        "verification",
        "Measure query-memory envelopes on generated fixtures.",
        "devtools.query_memory_budget",
        use_when="Assert memory budgets around a concrete query or archive-facing command.",
        examples=("devtools query-memory-budget --max-rss-mb 1536 -- polylogue --plain stats",),
    ),
    CommandSpec(
        "semantic-axis-evidence",
        "verification",
        "Generate verification-lab performance evidence across synthetic semantic scale tiers.",
        "devtools.semantic_axis_evidence",
        use_when=(
            "Produce comparative performance evidence that describes growth shape over semantic axes "
            "instead of machine-specific absolute budgets."
        ),
        examples=(
            "devtools semantic-axis-evidence --campaign fts-rebuild --axis messages --scales small medium",
            "devtools semantic-axis-evidence --campaign session-product-materialization --axis conversations --scales small medium",
        ),
    ),
    CommandSpec(
        "lab-corpus",
        "verification",
        "Generate verification-lab synthetic corpus fixtures and demo archives.",
        "devtools.lab_corpus",
        use_when="Seed synthetic corpus files or complete demo workspaces for lab exercises.",
        examples=(
            "devtools lab-corpus generate --provider chatgpt --count 5",
            "devtools lab-corpus seed --env-only",
        ),
    ),
    CommandSpec(
        "lab-scenario",
        "verification",
        "Run verification-lab showcase scenario sets and baseline checks.",
        "devtools.lab_scenario",
        use_when="Run showcase exercise smoke scenarios and committed baseline checks outside the product CLI.",
        examples=(
            "devtools lab-scenario run archive-smoke --tier 0",
            "devtools lab-scenario verify-baselines",
        ),
    ),
    CommandSpec(
        "schema-generate",
        "verification",
        "Generate provider schema packages and optional evidence clusters.",
        "devtools.schema_generate",
        use_when="Refresh provider schema package artifacts from archive observations outside the product CLI.",
        examples=("devtools schema-generate --provider chatgpt --cluster",),
    ),
    CommandSpec(
        "schema-promote",
        "verification",
        "Promote a schema evidence cluster into a registered package version.",
        "devtools.schema_promote",
        use_when="Turn reviewed schema evidence clusters into committed provider schema packages.",
        examples=("devtools schema-promote --provider chatgpt --cluster chatgpt-message-v2",),
    ),
    CommandSpec(
        "schema-audit",
        "verification",
        "Run committed provider schema package quality checks.",
        "devtools.schema_audit",
        use_when="Check committed schema package quality gates without presenting them as normal archive usage.",
        examples=("devtools schema-audit --provider chatgpt --json",),
    ),
    CommandSpec(
        "regression-capture",
        "verification",
        "Capture pipeline-probe summaries as durable local regression cases.",
        "devtools.regression_capture",
        use_when="Turn a live or probe failure JSON summary into a replayable local regression artifact.",
        examples=(
            "devtools regression-capture --input probe.json --name parse-drift",
            "devtools pipeline-probe --json | devtools regression-capture --name parse-drift --tag live",
        ),
    ),
    CommandSpec(
        "inject-semantic-annotations",
        "maintenance",
        "Annotate baseline provider schemas with semantic-role metadata.",
        "devtools.inject_semantic_annotations",
    ),
    CommandSpec(
        "build-package",
        "maintenance",
        "Build the default Nix package with the out-link under .local/result.",
        "devtools.build_package",
        use_when="Produce the Nix package artifact with its out-link kept under the repo-local output root.",
        examples=("devtools build-package",),
    ),
    CommandSpec(
        "mutmut-campaign",
        "campaigns",
        "Run focused mutation campaigns and maintain their local index.",
        "devtools.mutmut_campaign",
        use_when="Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.",
        examples=("devtools mutmut-campaign list", "devtools mutmut-campaign run filters"),
        featured=True,
    ),
    CommandSpec(
        "benchmark-campaign",
        "campaigns",
        "Run or compare benchmark campaigns.",
        "devtools.benchmark_campaign",
        use_when="Record durable benchmark artifacts or compare a candidate run against a baseline artifact.",
        examples=(
            "devtools benchmark-campaign list",
            "devtools benchmark-campaign run search-filters",
            "devtools benchmark-campaign compare baseline.json candidate.json",
        ),
        featured=True,
    ),
    CommandSpec(
        "run-benchmark-campaigns",
        "campaigns",
        "Run synthetic benchmark campaigns over generated archives.",
        "devtools.run_campaign",
    ),
)

COMMANDS: dict[str, CommandSpec] = {spec.name: spec for spec in COMMAND_SPECS}


def control_plane_command(*args: str) -> str:
    parts = [CONTROL_PLANE, *args]
    return " ".join(part for part in parts if part)


def control_plane_argv(*args: str) -> tuple[str, ...]:
    return tuple(part for part in (CONTROL_PLANE, *args) if part)


def featured_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> tuple[CommandSpec, ...]:
    return tuple(spec for spec in commands if spec.featured)


def verification_lab_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> tuple[CommandSpec, ...]:
    by_name = {spec.name: spec for spec in commands}
    return tuple(by_name[name] for name in VERIFICATION_LAB_COMMAND_NAMES)


def grouped_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> OrderedDict[str, list[CommandSpec]]:
    grouped: OrderedDict[str, list[CommandSpec]] = OrderedDict((category, []) for category in CATEGORY_ORDER)
    for spec in commands:
        grouped.setdefault(spec.category, [])
        grouped[spec.category].append(spec)
    for _category, specs in grouped.items():
        specs.sort(key=lambda item: item.name)
    return OrderedDict((category, specs) for category, specs in grouped.items() if specs)


__all__ = [
    "CATEGORY_ORDER",
    "COMMANDS",
    "COMMAND_SPECS",
    "CONTROL_PLANE",
    "CommandMain",
    "CommandSpec",
    "control_plane_argv",
    "control_plane_command",
    "featured_command_specs",
    "grouped_command_specs",
    "VERIFICATION_LAB_COMMAND_NAMES",
    "verification_lab_command_specs",
]
