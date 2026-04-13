"""Shared command catalog for repository developer tools."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

CommandMain = Callable[[list[str] | None], int]
CONTROL_PLANE = "devtools"

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
        return getattr(module, self.entrypoint)

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
    CommandSpec("run-validation-lanes", "verification", "Run named validation lanes.", "devtools.run_validation_lanes"),
    CommandSpec("run-scale-lanes", "verification", "Run scale-validation lanes.", "devtools.run_scale_lanes"),
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
        examples=("devtools scenario-projections", "devtools scenario-projections --json"),
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
        "pipeline-probe",
        "verification",
        "Run synthetic pipeline probes against generated archives.",
        "devtools.pipeline_probe",
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
]
