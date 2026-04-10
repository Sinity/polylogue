"""Shared command catalog for repository developer tools."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

CommandMain = Callable[[list[str] | None], int]

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

    @property
    def invocation(self) -> str:
        return f"python -m devtools {self.name}"

    def resolve_main(self) -> CommandMain:
        module = importlib.import_module(self.module)
        return getattr(module, self.entrypoint)

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        data["invocation"] = self.invocation
        return data


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("status", "core", "Render the devshell status view.", "devtools.project_motd"),
    CommandSpec("motd", "core", "Alias for `status`.", "devtools.project_motd"),
    CommandSpec(
        "render-all",
        "generated surfaces",
        "Refresh or verify generated docs and agent files.",
        "devtools.render_all",
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
        "verify-showcase", "verification", "Verify committed showcase/demo surfaces.", "devtools.verify_showcase"
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
    ),
    CommandSpec(
        "inject-semantic-annotations",
        "maintenance",
        "Annotate baseline provider schemas with semantic-role metadata.",
        "devtools.inject_semantic_annotations",
    ),
    CommandSpec(
        "mutmut-campaign",
        "campaigns",
        "Run focused mutation campaigns and maintain their local index.",
        "devtools.mutmut_campaign",
    ),
    CommandSpec(
        "benchmark-campaign",
        "campaigns",
        "Run or compare benchmark campaigns.",
        "devtools.benchmark_campaign",
    ),
    CommandSpec(
        "run-benchmark-campaigns",
        "campaigns",
        "Run synthetic benchmark campaigns over generated archives.",
        "devtools.run_campaign",
    ),
)

COMMANDS: dict[str, CommandSpec] = {spec.name: spec for spec in COMMAND_SPECS}


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
    "CommandMain",
    "CommandSpec",
    "grouped_command_specs",
]
