"""Generative exercise coverage for the showcase system.

Introspects the CLI's Click command tree to discover filter flags and
generates exercises from flag combinations.  Also generates format matrix
exercises and schema verification exercises.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import click

from polylogue.cli.click_app import cli as root_cli
from polylogue.cli.command_inventory import CommandPath, iter_command_paths
from polylogue.scenarios import build_product_contract_surfaces, polylogue_execution
from polylogue.showcase.dimensions import query_read, schema_exercise
from polylogue.showcase.exercise_models import AssertionSpec, Exercise


def discover_filter_flags(cli_group: click.Group) -> list[dict[str, Any]]:
    """Introspect Click command tree and discover filter/query flags.

    Returns a list of flag descriptors with name, type, and default values.
    """
    # The main query surface is the root group (polylogue itself)
    flags: list[dict[str, Any]] = []

    for param in cli_group.params:
        if not isinstance(param, click.Option):
            continue
        name = param.name or ""
        # Filter-like flags
        if any(
            token in name
            for token in (
                "has_",
                "min_",
                "max_",
                "provider",
                "since",
                "until",
                "sort",
                "reverse",
                "exclude",
            )
        ):
            flag_info: dict[str, Any] = {
                "name": name,
                "cli_name": param.opts[0] if param.opts else f"--{name}",
                "is_flag": param.is_flag,
                "type": str(param.type),
                "default": param.default,
            }
            # Determine a sane test value
            if param.is_flag:
                flag_info["test_value"] = None  # just use the flag
            elif "int" in str(param.type).lower():
                flag_info["test_value"] = "5"
            elif "date" in name or "since" in name or "until" in name:
                flag_info["test_value"] = "2020-01-01"
            elif "provider" in name:
                flag_info["test_value"] = "chatgpt"
            else:
                flag_info["test_value"] = None
            flags.append(flag_info)

    return flags


def _make_flag_args(flag: dict[str, Any]) -> list[str]:
    """Build CLI args for a single flag."""
    cli_name = flag["cli_name"]
    if flag["is_flag"]:
        return [cli_name]
    if flag["test_value"] is not None:
        return [cli_name, str(flag["test_value"])]
    return [cli_name]


def generate_filter_scenarios(cli_group: click.Group | None = None) -> tuple[Exercise, ...]:
    """Generate smoke and pairwise filter scenarios from CLI flags."""
    flags = discover_filter_flags(cli_group or root_cli)
    scenarios: list[Exercise] = []

    # Individual smoke: each flag alone
    for flag in flags:
        dims = query_read(complexity="basic")
        args = _make_flag_args(flag) + ["list", "-n", "3"]
        scenarios.append(
            Exercise(
                name=f"gen-filter-{flag['name']}",
                group="generated-filters",
                description=f"Generated: filter with {flag['cli_name']}",
                execution=polylogue_execution(*args),
                needs_data=True,
                tier=dims.derived_tier,
                env="any",
                origin="generated.filters",
                tags=("generated", "filters"),
            )
        )

    # Pairwise combinations (compatible flags only)
    for a, b in itertools.combinations(flags, 2):
        # Skip incompatible pairs
        if a["name"] == b["name"]:
            continue
        if "provider" in a["name"] and "exclude" in b["name"]:
            continue
        if "exclude" in a["name"] and "provider" in b["name"]:
            continue

        dims = query_read(complexity="combinatorial")
        args = _make_flag_args(a) + _make_flag_args(b) + ["list", "-n", "3"]
        pair_name = f"gen-filter-{a['name']}+{b['name']}"
        scenarios.append(
            Exercise(
                name=pair_name,
                group="generated-filters",
                description=f"Generated: {a['cli_name']} + {b['cli_name']}",
                execution=polylogue_execution(*args),
                needs_data=True,
                tier=dims.derived_tier,
                env="any",
                origin="generated.filters",
                tags=("generated", "filters", "pairwise"),
            )
        )

    return tuple(scenarios)


def generate_filter_exercises(cli_group: click.Group | None = None) -> list[Exercise]:
    """Generate smoke and pairwise filter exercises from CLI flags."""
    return list(generate_filter_scenarios(cli_group))


def inventory_command_paths() -> tuple[CommandPath, ...]:
    """Return the recursive command-path inventory for the main CLI."""
    return iter_command_paths(root_cli)


def command_help_exercise_names() -> set[str]:
    """Return the canonical showcase exercise names for command-path help."""
    return {scenario.name for scenario in generate_command_help_scenarios()}


def generate_command_help_scenarios() -> tuple[Exercise, ...]:
    """Generate tier-0 help scenarios from the recursive Click command tree."""
    scenarios: list[Exercise] = []
    for command_path in inventory_command_paths():
        display_name = command_path.display_name
        scenarios.append(
            Exercise(
                name=command_path.help_exercise_name,
                group="structural",
                description=f"{display_name} help",
                execution=polylogue_execution(*command_path.path, "--help"),
                assertion=AssertionSpec(stdout_contains=(f"polylogue {display_name}",)),
                tier=0,
                origin="generated.command-help",
                operation_targets=("cli.help",),
                tags=("generated", "help", "structural"),
            )
        )
    return tuple(scenarios)


def generate_command_help_exercises() -> list[Exercise]:
    """Generate tier-0 help exercises from the recursive Click command tree."""
    return list(generate_command_help_scenarios())


def _has_json_flag(cmd: click.Command) -> bool:
    """Check if a Click command has a --json flag."""
    return any(isinstance(param, click.Option) and "--json" in param.opts for param in cmd.params)


def _json_contract_scenario(
    name: str,
    description: str,
    *args: str,
    needs_data: bool,
    tier: int,
    env: str,
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> Exercise:
    return Exercise(
        name=name,
        group="subcommands",
        description=description,
        execution=polylogue_execution(*args),
        assertion=AssertionSpec(stdout_is_valid_json=True),
        needs_data=needs_data,
        tier=tier,
        env=env,
        output_ext=".json",
        artifact_class="json",
        origin="generated.json-contract",
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=("cli.json-contract", *operation_targets),
        tags=("generated", "json-contract", *tags),
    )


def _product_json_contract_scenarios() -> tuple[Exercise, ...]:
    return tuple(
        _json_contract_scenario(
            spec.name,
            spec.description,
            *spec.args,
            needs_data=spec.needs_data,
            tier=spec.tier,
            env=spec.env,
            tags=spec.tags,
        )
        for spec in build_product_contract_surfaces()
    )


JSON_CONTRACT_SCENARIOS: tuple[Exercise, ...] = (
    _json_contract_scenario("json-doctor", "doctor JSON contract", "doctor", "--json", needs_data=False, tier=0, env="any"),
    _json_contract_scenario(
        "json-doctor-action-event-preview",
        "doctor JSON contract",
        "doctor",
        "--json",
        "--repair",
        "--preview",
        "--target",
        "action_event_read_model",
        needs_data=False,
        tier=0,
        env="any",
        path_targets=("action-event-repair-loop",),
        artifact_targets=("action_event_rows", "action_event_fts", "action_event_health"),
        operation_targets=("project-action-event-health",),
        tags=("maintenance", "action-events"),
    ),
    _json_contract_scenario(
        "json-doctor-session-products-preview",
        "doctor JSON contract",
        "doctor",
        "--json",
        "--repair",
        "--preview",
        "--target",
        "session_products",
        needs_data=False,
        tier=0,
        env="any",
        path_targets=("session-product-repair-loop",),
        artifact_targets=("session_product_rows", "session_product_fts", "session_product_health"),
        operation_targets=("project-session-product-health",),
        tags=("maintenance", "session-products"),
    ),
    _json_contract_scenario("json-tags", "tags JSON contract", "tags", "--json", needs_data=False, tier=0, env="any"),
    _json_contract_scenario(
        "json-audit",
        "audit JSON contract",
        "audit",
        "--only",
        "audit",
        "--json",
        needs_data=False,
        tier=0,
        env="any",
    ),
    _json_contract_scenario("json-schema-list", "schema list JSON contract", "schema", "list", "--json", needs_data=False, tier=0, env="any"),
    _json_contract_scenario("json-schema-audit", "schema audit JSON contract", "schema", "audit", "--json", needs_data=False, tier=0, env="any"),
    *_product_json_contract_scenarios(),
    _json_contract_scenario("json-run-embed", "run embed JSON contract", "run", "embed", "--stats", "--json", needs_data=True, tier=1, env="seeded"),
)


def _command_path_from_polylogue_args(args: tuple[str, ...]) -> tuple[str, ...]:
    path: list[str] = []
    for item in args:
        if item.startswith("-"):
            break
        path.append(item)
    return tuple(path)


def json_contract_exercise_names() -> set[str]:
    """Return the canonical showcase exercise names for curated JSON-contract commands."""
    return {scenario.name for scenario in generate_json_contract_scenarios()}


def generate_json_contract_scenarios() -> tuple[Exercise, ...]:
    """Generate JSON contract scenarios for curated runnable commands."""
    available_paths = {
        tuple(command_path.path)
        for command_path in inventory_command_paths()
        if _has_json_flag(command_path.command)
    }
    return tuple(
        scenario
        for scenario in JSON_CONTRACT_SCENARIOS
        if _command_path_from_polylogue_args(scenario.execution.polylogue_args) in available_paths
    )


def generate_json_contract_exercises() -> list[Exercise]:
    """Generate JSON contract exercises for curated runnable commands."""
    return list(generate_json_contract_scenarios())


# ---------------------------------------------------------------------------
# Format matrix
# ---------------------------------------------------------------------------

_FORMAT_SPECS: dict[str, dict[str, Any]] = {
    "json": {"well_formed": True, "contains": []},
    "markdown": {"well_formed": False, "contains": ["#"]},
    "html": {"well_formed": False, "contains": ["<"]},
    "yaml": {"well_formed": False, "contains": [":"]},
    "plaintext": {"well_formed": False, "contains": []},
    "csv": {"well_formed": False, "contains": [","]},
    "obsidian": {"well_formed": False, "contains": ["---"]},
    "org": {"well_formed": False, "contains": ["#+"]},
}


def _is_valid_json_check(output: str, _exit_code: int) -> str | None:
    try:
        json.loads(output)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    return None


def generate_format_scenarios() -> tuple[Exercise, ...]:
    """Generate format × mode scenarios from the format registry."""
    scenarios: list[Exercise] = []

    modes = [
        ("latest", ["--latest"]),
        ("list", ["list", "-n", "1"]),
        ("count", ["count"]),
    ]

    for fmt, spec in _FORMAT_SPECS.items():
        for mode_name, mode_args in modes:
            # Count mode ignores format — skip
            if mode_name == "count":
                continue

            dims = query_read(output_format=fmt, complexity="basic")
            args = mode_args + ["-f", fmt]
            validation_kwargs: dict[str, Any] = {}

            if spec["well_formed"]:
                validation_kwargs["custom"] = _is_valid_json_check
            # contains checks only apply to "latest" mode which renders
            # full content; "list" mode outputs plain conversation titles
            if spec["contains"] and mode_name == "latest":
                validation_kwargs["stdout_contains"] = tuple(spec["contains"])

            scenarios.append(
                Exercise(
                    name=f"gen-fmt-{fmt}-{mode_name}",
                    group="generated-formats",
                    description=f"Generated: {fmt} format in {mode_name} mode",
                    execution=polylogue_execution(*args),
                    assertion=AssertionSpec(**validation_kwargs),
                    needs_data=True,
                    tier=dims.derived_tier,
                    env="any",
                    origin="generated.formats",
                    tags=("generated", "formats", fmt, mode_name),
                )
            )

    return tuple(scenarios)


def generate_format_exercises() -> list[Exercise]:
    """Generate format × mode exercises from the format registry."""
    return list(generate_format_scenarios())


# ---------------------------------------------------------------------------
# Schema exercises
# ---------------------------------------------------------------------------


def generate_schema_scenarios() -> tuple[Exercise, ...]:
    """Generate schema verification scenarios."""
    from polylogue.schemas.observation import PROVIDERS

    scenarios: list[Exercise] = []

    # Tier 0: schema list returns valid JSON
    dims_smoke = schema_exercise(complexity="smoke", io_mode="read")
    scenarios.append(
        Exercise(
            name="gen-schema-list",
            group="generated-schema",
            description="Generated: schema list --json returns valid JSON",
            execution=polylogue_execution("schema", "list", "--json"),
            assertion=AssertionSpec(stdout_is_valid_json=True),
            tier=dims_smoke.derived_tier,
            env="any",
            output_ext=".json",
            artifact_class="json",
            origin="generated.schema",
            tags=("generated", "schema", "list"),
        )
    )

    # Tier 1: schema explain for each provider
    dims_explain = schema_exercise(complexity="basic", io_mode="read")
    for provider in PROVIDERS:
        scenarios.append(
            Exercise(
                name=f"gen-schema-explain-{provider}",
                group="generated-schema",
                description=f"Generated: schema explain --provider {provider}",
                execution=polylogue_execution("schema", "explain", "--provider", provider),
                tier=dims_explain.derived_tier,
                env="any",
                origin="generated.schema",
                tags=("generated", "schema", provider),
            )
        )

    return tuple(scenarios)


def generate_schema_exercises() -> list[Exercise]:
    """Generate schema verification exercises."""
    return list(generate_schema_scenarios())


_PROVIDER_FEATURES: dict[str, set[str]] = {
    "chatgpt": {"tool-use", "thinking"},
    "claude-ai": {"tool-use", "thinking"},
    "claude-code": {"tool-use", "thinking"},
    "codex": {"tool-use"},
    "gemini": {"tool-use"},
}


def generate_provider_feature_scenarios() -> tuple[Exercise, ...]:
    """Generate provider × content-type cross-product scenarios."""
    scenarios: list[Exercise] = []
    for provider, features in _PROVIDER_FEATURES.items():
        for feature in sorted(features):
            flag = f"--has-{feature}"
            scenarios.append(
                Exercise(
                    name=f"gen-provider-{provider}-has-{feature}",
                    group="generated-filters",
                    description=f"Generated: {provider} has {feature}",
                    execution=polylogue_execution("--provider", provider, flag, "count"),
                    tier=1,
                    env="seeded",
                    needs_data=True,
                    origin="generated.provider-features",
                    tags=("generated", "filters", "provider-features", provider, feature),
                )
            )
    return tuple(scenarios)


def generate_provider_feature_exercises() -> list[Exercise]:
    """Generate provider × content-type cross-product exercises."""
    return list(generate_provider_feature_scenarios())


def generate_qa_extra_scenarios() -> tuple[Exercise, ...]:
    """Generate the extra scenario families exercised by the QA workflow."""
    return (
        *generate_schema_scenarios(),
        *generate_format_scenarios(),
    )


def generate_all_scenarios(cli_group: click.Group | None = None) -> tuple[Exercise, ...]:
    """Generate all scenario categories."""
    scenarios: list[Exercise] = []
    if cli_group is not None:
        scenarios.extend(generate_filter_scenarios(cli_group))
    scenarios.extend(generate_command_help_scenarios())
    scenarios.extend(generate_json_contract_scenarios())
    scenarios.extend(generate_format_scenarios())
    scenarios.extend(generate_schema_scenarios())
    scenarios.extend(generate_provider_feature_scenarios())
    return tuple(scenarios)


def generate_all_exercises(cli_group: click.Group | None = None) -> list[Exercise]:
    """Generate all exercise categories."""
    return list(generate_all_scenarios(cli_group))


__all__ = [
    "discover_filter_flags",
    "generate_all_exercises",
    "generate_all_scenarios",
    "generate_filter_exercises",
    "generate_filter_scenarios",
    "generate_format_exercises",
    "generate_format_scenarios",
    "generate_qa_extra_scenarios",
    "generate_provider_feature_exercises",
    "generate_provider_feature_scenarios",
    "command_help_exercise_names",
    "generate_command_help_exercises",
    "generate_command_help_scenarios",
    "generate_json_contract_exercises",
    "generate_json_contract_scenarios",
    "generate_schema_exercises",
    "generate_schema_scenarios",
    "inventory_command_paths",
    "json_contract_exercise_names",
]
