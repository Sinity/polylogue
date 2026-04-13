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
from polylogue.showcase.dimensions import query_read, schema_exercise
from polylogue.showcase.exercise_models import Exercise, Validation
from polylogue.showcase.scenario_models import ExerciseScenario, compile_exercise_scenarios


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


def generate_filter_exercises(cli_group: click.Group) -> list[Exercise]:
    """Generate smoke and pairwise filter exercises from CLI flags."""
    flags = discover_filter_flags(cli_group)
    exercises: list[Exercise] = []

    # Individual smoke: each flag alone
    for flag in flags:
        dims = query_read(complexity="basic")
        args = _make_flag_args(flag) + ["list", "-n", "3"]
        exercises.append(
            Exercise(
                name=f"gen-filter-{flag['name']}",
                group="generated-filters",
                description=f"Generated: filter with {flag['cli_name']}",
                args=args,
                needs_data=True,
                tier=dims.derived_tier,
                env="any",
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
        exercises.append(
            Exercise(
                name=pair_name,
                group="generated-filters",
                description=f"Generated: {a['cli_name']} + {b['cli_name']}",
                args=args,
                needs_data=True,
                tier=dims.derived_tier,
                env="any",
            )
        )

    return exercises


def inventory_command_paths() -> tuple[CommandPath, ...]:
    """Return the recursive command-path inventory for the main CLI."""
    return iter_command_paths(root_cli)


def command_help_exercise_names() -> set[str]:
    """Return the canonical showcase exercise names for command-path help."""
    return {scenario.scenario_id for scenario in generate_command_help_scenarios()}


def generate_command_help_scenarios() -> tuple[ExerciseScenario, ...]:
    """Generate tier-0 help scenarios from the recursive Click command tree."""
    scenarios: list[ExerciseScenario] = []
    for command_path in inventory_command_paths():
        display_name = command_path.display_name
        scenarios.append(
            ExerciseScenario(
                scenario_id=command_path.help_exercise_name,
                group="structural",
                description=f"{display_name} help",
                args=(*command_path.path, "--help"),
                validation=Validation(stdout_contains=(f"polylogue {display_name}",)),
                tier=0,
                origin="generated.command-help",
                operation_targets=("cli.help",),
                tags=("generated", "help", "structural"),
            )
        )
    return tuple(scenarios)


def generate_command_help_exercises() -> list[Exercise]:
    """Generate tier-0 help exercises from the recursive Click command tree."""
    return list(compile_exercise_scenarios(generate_command_help_scenarios()))


def _has_json_flag(cmd: click.Command) -> bool:
    """Check if a Click command has a --json flag."""
    return any(isinstance(param, click.Option) and "--json" in param.opts for param in cmd.params)


_JSON_CONTRACT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "json-doctor",
        "path": ("doctor",),
        "args": ["doctor", "--json"],
        "needs_data": False,
        "tier": 0,
        "env": "any",
    },
    {
        "scenario_id": "json-doctor-action-event-preview",
        "path": ("doctor",),
        "args": ["doctor", "--json", "--repair", "--preview", "--target", "action_event_read_model"],
        "needs_data": False,
        "tier": 0,
        "env": "any",
        "path_targets": ["action-event-repair-loop"],
        "artifact_targets": ["action_event_rows", "action_event_fts", "action_event_health"],
        "operation_targets": ["project-action-event-health"],
        "tags": ["maintenance", "action-events"],
    },
    {
        "scenario_id": "json-doctor-session-products-preview",
        "path": ("doctor",),
        "args": ["doctor", "--json", "--repair", "--preview", "--target", "session_products"],
        "needs_data": False,
        "tier": 0,
        "env": "any",
        "path_targets": ["session-product-repair-loop"],
        "artifact_targets": ["session_product_rows", "session_product_fts", "session_product_health"],
        "operation_targets": ["project-session-product-health"],
        "tags": ["maintenance", "session-products"],
    },
    {
        "scenario_id": "json-tags",
        "path": ("tags",),
        "args": ["tags", "--json"],
        "needs_data": False,
        "tier": 0,
        "env": "any",
    },
    {
        "scenario_id": "json-audit",
        "path": ("audit",),
        "args": ["audit", "--only", "audit", "--json"],
        "needs_data": False,
        "tier": 0,
        "env": "any",
    },
    {
        "scenario_id": "json-schema-list",
        "path": ("schema", "list"),
        "args": ["schema", "list", "--json"],
        "needs_data": False,
        "tier": 0,
        "env": "any",
    },
    {
        "scenario_id": "json-schema-audit",
        "path": ("schema", "audit"),
        "args": ["schema", "audit", "--json"],
        "needs_data": False,
        "tier": 0,
        "env": "any",
    },
    {
        "scenario_id": "json-products-profiles",
        "path": ("products", "profiles"),
        "args": ["products", "profiles", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-products-work-events",
        "path": ("products", "work-events"),
        "args": ["products", "work-events", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-products-phases",
        "path": ("products", "phases"),
        "args": ["products", "phases", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-products-threads",
        "path": ("products", "threads"),
        "args": ["products", "threads", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-products-tags",
        "path": ("products", "tags"),
        "args": ["products", "tags", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-products-day-summaries",
        "path": ("products", "day-summaries"),
        "args": ["products", "day-summaries", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-products-week-summaries",
        "path": ("products", "week-summaries"),
        "args": ["products", "week-summaries", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-products-analytics",
        "path": ("products", "analytics"),
        "args": ["products", "analytics", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
    {
        "scenario_id": "json-run-embed",
        "path": ("run", "embed"),
        "args": ["run", "embed", "--stats", "--json"],
        "needs_data": True,
        "tier": 1,
        "env": "seeded",
    },
)


def _coerce_optional_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def _json_contract_specs_by_path() -> dict[tuple[str, ...], tuple[dict[str, Any], ...]]:
    specs_by_path: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for spec in _JSON_CONTRACT_SPECS:
        specs_by_path.setdefault(tuple(spec["path"]), []).append(spec)
    return {path: tuple(specs) for path, specs in specs_by_path.items()}


def json_contract_exercise_names() -> set[str]:
    """Return the canonical showcase exercise names for curated JSON-contract commands."""
    return {scenario.scenario_id for scenario in generate_json_contract_scenarios()}


def generate_json_contract_scenarios() -> tuple[ExerciseScenario, ...]:
    """Generate JSON contract scenarios for curated runnable commands."""
    scenarios: list[ExerciseScenario] = []
    by_path = _json_contract_specs_by_path()
    for cp in inventory_command_paths():
        specs = by_path.get(tuple(cp.path))
        if specs is None or not _has_json_flag(cp.command):
            continue
        for spec in specs:
            scenarios.append(
                ExerciseScenario(
                    scenario_id=str(spec.get("scenario_id", f"json-{'-'.join(cp.path)}")),
                    group="subcommands",
                    description=f"{cp.display_name} JSON contract",
                    args=tuple(spec["args"]),
                    validation=Validation(stdout_is_valid_json=True),
                    needs_data=bool(spec["needs_data"]),
                    tier=int(spec["tier"]),
                    env=str(spec["env"]),
                    output_ext=".json",
                    artifact_class="json",
                    origin="generated.json-contract",
                    path_targets=_coerce_optional_string_tuple(spec.get("path_targets")),
                    artifact_targets=_coerce_optional_string_tuple(spec.get("artifact_targets")),
                    operation_targets=("cli.json-contract", *_coerce_optional_string_tuple(spec.get("operation_targets"))),
                    tags=("generated", "json-contract", *_coerce_optional_string_tuple(spec.get("tags"))),
                )
            )
    return tuple(scenarios)


def generate_json_contract_exercises() -> list[Exercise]:
    """Generate JSON contract exercises for curated runnable commands."""
    return list(compile_exercise_scenarios(generate_json_contract_scenarios()))


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


def generate_format_exercises() -> list[Exercise]:
    """Generate format × mode exercises from the format registry."""
    exercises: list[Exercise] = []

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

            exercises.append(
                Exercise(
                    name=f"gen-fmt-{fmt}-{mode_name}",
                    group="generated-formats",
                    description=f"Generated: {fmt} format in {mode_name} mode",
                    args=args,
                    validation=Validation(**validation_kwargs),
                    needs_data=True,
                    tier=dims.derived_tier,
                    env="any",
                )
            )

    return exercises


# ---------------------------------------------------------------------------
# Schema exercises
# ---------------------------------------------------------------------------


def generate_schema_exercises() -> list[Exercise]:
    """Generate schema verification exercises."""
    from polylogue.schemas.observation import PROVIDERS

    exercises: list[Exercise] = []

    # Tier 0: schema list returns valid JSON
    dims_smoke = schema_exercise(complexity="smoke", io_mode="read")
    exercises.append(
        Exercise(
            name="gen-schema-list",
            group="generated-schema",
            description="Generated: schema list --json returns valid JSON",
            args=["schema", "list", "--json"],
            validation=Validation(stdout_is_valid_json=True),
            tier=dims_smoke.derived_tier,
            env="any",
            output_ext=".json",
        )
    )

    # Tier 1: schema explain for each provider
    dims_explain = schema_exercise(complexity="basic", io_mode="read")
    for provider in PROVIDERS:
        exercises.append(
            Exercise(
                name=f"gen-schema-explain-{provider}",
                group="generated-schema",
                description=f"Generated: schema explain --provider {provider}",
                args=["schema", "explain", "--provider", provider],
                tier=dims_explain.derived_tier,
                env="any",
            )
        )

    return exercises


_PROVIDER_FEATURES: dict[str, set[str]] = {
    "chatgpt": {"tool-use", "thinking"},
    "claude-ai": {"tool-use", "thinking"},
    "claude-code": {"tool-use", "thinking"},
    "codex": {"tool-use"},
    "gemini": {"tool-use"},
}


def generate_provider_feature_exercises() -> list[Exercise]:
    """Generate provider × content-type cross-product exercises."""
    exercises: list[Exercise] = []
    for provider, features in _PROVIDER_FEATURES.items():
        for feature in sorted(features):
            flag = f"--has-{feature}"
            exercises.append(
                Exercise(
                    name=f"gen-provider-{provider}-has-{feature}",
                    group="generated-filters",
                    description=f"Generated: {provider} has {feature}",
                    args=["--provider", provider, flag, "count"],
                    tier=1,
                    env="seeded",
                    needs_data=True,
                )
            )
    return exercises


def generate_all_exercises(cli_group: click.Group | None = None) -> list[Exercise]:
    """Generate all exercise categories."""
    exercises: list[Exercise] = []
    if cli_group is not None:
        exercises.extend(generate_filter_exercises(cli_group))
    exercises.extend(generate_command_help_exercises())
    exercises.extend(generate_json_contract_exercises())
    exercises.extend(generate_format_exercises())
    exercises.extend(generate_schema_exercises())
    exercises.extend(generate_provider_feature_exercises())
    return exercises


__all__ = [
    "discover_filter_flags",
    "generate_all_exercises",
    "generate_filter_exercises",
    "generate_format_exercises",
    "generate_provider_feature_exercises",
    "command_help_exercise_names",
    "generate_command_help_exercises",
    "generate_command_help_scenarios",
    "generate_json_contract_exercises",
    "generate_json_contract_scenarios",
    "generate_schema_exercises",
    "inventory_command_paths",
    "json_contract_exercise_names",
]
