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

from polylogue.showcase.dimensions import (
    ExerciseDimensions,
    query_read,
    query_write,
    schema_exercise,
    structural_smoke,
)
from polylogue.showcase.exercises import Exercise, Validation


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
        if any(token in name for token in (
            "has_", "min_", "max_", "provider", "since", "until",
            "sort", "reverse", "exclude",
        )):
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
        args = ["--list", "-n", "3"] + _make_flag_args(flag)
        exercises.append(Exercise(
            name=f"gen-filter-{flag['name']}",
            group="generated-filters",
            description=f"Generated: filter with {flag['cli_name']}",
            args=args,
            needs_data=True,
            tier=dims.derived_tier,
            env="any",
        ))

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
        args = ["--list", "-n", "3"] + _make_flag_args(a) + _make_flag_args(b)
        pair_name = f"gen-filter-{a['name']}+{b['name']}"
        exercises.append(Exercise(
            name=pair_name,
            group="generated-filters",
            description=f"Generated: {a['cli_name']} + {b['cli_name']}",
            args=args,
            needs_data=True,
            tier=dims.derived_tier,
            env="any",
        ))

    return exercises


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
        ("list", ["--list", "-n", "1"]),
        ("count", ["--count"]),
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

            exercises.append(Exercise(
                name=f"gen-fmt-{fmt}-{mode_name}",
                group="generated-formats",
                description=f"Generated: {fmt} format in {mode_name} mode",
                args=args,
                validation=Validation(**validation_kwargs),
                needs_data=True,
                tier=dims.derived_tier,
                env="any",
            ))

    return exercises


# ---------------------------------------------------------------------------
# Schema exercises
# ---------------------------------------------------------------------------

def generate_schema_exercises() -> list[Exercise]:
    """Generate schema verification exercises."""
    from polylogue.schemas.sampling import PROVIDERS

    exercises: list[Exercise] = []

    # Tier 0: schema list returns valid JSON
    dims_smoke = schema_exercise(complexity="smoke", io_mode="read")
    exercises.append(Exercise(
        name="gen-schema-list",
        group="generated-schema",
        description="Generated: schema list --json returns valid JSON",
        args=["schema", "list", "--json"],
        validation=Validation(stdout_is_valid_json=True),
        tier=dims_smoke.derived_tier,
        env="any",
        output_ext=".json",
    ))

    # Tier 1: schema explain for each provider
    dims_explain = schema_exercise(complexity="basic", io_mode="read")
    for provider in PROVIDERS:
        exercises.append(Exercise(
            name=f"gen-schema-explain-{provider}",
            group="generated-schema",
            description=f"Generated: schema explain --provider {provider}",
            args=["schema", "explain", "--provider", provider],
            tier=dims_explain.derived_tier,
            env="any",
        ))

    return exercises


def generate_all_exercises(cli_group: click.Group | None = None) -> list[Exercise]:
    """Generate all exercise categories."""
    exercises: list[Exercise] = []
    if cli_group is not None:
        exercises.extend(generate_filter_exercises(cli_group))
    exercises.extend(generate_format_exercises())
    exercises.extend(generate_schema_exercises())
    return exercises


__all__ = [
    "discover_filter_flags",
    "generate_all_exercises",
    "generate_filter_exercises",
    "generate_format_exercises",
    "generate_schema_exercises",
]
