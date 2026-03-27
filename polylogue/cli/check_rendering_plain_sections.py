"""Section builders for plain check output."""

from __future__ import annotations

from polylogue.cli.check_rendering_plain_artifacts import (
    append_artifact_observation_lines,
    append_artifact_proof_lines,
    append_schema_lines,
)
from polylogue.cli.check_rendering_plain_health import (
    append_derived_model_lines,
    append_runtime_lines,
    build_health_lines,
)
from polylogue.cli.check_rendering_plain_semantics import (
    append_roundtrip_lines,
    append_semantic_lines,
)
from polylogue.cli.check_workflow import CheckCommandOptions, CheckCommandResult
from polylogue.cli.types import AppEnv


def build_report_lines(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> list[str]:
    """Build the full plain-mode report body."""
    lines = build_health_lines(env, result, options)
    append_derived_model_lines(lines, result)
    append_schema_lines(lines, result)
    append_artifact_proof_lines(lines, result)
    append_artifact_observation_lines(lines, result)
    append_semantic_lines(lines, result)
    append_runtime_lines(lines, result, plain=env.ui.plain)
    append_roundtrip_lines(lines, result)
    return lines


__all__ = ["build_report_lines"]
