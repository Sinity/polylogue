"""Small public root for plain/rich check rendering helpers."""

from __future__ import annotations

from polylogue.cli.check_rendering_maintenance import emit_maintenance_output
from polylogue.cli.check_rendering_plain_sections import build_report_lines
from polylogue.cli.check_rendering_plain_support import status_icon
from polylogue.cli.types import AppEnv

from .check_workflow import CheckCommandOptions, CheckCommandResult


def _status_icon(status, *, plain: bool) -> str:
    return status_icon(status, plain=plain)


def render_plain_output(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> None:
    env.ui.summary("Health Check", build_report_lines(env, result, options))
    emit_maintenance_output(env, result, options)
