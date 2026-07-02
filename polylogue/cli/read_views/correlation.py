"""Correlation read-view handler."""

from __future__ import annotations

from typing import cast

from polylogue.cli.read_view_registry import CORRELATION_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import ReadViewCorrelationOptions, ReadViewInvocation, ReadViewOptionValues
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def build_correlation_options(values: ReadViewOptionValues) -> ReadViewCorrelationOptions:
    """Build options owned by the correlation read view."""

    return ReadViewCorrelationOptions(
        repo_path=cast(str | None, values.get("repo_path")),
        since_hours=cast(int, values.get("since_hours", 2)),
        confidence_threshold=cast(float, values.get("confidence_threshold", 0.3)),
        github_api=cast(bool, values.get("github_api", True)),
        otlp=cast(bool, values.get("otlp", False)),
    )


def run_read_correlation(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render GitHub/Git/OTLP correlation evidence around one session."""

    from polylogue.insights.correlation_view import run_correlation_view

    del request
    assert invocation.session_id is not None
    options = cast(ReadViewCorrelationOptions, invocation.options or ReadViewCorrelationOptions())
    run_correlation_view(
        env,
        session_id=invocation.session_id,
        repo_path=options.repo_path,
        since_hours=options.since_hours,
        output_format=invocation.output_format,
        confidence_threshold=options.confidence_threshold,
        github_api=options.github_api,
        otlp=options.otlp,
    )


__all__ = ["CORRELATION_READ_VIEW_OPTION_NAMES", "build_correlation_options", "run_read_correlation"]
