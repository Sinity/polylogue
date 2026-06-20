"""Correlation read-view handler."""

from __future__ import annotations

from typing import cast

from polylogue.cli.read_views.base import ReadViewCorrelationOptions, ReadViewInvocation
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


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


__all__ = ["run_read_correlation"]
