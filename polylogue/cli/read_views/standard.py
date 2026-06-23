"""Standard summary/transcript/browser read-view handlers."""

from __future__ import annotations

import webbrowser
from dataclasses import replace
from urllib.parse import quote

from polylogue.cli.read_views.base import ReadViewInvocation, execute_query_request
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def _is_exact_ref_read(request: RootModeRequest, invocation: ReadViewInvocation) -> bool:
    if not invocation.session_id:
        return False
    spec = request.query_spec()
    if not spec.session_id:
        return False
    return not any(
        (
            spec.query_terms,
            spec.contains_terms,
            spec.exclude_text_terms,
            spec.similar_text,
            spec.similar_session_id,
        )
    )


def _request_for_standard_read(request: RootModeRequest, invocation: ReadViewInvocation) -> RootModeRequest:
    if not _is_exact_ref_read(request, invocation):
        return request
    return request.with_param_updates(conv_id=invocation.session_id).with_query_terms(())


def run_read_summary_or_transcript(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Standard query/list renderer used by summary and transcript views."""

    fmt = invocation.output_format or "markdown"
    updated = _request_for_standard_read(request, invocation).with_param_updates(output_format=fmt)
    if invocation.destination in ("stdout", "terminal"):
        execute_query_request(env, updated)
    elif invocation.destination == "clipboard":
        execute_query_request(env, updated.with_param_updates(output="clipboard"))
    elif invocation.destination == "file":
        if not invocation.out_path:
            import click

            raise click.UsageError("--to file requires --out <path>.")
        execute_query_request(env, updated.with_param_updates(output=invocation.out_path))
    else:
        execute_query_request(env, updated)


def run_read_browser(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Open the first matched session in the daemon web reader."""

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.cli.query import _create_query_vector_provider
    from polylogue.paths import archive_file_set_root_for_paths

    config = env.config

    async def _find_first() -> str | None:
        spec = replace(request.query_spec(), limit=1)
        archive_root = archive_file_set_root_for_paths(
            archive_root_path=config.archive_root,
            db_anchor=config.db_path,
        )
        vector_provider = _create_query_vector_provider(config, db_path=archive_root / "embeddings.db")
        filter_chain = spec.build_filter(config, vector_provider=vector_provider)
        first_id: str | None = None
        if filter_chain.can_use_summaries():
            summaries: list[SessionSummary] = list(await filter_chain.list_summaries())
            if summaries:
                first_id = str(summaries[0].id)
        else:
            sessions: list[Session] = list(await filter_chain.list())
            if sessions:
                first_id = str(sessions[0].id)
        return first_id

    session_id = run_coroutine_sync(_find_first())
    if session_id is None:
        effective_format = invocation.output_format or request.params.get("output_format")
        if effective_format == "json":
            from polylogue.cli.shared.machine_errors import error_no_results

            error_no_results("No sessions matched.").emit(exit_code=2)
        env.ui.error("No sessions matched.")
        return

    daemon_url = str(getattr(env, "daemon_url", None) or "http://127.0.0.1:8766").rstrip("/")
    web_url = f"{daemon_url}/s/{quote(session_id, safe='')}"
    webbrowser.open(web_url)
    env.ui.console.print(f"Opened: {web_url}")


__all__ = ["run_read_browser", "run_read_summary_or_transcript"]
