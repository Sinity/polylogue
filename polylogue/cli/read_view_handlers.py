"""Executable read-view handlers for the query-first CLI."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import click

from polylogue.archive.viewport import read_view_choices
from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.insights.transforms import RecoveryReportPreset

ReadViewSessionPolicy = Literal["optional", "required", "query_or_session", "none"]


@dataclass(frozen=True, slots=True)
class ReadViewInvocation:
    """Normalized CLI arguments for one read-view execution."""

    view: str
    session_id: str | None
    output_format: str | None
    destination: str
    out_path: str | None
    limit: int | None = None
    offset: int = 0
    message_role: tuple[str, ...] = ()
    message_type: str | None = None
    no_code_blocks: bool = False
    no_tool_calls: bool = False
    no_tool_outputs: bool = False
    no_file_reads: bool = False
    prose_only: bool = False
    window_hours: int = 24
    repo_path: str | None = None
    since_hours: int = 2
    confidence_threshold: float = 0.3
    github_api: bool = True
    otlp: bool = False
    related_limit: int = 5
    recovery_report: str | None = None
    project_path: str | None = None
    project_repo: str | None = None
    since: str | None = None
    until: str | None = None
    pack_origin: str | None = None
    pack_query: str | None = None
    max_sessions: int = 5
    max_messages: int = 20
    no_redact: bool = False


ReadViewHandlerFunc = Callable[[AppEnv, "RootModeRequest", ReadViewInvocation], None]


@dataclass(frozen=True, slots=True)
class ReadViewHandler:
    """Executable handler contract for a read-view id."""

    view_id: str
    session_policy: ReadViewSessionPolicy
    handler: ReadViewHandlerFunc
    default_format: str | None = None

    def validate(self, invocation: ReadViewInvocation, request: RootModeRequest) -> None:
        """Validate cross-view selection rules before executing the handler."""

        if self.session_policy == "required" and invocation.session_id is None:
            raise click.UsageError(
                f"read --view {self.view_id} requires a session ID (use --id, id:prefix, or --latest)."
            )
        if self.session_policy == "query_or_session" and invocation.session_id is None:
            query_seed = " ".join(request.query_terms).strip()
            if not query_seed:
                raise click.UsageError(
                    f"read --view {self.view_id} requires a seed (use --id, id:prefix, --latest, or a query)."
                )


def run_read_view(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Execute a registered read view."""

    try:
        handler = READ_VIEW_HANDLERS[invocation.view]
    except KeyError as exc:  # pragma: no cover - Click choice prevents this.
        raise click.UsageError(f"Unknown read view: {invocation.view}") from exc
    handler.validate(invocation, request)
    handler.handler(env, request, invocation)


def run_bulk_export_view(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str | None,
    fields: str | None,
    destination: str,
    out_path: str | None,
) -> None:
    """Bulk export all matched sessions."""

    from polylogue.cli.bulk_export import run_bulk_export

    fmt = output_format or "ndjson"
    bulk_fmt = "jsonl" if fmt == "ndjson" else fmt

    if destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        from pathlib import Path

        buf = io.StringIO()

        def _captured_echo_bulk(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_bulk  # type: ignore[assignment]
        try:
            run_bulk_export(env, request, output_format=bulk_fmt, fields=fields)
        finally:
            click.echo = _orig_echo
        Path(out_path).write_text(buf.getvalue(), encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
        return

    run_bulk_export(env, request, output_format=bulk_fmt, fields=fields)


def _run_read_summary_or_transcript(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Standard query/list renderer used by summary and transcript views."""

    fmt = invocation.output_format or "markdown"
    updated = request.with_param_updates(output_format=fmt)
    if invocation.destination in ("stdout", "terminal"):
        _execute_query_request(env, updated)
    elif invocation.destination == "clipboard":
        _execute_query_request(env, updated.with_param_updates(output="clipboard"))
    elif invocation.destination == "file":
        if not invocation.out_path:
            raise click.UsageError("--to file requires --out <path>.")
        _execute_query_request(env, updated.with_param_updates(output=invocation.out_path))
    else:
        _execute_query_request(env, updated)


def _run_read_browser(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Open the first matched session in the daemon web reader."""

    import webbrowser
    from urllib.parse import quote

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.query import _create_query_vector_provider
    from polylogue.cli.query_contracts import build_query_execution_plan
    from polylogue.paths import archive_file_set_root_for_paths

    config = env.config

    async def _find_first() -> str | None:
        plan = build_query_execution_plan(request.query_params())
        archive_root = archive_file_set_root_for_paths(
            archive_root_path=config.archive_root,
            db_anchor=config.db_path,
        )
        vector_provider = _create_query_vector_provider(config, db_path=archive_root / "embeddings.db")
        filter_chain = plan.selection.build_filter(config, vector_provider=vector_provider)
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


def _run_read_messages(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route messages view to messages renderer with destination handling."""

    from polylogue.cli.messages import run_messages

    assert invocation.session_id is not None
    limit = invocation.limit if invocation.limit is not None else 50

    if invocation.destination in ("file", "clipboard"):
        buf = io.StringIO()

        def _captured_echo(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo  # type: ignore[assignment]
        try:
            run_messages(
                env,
                request,
                session_id=invocation.session_id,
                message_role=invocation.message_role,
                message_type=invocation.message_type,
                limit=limit,
                offset=invocation.offset,
                no_code_blocks=invocation.no_code_blocks,
                no_tool_calls=invocation.no_tool_calls,
                no_tool_outputs=invocation.no_tool_outputs,
                no_file_reads=invocation.no_file_reads,
                prose_only=invocation.prose_only,
                output_format=invocation.output_format,
            )
        finally:
            click.echo = _orig_echo
        _deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_messages(
        env,
        request,
        session_id=invocation.session_id,
        message_role=invocation.message_role,
        message_type=invocation.message_type,
        limit=limit,
        offset=invocation.offset,
        no_code_blocks=invocation.no_code_blocks,
        no_tool_calls=invocation.no_tool_calls,
        no_tool_outputs=invocation.no_tool_outputs,
        no_file_reads=invocation.no_file_reads,
        prose_only=invocation.prose_only,
        output_format=invocation.output_format,
    )


def _run_read_raw(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Route raw view to raw renderer with destination handling."""

    from polylogue.cli.messages import run_raw

    assert invocation.session_id is not None
    limit = invocation.limit if invocation.limit is not None else 50
    output_format = invocation.output_format or "json"

    if invocation.destination in ("file", "clipboard", "stdout"):
        buf = io.StringIO()

        def _captured_echo_raw(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_raw  # type: ignore[assignment]
        try:
            run_raw(
                env,
                request,
                session_id=invocation.session_id,
                limit=limit,
                offset=invocation.offset,
                output_format=output_format,
            )
        finally:
            click.echo = _orig_echo
        _deliver_content(env, buf.getvalue(), destination=invocation.destination, out_path=invocation.out_path)
        return

    run_raw(
        env,
        request,
        session_id=invocation.session_id,
        limit=limit,
        offset=invocation.offset,
        output_format=output_format,
    )


def _run_read_context(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Compose the context preamble for the seed session."""

    from polylogue.context.preamble import compose_context_preamble

    del request
    assert invocation.session_id is not None
    preamble = compose_context_preamble(
        env,
        session_id=invocation.session_id,
        related_limit=max(1, invocation.related_limit),
    )
    _deliver_content(env, preamble + "\n", destination=invocation.destination, out_path=invocation.out_path)


def _run_read_context_pack(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the project/query-scoped context pack."""

    from polylogue.context.pack import run_context_pack_view

    del request
    run_context_pack_view(
        env,
        project_path=invocation.project_path,
        project_repo=invocation.project_repo,
        since=invocation.since,
        until=invocation.until,
        origin=invocation.pack_origin,
        query=invocation.pack_query,
        max_sessions=invocation.max_sessions,
        max_messages=invocation.max_messages,
        no_redact=invocation.no_redact,
    )


def _run_read_recovery(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the deterministic recovery digest for one archived session."""

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.shared.helper_support import fail
    from polylogue.cli.shared.machine_errors import success
    from polylogue.surfaces.payloads import model_json_document

    del request
    assert invocation.session_id is not None
    if invocation.recovery_report is not None:
        if invocation.recovery_report == "work-packet" and invocation.output_format == "json":
            packet = run_coroutine_sync(env.polylogue.recovery_work_packet(invocation.session_id))
            if packet is None:
                fail("read", f"Session not found: {invocation.session_id}")
            payload = success({"recovery_work_packet": model_json_document(packet, exclude_none=True)}).to_json()
            _deliver_content(env, payload + "\n", destination=invocation.destination, out_path=invocation.out_path)
            return
        rendered_report = run_coroutine_sync(
            env.polylogue.recovery_report(
                invocation.session_id,
                cast("RecoveryReportPreset", invocation.recovery_report),
            )
        )
        if rendered_report is None:
            fail("read", f"Session not found: {invocation.session_id}")
        _deliver_content(env, rendered_report, destination=invocation.destination, out_path=invocation.out_path)
        return
    digest = run_coroutine_sync(env.polylogue.recovery_digest(invocation.session_id))
    if digest is None:
        fail("read", f"Session not found: {invocation.session_id}")
    if invocation.output_format == "json":
        payload = success({"recovery": model_json_document(digest, exclude_none=True)}).to_json()
        _deliver_content(env, payload + "\n", destination=invocation.destination, out_path=invocation.out_path)
        return
    _deliver_content(env, digest.resume_markdown, destination=invocation.destination, out_path=invocation.out_path)


def _neighbor_score_label(score: float) -> str:
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _neighbor_candidate_heading(candidate: SessionNeighborCandidate) -> str:
    summary = candidate.summary
    date = f" {summary.display_date.isoformat()}" if summary.display_date else ""
    return (
        f"{candidate.rank}. {candidate.session_id} "
        f"[{summary.origin.value}] {summary.display_title}{date} "
        f"(score {_neighbor_score_label(candidate.score)})"
    )


def _render_neighbors_plain(candidates: list[SessionNeighborCandidate]) -> str:
    if not candidates:
        return "No neighboring candidates found.\n"
    lines = [f"Neighbor candidates ({len(candidates)}):"]
    for candidate in candidates:
        lines.append(_neighbor_candidate_heading(candidate))
        for reason in candidate.reasons:
            evidence = f" ({reason.evidence})" if reason.evidence else ""
            lines.append(f"   - {reason.kind}: {reason.detail}{evidence}")
    return "\n".join(lines) + "\n"


def _run_read_neighbors(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render explainable neighbor/near-duplicate candidates for a seed session."""

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.archive.session.neighbor_candidates import NeighborDiscoveryError
    from polylogue.cli.shared.helper_support import fail
    from polylogue.cli.shared.machine_errors import emit_success
    from polylogue.core.enums import Origin
    from polylogue.core.sources import provider_from_origin
    from polylogue.surfaces.payloads import SessionNeighborCandidatePayload, model_json_document

    query_seed = " ".join(request.query_terms).strip() or None
    if not invocation.session_id and not query_seed:
        fail("read", "read --view neighbors requires a seed (use --id, id:prefix, --latest, or a query).")

    origin = request.params.get("origin")
    provider = provider_from_origin(Origin(str(origin))).value if origin else None

    try:
        candidates = run_coroutine_sync(
            env.polylogue.neighbor_candidates(
                session_id=invocation.session_id,
                query=query_seed,
                provider=provider,
                limit=max(1, invocation.limit if invocation.limit is not None else 10),
                window_hours=max(1, invocation.window_hours),
            )
        )
    except NeighborDiscoveryError as exc:
        fail("read", str(exc))

    if invocation.output_format == "json":
        emit_success(
            {
                "neighbors": [
                    model_json_document(
                        SessionNeighborCandidatePayload.from_candidate(candidate),
                        exclude_none=True,
                    )
                    for candidate in candidates
                ]
            }
        )
        return

    _deliver_content(
        env, _render_neighbors_plain(candidates), destination=invocation.destination, out_path=invocation.out_path
    )


def _run_read_correlation(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render GitHub/Git/OTLP correlation evidence around one session."""

    from polylogue.insights.correlation_view import run_correlation_view

    del request
    assert invocation.session_id is not None
    run_correlation_view(
        env,
        session_id=invocation.session_id,
        repo_path=invocation.repo_path,
        since_hours=invocation.since_hours,
        output_format=invocation.output_format,
        confidence_threshold=invocation.confidence_threshold,
        github_api=invocation.github_api,
        otlp=invocation.otlp,
    )


def _execute_query_request(env: AppEnv, request: RootModeRequest) -> None:
    """Execute query with verb-modified params."""

    from polylogue.cli.query import execute_query_request

    execute_query_request(env, request)


def _deliver_content(env: AppEnv, content: str, *, destination: str, out_path: str | None) -> None:
    """Deliver captured content to the requested destination."""

    if destination == "file":
        from pathlib import Path

        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        Path(out_path).write_text(content, encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
    elif destination == "clipboard":
        from polylogue.cli.query_output import copy_to_clipboard

        copy_to_clipboard(env, content)
    else:
        click.echo(content, nl=False)


READ_VIEW_HANDLERS: dict[str, ReadViewHandler] = {
    "summary": ReadViewHandler("summary", "optional", _run_read_summary_or_transcript, default_format="markdown"),
    "transcript": ReadViewHandler("transcript", "optional", _run_read_summary_or_transcript, default_format="markdown"),
    "messages": ReadViewHandler("messages", "required", _run_read_messages, default_format="text"),
    "raw": ReadViewHandler("raw", "required", _run_read_raw, default_format="json"),
    "context": ReadViewHandler("context", "required", _run_read_context, default_format="json"),
    "context-pack": ReadViewHandler("context-pack", "none", _run_read_context_pack, default_format="markdown"),
    "recovery": ReadViewHandler("recovery", "required", _run_read_recovery, default_format="markdown"),
    "neighbors": ReadViewHandler("neighbors", "query_or_session", _run_read_neighbors, default_format="text"),
    "correlation": ReadViewHandler("correlation", "required", _run_read_correlation, default_format="text"),
}


def read_view_handler_ids() -> tuple[str, ...]:
    """Return executable read-view handler ids."""

    return tuple(READ_VIEW_HANDLERS)


def validate_read_view_handler_registry() -> None:
    """Fail fast if profile metadata and executable handlers drift."""

    profile_ids = set(read_view_choices())
    handler_ids = set(READ_VIEW_HANDLERS)
    missing = sorted(profile_ids - handler_ids)
    extra = sorted(handler_ids - profile_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing handlers: {', '.join(missing)}")
        if extra:
            details.append(f"handlers without profiles: {', '.join(extra)}")
        raise RuntimeError("read-view handler registry drift: " + "; ".join(details))


validate_read_view_handler_registry()

__all__ = [
    "READ_VIEW_HANDLERS",
    "ReadViewHandler",
    "ReadViewInvocation",
    "read_view_handler_ids",
    "run_bulk_export_view",
    "run_read_view",
    "validate_read_view_handler_registry",
]
