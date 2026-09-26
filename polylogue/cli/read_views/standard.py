"""Standard summary/transcript/browser read-view handlers."""

from __future__ import annotations

import json
import tempfile
import time
import webbrowser
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import quote

import click
import yaml

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.hydration import archive_summary_to_domain
from polylogue.archive.query.transaction import run_archive_read_sync
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content, execute_query_request
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.rendering.formatting import format_session
from polylogue.storage.archive_identity import archive_file_set_root
from polylogue.surfaces.projection_spec import ProjectionSpec, RenderDestination
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceEvent,
    TemporalEvidenceWindow,
    action_row_to_temporal_event,
    build_temporal_evidence_window,
    message_row_to_temporal_event,
    summary_to_temporal_event,
)

TemporalPhaseRecorder = Callable[[str, float, Mapping[str, object]], None]


def _warn_on_written_file_secret_candidates(env: AppEnv, out_path: str | None) -> None:
    """Post-write secret-candidate scan for the streaming-markdown fast path.

    ``stream_exact_session_markdown`` writes directly to a file handle, so
    (unlike the buffered ``deliver_content``/query-set export paths) there is
    no in-memory string to scan before the write happens. Reading the file
    back here still closes the same gap the leak-surfaces audit flagged
    (polylogue-t9xd L11): before this, a transcript export taking this fast
    path was never scanned at all.
    """
    if out_path is None:
        return
    from polylogue.security.secret_scan import describe_path_scan_result, scan_path_for_secret_candidates

    notice = describe_path_scan_result(scan_path_for_secret_candidates(Path(out_path)))
    if notice is None:
        return
    env.ui.console.print(f"[yellow]{notice}[/yellow]")


def _record_temporal_phase(
    recorder: TemporalPhaseRecorder | None,
    name: str,
    started_at: float,
    details: Mapping[str, object] | None = None,
) -> None:
    if recorder is None:
        return
    recorder(name, (time.perf_counter() - started_at) * 1000, details or {})


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
    updated = (
        _request_for_standard_read(request, invocation)
        .with_param_updates(output_format=fmt)
        .with_param_updates(view=invocation.view)
    )
    if invocation.destination in (RenderDestination.STDOUT, RenderDestination.TERMINAL):
        execute_query_request(env, updated)
    elif invocation.destination == RenderDestination.CLIPBOARD:
        execute_query_request(env, updated.with_param_updates(output="clipboard"))
    elif invocation.destination == RenderDestination.BROWSER:
        # Route through the same query-output delivery contract that
        # `find QUERY --to browser` (without `read`) already uses --
        # `QueryOutputSpec`/`deliver_query_output` (query_output.py) resolve
        # ``output="browser"`` to the existing browser-opening path
        # (query_output.open_in_browser). Previously this branch was
        # missing and fell through to the plain-stdout `else`, so
        # `read --to browser` silently printed to the terminal
        # (polylogue-bvnz).
        execute_query_request(env, updated.with_param_updates(output="browser"))
    elif invocation.destination == RenderDestination.FILE:
        if not invocation.out_path:
            raise click.UsageError("--to file requires --out <path>.")
        execute_query_request(env, updated.with_param_updates(output=invocation.out_path))
    else:
        raise click.UsageError(f"Unrecognized read destination: {invocation.destination!r}.")


def run_read_dialogue(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render one session as authored dialogue using the shared content projection."""

    assert invocation.session_id is not None
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    fmt = invocation.output_format or "markdown"
    if invocation.destination == RenderDestination.FILE and fmt == "markdown" and invocation.projection_spec is None:
        if not invocation.out_path:
            raise click.UsageError("--to file requires --out <path>.")
        if _stream_dialogue_markdown(env, request, invocation.session_id, Path(invocation.out_path)):
            _warn_on_written_file_secret_candidates(env, invocation.out_path)
            env.ui.console.print(f"Wrote to {invocation.out_path}")
        else:
            env.ui.error(f"Session not found: {invocation.session_id}")
        return
    session = _read_dialogue_session(env, request, invocation.session_id, projection)
    if session is None:
        env.ui.error(f"Session not found: {invocation.session_id}")
        return
    content = _format_dialogue_session(session, fmt, projection=projection)
    # open_in_browser() re-derives HTML from `session` for non-html formats
    # rather than using `content` directly, so it must receive the same
    # windowed session `_format_dialogue_session` rendered from -- passing
    # the raw, un-windowed `session` here would silently show the full
    # dialogue in the browser even when a projection/window was requested
    # (polylogue-bvnz follow-up, CodeRabbit review on #3504).
    windowed_session = _window_dialogue_session(session, projection)
    deliver_content(
        env,
        content,
        destination=invocation.destination,
        out_path=invocation.out_path,
        output_format=fmt,
        session=windowed_session,
    )


def _stream_dialogue_markdown(env: AppEnv, request: RootModeRequest, session_id: str, out_path: Path) -> bool:
    """Export operation pages while retaining only one rendered page."""

    from polylogue.archive.message.messages import MessageCollection

    pages = _iter_dialogue_pages(env, request, session_id, None)
    first = next(pages, None)
    if first is None:
        return False
    empty = MessageCollection(messages=[])
    header_session = first.model_copy(update={"messages": empty, "attachments": []})
    header = format_session(header_session, "markdown", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_path.parent, delete=False) as output:
        temporary_path = Path(output.name)
        try:
            output.write(header)
            wrote_messages = False
            for page in chain((first,), pages):
                if not page.messages:
                    continue
                rendered = format_session(page.model_copy(update={"attachments": []}), "markdown", None)
                if not rendered.startswith(header):
                    raise ValueError("read.dialogue changed markdown header during paging")
                section = rendered[len(header) :]
                if not section:
                    continue
                if wrote_messages:
                    output.write("\n")
                output.write(section)
                wrote_messages = True
            if first.attachments:
                attachments = format_session(first.model_copy(update={"messages": empty}), "markdown", None)
                if not attachments.startswith(header):
                    raise ValueError("read.dialogue changed markdown attachment header")
                if wrote_messages:
                    output.write("\n")
                output.write(attachments[len(header) :])
            output.flush()
            temporary_path.replace(out_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return True


def _read_dialogue_session(
    env: AppEnv, request: RootModeRequest, session_id: str, projection: ProjectionSpec | None
) -> Session | None:
    """Compose bounded daemon pages into the dialogue renderer's session model."""

    from polylogue.archive.message.messages import MessageCollection

    pages = _iter_dialogue_pages(env, request, session_id, projection)
    first = next(pages, None)
    if first is None:
        return None
    messages = list(first.messages)
    for page in pages:
        messages.extend(page.messages)
    return first.model_copy(update={"messages": MessageCollection(messages=messages)})


def _iter_dialogue_pages(
    env: AppEnv, request: RootModeRequest, session_id: str, projection: ProjectionSpec | None
) -> Iterator[Session]:
    """Validate and yield each snapshot-bound operation page in order."""

    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.message.models import Message
    from polylogue.cli.lowering import _selection_params
    from polylogue.cli.operation_kernel import (
        OperationEnvelopeError,
        OperationKernelError,
        OperationRequest,
    )
    from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read

    params = _selection_params(request)
    params["query"] = list(request.query_terms)
    body_projection = projection.model_dump(mode="json") if projection is not None else {}
    first: Session | None = None
    offset = 0
    continuation: str | None = None
    total: int | None = None
    while True:
        try:
            result, served_by = dispatch_read(
                env.config,
                OperationRequest(
                    "read.dialogue",
                    {
                        "session_id": session_id,
                        "params": params,
                        "projection": body_projection,
                        "offset": offset,
                        "limit": 100,
                        "continuation": continuation,
                    },
                ),
                daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
            )
        except OperationKernelError as exc:
            from polylogue.cli.render.outcome import exit_for_read_failure

            exit_for_read_failure(exc)
        body = result.get("payload")
        if result.get("view") != "dialogue" or not isinstance(body, Mapping):
            raise OperationEnvelopeError("read.dialogue returned an invalid view envelope")
        raw_session = body.get("session")
        if raw_session is None and first is None:
            return
        if not isinstance(raw_session, Mapping):
            raise OperationEnvelopeError("read.dialogue returned no session page")
        page = Session.model_validate(raw_session)
        page = page.model_copy(
            update={
                "messages": MessageCollection(messages=[Message.model_validate(message) for message in page.messages])
            }
        )
        if first is None:
            first = page
            if request.verbose:
                click.echo(f"served-by: {served_by.line()}", err=True)
        elif page.id != first.id:
            raise OperationEnvelopeError("read.dialogue changed session during paging")
        raw_total = body.get("total_message_count")
        if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total < 0:
            raise OperationEnvelopeError("read.dialogue returned an invalid message count")
        if total is not None and total != raw_total:
            raise OperationEnvelopeError("read.dialogue changed total during paging")
        total = raw_total
        next_offset = body.get("next_offset")
        if next_offset is None:
            yield page
            break
        if isinstance(next_offset, bool) or not isinstance(next_offset, int) or next_offset <= offset:
            raise OperationEnvelopeError("read.dialogue returned an invalid continuation offset")
        if next_offset > raw_total:
            raise OperationEnvelopeError("read.dialogue continuation exceeded its message count")
        raw_continuation = body.get("continuation")
        if not isinstance(raw_continuation, str) or not raw_continuation:
            raise OperationEnvelopeError("read.dialogue omitted its snapshot-bound continuation")
        continuation = raw_continuation
        offset = next_offset
        yield page


def _format_dialogue_session(
    session: Session,
    output_format: str,
    *,
    projection: ProjectionSpec | None = None,
) -> str:
    if output_format == "json":
        return json.dumps(_dialogue_payload(session, projection=projection), indent=2)
    if output_format == "yaml":
        return str(
            yaml.dump(
                _dialogue_payload(session, projection=projection),
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        )
    session = _window_dialogue_session(session, projection)
    return format_session(session, output_format, None)


def _dialogue_messages(session: Session) -> list[Any]:
    return [message for message in session.messages if message.text]


def _projection_dialogue_window(messages: list[Any], projection: ProjectionSpec | None) -> list[Any]:
    if projection is None:
        return messages
    offset = projection.body_offset or 0
    if offset:
        messages = messages[offset:]
    if projection.body_limit is not None:
        messages = messages[: projection.body_limit]
    if projection.max_tokens is not None:
        remaining = projection.max_tokens
        bounded: list[Any] = []
        for message in messages:
            text = getattr(message, "text", "") or ""
            token_estimate = max(1, len(str(text).split()))
            if bounded and token_estimate > remaining:
                break
            bounded.append(message)
            remaining -= token_estimate
            if remaining <= 0:
                break
        return bounded
    return messages


def _window_dialogue_session(session: Session, projection: ProjectionSpec | None) -> Session:
    if projection is None or (
        projection.body_limit is None and projection.body_offset is None and projection.max_tokens is None
    ):
        return session
    return session.model_copy(
        update={"messages": tuple(_projection_dialogue_window(_dialogue_messages(session), projection))}
    )


def _dialogue_payload(session: Session, *, projection: ProjectionSpec | None = None) -> dict[str, object]:
    all_messages = _dialogue_messages(session)
    messages = _projection_dialogue_window(all_messages, projection)
    omitted_before = projection.body_offset or 0 if projection is not None else 0
    omitted_after = max(0, len(all_messages) - omitted_before - len(messages))
    return {
        "id": str(session.id),
        "origin": session.origin.value,
        "title": session.display_title,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "message_count": len(all_messages),
        "rendered_message_count": len(messages),
        "omitted_before": omitted_before,
        "omitted_after": omitted_after,
        "projection": {
            "body_limit": projection.body_limit if projection is not None else None,
            "body_offset": projection.body_offset if projection is not None else None,
            "max_tokens": projection.max_tokens if projection is not None else None,
        },
        "messages": [
            {
                "id": message.id,
                "role": message.role.value,
                "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                "material_origin": message.material_origin.value,
                "text": message.text,
            }
            for message in messages
        ],
    }


def exact_read_summaries(config: Config, request: RootModeRequest) -> list[SessionSummary] | None:
    """Resolve a single ``--id`` read without enumerating generic query rows."""

    if request.query_terms:
        return None
    spec = request.query_spec()
    if spec.session_id is None:
        return None
    session_id = spec.session_id

    # polylogue-yla8.1 split-root contract: config.db_path always names a
    # concrete index.db (explicit override or resolved active generation).
    archive_root = archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)
    try:
        return run_archive_read_sync(
            archive_root,
            operation="cli.read.exact_summary",
            arguments={"session_id": session_id},
            work=lambda archive: [
                archive_summary_to_domain(archive.read_summary(archive.resolve_session_id(session_id)))
            ],
            projection="session-summary",
        )
    except KeyError:
        return []


def _message_temporal_events_for_summaries(
    config: Config,
    summaries: list[SessionSummary],
    *,
    per_session_limit: int = 8,
) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    if not summaries:
        return [], ()
    # polylogue-yla8.1 split-root contract: config.db_path always names a
    # concrete index.db (explicit override or resolved active generation).
    archive_root = archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)
    events: list[TemporalEvidenceEvent] = []
    caveats: list[str] = []
    total_limit = max(per_session_limit * len(summaries), 0)
    rows = run_archive_read_sync(
        archive_root,
        operation="cli.read.temporal_messages",
        arguments={"session_ids": [str(summary.id) for summary in summaries], "limit": total_limit},
        work=lambda archive: archive.query_session_messages(
            [str(summary.id) for summary in summaries],
            limit=total_limit,
            sort_direction="asc",
        ),
        page_size=total_limit,
        projection="temporal-messages",
        stable_order="time,message_id",
    )
    if len(rows) >= total_limit and sum(summary.message_count or 0 for summary in summaries) > total_limit:
        caveats.append("message_events_capped")
    events.extend(event for row in rows if (event := message_row_to_temporal_event(row)) is not None)
    return events, tuple(caveats)


def _action_temporal_events_for_summaries(
    config: Config,
    summaries: list[SessionSummary],
    *,
    per_session_limit: int = 4,
) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    if not summaries:
        return [], ()
    # polylogue-yla8.1 split-root contract: config.db_path always names a
    # concrete index.db (explicit override or resolved active generation).
    archive_root = archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)
    events: list[TemporalEvidenceEvent] = []
    caveats: list[str] = []
    total_limit = max(per_session_limit * len(summaries), 0)
    session_ids = [str(summary.id) for summary in summaries]
    rows = run_archive_read_sync(
        archive_root,
        operation="cli.read.temporal_actions",
        arguments={"session_ids": session_ids, "limit": total_limit},
        work=lambda archive: archive.query_session_action_occurrences(
            session_ids, limit=total_limit, sort_direction="asc"
        ),
        page_size=total_limit,
        projection="temporal-actions",
        stable_order="time,action_id",
    )
    if len(rows) >= total_limit:
        caveats.append("action_events_capped")
    events.extend(event for row in rows if (event := action_row_to_temporal_event(row)) is not None)
    return events, tuple(caveats)


def _render_temporal_window_markdown(window: TemporalEvidenceWindow) -> str:
    lines = [
        "# Temporal Evidence Window",
        "",
        f"- Events: {window.event_count}",
        f"- Bucket: {window.bucket.value}",
        f"- Families: {', '.join(f'{key}={value}' for key, value in window.family_counts.items()) or 'none'}",
        f"- Kinds: {', '.join(f'{key}={value}' for key, value in window.kind_counts.items()) or 'none'}",
    ]
    if window.caveats:
        lines.append(f"- Caveats: {', '.join(window.caveats)}")
    lines.extend(["", "## Buckets"])
    if window.buckets:
        for bucket in window.buckets[:25]:
            lines.append(f"- {bucket.bucket_start.isoformat()} {bucket.family}/{bucket.kind}: {bucket.count}")
        if len(window.buckets) > 25:
            lines.append(f"- ... {len(window.buckets) - 25} more buckets")
    else:
        lines.append("- none")
    lines.extend(["", "## Events"])
    if window.events:
        for event in window.events[:50]:
            label = event.label.replace("\n", " ").strip()
            lines.append(
                f"- {event.occurred_at.isoformat()} [{event.family}/{event.kind}] {label} ({event.source_ref})"
            )
        if len(window.events) > 50:
            lines.append(f"- ... {len(window.events) - 50} more events")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_read_temporal_window(
    config: Config,
    request: RootModeRequest,
    *,
    phase_recorder: TemporalPhaseRecorder | None = None,
) -> TemporalEvidenceWindow:
    """Project selected session summaries into a temporal evidence window."""

    from polylogue.cli.query import _create_query_vector_provider

    started = time.perf_counter()
    spec = request.query_spec()
    if spec.limit is None:
        spec = replace(spec, limit=50)
    # polylogue-yla8.1 split-root contract: config.db_path always names a
    # concrete index.db (explicit override or resolved active generation).
    archive_root = archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)
    vector_provider = _create_query_vector_provider(config, db_path=archive_root / "embeddings.db")
    _record_temporal_phase(
        phase_recorder,
        "prepare",
        started,
        {"archive_root": str(archive_root), "limit": spec.limit},
    )

    started = time.perf_counter()
    summaries = exact_read_summaries(config, request)
    if summaries is None:
        summaries = run_coroutine_sync(spec.list_summaries(config, vector_provider=vector_provider))
    _record_temporal_phase(phase_recorder, "select_sessions", started, {"session_count": len(summaries)})

    started = time.perf_counter()
    events = [event for summary in summaries if (event := summary_to_temporal_event(summary)) is not None]
    _record_temporal_phase(phase_recorder, "project_sessions", started, {"event_count": len(events)})

    started = time.perf_counter()
    message_events, caveats = _message_temporal_events_for_summaries(config, summaries)
    _record_temporal_phase(
        phase_recorder,
        "project_messages",
        started,
        {"event_count": len(message_events), "caveats": list(caveats)},
    )

    started = time.perf_counter()
    action_events, action_caveats = _action_temporal_events_for_summaries(config, summaries)
    _record_temporal_phase(
        phase_recorder,
        "project_actions",
        started,
        {"event_count": len(action_events), "caveats": list(action_caveats)},
    )

    started = time.perf_counter()
    window = build_temporal_evidence_window(
        [*events, *message_events, *action_events], caveats=(*caveats, *action_caveats)
    )
    _record_temporal_phase(
        phase_recorder,
        "build_window",
        started,
        {"event_count": window.event_count, "family_counts": dict(window.family_counts)},
    )
    return window


def run_read_temporal(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Project selected session summaries into a temporal evidence window."""

    from polylogue.cli.lowering import _selection_params
    from polylogue.cli.operation_kernel import (
        OperationEnvelopeError,
        OperationKernelError,
        OperationRequest,
    )
    from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read

    params = _selection_params(request)
    params["query"] = list(request.query_terms)
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    try:
        result, served_by = dispatch_read(
            env.config,
            OperationRequest(
                "read.temporal",
                {
                    "session_id": invocation.session_id,
                    "params": params,
                    "projection": projection.model_dump(mode="json") if projection is not None else {},
                },
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    body = result.get("payload")
    if result.get("view") != "temporal" or not isinstance(body, Mapping):
        raise OperationEnvelopeError("read.temporal returned an invalid view envelope")
    if not isinstance(body.get("temporal_window"), Mapping):
        raise OperationEnvelopeError("read.temporal returned no temporal window")
    window = TemporalEvidenceWindow.model_validate(body["temporal_window"])
    if request.verbose:
        click.echo(f"served-by: {served_by.line()}", err=True)
    fmt = invocation.output_format or "markdown"
    if fmt == "json":
        content = json.dumps({"temporal_window": window.model_dump(mode="json")}, indent=2) + "\n"
    else:
        content = _render_temporal_window_markdown(window)
    deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path, output_format=fmt)


def run_read_browser(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Open the first matched session in the daemon web reader."""

    from polylogue.cli.query import _create_query_vector_provider

    config = env.config

    async def _find_first() -> str | None:
        spec = replace(request.query_spec(), limit=1)
        # polylogue-yla8.1 split-root contract: config.db_path always names a
        # concrete index.db (explicit override or resolved active generation).
        archive_root = archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)
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


__all__ = [
    "TemporalPhaseRecorder",
    "build_read_temporal_window",
    "exact_read_summaries",
    "run_read_dialogue",
    "run_read_browser",
    "run_read_summary_or_transcript",
    "run_read_temporal",
]
