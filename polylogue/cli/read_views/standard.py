"""Standard summary/transcript/browser read-view handlers."""

from __future__ import annotations

import json
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content, execute_query_request
from polylogue.cli.read_views.streaming_markdown import stream_exact_session_markdown
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamps import parse_archive_datetime
from polylogue.core.types import SessionId
from polylogue.rendering.formatting import format_session
from polylogue.surfaces.projection_spec import ProjectionSpec
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceEvent,
    TemporalEvidenceWindow,
    action_row_to_temporal_event,
    build_temporal_evidence_window,
    message_row_to_temporal_event,
    summary_to_temporal_event,
)

TemporalPhaseRecorder = Callable[[str, float, Mapping[str, object]], None]


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
    if (
        invocation.view == "transcript"
        and invocation.destination == "file"
        and invocation.session_id is not None
        and fmt == "markdown"
        and invocation.out_path
        and stream_exact_session_markdown(
            env.config.archive_root,
            invocation.session_id,
            Path(invocation.out_path),
            prose_only=False,
        )
    ):
        env.ui.console.print(f"Wrote to {invocation.out_path}")
        return
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


def run_read_dialogue(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render one session as authored dialogue using the shared content projection."""

    del request
    assert invocation.session_id is not None
    if (
        invocation.destination == "file"
        and (invocation.output_format or "markdown") == "markdown"
        and invocation.out_path
        and invocation.projection_spec is None
        and stream_exact_session_markdown(
            env.config.archive_root,
            invocation.session_id,
            Path(invocation.out_path),
            prose_only=True,
        )
    ):
        env.ui.console.print(f"Wrote to {invocation.out_path}")
        return
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    session = run_coroutine_sync(
        env.polylogue.get_session(invocation.session_id, content_projection=ContentProjectionSpec.prose_only())
    )
    if session is None:
        env.ui.error(f"Session not found: {invocation.session_id}")
        return
    fmt = invocation.output_format or "markdown"
    content = _format_dialogue_session(session, fmt, projection=projection)
    deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)


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
        "title": session.title,
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


def _archive_summary_to_domain(summary: Any) -> SessionSummary:
    return SessionSummary(
        id=SessionId(str(summary.session_id)),
        origin=origin_from_provider(summary.provider),
        title=summary.title,
        created_at=parse_archive_datetime(summary.created_at),
        updated_at=parse_archive_datetime(summary.updated_at),
        working_directories=tuple(summary.working_directories),
        git_branch=summary.git_branch,
        git_repository_url=summary.git_repository_url,
        provider_project_ref=summary.provider_project_ref,
        message_count=summary.message_count,
        tags_m2m=summary.tags,
    )


def exact_read_summaries(config: Config, request: RootModeRequest) -> list[SessionSummary] | None:
    """Resolve a single ``--id`` read without enumerating generic query rows."""

    if request.query_terms:
        return None
    spec = request.query_spec()
    if spec.session_id is None:
        return None

    from polylogue.paths import archive_file_set_root_for_paths
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )
    try:
        with ArchiveStore.open_existing(archive_root) as archive:
            session_id = archive.resolve_session_id(spec.session_id)
            return [_archive_summary_to_domain(archive.read_summary(session_id))]
    except KeyError:
        return []


def _message_temporal_events_for_summaries(
    config: Config,
    summaries: list[SessionSummary],
    *,
    per_session_limit: int = 8,
) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    from polylogue.paths import archive_file_set_root_for_paths
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    if not summaries:
        return [], ()
    archive_root = archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )
    events: list[TemporalEvidenceEvent] = []
    caveats: list[str] = []
    total_limit = max(per_session_limit * len(summaries), 0)
    with ArchiveStore.open_existing(archive_root) as archive:
        rows = archive.query_session_messages(
            [str(summary.id) for summary in summaries],
            limit=total_limit,
            sort_direction="asc",
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
    from polylogue.paths import archive_file_set_root_for_paths
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    if not summaries:
        return [], ()
    archive_root = archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )
    events: list[TemporalEvidenceEvent] = []
    caveats: list[str] = []
    total_limit = max(per_session_limit * len(summaries), 0)
    session_ids = [str(summary.id) for summary in summaries]
    with ArchiveStore.open_existing(archive_root) as archive:
        rows = archive.query_session_action_occurrences(session_ids, limit=total_limit, sort_direction="asc")
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

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.query import _create_query_vector_provider
    from polylogue.paths import archive_file_set_root_for_paths

    started = time.perf_counter()
    spec = request.query_spec()
    if spec.limit is None:
        spec = replace(spec, limit=50)
    archive_root = archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )
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

    window = build_read_temporal_window(env.config, request)
    fmt = invocation.output_format or "markdown"
    if fmt == "json":
        content = json.dumps({"temporal_window": window.model_dump(mode="json")}, indent=2) + "\n"
    else:
        content = _render_temporal_window_markdown(window)
    deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)


def run_read_browser(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Open the first matched session in the daemon web reader."""

    from polylogue.api.sync.bridge import run_coroutine_sync
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


__all__ = [
    "TemporalPhaseRecorder",
    "build_read_temporal_window",
    "exact_read_summaries",
    "run_read_dialogue",
    "run_read_browser",
    "run_read_summary_or_transcript",
    "run_read_temporal",
]
