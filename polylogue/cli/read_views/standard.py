"""Standard summary/transcript/browser read-view handlers."""

from __future__ import annotations

import json
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import replace
from urllib.parse import quote

from polylogue.archive.session.domain_models import SessionSummary
from polylogue.cli.read_views.base import ReadViewInvocation, deliver_content, execute_query_request
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
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


def _session_scope_for_summaries(summaries: list[SessionSummary]) -> str:
    return " OR ".join(f"session:{summary.id}" for summary in summaries)


def _message_temporal_events_for_summaries(
    config: Config,
    summaries: list[SessionSummary],
    *,
    per_session_limit: int = 8,
) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    from polylogue.archive.query.expression import parse_unit_source_expression
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
    expression = (
        f"sessions where {_session_scope_for_summaries(summaries)} | "
        "messages where time >= 1970-01-01T00:00:00+00:00 | sort by time asc"
    )
    source = parse_unit_source_expression(expression)
    if source is None:
        return [], ("message_temporal_parse_failed",)
    with ArchiveStore.open_existing(archive_root) as archive:
        rows = archive.query_messages(source.predicate, limit=total_limit, sort="time", sort_direction="asc")
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
        rows = archive.query_session_actions(session_ids, limit=total_limit, sort_direction="asc")
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
    from polylogue.archive.session.domain_models import Session
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
    "run_read_browser",
    "run_read_summary_or_transcript",
    "run_read_temporal",
]
