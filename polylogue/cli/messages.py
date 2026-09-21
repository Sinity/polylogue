"""CLI execution for messages and raw verbs."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from time import monotonic
from types import SimpleNamespace
from typing import Any, Literal, cast

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import DEFAULT_MESSAGE_PAGE_LIMIT
from polylogue.cli.operation_kernel import OperationKernelError
from polylogue.cli.read_dispatch import ServedBy, daemon_route_disabled, dispatch_read
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.operations.authority import authority_for_config
from polylogue.rendering.semantic_card_models import LineageDescriptor
from polylogue.rendering.semantic_cards import (
    build_semantic_transcript,
    lineage_descriptor_from_session,
)
from polylogue.rendering.semantic_markdown import render_semantic_transcript_markdown
from polylogue.surfaces.outcome import lineage_page_outcome
from polylogue.surfaces.payloads import SessionMessagesResponsePayload, model_json_document

#: One ``session.read`` messages window.  A whole transcript can exceed the
#: bounded operation result, so a wider request is composed from a sequence of
#: these rather than asked for in one window the transport cannot carry.
_MESSAGE_READ_WINDOW = 200

#: The ``session.read`` refusal that means "this reference names no session".
#: It is the operation's own wording; the CLI renders it in its established
#: terms rather than as a read failure with a traceback.
_SESSION_NOT_FOUND = "session not found"


@dataclasses.dataclass(frozen=True, slots=True)
class _MessageWindow:
    """One composed message page, as the declared read answered it."""

    session: Mapping[str, object]
    rows: list[Mapping[str, object]]
    total: int
    offset: int
    next_offset: int | None
    continuation: str | None
    lineage_complete: bool
    lineage_truncation_reason: str | None
    served_by: ServedBy


def read_message_windows(
    config: Config,
    session_id: str,
    *,
    limit: int,
    offset: int,
    full: bool,
    continuation: str | None,
    daemon_disabled: bool,
    around: str | None = None,
) -> Iterator[_MessageWindow]:
    """Yield the declared ``session.read`` message windows one request needs.

    The operation is windowed by construction, so composing a wider page is
    the adapter's job: ``full`` asks for every remaining message and a bounded
    page wider than one window is read as a bounded sequence of windows.  A
    window that adds no rows ends the loop -- it cannot advance the
    composition, so continuing on one would hang rather than wait.

    ``around`` decides the *first* window's offset and is carried only on that
    request: the operation reports the coordinate it resolved to, and every
    further window advances from there by coordinate, so a composed read walks
    forward from the anchor rather than re-resolving it on each page.
    """

    from polylogue.cli.lowering import lower_session_read

    token = continuation
    remaining: int | None = None if full else max(limit, 0)
    delivered = 0
    anchor = around
    while True:
        if remaining is not None and remaining <= 0:
            return
        window_limit = _MESSAGE_READ_WINDOW if remaining is None else min(remaining, _MESSAGE_READ_WINDOW)
        if token is not None:
            request = lower_session_read(session_id, kind="messages", continuation=token)
        elif anchor is not None:
            request = lower_session_read(session_id, kind="messages", limit=window_limit, around=anchor)
        else:
            request = lower_session_read(session_id, kind="messages", limit=window_limit, offset=offset + delivered)
        payload, served_by = dispatch_read(config, request, daemon_disabled=daemon_disabled)
        raw_rows = payload.get("messages")
        rows = [row for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
        session = payload.get("session")
        next_offset = payload.get("next_offset")
        window = _MessageWindow(
            session=session if isinstance(session, Mapping) else {},
            rows=rows,
            total=int(cast(int, payload.get("total") or 0)),
            offset=int(cast(int, payload.get("offset") or 0)),
            next_offset=int(next_offset) if isinstance(next_offset, int) else None,
            continuation=cast("str | None", payload.get("continuation")),
            lineage_complete=bool(payload.get("lineage_complete", True)),
            lineage_truncation_reason=cast("str | None", payload.get("lineage_truncation_reason")),
            served_by=served_by,
        )
        yield window
        if anchor is not None:
            # The anchor resolved to a coordinate; the composition continues
            # from that coordinate rather than re-resolving the same message.
            offset = window.offset
            anchor = None
        delivered += len(rows)
        if remaining is not None:
            remaining -= len(rows)
        if not rows or window.continuation is None:
            return
        token = window.continuation


#: The declared refusals that mean "this continuation does not name this
#: window".  Named from the exception classes that own them so the CLI cannot
#: drift from the token every other surface reports (polylogue-ijbwq).
def _continuation_refusal_codes() -> frozenset[str]:
    from polylogue.archive.query.transaction import (
        QueryContinuationExpiredError,
        QueryContinuationInvalidError,
        QueryContinuationStaleError,
    )

    return frozenset(
        {
            QueryContinuationStaleError.code,
            QueryContinuationInvalidError.code,
            QueryContinuationExpiredError.code,
        }
    )


def message_read_failure(env: AppEnv, exc: OperationKernelError, *, session_id: str) -> None:
    """Render one declared read's refusal in the messages verb's own terms.

    A reference that names no session keeps the wording and the exit status
    this verb has always used, and a refused continuation keeps the typed code
    the MCP and HTTP surfaces report for the same token rather than silently
    re-reading a shifted window.  Every other refusal exits through the one
    CLI read-failure terminal rather than reaching the operator as a
    traceback.
    """

    from polylogue.cli.render.outcome import exit_for_read_failure

    detail = str(getattr(exc, "detail", None) or exc)
    code = str(getattr(exc, "code", "") or "")
    if code in _continuation_refusal_codes():
        raise click.ClickException(f"{code}: {detail}") from exc
    if _SESSION_NOT_FOUND in detail.lower():
        env.ui.error(f"Session not found: {session_id}")
        return
    exit_for_read_failure(exc)


def run_messages(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
    limit: int = DEFAULT_MESSAGE_PAGE_LIMIT,
    offset: int = 0,
    full: bool = False,
    output_format: str | None = None,
    continuation: str | None = None,
    around: str | None = None,
) -> None:
    """Execute the messages verb over the declared ``session.read`` window.

    The verb lowers and renders; which executor answers is the operation
    kernel's decision, so a reachable daemon serves this page exactly as it
    serves every sibling read and the route never branches on whether one is
    running.
    """

    started_at = monotonic()
    config = cast(Config, request.config())
    daemon_disabled = daemon_route_disabled(flag=bool(request.params.get("no_daemon")))

    windows: list[_MessageWindow] = []
    try:
        for window in read_message_windows(
            config,
            session_id,
            limit=limit,
            offset=offset,
            full=full,
            continuation=continuation,
            daemon_disabled=daemon_disabled,
            around=around,
        ):
            windows.append(window)
    except OperationKernelError as exc:
        message_read_failure(env, exc, session_id=session_id)
        return
    if not windows:
        return

    last = windows[-1]
    messages: list[Mapping[str, object]] = [row for window in windows for row in window.rows]
    session = windows[0].session
    # A composed read delivers exactly the rows it gathered; a bounded page
    # delivers the window it asked for.  Reporting the request's own bound for
    # a composed read would name a window nobody asked for.
    effective_limit = len(messages) if full else limit
    if bool(request.params.get("verbose")):
        click.echo(f"served-by: {last.served_by.line()}", err=True)

    fmt = output_format or "markdown"
    if fmt == "json":
        import json as _json

        # Finite machine-output contract (#1818): one JSON value.
        payload = model_json_document(
            SessionMessagesResponsePayload(
                session_id=session_id,
                messages=tuple(cast("Any", row) for row in messages),
                total=last.total,
                limit=effective_limit,
                offset=windows[0].offset,
                next_offset=last.next_offset,
                continuation=last.continuation,
                lineage_complete=last.lineage_complete,
                lineage_truncation_reason=last.lineage_truncation_reason,
                authority=authority_for_config(
                    config, server_identity=_authority_identity(last.served_by), started_at=started_at
                ),
                outcome=lineage_page_outcome(
                    matched=last.total,
                    complete=last.lineage_complete,
                    truncation_reason=last.lineage_truncation_reason,
                ),
            ),
            exclude_none=True,
        )
        # Machine output goes through click.echo (raw stdout), NOT
        # env.ui.print: the Rich console defaults markup=True and would
        # interpret/strip bracket sequences like "[bold]" inside message
        # text, corrupting the exact bytes json.dumps produced (#1818).
        click.echo(_json.dumps(payload, indent=2))
        return
    if fmt == "ndjson":
        import json as _json

        # Streaming machine-output contract (#1818): one JSON document
        # per line. Each line is self-contained, carrying session_id so
        # downstream consumers do not need an out-of-band envelope.
        # Raw click.echo (not env.ui.print) so Rich markup never mangles
        # message text inside the JSON document.
        for row in messages:
            click.echo(_json.dumps({"session_id": session_id, **dict(row)}))
        return

    rendered = render_semantic_transcript_markdown(
        build_semantic_transcript(
            messages,
            session_id=session_id,
            lineage=_message_lineage(session, complete=last.lineage_complete, reason=last.lineage_truncation_reason),
            provider_family=cast("str | None", session.get("origin")) or None,
        )
    )
    if rendered:
        # ``read --view messages --to file|clipboard`` captures click.echo at
        # the existing destination adapter. Rich output would bypass that
        # contract and reinterpret markup.
        click.echo(rendered, nl=False)


def _authority_identity(served_by: ServedBy) -> Literal["daemon", "direct"]:
    """Name the executor the authority envelope admits, defaulting to direct."""

    return "daemon" if served_by.identity == "daemon" else "direct"


def _message_lineage(
    session: Mapping[str, object],
    *,
    complete: bool,
    reason: str | None,
) -> LineageDescriptor | None:
    """Project the read's own session header onto the renderer's lineage card.

    polylogue-ppkj: the session row carries no read-time completeness, so the
    signal the window reported is overlaid rather than re-derived, which is
    what lets a truncated composed transcript render as truncated instead of
    as a short conversation.
    """

    if not session:
        return None
    return dataclasses.replace(
        lineage_descriptor_from_session(SimpleNamespace(**dict(session))),
        lineage_complete=complete,
        lineage_truncation_reason=reason,
    )


def run_raw(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
    limit: int = DEFAULT_MESSAGE_PAGE_LIMIT,
    offset: int = 0,
    output_format: str = "json",
) -> None:
    """Execute the raw verb."""
    from polylogue.api import Polylogue

    async def _run() -> None:
        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            artifacts, total = await api.get_raw_artifacts_for_session(
                session_id,
                limit=limit,
                offset=offset,
            )

            if not artifacts:
                env.ui.error(f"No raw artifacts found for session: {session_id}")
                return

            if output_format == "json":
                import json as _json

                payload = {
                    "session_id": session_id,
                    "artifacts": [
                        {
                            "raw_id": r.get("raw_id", ""),
                            "source_name": r.get("source_name", ""),
                            "source_path": r.get("source_path", ""),
                            "blob_size": r.get("blob_size", 0),
                        }
                        for r in artifacts
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
                # Machine output uses raw stdout so Rich markup never rewrites
                # JSON bytes and read-view delivery can capture file/clipboard
                # targets consistently.
                click.echo(_json.dumps(payload, indent=2))
            else:
                import yaml

                click.echo(yaml.dump(artifacts))

    run_coroutine_sync(_run())


def run_session_events(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
    event_type: str | None = None,
    limit: int | None = None,
    output_format: str = "json",
) -> None:
    """Execute the events verb.

    Renders the raw session-timeline evidence (``Session.session_events``):
    provider evidence that rides the session timeline instead of a dialogue
    message -- Codex ``world_state``/``agent_policy``/``turn_context`` policy
    facts, Claude Code sidecar events, Hermes tool-availability/step spans,
    and similar. Previously populated on every full session read but never
    rendered on any surface (this read view is the fix).
    """
    from polylogue.api import Polylogue

    async def _run() -> None:
        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            events = await api.get_session_events(session_id, event_type=event_type, limit=limit)

            if events is None:
                env.ui.error(f"Session not found: {session_id}")
                return

            payload = {
                "session_id": session_id,
                "event_type": event_type,
                "total": len(events),
                "events": events,
            }

            if output_format == "json":
                import json as _json

                # Machine output uses raw stdout so Rich markup never rewrites
                # JSON bytes and read-view delivery can capture file/clipboard
                # targets consistently.
                click.echo(_json.dumps(payload, indent=2))
            else:
                import yaml

                click.echo(yaml.dump(payload))

    run_coroutine_sync(_run())


__all__ = [
    "message_read_failure",
    "read_message_windows",
    "run_messages",
    "run_raw",
    "run_session_events",
]
