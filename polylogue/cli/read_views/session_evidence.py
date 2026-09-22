"""Per-session evidence read views, served by one declared ``session.read``.

``hooks``, ``file-edits``, ``agent-policies`` and ``web-content`` render four
index-tier relations that ride one exact session reference and have no
query-grammar unit of their own (design D3).  They are the same request with a
different relation named, so they are one adapter rather than four: each lowers
``session.read`` with its own kind, dispatches through the operation kernel and
renders the evidence body the operation answered with.

``events`` and ``raw`` ride the same route but a *paged* contract: both
accepted a row bound before this, and ``events`` reported the clipped count as
its ``total``, so neither could be lowered onto a kind that is answered whole.
They lower onto the windowed-evidence kinds instead, and each composes its own
document from the reported page -- the row keys the view has always rendered,
plus the bound (``returned``/``next_offset``/``complete``/``continuation``)
that makes a truncated body distinguishable from a finished one.

None of them opens an archive, and none branches on whether a daemon is
running: which executor answered is the kernel's decision, reported by the
result's own authority and named on stderr under ``--verbose`` exactly as
``read --view messages`` names it (polylogue-r3cuz).
"""

from __future__ import annotations

import json
from typing import cast

import click

from polylogue.cli.read_views.base import (
    ReadViewEventsOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.surfaces.projection_spec import RenderDestination

__all__ = [
    "build_events_options",
    "run_read_agent_policies",
    "run_read_events",
    "run_read_file_edits",
    "run_read_hooks",
    "run_read_raw",
    "run_read_web_content",
    "run_session_evidence_view",
]


def run_session_evidence_view(
    env: AppEnv,
    request: RootModeRequest,
    invocation: ReadViewInvocation,
    *,
    kind: str,
) -> None:
    """Render one declared per-session evidence relation.

    A refusal names itself and exits non-zero; it must never render as an
    empty body that reads "this session recorded nothing".  Refusals are
    classed exactly as the transcript path classes them, through the one CLI
    read-failure terminal (polylogue-jtrtj).
    """

    from polylogue.cli.lowering import lower_session_read
    from polylogue.cli.operation_kernel import (
        OperationEnvelopeError,
        OperationFailedError,
        OperationUnavailableError,
    )
    from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read

    assert invocation.session_id is not None
    output_format = invocation.output_format or "json"
    config = cast(Config, request.config())

    try:
        payload, served_by = dispatch_read(
            config,
            lower_session_read(invocation.session_id, kind=kind),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except (OperationFailedError, OperationUnavailableError) as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise OperationEnvelopeError(f"session.read {kind} result carries no evidence body")
    if bool(request.params.get("verbose")):
        click.echo(f"served-by: {served_by.line()}", err=True)

    if output_format == "json":
        # Machine output is rendered as raw bytes so Rich markup never rewrites
        # JSON and read-view delivery can capture file/clipboard targets.
        content = json.dumps(evidence, indent=2) + "\n"
    else:
        import yaml

        content = yaml.dump(evidence) + "\n"

    if invocation.destination in (RenderDestination.FILE, RenderDestination.CLIPBOARD, RenderDestination.STDOUT):
        deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)
        return
    click.echo(content, nl=False)


def run_read_hooks(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the per-session hook-event summary from ``session.read``."""

    run_session_evidence_view(env, request, invocation, kind="hooks")


def run_read_agent_policies(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render sandbox/approval/network policy facts from ``session.read``."""

    run_session_evidence_view(env, request, invocation, kind="agent-policies")


def _deliver_evidence_document(
    env: AppEnv,
    invocation: ReadViewInvocation,
    document: dict[str, object],
) -> None:
    """Render one evidence document as the read verb's destination contract asks."""

    # Machine output is rendered as raw bytes so Rich markup never rewrites
    # JSON and read-view delivery can capture file/clipboard targets.
    content = json.dumps(document, indent=2) + "\n"
    if invocation.destination in (RenderDestination.FILE, RenderDestination.CLIPBOARD, RenderDestination.STDOUT):
        deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)
        return
    click.echo(content, nl=False)


def _read_evidence_window(
    request: RootModeRequest,
    invocation: ReadViewInvocation,
    *,
    kind: str,
    default_limit: int | None = None,
) -> tuple[dict[str, object], str, object]:
    """Run one windowed-evidence read and hand back its reported page.

    Returns ``(window_body, session_id, served_by)``.  The page is the
    operation's ``evidence_window`` body verbatim: the bound it reports is the
    bound the caller is shown, so a view cannot re-derive a friendlier one.
    """

    from polylogue.cli.lowering import lower_session_read
    from polylogue.cli.operation_kernel import (
        OperationEnvelopeError,
        OperationFailedError,
        OperationUnavailableError,
    )
    from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read

    assert invocation.session_id is not None
    config = cast(Config, request.config())
    options = invocation.options
    continuation = cast("str | None", getattr(options, "continuation", None))
    limit = cast("int | None", getattr(options, "limit", None))
    if limit is None:
        limit = default_limit
    offset = cast(int, getattr(options, "offset", 0) or 0)

    try:
        payload, served_by = dispatch_read(
            config,
            lower_session_read(
                invocation.session_id,
                kind=kind,
                limit=None if continuation is not None else limit,
                offset=0 if continuation is not None else offset,
                continuation=continuation,
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except (OperationFailedError, OperationUnavailableError) as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    window = payload.get("evidence_window")
    if not isinstance(window, dict):
        raise OperationEnvelopeError(f"session.read {kind} result carries no evidence-window body")
    return dict(window), str(payload.get("session_id") or invocation.session_id), served_by


def _echo_served_by(request: RootModeRequest, served_by: object) -> None:
    if bool(request.params.get("verbose")):
        click.echo(f"served-by: {served_by.line()}", err=True)  # type: ignore[attr-defined]


def run_read_events(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render one page of ``Session.session_events`` from ``session.read``.

    ``total`` now names the relation's own row count rather than the clipped
    one this view used to report, and ``returned``/``next_offset``/
    ``complete``/``continuation`` state the bound alongside it.  That change is
    the point: under the old payload a ``--limit``-ed read and a whole one were
    the same document, so a caller could not tell that rows had been withheld.
    """

    window, session_id, served_by = _read_evidence_window(request, invocation, kind="events")
    _echo_served_by(request, served_by)
    _deliver_evidence_document(
        env,
        invocation,
        {
            "session_id": session_id,
            # This view has never narrowed by event type; the key is part of
            # its document and stays, naming the absence of a filter.
            "event_type": None,
            "total": window["total"],
            "returned": window["returned"],
            "limit": window["limit"],
            "offset": window["offset"],
            "next_offset": window["next_offset"],
            "continuation": window["continuation"],
            "complete": window["complete"],
            "events": window["rows"],
        },
    )


def run_read_raw(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render one page of source-tier acquisition rows from ``session.read``.

    The four artifact keys and the empty-relation message are the document
    this view has always answered with; what is new is that the page it
    reports is the page the declared read decided, with a continuation minted
    in the artifact family's own projection rather than the message window's.
    """

    from polylogue.archive.query.spec import DEFAULT_MESSAGE_PAGE_LIMIT

    # The raw view's own default page, preserved: the move changes which
    # executor answers and what the answer reports, not how much it returns
    # when the caller names no bound.
    window, session_id, served_by = _read_evidence_window(
        request, invocation, kind="raw", default_limit=DEFAULT_MESSAGE_PAGE_LIMIT
    )
    rows = cast("list[object]", window["rows"])
    if not rows:
        # Preserved verbatim from the in-process route: an artifact-less
        # session answers with this line, not with an empty body.
        env.ui.error(f"No raw artifacts found for session: {invocation.session_id}")
        return
    _echo_served_by(request, served_by)
    _deliver_evidence_document(
        env,
        invocation,
        {
            "session_id": session_id,
            "artifacts": rows,
            "total": window["total"],
            "returned": window["returned"],
            "limit": window["limit"],
            "offset": window["offset"],
            "next_offset": window["next_offset"],
            "continuation": window["continuation"],
            "complete": window["complete"],
        },
    )


def build_events_options(values: ReadViewOptionValues) -> ReadViewEventsOptions:
    """Build options owned by the events read view."""

    return ReadViewEventsOptions(
        limit=cast("int | None", values.get("limit")),
        offset=cast(int, values.get("offset") or 0),
        continuation=cast("str | None", values.get("continuation")),
    )


def run_read_file_edits(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render one page of captured Edit/Write/MultiEdit evidence from ``session.read``.

    Paged rather than answered whole: a file edit carries ``original_file``,
    the pre-edit contents of whatever the tool call touched, so one row can
    exceed the operation-result bound. Answered whole, such a session was
    materialized and then refused with "retry with a smaller limit" -- which a
    whole kind cannot accept, so the view had no reachable answer at all. The
    bound is reported alongside the rows, exactly as ``events`` reports it.
    """

    window, session_id, served_by = _read_evidence_window(request, invocation, kind="file-edits")
    _echo_served_by(request, served_by)
    _deliver_evidence_document(
        env,
        invocation,
        {
            "session_id": session_id,
            "total": window["total"],
            "returned": window["returned"],
            "limit": window["limit"],
            "offset": window["offset"],
            "next_offset": window["next_offset"],
            "continuation": window["continuation"],
            "complete": window["complete"],
            "file_edits": window["rows"],
        },
    )


def run_read_web_content(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render one page of typed web-export constructs from ``session.read``.

    Paged for the same reason as ``file-edits``: a construct carries ``text``,
    a fetched page or search-result body, so enough web evidence made the whole
    relation undeliverable with no retry that could succeed.
    """

    window, session_id, served_by = _read_evidence_window(request, invocation, kind="web-content")
    _echo_served_by(request, served_by)
    _deliver_evidence_document(
        env,
        invocation,
        {
            "session_id": session_id,
            "total": window["total"],
            "returned": window["returned"],
            "limit": window["limit"],
            "offset": window["offset"],
            "next_offset": window["next_offset"],
            "continuation": window["continuation"],
            "complete": window["complete"],
            "web_content_constructs": window["rows"],
        },
    )
