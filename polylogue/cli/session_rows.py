"""Selector rows from the declared ``cli.query`` operation.

Selection used to be the CLI's last private query executor: ``select``, the
cardinality probes and the bare-invocation landing screen each built a
``SessionQuerySpec``, opened a filter chain against the local archive (and, for
a similarity selection, a vector provider), and listed summaries in-process.
That is the same "what does this selection match" question ``find`` asks, and
answering it twice is how the two could disagree.

Every row below comes from one ``cli.query`` result through the kernel, so a
resident daemon answers a selection exactly as it answers a query, and the
daemon-absent case is the operation's own direct execution rather than a
separate code path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from polylogue.cli.select import SelectSessionRow

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.config import Config


def select_row_from_operation_row(item: Mapping[str, object]) -> SelectSessionRow:
    """Project one ``cli.query`` row onto the selector row.

    ``repo`` and ``cwd_display`` are read from the row directly: the operation
    already derived them from the session's git remote and working directories,
    and the shared row projection can only re-derive them from raw fields a
    list row does not carry.
    """

    from polylogue.surfaces.query_rows import session_row

    projected = session_row(item)
    repo = item.get("repo")
    cwd_display = item.get("cwd_display")
    return SelectSessionRow(
        session_id=projected.id,
        origin=projected.origin,
        title=projected.title,
        date=projected.date,
        message_count=projected.message_count,
        repo=str(repo) if isinstance(repo, str) and repo else projected.repo,
        cwd_display=str(cwd_display) if isinstance(cwd_display, str) and cwd_display else projected.cwd_display,
        outcome=projected.outcome,
        cost_usd=projected.cost_usd,
        relative_time=projected.relative_time,
    )


def query_session_ids(
    config: Config,
    request: RootModeRequest,
    *,
    limit: int,
    daemon_disabled: bool = False,
) -> list[str]:
    """Return the session ids ``request`` selects, at most ``limit`` of them."""

    return [row.session_id for row in query_session_rows(config, request, limit=limit, daemon_disabled=daemon_disabled)]


def query_session_rows(
    config: Config,
    request: RootModeRequest,
    *,
    limit: int,
    offset: int = 0,
    daemon_disabled: bool = False,
) -> list[SelectSessionRow]:
    """Return selector rows for ``request`` from one declared read.

    A ranked selection reports ``hits`` rather than ``items``; both are session
    pages, so both are read here and the row is taken from the hit's session.
    """

    payload = _query_page(config, request, limit=limit, offset=offset, daemon_disabled=daemon_disabled)
    return [select_row_from_operation_row(row) for row in _session_rows(payload)]


#: Page size used when walking a complete selection. The operation clamps an
#: over-large ``limit`` of its own accord, so asking for everything in one
#: request would silently truncate; this is the window the walk advances by.
COMPLETE_SELECTION_PAGE = 500


def query_complete_session_ids(
    config: Config,
    request: RootModeRequest,
    *,
    daemon_disabled: bool = False,
) -> list[str]:
    """Return every session id ``request`` selects, walking the page boundary.

    A mutating verb's cardinality guard and the mutation it authorises must see
    the same rows, so this deliberately resolves the complete matched set rather
    than one page: a single-page answer let ``delete --yes --all`` skip every
    match past the first page (#1873). The operation bounds one response, so
    completeness is the client's loop over ``next_offset`` rather than an
    unbounded request the operation would clamp without saying so.
    """

    ids: list[str] = []
    seen: set[str] = set()
    offset = 0
    while True:
        payload = _query_page(
            config, request, limit=COMPLETE_SELECTION_PAGE, offset=offset, daemon_disabled=daemon_disabled
        )
        rows = _session_rows(payload)
        if not rows:
            return ids
        for row in rows:
            session_id = str(row.get("id") or row.get("session_id") or "")
            if session_id and session_id not in seen:
                seen.add(session_id)
                ids.append(session_id)
        next_offset = payload.get("next_offset")
        if not isinstance(next_offset, int) or next_offset <= offset:
            return ids
        offset = next_offset


def query_session_rows_with_authority(
    config: Config,
    request: RootModeRequest,
    *,
    limit: int,
    offset: int = 0,
    daemon_disabled: bool = False,
) -> tuple[list[SelectSessionRow], str]:
    """Return selector rows plus the authority mode that actually answered.

    The bare landing screen used to print ``Archive: ready (daemon)`` whenever
    ``--no-daemon`` was absent, although the kernel falls back to the
    in-process reader when no socket answers (polylogue-jfabc;
    ``cli/daemon_probe.py`` documents exactly why provenance must not be
    inferred from success).  The result's own authority is the discriminator,
    so it is carried out of the read rather than guessed at the call site.
    """

    payload, authority = _query_page_with_authority(
        config, request, limit=limit, offset=offset, daemon_disabled=daemon_disabled
    )
    return [select_row_from_operation_row(row) for row in _session_rows(payload)], authority


def _query_page(
    config: Config,
    request: RootModeRequest,
    *,
    limit: int,
    offset: int,
    daemon_disabled: bool,
) -> Mapping[str, object]:
    payload, _authority = _query_page_with_authority(
        config, request, limit=limit, offset=offset, daemon_disabled=daemon_disabled
    )
    return payload


def _query_page_with_authority(
    config: Config,
    request: RootModeRequest,
    *,
    limit: int,
    offset: int,
    daemon_disabled: bool,
) -> tuple[Mapping[str, object], str]:
    from polylogue.cli.lowering import lower_cli_query
    from polylogue.cli.operation_kernel import OperationEnvelopeError, OperationKernelError, dispatch

    try:
        result = dispatch(config, lower_cli_query(request, limit=limit, offset=offset), daemon_disabled=daemon_disabled)
    except OperationKernelError as exc:
        # An ``--id``/``id:`` scope naming a session the archive does not have
        # is an empty selection, not a failure: ``select`` owes the operator
        # "No sessions matched." and exit 2, and a ``delete --dry-run`` owes an
        # empty preview. The handler states it as a refusal because for a
        # *read* of that session it is one; for a selection it is zero rows.
        if "session not found" not in str(getattr(exc, "detail", None) or exc).lower():
            raise
        return {"items": [], "total": 0}, "unknown"
    if not isinstance(result.value, dict):
        raise OperationEnvelopeError("cli.query returned a non-object result")
    authority = str(result.authority.get("server_identity") or result.authority.get("mode") or "unknown")
    return result.value, authority


def _session_rows(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    hits = payload.get("hits")
    if isinstance(hits, list):
        rows: list[Mapping[str, object]] = []
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            session = hit.get("session")
            rows.append(session if isinstance(session, Mapping) else hit)
        return rows
    items = payload.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, Mapping)]
    return []


__all__ = [
    "COMPLETE_SELECTION_PAGE",
    "query_complete_session_ids",
    "query_session_ids",
    "query_session_rows",
    "query_session_rows_with_authority",
    "select_row_from_operation_row",
]
