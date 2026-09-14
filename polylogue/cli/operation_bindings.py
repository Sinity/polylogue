"""Which CLI code lowers and renders each declared daemon operation.

Declaring an operation is not the same as serving a surface with it.  This
registry is the join between the operation declarations and the CLI code on
either side of the transport, so a declaration that no CLI route can build a
request for or render a result from is visible as such instead of looking like
progress.  Every :data:`~polylogue.operations.daemon_protocol.DAEMON_OPERATION_SPECS`
entry belongs to exactly one of three tables:

``CLI_OPERATION_BINDINGS``
    The CLI can lower a request for it and render its result today.
``CLI_PENDING_ADOPTION``
    Declared and directly executable, with the CLI route that will call it
    named.  Each entry states which migration step adopts it.
``CLI_EXTERNAL_OPERATIONS``
    Deliberately not a CLI concern (transport-owned control verbs, or
    operations whose only consumers are other surfaces).

References are dotted ``module:attribute`` strings rather than imports so this
module stays import-light on the CLI's cold path; the registry test resolves
every one of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast


class OperationBindingError(LookupError):
    """A declared operation has no classification, or an unresolvable one."""


@dataclass(frozen=True, slots=True)
class CliOperationBinding:
    """One operation joined to the CLI code that builds and renders it.

    ``lowering`` and ``renderers`` may name the same callable while a CLI route
    still does both in one pass; the seams separate at S3/S4.
    """

    lowering: str
    renderers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.renderers:
            raise OperationBindingError("an operation binding needs at least one renderer")


_ARCHIVE_QUERY = "polylogue.cli.archive_query"

CLI_OPERATION_BINDINGS: Mapping[str, CliOperationBinding] = {
    "cli.query": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_daemon_session_query_params",
        renderers=(
            f"{_ARCHIVE_QUERY}:_emit_daemon_list_payload",
            f"{_ARCHIVE_QUERY}:_emit_daemon_search_payload",
        ),
    ),
    "query.units": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_daemon_session_query_params",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_rows",),
    ),
    "query.aggregate": CliOperationBinding(
        lowering=f"{__name__}:lower_query_aggregate",
        renderers=(f"{__name__}:render_query_aggregate",),
    ),
    "session.read": CliOperationBinding(
        lowering=f"{__name__}:lower_session_read",
        renderers=(f"{__name__}:render_session_read",),
    ),
    "session.reference": CliOperationBinding(
        lowering=f"{__name__}:lower_session_reference",
        renderers=(f"{__name__}:render_session_reference",),
    ),
    "status": CliOperationBinding(
        lowering="polylogue.cli.commands.status:_status_operation_result",
        renderers=("polylogue.cli.commands.status:_show_daemon_status",),
    ),
    "facets": CliOperationBinding(
        lowering="polylogue.cli.commands.facets:_fetch_daemon_facets",
        renderers=("polylogue.cli.commands.facets:facets_command",),
    ),
    "mutation.session.tag": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_user_mutations",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_user_mutations",),
    ),
    "mutation.session.metadata": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_user_mutations",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_user_mutations",),
    ),
    "mutation.session.delete.preview": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_delete",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_delete",),
    ),
    "mutation.session.delete.authorize": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_delete",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_delete",),
    ),
    "mutation.session.delete.execute": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_emit_delete",
        renderers=(f"{_ARCHIVE_QUERY}:_emit_delete",),
    ),
    "mutation.session.delete.cancel": CliOperationBinding(
        lowering=f"{_ARCHIVE_QUERY}:_cancel_delete_preview",
        renderers=(f"{_ARCHIVE_QUERY}:_cancel_delete_preview",),
    ),
    "mutation.session.lifecycle-request": CliOperationBinding(
        lowering="polylogue.cli.commands.excise:excise_command",
        renderers=("polylogue.cli.commands.excise:excise_command",),
    ),
    "mutation.session.excision": CliOperationBinding(
        lowering="polylogue.cli.commands.excise:excise_command",
        renderers=("polylogue.cli.commands.excise:excise_command",),
    ),
    "mutation.identity-reset": CliOperationBinding(
        lowering="polylogue.cli.commands.reset:reset_command",
        renderers=("polylogue.cli.commands.reset:reset_command",),
    ),
    "maintenance.reset": CliOperationBinding(
        lowering="polylogue.cli.commands.reset:reset_command",
        renderers=("polylogue.cli.commands.reset:reset_command",),
    ),
    "maintenance.blob-gc.recover": CliOperationBinding(
        lowering="polylogue.cli.commands.maintenance._blob_gc:_submit",
        renderers=("polylogue.cli.commands.maintenance._blob_gc:blob_gc_command",),
    ),
    "mutation.raw-authority-blocker.resolve": CliOperationBinding(
        lowering="polylogue.cli.commands.maintenance._raw_identity:_submit",
        renderers=("polylogue.cli.commands.maintenance._raw_identity:raw_authority_blocker_resolve_command",),
    ),
}

CLI_PENDING_ADOPTION: Mapping[str, str] = {
    "completion": "archive-backed shell completion still opens a store in-process; S7 routes it here",
    "ingest": "`polylogue import` still POSTs the browser HTTP route; S11 routes it here",
}

CLI_EXTERNAL_OPERATIONS: Mapping[str, str] = {
    "operation.await": "transport-owned: DaemonClient.operation_to_completion drives it, not a CLI route",
    "operation.cancel": "transport-owned: DaemonClient cancels a submitted operation, not a CLI route",
    "operation.status": "no consumer on any surface; retained for receipt recovery tooling",
    "maintenance.insights.rebuild": "daemon-internal derivation; no CLI verb requests it",
}


def binding_for(operation: str) -> CliOperationBinding:
    """Return the CLI binding for ``operation`` or say why there is none."""

    binding = CLI_OPERATION_BINDINGS.get(operation)
    if binding is None:
        reason = CLI_PENDING_ADOPTION.get(operation) or CLI_EXTERNAL_OPERATIONS.get(operation)
        raise OperationBindingError(
            f"{operation} has no CLI binding: {reason}" if reason else f"{operation} is not classified"
        )
    return binding


def resolve_reference(reference: str) -> Any:
    """Import one ``module:attribute`` registry reference."""

    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise OperationBindingError(f"malformed binding reference: {reference!r}")
    try:
        return getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise OperationBindingError(f"binding reference does not resolve: {reference!r}") from exc


def unclassified_operations() -> tuple[str, ...]:
    """Return declared operations that appear in none of the three tables."""

    from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS

    classified = {*CLI_OPERATION_BINDINGS, *CLI_PENDING_ADOPTION, *CLI_EXTERNAL_OPERATIONS}
    return tuple(spec.name for spec in DAEMON_OPERATION_SPECS if spec.name not in classified)


# ---------------------------------------------------------------------------
# Lowerings and renderers for the read operations declared at S2.
#
# The CLI verbs still run their local executors; S3 repoints them here.  These
# exist now so the declarations above are backed by real client-side code
# rather than by an intention.
# ---------------------------------------------------------------------------


def lower_query_aggregate(params: Mapping[str, object]) -> dict[str, object]:
    """Build a ``query.aggregate`` request from root query params.

    The aggregate mode is read from the same ``stats_only``/``stats_by``/
    ``count_only`` flags the root query already carries, so one parse of argv
    serves both the selection and the aggregate choice.
    """

    selection = {
        key: value
        for key, value in params.items()
        if key not in {"stats_only", "stats_by", "count_only"} and value is not None
    }
    group_by = params.get("stats_by")
    if group_by:
        return {"mode": "stats_by", "group_by": str(group_by), "params": selection}
    if params.get("stats_only"):
        return {"mode": "stats", "params": selection}
    if params.get("count_only"):
        return {"mode": "count", "params": selection}
    raise OperationBindingError("no aggregate mode is selected")


def render_query_aggregate(result: Mapping[str, object]) -> list[str]:
    """Render one aggregate result as plain lines, one measurement per line."""

    mode = str(result.get("mode") or "")
    if mode == "count":
        return [f"{result.get('count', 0)}"]
    if mode == "stats_by":
        groups = result.get("groups")
        rows = groups.items() if isinstance(groups, Mapping) else ()
        return [f"{key}\t{value}" for key, value in sorted(rows, key=lambda row: (-int(row[1]), str(row[0])))]
    stats = result.get("stats")
    if not isinstance(stats, Mapping):
        raise OperationBindingError("stats result is missing its body")
    return [f"sessions\t{stats.get('total_sessions', 0)}", f"messages\t{stats.get('total_messages', 0)}"]


def lower_session_read(
    ref: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    projection: Mapping[str, object] | None = None,
    continuation: str | None = None,
) -> dict[str, object]:
    """Build one bounded ``session.read`` window request.

    A continuation supersedes the window coordinates it was minted from, so
    passing both is a caller error rather than a silently ignored argument.
    """

    if continuation is not None and (limit is not None or offset):
        raise OperationBindingError("a continuation already carries its window coordinates")
    payload: dict[str, object] = {"ref": ref}
    if limit is not None:
        payload["limit"] = limit
    if offset:
        payload["offset"] = offset
    if projection:
        payload["projection"] = dict(projection)
    if continuation is not None:
        payload["continuation"] = continuation
    return payload


def render_session_read(result: Mapping[str, object]) -> list[str]:
    """Render one transcript window, naming the window's own bounds.

    The window is stated because the operation result is bounded: a reader that
    cannot tell a whole transcript from its first page will report a truncated
    read as a complete one.
    """

    session = result.get("session")
    messages = session.get("messages") if isinstance(session, Mapping) else None
    rows = messages if isinstance(messages, list) else []
    lines = [
        f"{index}\t{row.get('role', '')}\t{row.get('text', '')}"
        for index, row in enumerate(rows, start=int(cast("int", result.get("offset") or 0)))
        if isinstance(row, Mapping)
    ]
    if not result.get("complete"):
        lines.append(f"... {result.get('next_offset')}/{result.get('total')} messages remain in the next window")
    return lines


def lower_session_reference(expression: str, *, limit: int | None = None) -> dict[str, object]:
    """Build a ``session.reference`` request for a bare ``from <ref>`` root."""

    payload: dict[str, object] = {"expression": expression}
    if limit is not None:
        payload["limit"] = limit
    return payload


def render_session_reference(result: Mapping[str, object]) -> list[str]:
    """Render resolved member refs, naming a truncated resolution as such."""

    members = result.get("members")
    lines = [str(member) for member in members] if isinstance(members, list) else []
    if result.get("truncated"):
        lines.append(f"... {result.get('member_count')} members in total")
    return lines


__all__ = [
    "CLI_EXTERNAL_OPERATIONS",
    "CLI_OPERATION_BINDINGS",
    "CLI_PENDING_ADOPTION",
    "CliOperationBinding",
    "OperationBindingError",
    "binding_for",
    "lower_query_aggregate",
    "lower_session_read",
    "lower_session_reference",
    "render_query_aggregate",
    "render_session_read",
    "render_session_reference",
    "resolve_reference",
    "unclassified_operations",
]
