"""Current archive query executor for the root CLI."""

from __future__ import annotations

import io
import json
import re
import webbrowser
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, NoReturn, cast
from urllib.parse import quote

import click

from polylogue.archive.message.types import validate_message_type_filter
from polylogue.archive.query.expression import (
    QueryUnitSource,
    WithUnitWindow,
    parse_unit_source_expression,
    split_with_projection_clause,
)
from polylogue.archive.query.filter_kwargs import spec_session_filter_kwargs
from polylogue.archive.query.metadata import query_unit_descriptor
from polylogue.archive.query.search_hits import bound_display_text
from polylogue.archive.query.spec import (
    DEFAULT_SESSION_LIST_LIMIT,
    QuerySpecError,
    SessionQuerySpec,
    clamp_query_limit,
    session_count_unit_label,
)
from polylogue.cli.lowering import aggregate_mode
from polylogue.cli.operation_kernel import OperationRequest
from polylogue.cli.query_contracts import QueryOutputSpec
from polylogue.cli.query_output_contracts import QueryOutputDocument
from polylogue.cli.render.outcome import EMPTY_EXIT_CODE, emit_empty_page, maybe_subcommand_typo_hint
from polylogue.cli.render.rows import (
    TIMING_ENV,
    emit_rows,
    emit_session_list_page,
    emit_session_search_page,
    project_payload,
    stats_by_line,
    summary_line_renderer,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.helpers import load_effective_config
from polylogue.cli.shared.machine_errors import error_no_results
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.logging import get_logger
from polylogue.surfaces.cursor_identity import search_cursor_request_identity
from polylogue.surfaces.outcome import (
    OutcomeEnvelope,
    decide_outcome,
    outcome_exit_code,
    render_outcome_line,
)

# The names below are used only as type annotations across this module and are
# constructed, if at all, behind a function-local import. This module no longer
# reaches ``polylogue.storage`` at all: the archive is opened by the operation
# executor, not here, which is what removed this file's six surface->substrate
# layering entries. Keeping the remaining annotations under TYPE_CHECKING
# preserves the cold-path import budget (polylogue-g3jk) — ``surfaces.payloads``
# alone is a full pydantic model graph. `from __future__ import annotations`
# (top of file) means annotation-only uses below never evaluate these names at
# runtime, so the gating is safe.
if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.archive.stats import ArchiveStats
    from polylogue.surfaces.payloads import (
        MutationOperation,
        QueryMissDiagnosticsPayload,
        SearchCursor,
    )


logger = get_logger(__name__)

_UNSUPPORTED_PARAM_MESSAGES: dict[str, str] = {}
_QueryUnitTextLine = Callable[[dict[str, object]], str]
_DAEMON_MUTATION_TIMEOUT_S: float | None = None
# A mutation the operator interrupted exits 130 (the shell's SIGINT
# convention): a cancelled write must never share an exit code with a
# completed one.
_CANCELLED_EXIT_CODE = 130
_NATIVE_REF_RE = re.compile(r"(?=.*\d)[A-Za-z0-9][A-Za-z0-9_.:-]{11,}")
# One ``session.read`` window.  A whole transcript can exceed the operation
# result bound, so the adapter reads it as a bounded sequence of these.
_SESSION_READ_WINDOW = 200


def _object_int(value: object) -> int:
    if value is None:
        return 0
    return int(str(value))


def execute_delete_by_session_ids(
    env: AppEnv,
    session_ids: list[str],
    *,
    force: bool,
    dry_run: bool = False,
) -> None:
    """Delete (or preview) a known set of session IDs, bypassing the query phase.

    Used by the delete verb after cardinality resolution — the IDs are already
    known so we skip the re-query (which would be capped at the default limit
    of 20, causing ``delete --yes --all`` to truncate silently). The dry-run
    preview routes through here too so the previewed set is the *same* full
    resolved set the real delete acts on (the guard, preview, and deleted sets
    must be identical, #1873).
    """
    params: dict[str, object] = {"force": force, "delete_matched": True, "dry_run": dry_run}
    if dry_run:
        _emit_delete(env, tuple(session_ids), params=params)
        return
    # Cardinality resolution already produced an immutable ID tuple. Do not
    # reopen a read transaction while the daemon executes the write: an old
    # WAL reader would pin checkpoints for the full batch-delete duration.
    _emit_delete(env, tuple(session_ids), params=params)


def execute_archive_query(env: AppEnv, request: RootModeRequest) -> None:
    """Execute the root query path."""
    timing_token = TIMING_ENV.set(env)
    env.begin_timing("execute")
    try:
        output = QueryOutputSpec.from_params(request.params)
        if output.destination_labels() == ("stdout",) or request.params.get("stream"):
            _execute_archive_query_stdout(env, request)
            return
        from polylogue.cli.query_output import deliver_query_output

        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _execute_archive_query_stdout(env, request)
        deliver_query_output(
            env,
            QueryOutputDocument(
                content=rendered.getvalue().rstrip("\n"),
                output_format=output.output_format,
                destinations=output.destinations,
            ),
        )
    finally:
        env.finish_timing("execute")
        TIMING_ENV.reset(timing_token)


@dataclass(frozen=True, slots=True)
class _ServedBy:
    """Which executor answered, as the result's own authority reports it."""

    identity: str
    elapsed_ms: int | None

    def line(self) -> str:
        if self.elapsed_ms is None:
            return self.identity
        transport = "uds, " if self.identity == "daemon" else ""
        return f"{self.identity} ({transport}{self.elapsed_ms}ms)"


def _dispatch_read(
    config: Config,
    request: OperationRequest,
    *,
    daemon_disabled: bool,
) -> tuple[dict[str, object], _ServedBy]:
    """Run one declared read and return its result body plus the daemon timing.

    Every root-query capability goes through here, so "which executor answered"
    is a transport fact recorded in the envelope rather than a semantic fork in
    the adapter.
    """
    from polylogue.cli.operation_kernel import OperationEnvelopeError, dispatch

    result = dispatch(config, request, daemon_disabled=daemon_disabled)
    if not isinstance(result.value, dict):
        raise OperationEnvelopeError(f"{request.operation} returned a non-object result")
    timing = result.envelope.get("timing") if result.envelope is not None else None
    elapsed_ms = timing.get("elapsed_ms") if isinstance(timing, Mapping) else None
    # The executor is named by the result's own authority, not by which branch
    # of this adapter ran: a rendered "daemon" provenance for a read the
    # in-process executor answered would be a claim the result does not support.
    identity = str(result.authority.get("server_identity") or result.authority.get("mode") or "unknown")
    return dict(result.value), _ServedBy(identity, elapsed_ms if isinstance(elapsed_ms, int) else None)


def _read_failure_detail(exc: Exception) -> str:
    return str(getattr(exc, "detail", None) or exc)


def _read_failure_as_usage_error(exc: Exception) -> NoReturn:
    """Re-raise a declared read's typed refusal as the CLI's own refusal.

    The handler states *what* it refused; naming it in the operator's terms and
    choosing the exit class is the adapter's job.

    Only a *request* the grammar cannot express is a usage mistake. Everything
    else -- a dropped daemon connection, a read that hit its deadline, a
    cancelled read, a typed operation failure -- exits through
    ``render.outcome.read_failure_exit_code`` with no usage banner. Routing all
    of them to ``click.UsageError`` exited 2, which is the *empty* status, so
    "the daemon went away" and "matched nothing" were indistinguishable to a
    caller branching on exit status (polylogue-jtrtj).
    """
    from polylogue.cli.render.outcome import exit_for_read_failure

    exit_for_read_failure(exc)


def _emit_reference_query(
    config: Config,
    expression: str,
    *,
    output_format: str,
    limit: int | None,
    daemon_disabled: bool,
) -> bool:
    """Resolve and render a bare ``from <ref>`` root, or decline it.

    Detection is syntax and stays here; resolution reads durable reference
    state and is the ``session.reference`` operation's job.
    """
    from polylogue.archive.query.expression import parse_reference_query_pipeline

    if not expression:
        return False
    pipeline = parse_reference_query_pipeline(expression)
    if pipeline is None:
        return False
    if pipeline.stages:
        raise click.UsageError(
            "reference pipeline stages are not supported by the CLI find surface; "
            "only the bare `from <ref>` operand is supported"
        )
    from polylogue.cli.lowering import lower_session_reference
    from polylogue.cli.operation_kernel import OperationKernelError

    try:
        payload, _ = _dispatch_read(
            config,
            lower_session_reference(expression, limit=limit),
            daemon_disabled=daemon_disabled,
        )
    except OperationKernelError as exc:
        _read_failure_as_usage_error(exc)
    if output_format in {"json", "ndjson"}:
        click.echo(
            json.dumps(
                {
                    "source": payload.get("source"),
                    "grain": payload.get("grain"),
                    "lineage": payload.get("lineage"),
                    "member_count": payload.get("member_count"),
                    "members": payload.get("members"),
                    "truncated": payload.get("truncated"),
                },
                sort_keys=True,
            )
        )
        return True
    members = payload.get("members")
    # Newline-separated, one member ref per line.  The previous local renderer
    # joined on the two-character escape ``"\\n"`` and emitted every ref on one
    # line with a literal backslash-n between them.
    click.echo("\n".join(str(member) for member in members) if isinstance(members, list) else "")
    return True


def _read_session_windows(
    config: Config,
    ref: str,
    *,
    daemon_disabled: bool,
    message_limit: int | None = None,
) -> dict[str, object]:
    """Compose one whole transcript out of the operation's bounded windows.

    ``session.read`` is windowed because a full transcript can exceed the
    declared result bound, so the adapter owns the loop.  ``complete`` — not an
    empty window — ends it: a truncated read must never render as a finished
    one.  A caller that only needs a prefix (``--stream --limit``) asks for
    exactly that prefix and stops.
    """
    from polylogue.cli.lowering import lower_session_read

    messages: list[dict[str, object]] = []
    session: dict[str, object] = {}
    continuation: str | None = None
    while True:
        window_limit: int | None = None
        if message_limit is not None:
            remaining = message_limit - len(messages)
            if remaining <= 0:
                break
            window_limit = min(remaining, _SESSION_READ_WINDOW)
        payload, _ = _dispatch_read(
            config,
            (
                lower_session_read(ref, continuation=continuation)
                if continuation is not None
                else lower_session_read(ref, limit=window_limit)
            ),
            daemon_disabled=daemon_disabled,
        )
        window = payload.get("session")
        if not isinstance(window, Mapping):
            raise click.ClickException("session.read returned no session body")
        if not session:
            session = {key: value for key, value in window.items() if key != "messages"}
        window_messages = window.get("messages")
        if isinstance(window_messages, list):
            messages.extend(item for item in window_messages if isinstance(item, dict))
        if payload.get("complete") or message_limit is not None:
            break
        next_continuation = payload.get("continuation")
        if not isinstance(next_continuation, str):
            break
        continuation = next_continuation
    session["messages"] = messages if message_limit is None else messages[:message_limit]
    return session


def _session_result_payload(session: Mapping[str, object]) -> dict[str, object]:
    """Project a ``session.read`` body onto the CLI's own session document.

    The operation carries more than this document needs (the per-message word
    and tool-use counts the summary view reads).  Naming the kept fields here
    keeps the machine document stable when the operation grows another one.
    """
    messages = session.get("messages")
    rows = [row for row in messages if isinstance(row, Mapping)] if isinstance(messages, list) else []
    return {
        "mode": "session",
        "session_id": session.get("session_id"),
        "native_id": session.get("native_id"),
        # polylogue-1c6j: this document used to emit the same Origin token
        # twice, once as ``origin`` and once as ``source``.  ``Source`` is a
        # distinct, richer acquisition identity in this codebase, so a second
        # key spelling it as the origin taught consumers a vocabulary the
        # archive does not have.  ``origin`` is the public filter token and
        # the only one here.
        "origin": session.get("origin"),
        "title": session.get("title"),
        "active_leaf_message_id": session.get("active_leaf_message_id"),
        "messages": [
            {key: value for key, value in row.items() if key not in {"word_count", "has_tool_use"}} for row in rows
        ],
    }


def _session_messages(session: Mapping[str, object]) -> list[Mapping[str, object]]:
    messages = session.get("messages")
    return [row for row in messages if isinstance(row, Mapping)] if isinstance(messages, list) else []


def _session_message_text(message: Mapping[str, object]) -> str:
    blocks = message.get("blocks")
    rows = [block for block in blocks if isinstance(block, Mapping)] if isinstance(blocks, list) else []
    return "\n".join(str(block.get("text") or "") for block in rows if block.get("text"))


def _emit_session_result(
    session: Mapping[str, object],
    *,
    output_format: str,
    fields: str | None,
    view: str = "transcript",
) -> None:
    payload = _session_result_payload(session)
    if output_format == "json":
        click.echo(json.dumps(project_payload(payload, fields), indent=2, sort_keys=True))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(project_payload(payload, fields), sort_keys=False, allow_unicode=True), nl=False)
        return
    if output_format == "ndjson":
        messages = payload["messages"]
        if not isinstance(messages, list):
            raise TypeError("session payload messages must be a list")
        for message in messages:
            click.echo(json.dumps(message, sort_keys=True))
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Full-session reads do not support --format {output_format}.")
    if view == "summary":
        click.echo(_session_summary_text(session))
        return
    click.echo(_session_text(session))


def _session_text(session: Mapping[str, object]) -> str:
    session_id = str(session.get("session_id") or "")
    lines = [f"# {session.get('title') or session_id}", "", f"`{session_id}`", ""]
    for message in _session_messages(session):
        lines.append(f"## {message.get('role')}")
        lines.append(_session_message_text(message))
        lines.append("")
    return "\n".join(lines).rstrip()


def _session_summary_text(session: Mapping[str, object]) -> str:
    """Condensed session synopsis: counts, roles, tool usage, first/last excerpts.

    Deliberately distinct from :func:`_session_text` (the full transcript) --
    ``read --view summary`` previously routed to the same renderer as
    ``read --view transcript`` and produced byte-identical output for any
    session (polylogue-zumd class: a surface claiming to do X silently did
    the whole-transcript Y instead).
    """
    session_id = str(session.get("session_id") or "")
    rows = _session_messages(session)
    role_counts: dict[str, int] = {}
    tool_use_message_count = 0
    total_words = 0
    for message in rows:
        role = str(message.get("role") or "")
        role_counts[role] = role_counts.get(role, 0) + 1
        total_words += _object_int(message.get("word_count"))
        if message.get("has_tool_use"):
            tool_use_message_count += 1

    lines = [
        f"# {session.get('title') or session_id}",
        "",
        f"`{session_id}`  ({session.get('origin')})",
        "",
        f"- messages: {len(rows)}",
    ]
    for role in sorted(role_counts):
        lines.append(f"  - {role}: {role_counts[role]}")
    lines.append(f"- words (sum of per-message word_count): {total_words}")
    lines.append(f"- messages with tool use: {tool_use_message_count}")
    created_at = session.get("created_at")
    updated_at = session.get("updated_at")
    if created_at or updated_at:
        lines.append(f"- created: {created_at or 'unknown'}  updated: {updated_at or 'unknown'}")

    def _first_authored_text(candidates: Iterable[Mapping[str, object]]) -> str:
        for message in candidates:
            if message.get("role") not in ("user", "assistant"):
                continue
            blocks = message.get("blocks")
            block_rows = [block for block in blocks if isinstance(block, Mapping)] if isinstance(blocks, list) else []
            # The archive's own display flattening (``archive_message_display_text``):
            # every non-empty block's text in block order, blank-line separated.
            text = "\n\n".join(str(block.get("text")) for block in block_rows if block.get("text"))
            if text.strip():
                return text
        return ""

    first_text = _first_authored_text(rows)
    last_text = _first_authored_text(reversed(rows))
    if first_text:
        lines += ["", "## First turn", "", bound_display_text(first_text, max_chars=500)]
    if last_text and last_text != first_text:
        lines += ["", "## Last turn", "", bound_display_text(last_text, max_chars=500)]
    return "\n".join(lines).rstrip()


def _emit_stream(session: Mapping[str, object], *, output_format: str) -> None:
    payload = _session_result_payload(session)
    raw_messages = payload["messages"]
    if not isinstance(raw_messages, list):
        raise TypeError("session payload messages must be a list")
    if output_format in {"json", "json-lines", "ndjson"}:
        for message in raw_messages:
            click.echo(json.dumps(message, sort_keys=True))
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Stream does not support --format {output_format}.")
    lines: list[str] = []
    for message in _session_messages(session):
        lines.append(f"## {message.get('role')}")
        lines.append(_session_message_text(message))
        lines.append("")
    click.echo("\n".join(lines).rstrip())


def _archive_stats_from_result(body: Mapping[str, object]) -> ArchiveStats:
    """Rebuild the stats dataclass from its wire form.

    ``embedding_dimensions`` is keyed by dimension *number*; JSON object keys
    are strings, so the round trip has to restore the integer keys or the
    rendered document silently changes shape by transport.
    """
    from dataclasses import fields as dataclass_fields

    from polylogue.archive.stats import ArchiveStats

    values = dict(body)
    dimensions = values.get("embedding_dimensions")
    if isinstance(dimensions, Mapping):
        values["embedding_dimensions"] = {int(key): _object_int(value) for key, value in dimensions.items()}
    declared = {field.name for field in dataclass_fields(ArchiveStats)}
    return ArchiveStats(**cast("Any", {key: value for key, value in values.items() if key in declared}))


def _emit_aggregate_result(
    payload: Mapping[str, object],
    *,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> None:
    """Render one ``query.aggregate`` result through the aggregate renderers."""
    mode = str(payload.get("mode") or "")
    if mode == "count":
        _emit_count(_object_int(payload.get("count")), output_format=output_format, origin=origin)
        return
    if mode == "stats_by":
        groups = payload.get("groups")
        _emit_stats_by(
            {str(key): _object_int(value) for key, value in groups.items()} if isinstance(groups, Mapping) else {},
            group_by=str(payload.get("group_by") or ""),
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
        )
        return
    body = payload.get("stats")
    _emit_stats(
        _archive_stats_from_result(body if isinstance(body, Mapping) else {}),
        output_format=output_format,
        origin=origin,
        query=query,
        fields=fields,
    )


def _transcript_or_page(
    config: Config,
    ref: str,
    *,
    daemon_disabled: bool,
    message_limit: int | None,
    certain: bool,
) -> dict[str, object] | None:
    """Read ``ref`` as a transcript, or decline when it was only ref-*shaped*.

    ``repo:polylogue`` and other structured field clauses pass the cheap
    syntactic ref probe.  They are not identity queries, so a miss falls
    through to ordinary page execution — but an *ambiguous* reference is a real
    identity failure and must not broaden into a text search.
    """
    from polylogue.cli.operation_kernel import OperationKernelError

    try:
        return _read_session_windows(config, ref, daemon_disabled=daemon_disabled, message_limit=message_limit)
    except OperationKernelError as exc:
        detail = _read_failure_detail(exc)
        if certain:
            if "not found" in detail:
                _fail(f"Session not found: {ref}")
            _read_failure_as_usage_error(exc)
        if "ambiguous" in detail:
            raise click.UsageError(detail) from exc
        return None


def _execute_archive_query_stdout(env: AppEnv, request: RootModeRequest) -> None:
    """Render root query output to stdout.

    Every read below is a declared operation dispatched through the kernel.
    There is no second, local implementation of what ``find`` means: this
    adapter chooses the operation, projects argv onto its payload (Seam A,
    :mod:`polylogue.cli.lowering`), and renders the result.
    """
    params = dict(request.params)
    cursor_request_identity = search_cursor_request_identity(params)
    _reject_unsupported_params(params)
    _validate_retrieval_params(params)
    config_started_at = perf_counter()
    config = load_effective_config(env)
    # polylogue-yla8.1 split-root contract: config.db_path always names a
    # concrete index.db (explicit override or resolved active generation).  The
    # dispatch resolves the file set that index belongs to, so a ``--db`` pin at
    # a non-active generation is answered from the generation the operator named
    # rather than from whatever is active.
    index_db_path = config.db_path
    env.record_timing("config", config_started_at)
    compile_started_at = perf_counter()
    typo_hint = maybe_subcommand_typo_hint(request.query_terms)
    raw_query = _query_text(request.query_terms, params)
    output_format = str(params.get("output_format") or "markdown")
    daemon_disabled = _daemon_disabled(flag=bool(params.get("no_daemon")))

    if _emit_reference_query(
        config,
        raw_query,
        output_format=output_format,
        limit=_optional_int(params.get("limit")),
        daemon_disabled=daemon_disabled,
    ):
        env.record_timing("compile", perf_counter() - compile_started_at)
        return

    fields = _optional_str(params.get("fields"))
    read_view = str(params.get("view") or "transcript")
    # Split a trailing ``with <units>`` projection clause off the FTS text so it
    # is not searched literally; the units drive the attached-unit projection.
    unit_source_query = raw_query
    with_units: tuple[str, ...] = ()
    with_unit_fields: dict[str, tuple[str, ...]] = {}
    with_unit_windows: dict[str, WithUnitWindow] = {}
    if unit_source_query and not (unit_source_query.startswith("{") or unit_source_query.startswith("[")):
        unit_source_query, with_units, with_unit_fields, with_unit_windows = split_with_projection_clause(
            unit_source_query
        )
    unit_source = (
        parse_unit_source_expression(unit_source_query)
        if unit_source_query and not _optional_str(params.get("similar_text"))
        else None
    )
    compiled_spec = (
        SessionQuerySpec.from_params(params)
        if unit_source is not None
        else _compiled_session_spec(request, params=params, raw_query=raw_query)
    )
    origins = compiled_spec.origins
    origin = origins[0] if len(origins) == 1 else None
    query = _query_text(compiled_spec.query_terms, {"contains": compiled_spec.contains_terms})
    if compiled_spec.with_units:
        with_units = compiled_spec.with_units
        with_unit_fields = compiled_spec.with_unit_fields
        with_unit_windows = compiled_spec.with_unit_windows
    _reject_date_bounds_that_cannot_parse(compiled_spec)
    env.record_timing("compile", compile_started_at)

    tags_to_add = _tuple_tokens(params.get("add_tag"))
    metadata_to_set = _metadata_pairs(params.get("set_meta"))
    since_session_id = compiled_spec.since_session_id
    # polylogue-fawr7: the compiled spec can carry an explicit ``limit:`` from
    # the query expression as well as from ``--limit``; both are explicit
    # requests and both answer to the shared public ceiling.
    limit = (
        clamp_query_limit(compiled_spec.limit, default=DEFAULT_SESSION_LIST_LIMIT)
        if compiled_spec.limit is not None and compiled_spec.limit > 0
        else _limit(params)
    )
    offset = compiled_spec.offset if compiled_spec.offset > 0 else _offset(params)
    cursor = _decode_cursor(_optional_str(params.get("cursor")))
    _validate_cursor_request_identity(cursor, cursor_request_identity)
    page_offset = cursor.r if cursor is not None else offset
    sample_count = _optional_int(params.get("sample"))
    if sample_count is not None:
        if cursor is not None:
            raise click.UsageError("Root query does not combine --sample with --cursor.")
        if sample_count <= 0:
            raise click.UsageError("Root query --sample must be positive.")
        limit = sample_count
        page_offset = 0
    stream = bool(params.get("stream"))
    stream_output_format = QueryOutputSpec.from_params(params).stream_format()
    sort = compiled_spec.sort
    reverse = compiled_spec.reverse
    similar_text = compiled_spec.similar_text
    similar_session_id = compiled_spec.similar_session_id
    retrieval_lane = _optional_str(params.get("retrieval_lane")) or compiled_spec.retrieval_lane
    delete_matched = bool(params.get("delete_matched"))
    aggregate = aggregate_mode(params)
    searching = bool(query or similar_text or similar_session_id)
    session_scope_id = compiled_spec.session_id or (
        str(params["conv_id"]) if params.get("conv_id") is not None else None
    )

    # --- Syntax refusals.  These name option combinations no operation will
    # ever be asked to answer, so they are decided before any transport.
    if cursor is not None and (aggregate is not None or session_scope_id is not None or params.get("latest")):
        raise click.UsageError("Root query --cursor is only supported for list and search pages.")
    if retrieval_lane == "hybrid" and not query:
        raise click.UsageError("Hybrid retrieval requires lexical query terms.")
    if tags_to_add and aggregate is not None:
        raise click.UsageError("--add-tag is only supported for matched sessions.")
    if delete_matched and aggregate is not None:
        raise click.UsageError("Delete is only supported for matched sessions.")
    if delete_matched and tags_to_add:
        raise click.UsageError("Root query cannot combine delete with --add-tag.")
    if delete_matched and metadata_to_set:
        raise click.UsageError("Root query cannot combine delete with --set.")
    if sample_count is not None and aggregate is not None:
        raise click.UsageError("Root query does not combine --sample with stats.")
    if sample_count is not None and searching:
        raise click.UsageError("Root query does not combine --sample with search terms.")
    if unit_source is not None and any(
        (
            aggregate is not None,
            stream,
            tags_to_add,
            metadata_to_set,
            delete_matched,
            params.get("open_result"),
            params.get("conv_id"),
            sample_count is not None,
            since_session_id is not None,
            cursor is not None,
            sort is not None,
            reverse,
        )
    ):
        unit_label = _unit_source_display_name(unit_source)
        raise click.UsageError(
            f"{unit_label} where queries return {unit_source.unit} rows and do not combine "
            "with session-only actions, aggregate modes, sort, reverse, or cursor."
        )

    # A missing local index tier is only decisive when nothing else can answer.
    # A resident daemon owns its own file set and may well be able to serve this
    # page, so the refusal is deferred until the dispatch below actually fails
    # for want of an archive (:func:`_missing_archive_refusal`).
    if daemon_disabled and not index_db_path.exists():
        _missing_archive_refusal(
            params,
            index_db_path=index_db_path,
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
            typo_hint=typo_hint,
        )

    from polylogue.cli.lowering import lower_cli_query, lower_query_aggregate, lower_query_units
    from polylogue.cli.operation_kernel import OperationKernelError

    db_open_started_at = perf_counter()

    if unit_source is not None:
        try:
            payload, _ = _dispatch_read(
                config,
                lower_query_units(request, expression=unit_source_query, limit=limit, offset=page_offset),
                daemon_disabled=daemon_disabled,
            )
        except OperationKernelError as exc:
            _read_failure_as_usage_error(exc)
        env.record_timing("db-open", db_open_started_at)
        raw_items = payload.get("items")
        items = [item for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []
        if not items:
            _emit_unit_no_results(payload, unit=unit_source.unit, output_format=output_format)
        text_line = (
            _aggregate_query_line
            if payload.get("mode") == "query-unit-aggregate"
            else _query_unit_text_line(unit_source.unit)
        )
        emit_rows(payload, items, output_format=output_format, text_line=text_line, fields=fields)
        return

    if aggregate is not None:
        try:
            payload, _ = _dispatch_read(
                config,
                lower_query_aggregate(request, mode=aggregate),
                daemon_disabled=daemon_disabled,
            )
        except ArchiveTierUnavailableError:
            # An aggregate over an archive that does not exist yet has a
            # correct answer -- zero -- and is the first thing a fresh install
            # runs (polylogue-wwjy6).
            _missing_archive_refusal(
                params,
                index_db_path=index_db_path,
                output_format=output_format,
                origin=origin,
                query=query,
                fields=fields,
                typo_hint=typo_hint,
            )
            return
        except OperationKernelError as exc:
            _read_failure_as_usage_error(exc)
        env.record_timing("db-open", db_open_started_at)
        _emit_aggregate_result(payload, output_format=output_format, origin=origin, query=query, fields=fields)
        return

    # --- An exact session reference selects one transcript, not a page.  A
    # scoped search (``--id X <terms>``) is still a page, inside that session.
    transcript_ref: str | None = None
    certain_ref = False
    if session_scope_id is not None and not searching:
        transcript_ref, certain_ref = session_scope_id, True
    elif session_scope_id is None and query and not similar_text and _single_query_token_looks_like_ref(query):
        transcript_ref = query
    if transcript_ref is not None and certain_ref and params.get("open_result"):
        # Opening a session needs its identity, not its content: a one-message
        # window resolves the reference (and proves the session exists) without
        # reading a transcript the launcher immediately discards.
        opened = _transcript_or_page(
            config, transcript_ref, daemon_disabled=daemon_disabled, message_limit=1, certain=True
        )
        _open_session(
            env,
            str((opened or {}).get("session_id") or transcript_ref),
            output_format=output_format,
            print_url=bool(params.get("print_url")),
        )
        return
    if transcript_ref is not None:
        session = _transcript_or_page(
            config,
            transcript_ref,
            daemon_disabled=daemon_disabled,
            message_limit=_stream_message_limit(params) if stream else None,
            certain=certain_ref,
        )
        if session is not None:
            env.record_timing("db-open", db_open_started_at)
            session_id = str(session.get("session_id") or transcript_ref)
            if stream:
                _emit_stream(session, output_format=stream_output_format)
                return
            if tags_to_add or metadata_to_set:
                _emit_user_mutations(env, (session_id,), tags_to_add=tags_to_add, metadata_to_set=metadata_to_set)
                return
            if delete_matched:
                _emit_delete(env, (session_id,), params=params)
                return
            if params.get("open_result"):
                _open_session(env, session_id, output_format=output_format, print_url=bool(params.get("print_url")))
                return
            _emit_session_result(session, output_format=output_format, fields=fields, view=read_view)
            return

    # --- Ordinary page: one declared session query, rendered as list or search.
    try:
        payload, served_by = _dispatch_read(
            config,
            lower_cli_query(
                request,
                limit=limit,
                offset=page_offset,
                sample=sample_count,
                with_units=with_units,
                with_unit_fields=with_unit_fields,
                with_unit_windows=with_unit_windows,
            ),
            daemon_disabled=daemon_disabled,
        )
    except ArchiveTierUnavailableError:
        # The index tier is absent: a first-run condition, not a failed
        # request. Browse and aggregate modes have a correct empty answer and
        # a search names the path it looked for -- the same refusal the
        # daemon-disabled pre-check above takes (polylogue-wwjy6).
        _missing_archive_refusal(
            params,
            index_db_path=index_db_path,
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
            typo_hint=typo_hint,
        )
        return
    except OperationKernelError as exc:
        detail = _read_failure_detail(exc)
        if session_scope_id is not None and "session not found" in detail.lower():
            # An explicit ``--id`` that resolves to nothing is a missing
            # session, not a malformed command line: same wording and exit
            # class as an exact-ref read of the same reference.
            _fail(f"Session not found: {session_scope_id}")
        if not index_db_path.exists():
            _missing_archive_refusal(
                params,
                index_db_path=index_db_path,
                output_format=output_format,
                origin=origin,
                query=query,
                fields=fields,
                typo_hint=typo_hint,
            )
            return
        _read_failure_as_usage_error(exc)
    env.record_timing("db-open", db_open_started_at)
    if bool(params.get("verbose")):
        click.echo(f"served-by: {served_by.line()}", err=True)

    ranked = isinstance(payload.get("hits"), list)
    raw_rows = payload.get("hits") if ranked else payload.get("items")
    rows = [row for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
    matched_session_ids = tuple(
        session_id for session_id in (str(row.get("id") or row.get("session_id") or "") for row in rows) if session_id
    )

    if stream:
        if not matched_session_ids:
            _fail("Stream found no matching session.")
        _emit_stream(
            _read_session_windows(
                config,
                matched_session_ids[0],
                daemon_disabled=daemon_disabled,
                message_limit=_stream_message_limit(params),
            ),
            output_format=stream_output_format,
        )
        return
    if tags_to_add or metadata_to_set:
        _emit_user_mutations(env, matched_session_ids, tags_to_add=tags_to_add, metadata_to_set=metadata_to_set)
        return
    if delete_matched:
        _emit_delete(env, matched_session_ids, params=params)
        return
    if params.get("open_result"):
        if not matched_session_ids:
            if searching:
                _fail("Open found no matching session.")
            _emit_open_no_results(output_format=output_format, origin=origin)
        _open_session(
            env,
            matched_session_ids[0],
            output_format=output_format,
            print_url=bool(params.get("print_url")),
        )
        return

    if ranked:
        emit_session_search_page(
            payload,
            query=similar_text or query or similar_session_id or "",
            limit=limit,
            offset=page_offset,
            output_format=output_format,
            origin=origin,
            fields=fields,
            typo_hint=typo_hint,
            # A zero-hit page still owes the operator a diagnosis and no read
            # operation declares one yet, so the adapter offers the same
            # clause-drop/relaxation/FTS-disagreement diagnosis the TUI, daemon
            # and API surfaces use.  It is a callable because the renderer asks
            # for it only when the page is empty and the operation named none.
            diagnose_miss=lambda: _search_miss_diagnostics(env, compiled_spec, why=bool(params.get("why"))),
            source=served_by.identity,
        )
        return
    _emit_list_miss_if_field_syntax(
        env,
        items=list(rows),
        total=_object_int(payload.get("total")) if payload.get("total") is not None else None,
        raw_query=raw_query,
        compiled_spec=compiled_spec,
        why=bool(params.get("why")),
        output_format=output_format,
        origin=origin,
        limit=limit,
        offset=page_offset,
        root=True,
        typo_hint=typo_hint,
    )
    emit_session_list_page(
        payload,
        limit=limit,
        offset=page_offset,
        output_format=output_format,
        origin=origin,
        fields=fields,
        source=served_by.identity,
    )


def _reject_date_bounds_that_cannot_parse(spec: SessionQuerySpec) -> None:
    """Refuse an unparseable ``--since``/``--until`` literal as a CLI usage fault.

    Whether a date literal parses is a property of what the operator typed, so
    it is answered here with the wording and exit class the CLI has always
    used, rather than surfacing as the handler's generic invalid-request
    refusal. The lowering is pure and repeated only to ask the question.
    """
    try:
        spec_session_filter_kwargs(spec)
    except QuerySpecError as exc:
        raise click.ClickException(f"Cannot parse date: {exc.value!r}") from exc


def _reject_unsupported_params(params: dict[str, object]) -> None:
    for key, message in _UNSUPPORTED_PARAM_MESSAGES.items():
        if _has_value(params.get(key)):
            raise click.UsageError(message)


def _validate_retrieval_params(params: dict[str, object]) -> None:
    lane = _optional_str(params.get("retrieval_lane"))
    if lane not in {None, "auto", "dialogue", "semantic", "hybrid"}:
        raise click.UsageError("Root query retrieval lane must be auto, dialogue, semantic, or hybrid.")


def _single_query_token_looks_like_ref(query: str) -> bool:
    token = query.strip()
    return bool(token and " " not in token and (":" in token or _NATIVE_REF_RE.fullmatch(token)))


def _daemon_disabled(*, flag: bool = False) -> bool:
    if flag:
        return True
    from polylogue.config import load_polylogue_config

    settings = load_polylogue_config()
    if settings.no_daemon:
        return True
    return settings.daemon_client_mode == "off"


def _submit_mutation_operation(
    config: Config,
    operation: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Run one declared write or control operation and return its result.

    The daemon is the sole writer, so a mutation has no direct executor: the
    kernel refuses with ``OperationUnavailableError`` when no daemon answers
    rather than falling back to a second write authority. Once the request is
    on the socket, an absent receipt is indeterminate, never a retryable
    absence.
    """
    from polylogue.cli.operation_kernel import OperationUnavailableError, configured_mutation_operation

    if _daemon_disabled():
        raise OperationUnavailableError(f"daemon is unavailable for operation: {operation}")
    return configured_mutation_operation(config, operation, payload)


def _daemon_preview_refs(payload: Mapping[str, object]) -> tuple[str, ...] | None:
    """Read the preview refs a daemon delete payload names, or None if it names none.

    The daemon echoes ``preview_ref`` for a single ref and ``preview_refs``
    for several, so both shapes are authoritative.
    """
    refs = payload.get("preview_refs")
    if isinstance(refs, list) and refs and all(isinstance(ref, str) and ref for ref in refs):
        return tuple(refs)
    ref = payload.get("preview_ref")
    if isinstance(ref, str) and ref:
        return (ref,)
    return None


def _decode_cursor(token: str | None) -> SearchCursor | None:
    if token is None:
        return None
    from polylogue.surfaces.payloads import InvalidSearchCursorError, decode_search_cursor

    try:
        cursor = decode_search_cursor(token)
    except InvalidSearchCursorError as exc:
        raise click.UsageError(f"invalid --cursor: {exc}") from exc
    return cursor


def _validate_cursor_request_identity(cursor: SearchCursor | None, request_identity: str) -> None:
    if cursor is not None and cursor.query_hash is not None and cursor.query_hash != request_identity:
        raise click.UsageError("invalid --cursor: cursor belongs to a different ranked-search request")


def _open_session(env: AppEnv, session_id: str, *, output_format: str, print_url: bool) -> None:
    daemon_url = str(getattr(env, "daemon_url", None) or "http://127.0.0.1:8766").rstrip("/")
    web_url = f"{daemon_url}/?session={quote(session_id, safe='')}"
    if print_url:
        if output_format == "json":
            click.echo(json.dumps({"url": web_url}, indent=2))
        else:
            click.echo(web_url)
        return

    webbrowser.open(web_url)
    env.ui.console.print(f"Opened: {web_url}")


def _stream_message_limit(params: dict[str, object]) -> int | None:
    value = params.get("limit")
    if isinstance(value, int) and value > 0:
        return value
    return None


def _has_value(value: object) -> bool:
    if value is None or value is False:
        return False
    return not (value == "" or value == () or value == [])


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _csv_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    # Click ``multiple=True`` options arrive as a (possibly empty) tuple/list;
    # each element may itself be a comma-separated string. Tokenize per element so
    # an empty tuple yields no tokens — str(()) would otherwise inject "()" and
    # turn every empty multi-option into a filter that matches nothing.
    elements: tuple[object, ...] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return tuple(token.strip() for element in elements for token in str(element).split(",") if token.strip())


def _tuple_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(token).strip() for token in value if str(token).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def _metadata_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    pairs: list[tuple[str, str]] = []
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, str | bytes) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
            else:
                raise click.UsageError("--set expects key/value pairs.")
    else:
        raise click.UsageError("--set expects key/value pairs.")
    return tuple(pairs)


def _tool_tokens(value: object) -> tuple[str, ...]:
    return tuple(token.lower() for token in _csv_tokens(value))


def _message_type(value: object) -> str | None:
    if not value:
        return None
    try:
        return validate_message_type_filter(value).value
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _sort(value: object) -> str | None:
    if not value:
        return None
    sort = str(value)
    if sort not in {"date", "messages", "words", "longest", "tokens", "random"}:
        raise click.UsageError("Root query sort must be one of date, messages, words, longest, tokens, random.")
    return sort


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _limit(params: dict[str, object]) -> int:
    """Resolve the page size: refuse a nonpositive request, clamp to the ceiling.

    ``--limit 0`` used to fall through to ``DEFAULT_SESSION_LIST_LIMIT``, so a
    caller asking for nothing silently got a default-sized read; a nonpositive
    limit is a usage fault, not a default (polylogue-45pkf). A positive limit
    used to be returned verbatim, making the explicit CLI limit the one read
    route that could exceed ``MAX_QUERY_LIMIT`` while MCP, daemon HTTP and the
    operation route all clamped (polylogue-fawr7).
    """
    value = params.get("limit")
    if isinstance(value, int):
        if value <= 0:
            raise click.UsageError("--limit must be a positive integer.")
        return clamp_query_limit(value, default=DEFAULT_SESSION_LIST_LIMIT)
    return DEFAULT_SESSION_LIST_LIMIT


def _offset(params: dict[str, object]) -> int:
    value = params.get("offset")
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _query_text(query_terms: tuple[str, ...], params: dict[str, object]) -> str:
    terms = [term for term in query_terms if term]
    contains = params.get("contains")
    if isinstance(contains, Iterable) and not isinstance(contains, str | bytes):
        terms.extend(str(term) for term in contains if term)
    return " ".join(terms).strip()


def _compiled_session_spec(request: RootModeRequest, *, params: dict[str, object], raw_query: str) -> SessionQuerySpec:
    """Compile CLI selection terms while preserving CLI-only semantic lane spelling."""
    if params.get("retrieval_lane") != "semantic":
        return request.query_spec()
    spec_params = dict(params)
    spec_params["retrieval_lane"] = "auto"
    if raw_query and not spec_params.get("similar_text"):
        spec_params["similar_text"] = raw_query
    return RootModeRequest(params=spec_params, query_terms=()).query_spec()


def _missing_archive_refusal(
    params: dict[str, object],
    *,
    index_db_path: Path,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
    typo_hint: str | None,
) -> None:
    """Render, or refuse, a read against an archive that is not there.

    Browse and aggregate modes have a correct empty answer; a search does not,
    and says so with the path it looked for.
    """
    if _emit_missing_archive_empty_read(
        params,
        output_format=output_format,
        origin=origin,
        query=query,
        fields=fields,
    ):
        return
    # One first-run condition, one producer.  This branch used to ``click.echo``
    # its own wording and ``SystemExit(1)``, which bypassed ``machine_errors``
    # entirely: ``polylogue --format json find X`` on a fresh root emitted
    # unparseable plain text while ``read``/``select``/the bare screen let the
    # typed ``ArchiveTierUnavailableError`` reach ``machine_main`` and produce
    # the structured envelope.  Two texts and two machine contracts for one
    # condition (polylogue-ry6g5); raising the same typed refusal the storage
    # layer raises collapses them.
    guidance = "run `polylogue ingest` to create the archive, or point --archive-root at an existing one"
    if typo_hint is not None:
        guidance = f"{guidance}\n{typo_hint}"
    raise ArchiveTierUnavailableError(
        tier="index",
        path=str(index_db_path),
        reason="database file not found",
        guidance=guidance,
    )


def _emit_missing_archive_empty_read(
    params: dict[str, object],
    *,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> bool:
    if params.get("count_only"):
        _emit_count(0, output_format=output_format, origin=origin)
        return True
    if params.get("stats_by"):
        _emit_stats_by(
            {},
            group_by=str(params["stats_by"]),
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
        )
        return True
    if params.get("stats_only"):
        from polylogue.archive.stats import ArchiveStats

        _emit_stats(
            ArchiveStats(total_sessions=0, total_messages=0),
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
        )
        return True
    if params.get("list_mode") and not query:
        # Browse mode over an archive that does not exist yet: "show me
        # everything, there is nothing" is a valid success, not an error.
        envelope: dict[str, object] = {
            "mode": "list",
            "origin": origin,
            "items": [],
            "total": 0,
            "total_unit": session_count_unit_label(True),
            "limit": _limit(params),
            "offset": _offset(params),
            "next_offset": None,
            "next_cursor": None,
            "outcome": decide_outcome(matched=0).to_dict(),
        }
        emit_rows(envelope, [], output_format=output_format, text_line=summary_line_renderer([]), fields=fields)
        return True
    return False


def _emit_count(count: int, *, output_format: str, origin: str | None) -> None:
    if output_format == "json":
        click.echo(json.dumps({"mode": "count", "origin": origin, "count": count}, indent=2))
        return
    click.echo(count)


def _emit_stats(
    stats: ArchiveStats,
    *,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> None:
    from polylogue.cli.render.outcome import convergence_warning_line

    convergence_warning = convergence_warning_line()
    payload = {
        "mode": "stats",
        "origin": origin,
        "query": query or None,
        "outcome": decide_outcome(matched=stats.total_sessions).to_dict(),
        **stats.to_dict(),
    }
    if convergence_warning is not None:
        payload["archive_converging"] = True
        payload["convergence_warning"] = convergence_warning
    if output_format == "json":
        click.echo(json.dumps(project_payload(payload, fields), indent=2, sort_keys=True))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(project_payload(payload, fields), sort_keys=False, allow_unicode=True), nl=False)
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Stats do not support --format {output_format}.")
    outcome_line = render_outcome_line(OutcomeEnvelope.model_validate(payload["outcome"]))
    lines = [
        f"Sessions: {stats.total_sessions}",
        f"Messages: {stats.total_messages}",
        f"Attachments: {stats.total_attachments}",
        f"Origins: {stats.origin_count}",
        f"Average messages: {stats.avg_messages_per_session:.1f}",
    ]
    if convergence_warning is not None:
        lines.insert(0, convergence_warning)
    if outcome_line is not None:
        lines.insert(0, outcome_line)
    click.echo("\n".join(lines))


def _emit_stats_by(
    grouped: dict[str, int],
    *,
    group_by: str,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> None:
    items = [{"group": key, "count": count} for key, count in grouped.items()]
    envelope: dict[str, object] = {
        "mode": "stats_by",
        "group_by": group_by,
        "origin": origin,
        "query": query or None,
        "items": items,
        "total": sum(grouped.values()),
        "outcome": decide_outcome(matched=sum(grouped.values())).to_dict(),
    }
    emit_rows(envelope, items, output_format=output_format, text_line=stats_by_line, fields=fields)


def _emit_mutation(changed: int, *, operation: MutationOperation) -> None:
    from polylogue.surfaces.payloads import MutationResultPayload

    click.echo(
        MutationResultPayload(status="ok", operation=operation, affected_count=changed).to_json(exclude_none=True)
    )


def _emit_user_mutations(
    env: AppEnv,
    session_ids: tuple[str, ...],
    *,
    tags_to_add: tuple[str, ...],
    metadata_to_set: tuple[tuple[str, str], ...],
) -> None:
    """Apply matched-session tag/metadata writes through the mutation authority.

    ``user.db`` is durable and irreplaceable, so this route lowers to the
    declared ``mutation.session.tag``/``mutation.session.metadata``
    operations. The daemon owns the PREPARE/AUTHORIZE/EXECUTE cycle behind
    them; an adapter that opened its own writable handle would be a second
    write authority whose preview, authorization and audit records the
    daemon's journal does not have.

    Selection is finished by the time this runs, and it came from a declared
    read that has already released its own reader, so no snapshot of this
    process holds ``user.db`` attached while the daemon's writer works.
    """
    from polylogue.cli.operation_kernel import OperationKernelError
    from polylogue.surfaces.payloads import MutationResultPayload

    config = load_effective_config(env)

    def _apply(operation: str, payload: dict[str, object]) -> int:
        try:
            result = _submit_mutation_operation(config, operation, payload)
        except OperationKernelError as exc:
            raise _mutation_refusal(exc, operation) from exc
        return _object_int(result.get("affected_count"))

    changes: dict[str, int] = {}
    if metadata_to_set:
        changes["metadata"] = _apply(
            "mutation.session.metadata",
            {"session_ids": list(session_ids), "pairs": [[key, value] for key, value in metadata_to_set]},
        )
    if tags_to_add:
        changes["tags"] = _apply(
            "mutation.session.tag",
            {"session_ids": list(session_ids), "tags": list(tags_to_add)},
        )
    if set(changes) == {"tags"}:
        _emit_mutation(changes["tags"], operation="add_tag")
        return
    if set(changes) == {"metadata"}:
        _emit_mutation(changes["metadata"], operation="set_meta")
        return
    # Combined tag+metadata mutation: ``tag_count`` carries the number of
    # sessions that had a tag added, ``applied_count`` the number that had
    # metadata set (the two halves of the former ``changed`` dict).
    click.echo(
        MutationResultPayload(
            status="ok",
            operation="mutate",
            tag_count=changes.get("tags"),
            applied_count=changes.get("metadata"),
        ).to_json(exclude_none=True)
    )


def _emit_delete(env: AppEnv, session_ids: tuple[str, ...], *, params: dict[str, object]) -> None:
    """Delete sessions through the shared OperationExecutor mutation authority.

    Every surface that can permanently delete a session (this CLI route and
    MCP ``write(operation='delete_session')`` in ``mcp/server_cutover.py``)
    drives the same :class:`SessionDeleteActuator` through
    :class:`OperationExecutor` (polylogue-t46.9/kwsb.2) instead of calling
    ``ArchiveStore.delete_sessions`` directly, so preview/authorization/
    receipt semantics cannot diverge between adapters.
    """
    from polylogue.surfaces.payloads import MutationResultPayload

    dry_run = bool(params.get("dry_run"))
    force = bool(params.get("force"))
    count = len(session_ids)

    if dry_run:
        # ``session_count`` = matched, ``affected_count`` = deleted (0 in a
        # preview); ``session_ids`` enumerates the sessions that would be deleted.
        click.echo(
            MutationResultPayload(
                status="preview",
                operation="delete",
                session_count=count,
                affected_count=0,
                session_ids=tuple(session_ids),
            ).to_json(exclude_none=True)
        )
        return
    if count == 0:
        click.echo(
            MutationResultPayload(status="ok", operation="delete", session_count=0, affected_count=0).to_json(
                exclude_none=True
            )
        )
        return
    if not force and env.ui.plain:
        click.echo(
            MutationResultPayload(
                status="aborted",
                operation="delete",
                session_count=count,
                affected_count=0,
                detail="confirmation_required",
            ).to_json(exclude_none=True)
        )
        return
    config = load_effective_config(env)
    from polylogue.cli.operation_kernel import OperationKernelError

    try:
        daemon_preview = _submit_mutation_operation(
            config,
            "mutation.session.delete.preview",
            {"session_ids": list(session_ids)},
        )
    except OperationKernelError as exc:
        raise _delete_refusal(exc, "prepare") from exc

    prepared_session_ids = _prepared_delete_session_ids(daemon_preview)
    daemon_preview_refs = _daemon_preview_refs(daemon_preview)
    if daemon_preview_refs is None:
        raise click.ClickException("daemon returned an invalid delete preview")
    if not force:
        click.echo(f"About to delete {len(prepared_session_ids)} session(s):", err=True)
        for session_id in prepared_session_ids[:5]:
            click.echo(f"  - {session_id}", err=True)
        if len(prepared_session_ids) > 5:
            click.echo(f"  ... and {len(prepared_session_ids) - 5} more", err=True)
        try:
            proceed = env.ui.confirm("Proceed?", default=False)
        except (KeyboardInterrupt, click.Abort):
            proceed = False
            interrupted = True
        else:
            interrupted = False
        if not proceed:
            _cancel_delete_preview(config, daemon_preview_refs, count=count, interrupted=interrupted)
            return
    try:
        daemon_authorization = _submit_mutation_operation(
            config,
            "mutation.session.delete.authorize",
            {"preview_refs": list(daemon_preview_refs)},
        )
    except KeyboardInterrupt:
        # The confirmed write never reached the socket, so the preview is
        # still the daemon's to release and the operator gets a cancelled
        # receipt instead of a half-rendered success.
        _cancel_delete_preview(config, daemon_preview_refs, count=count, interrupted=True)
        return
    except OperationKernelError as exc:
        raise _delete_refusal(exc, "authorize") from exc
    daemon_authorization_refs = _delete_authorization_refs(daemon_authorization, len(daemon_preview_refs))
    try:
        daemon_payload = _submit_mutation_operation(
            config,
            "mutation.session.delete.execute",
            {"authorization_refs": list(daemon_authorization_refs)},
        )
    except OperationKernelError as exc:
        raise _delete_refusal(exc, "execute") from exc
    deleted = _object_int(daemon_payload.get("affected_count"))
    # ``session_count`` = matched, ``affected_count`` = sessions actually deleted.
    click.echo(
        MutationResultPayload(
            status="deleted" if deleted else "ok",
            operation="delete",
            session_count=count,
            affected_count=deleted,
        ).to_json(exclude_none=True)
    )


def _mutation_refusal(exc: Exception, operation: str) -> click.ClickException:
    """Translate a typed operation failure into the mutation route's message."""
    from polylogue.cli.operation_kernel import (
        OperationFailedError,
        OperationIndeterminateError,
        OperationUnavailableError,
    )

    if isinstance(exc, OperationIndeterminateError):
        return click.ClickException(
            f"{operation} outcome is indeterminate after the daemon accepted the request; "
            "do not retry offline, inspect daemon audit state before retrying"
        )
    if isinstance(exc, OperationUnavailableError):
        # The daemon is the standard, not an optional addon: there is no local
        # fallback to offer, so the refusal names the one action that makes the
        # command work instead of leaving the operator to guess.
        return click.ClickException(
            f"daemon is unavailable; it must execute {operation}. Start one with `polylogued run` "
            "(and drop --no-daemon / POLYLOGUE_NO_DAEMON if either is set)."
        )
    if isinstance(exc, OperationFailedError):
        return click.ClickException(f"daemon refused {operation} ({exc.code}): {exc.detail}")
    return click.ClickException(f"{operation} failed: {exc}")


def submit_cli_mutation(env: AppEnv, operation: str, payload: dict[str, object]) -> dict[str, object]:
    """Run one declared write for a CLI verb, or refuse in the route's voice.

    The single entry point every non-query CLI mutation uses, so that "the CLI
    never holds write authority" is one fact about one function rather than a
    property re-established per command. A caller that wants a receipt reads
    the returned result; one that only needs the write to have happened ignores
    it and lets the typed refusal propagate.
    """
    from polylogue.cli.operation_kernel import OperationKernelError

    try:
        return _submit_mutation_operation(load_effective_config(env), operation, payload)
    except OperationKernelError as exc:
        raise _mutation_refusal(exc, operation) from exc


def _delete_refusal(exc: Exception, stage: str) -> click.ClickException:
    """Translate a typed operation failure into the delete route's message."""
    from polylogue.cli.operation_kernel import (
        OperationFailedError,
        OperationIndeterminateError,
        OperationUnavailableError,
    )

    if isinstance(exc, OperationIndeterminateError):
        return click.ClickException(
            f"delete {stage} outcome is indeterminate after the daemon accepted the request; "
            "do not retry offline, inspect daemon audit state before retrying"
        )
    if isinstance(exc, OperationUnavailableError):
        return click.ClickException(f"daemon is unavailable; it must {stage} the delete")
    if isinstance(exc, OperationFailedError):
        if exc.code == "delete_partially_applied":
            return click.ClickException(
                f"delete partially applied: {exc.detail}; "
                f"completed_chunks={exc.data.get('completed_chunks')}; "
                f"affected_count={exc.data.get('affected_count')}"
            )
        return click.ClickException(f"daemon refused delete {stage} ({exc.code}): {exc.detail}")
    return click.ClickException(f"delete {stage} failed: {exc}")


def _delete_authorization_refs(daemon_authorization: dict[str, object], expected: int) -> tuple[str, ...]:
    """Read authenticated durable references; a count mismatch is not authorization."""

    tokens = daemon_authorization.get("authorization_refs")
    if isinstance(tokens, list) and all(isinstance(token, str) and token for token in tokens):
        issued = tuple(tokens)
    else:
        token = daemon_authorization.get("authorization_ref")
        if not isinstance(token, str) or not token:
            raise click.ClickException("daemon returned an invalid delete authorization")
        issued = (token,)
    if len(issued) != expected:
        raise click.ClickException("daemon returned an invalid delete authorization")
    return issued


def _cancel_delete_preview(
    config: Config,
    preview_refs: tuple[str, ...],
    *,
    count: int,
    interrupted: bool,
) -> None:
    """Release an unconfirmed preview through the operation's cancel route."""
    from polylogue.cli.operation_kernel import OperationKernelError
    from polylogue.surfaces.payloads import MutationResultPayload

    try:
        cancellation = _submit_mutation_operation(
            config,
            "mutation.session.delete.cancel",
            {"preview_refs": list(preview_refs)},
        )
    except OperationKernelError as exc:
        raise click.ClickException(f"daemon did not cancel the delete preview: {exc}") from exc
    # An acknowledgement that names other previews, or none, is not evidence
    # that this delete was cancelled.
    acknowledged_refs = _daemon_preview_refs(cancellation)
    if (
        cancellation.get("status") != "cancelled"
        or acknowledged_refs is None
        or set(acknowledged_refs) != set(preview_refs)
    ):
        raise click.ClickException("daemon returned an invalid delete cancellation acknowledgement")
    click.echo(
        MutationResultPayload(status="aborted", operation="delete", session_count=count, affected_count=0).to_json(
            exclude_none=True
        )
    )
    if interrupted:
        raise click.exceptions.Exit(_CANCELLED_EXIT_CODE)


def _prepared_delete_session_ids(
    daemon_preview: dict[str, object],
) -> tuple[str, ...]:
    """Use the daemon's canonical preview, never a client-side substitution."""

    raw_session_ids = daemon_preview.get("session_ids")
    if not isinstance(raw_session_ids, list) or any(
        not isinstance(value, str) or not value for value in raw_session_ids
    ):
        raise click.ClickException("daemon returned an invalid delete preview")
    session_ids = tuple(raw_session_ids)
    if not session_ids or len(set(session_ids)) != len(session_ids):
        raise click.ClickException("daemon returned a non-canonical delete preview")
    return session_ids


def _search_miss_diagnostics(
    env: AppEnv,
    spec: SessionQuerySpec,
    *,
    why: bool,
) -> QueryMissDiagnosticsPayload | None:
    """Best-effort zero-hit diagnosis for the root query search path.

    **The one in-process read left in this module.** Every other root-query
    read is a declared operation dispatched through the kernel; no operation
    declares a query-miss diagnosis, so this still bridges into the async
    facade to share the clause-drop/relaxation/FTS-disagreement diagnosis with
    the TUI/daemon/API surfaces (polylogue-jnj.12) rather than becoming a
    fourth reimplementation. Declaring it is follow-up work, not a reason to
    drop the explanation a zero-hit page owes the operator.

    Degrades to ``None`` on any failure -- a failed diagnosis must never turn
    a legitimate zero-hit result into an error.
    """
    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.surfaces.payloads import QueryMissDiagnosticsPayload

    try:
        raw_diagnostics = run_coroutine_sync(env.polylogue.diagnose_query_miss(spec, full=why))
    except Exception:
        logger.exception("_search_miss_diagnostics: diagnose_query_miss failed")
        return None
    return QueryMissDiagnosticsPayload.from_diagnostics(raw_diagnostics)


def _list_no_results_envelope(
    *,
    origin: str | None,
    limit: int,
    offset: int,
    root: bool | None,
) -> dict[str, object]:
    return {
        "mode": "list",
        "origin": origin,
        "items": [],
        "total": 0,
        "total_unit": session_count_unit_label(root),
        "limit": limit,
        "offset": offset,
        "next_offset": None,
        "next_cursor": None,
        "outcome": decide_outcome(matched=0).to_dict(),
    }


def _emit_list_miss_if_field_syntax(
    env: AppEnv,
    *,
    items: Sequence[object],
    total: int | None,
    raw_query: str,
    compiled_spec: SessionQuerySpec,
    why: bool,
    output_format: str,
    origin: str | None,
    limit: int,
    offset: int,
    root: bool | None,
    typo_hint: str | None,
) -> None:
    """Route a zero-row list-mode page through the shared miss diagnostics.

    List/browse mode is deliberately silent-success (exit 0, no message) when
    the request carried no query expression at all -- "show me everything,
    there is nothing" per the ``mode: list`` contract just above. But a
    field-syntax-only query expression (e.g. ``repo:polylogue``, ``since:7d``
    with no bare text term) compiles down to an empty ``query`` string and
    used to fall through to this same silent-success rendering: zero bytes of
    stdout, exit 0, no diagnostics, even with ``--why`` -- indistinguishable
    from the query being silently misrouted (polylogue-hlww). ``raw_query``
    being non-empty here means the user typed *some* query expression (it
    just had no free-text residue after field-syntax compilation), so treat
    that case as a miss like a lexical search: the shared
    ``Why this may have missed:`` block, exit 2.

    ``items`` alone cannot distinguish a genuine zero-match query from an
    exhausted page (a nonzero ``--offset`` past the last row of a query that
    DID match): both render an empty page. ``total`` is the unpaginated match
    count when known (``None`` for lanes without a cheap count primitive,
    e.g. vector/hybrid retrieval) -- a page with no items but a positive
    total is valid empty pagination, not a miss, and must not fabricate
    ``total: 0`` or exit 2.
    """
    if items or (total is not None and total > 0) or not raw_query.strip():
        return
    emit_empty_page(
        _list_no_results_envelope(origin=origin, limit=limit, offset=offset, root=root),
        output_format=output_format,
        typo_hint=typo_hint,
        diagnostics=_search_miss_diagnostics(env, compiled_spec, why=why),
    )


def _emit_open_no_results(*, output_format: str, origin: str | None) -> NoReturn:
    if output_format == "json":
        error_no_results("No sessions matched.").emit(exit_code=EMPTY_EXIT_CODE)
    emit_empty_page(
        {
            "mode": "open",
            "origin": origin,
            "items": [],
            "total": 0,
        },
        output_format=output_format,
    )


def _unit_source_display_name(source: QueryUnitSource) -> str:
    descriptor = query_unit_descriptor(source.unit)
    if descriptor is None:
        return source.unit
    return descriptor.plural_source


def _emit_unit_no_results(envelope: dict[str, object], *, unit: str, output_format: str) -> NoReturn:
    empty = {**envelope, "items": [], "total": 0}
    outcome = OutcomeEnvelope.model_validate(empty["outcome"])
    if output_format == "json":
        click.echo(json.dumps(empty, indent=2, sort_keys=True))
    elif output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(empty, sort_keys=False, allow_unicode=True), nl=False)
    elif output_format in {"ndjson", "csv"}:
        pass
    else:
        click.echo(f"No {unit}s matched.")
    raise SystemExit(outcome_exit_code(outcome))


def _message_query_line(item: dict[str, object]) -> str:
    return f"{item['message_id']} [{item['role']}] {bound_display_text(item.get('text'))}"


def _action_query_line(item: dict[str, object]) -> str:
    action = item.get("semantic_type") or item.get("tool_name") or "action"
    detail = item.get("tool_path") or item.get("tool_command") or item.get("output_text") or ""
    return f"{item['tool_use_block_id']} [{action}] {bound_display_text(detail)}"


def _block_query_line(item: dict[str, object]) -> str:
    detail = item.get("text") or item.get("tool_path") or item.get("tool_command") or ""
    return f"{item['block_id']} [{item['block_type']}] {bound_display_text(detail)}"


def _file_query_line(item: dict[str, object]) -> str:
    detail = f"actions={item.get('action_count', 0)}"
    first_ref = item.get("first_tool_use_block_id") or item.get("first_message_id") or item.get("session_id")
    if first_ref:
        detail = f"{detail} first={first_ref}"
    return f"{item['path']} [{item['origin']}] {bound_display_text(detail)}"


def _assertion_query_line(item: dict[str, object]) -> str:
    detail = item.get("body_text") or item.get("key") or item.get("value") or item.get("target_ref") or ""
    return f"{item['assertion_id']} [{item['kind']}/{item['status']}] {bound_display_text(detail)}"


def _aggregate_query_line(item: dict[str, object]) -> str:
    group_by = item.get("group_by") or "all"
    group_key = item.get("group_key") or "all"
    return f"{group_by}={group_key} count={item['count']}"


def _run_query_line(item: dict[str, object]) -> str:
    detail_parts = [str(part) for part in (item.get("agent_ref"), item.get("title")) if part]
    detail = " ".join(detail_parts) or item.get("run_ref") or ""
    return f"{item['run_ref']} [{item['role']}/{item['status']}] {bound_display_text(detail)}"


def _observed_event_query_line(item: dict[str, object]) -> str:
    detail = item.get("summary") or item.get("subject_ref") or item.get("event_ref") or ""
    return f"{item['event_ref']} [{item['kind']}/{item['delivery_state']}] {bound_display_text(detail)}"


def _context_snapshot_query_line(item: dict[str, object]) -> str:
    detail = item.get("metadata") or item.get("segment_refs") or item.get("evidence_refs") or ""
    return f"{item['snapshot_ref']} [{item['boundary']}/{item['inheritance_mode']}] {bound_display_text(detail)}"


def _delegation_query_line(item: dict[str, object]) -> str:
    detail = item.get("instruction_preview") or item.get("artifact_preview") or item.get("child_session_id") or ""
    return f"{item['delegation_ref']} [{item['mapping_state']}/{item['result_status']}] {bound_display_text(detail)}"


_QUERY_UNIT_TEXT_LINES: dict[str, _QueryUnitTextLine] = {
    "message": _message_query_line,
    "action": _action_query_line,
    "block": _block_query_line,
    "file": _file_query_line,
    "assertion": _assertion_query_line,
    "run": _run_query_line,
    "observed-event": _observed_event_query_line,
    "context-snapshot": _context_snapshot_query_line,
    "delegation": _delegation_query_line,
}


def _query_unit_text_line(unit: str) -> _QueryUnitTextLine:
    descriptor = query_unit_descriptor(unit)
    renderer = descriptor.cli_plain_renderer if descriptor else None
    if renderer is None:
        raise click.UsageError(f"Unsupported query unit: {unit}")
    try:
        return _QUERY_UNIT_TEXT_LINES[renderer]
    except KeyError as exc:
        raise click.UsageError(f"Unsupported query unit renderer: {renderer}") from exc


def _fail(message: str) -> NoReturn:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(1)


__all__ = ["execute_archive_query", "execute_delete_by_session_ids"]
