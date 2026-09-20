"""Terminal-outcome rendering: the one place a CLI read decides its exit code.

This module absorbs three previously separate owners of "the query matched
nothing, now what": ``cli/query_feedback.py`` (spec-driven no-results for the
verb surfaces), ``cli/convergence_feedback.py`` (the partial-archive warning
line each of them printed), and ``archive_query._emit_no_results`` (the
envelope-driven empty page of the root query surface).

The convergence *probe* stayed behind in ``cli/convergence_feedback.py`` and is
imported here. It reads ``storage.archive_readiness`` directly, which is
exactly what the Seam B import rule forbids a renderer to do, so folding it in
would have moved a substrate read into the render package to save a module.
Its real home is behind the ``status`` operation, which already reports
``component_readiness.raw_materialization``; until a CLI read consumes that,
this module renders the warning and the probe owns producing it.

One rule governs the exit status, and it is why the three were merged: the
process exit code is computed **only** by
:func:`polylogue.surfaces.outcome.outcome_exit_code` from the terminal outcome
the operation boundary decided. No renderer picks a status out of
``OUTCOME_EXIT_CODES`` by name, and no renderer re-decides an outcome a daemon
supplied -- in particular a degraded zero-row answer must never be flattened
into ``empty`` merely because this layer has no rows to print.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NoReturn

import click

from polylogue.cli.convergence_feedback import convergence_warning_line
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome, outcome_exit_code

if TYPE_CHECKING:
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.cli.shared.types import AppEnv
    from polylogue.surfaces.payloads import QueryMissDiagnosticsPayload

#: Exit status of a zero-row terminal result, derived rather than named: the
#: empty outcome is decided once and mapped once, so this module holds no
#: second copy of the exit table.
EMPTY_EXIT_CODE = outcome_exit_code(decide_outcome(matched=0))


# ---------------------------------------------------------------------------
# Subcommand typo hint (from cli/query_feedback.py)
# ---------------------------------------------------------------------------


def maybe_subcommand_typo_hint(query_terms: Sequence[str]) -> str | None:
    """Return a 'did you mean to run a subcommand?' hint for a single bare token.

    Query-first dispatch is surprising when a user types something like
    ``polylogue analyze-by-origin`` expecting a subcommand. If ``query_terms``
    is a single token and matches a registered subcommand by Levenshtein
    distance, we surface that explicitly in the no-results output. Both the
    archive no-results paths consume this helper so the hint stays identical
    across surfaces.
    """
    terms: tuple[str, ...] = tuple(str(term) for term in query_terms)
    if len(terms) != 1:
        return None
    token = terms[0].strip()
    if not token or " " in token:
        return None

    # Lazy import to keep startup cheap.
    from polylogue.cli.click_app import cli
    from polylogue.cli.parser_diagnostics import looks_like_subcommand_typo

    registered = sorted(cli.commands.keys())
    suggestions = looks_like_subcommand_typo(token, registered)
    if not suggestions:
        return None
    return (
        "Note: query-first dispatch interpreted "
        f"`{token}` as a search query. If you meant a subcommand, try: "
        + ", ".join(f"`polylogue {s}`" for s in suggestions)
    )


def _maybe_subcommand_typo_hint(selection: SessionQuerySpec | None) -> str | None:
    """Adapter for the legacy spec-driven no-results path."""
    if selection is None:
        return None
    return maybe_subcommand_typo_hint(getattr(selection, "query_terms", None) or ())


# ---------------------------------------------------------------------------
# Miss diagnostics
# ---------------------------------------------------------------------------


def diagnostics_dict(
    diagnostics: QueryMissDiagnosticsPayload | Mapping[str, object] | None,
) -> dict[str, object] | None:
    if diagnostics is None:
        return None
    if isinstance(diagnostics, Mapping):
        return dict(diagnostics)
    return diagnostics.model_dump(mode="json")


def print_diagnostics_lines(diagnostics: dict[str, object]) -> None:
    """Render the ``Why this may have missed:`` block shared across query surfaces."""
    reasons = diagnostics.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        return
    click.echo("Why this may have missed:")
    for reason in reasons:
        if not isinstance(reason, Mapping):
            continue
        summary = reason.get("summary")
        if summary:
            click.echo(f"  - {summary}")
        detail = reason.get("detail")
        if detail:
            click.echo(f"    {detail}")


# ---------------------------------------------------------------------------
# Terminal exits
# ---------------------------------------------------------------------------


def exit_for(outcome: OutcomeEnvelope) -> NoReturn:
    """Leave the process with the status this terminal outcome names.

    The single exit-code authority for every CLI read. ``OUTCOME_EXIT_CODES``
    is never indexed by a renderer: the outcome decides, so a surface cannot
    report ``empty`` (2) for a page the operation called ``degraded`` (1).
    """

    raise SystemExit(outcome_exit_code(outcome))


#: Exit status of a read the operator interrupted. 128 + SIGINT, the shell
#: convention, and deliberately outside ``OUTCOME_EXIT_CODES``: cancellation is
#: not a terminal *outcome* of a read, it is the absence of one, so it cannot
#: be spelled as an outcome state without inventing a fifth.
CANCELLED_EXIT_CODE = 130

#: Exit status of a read that failed. Derived from the ``error`` outcome so
#: this module holds no second copy of the table.
FAILED_READ_EXIT_CODE = outcome_exit_code(decide_outcome(matched=0, error="read_failed"))

#: Remedies keyed by the operation-kernel failure code, so the operator is
#: never told only *what* broke.
_READ_FAILURE_REMEDIES: dict[str, str] = {
    # ``polylogue run`` is not a command: the daemon entry point is the
    # ``polylogued`` console script. A remedy naming a verb that does not
    # exist leaves the operator worse off than no remedy at all.
    "daemon_required": "start the daemon with `polylogued run`",
    "daemon_transport_error": "the daemon connection dropped mid-read; check `polylogue ops status` and retry",
    # ``daemon_execution`` frames a deadline as the exception's own type name.
    "QueryTimeoutError": "the read hit its deadline; narrow the selection (--limit/--since) and retry",
    "deadline_exceeded": "narrow the selection (--limit/--since) or raise the deadline, then retry",
    "cancelled": "the read was cancelled before it produced an answer; re-run it to get one",
    "result_too_large": "narrow the window with --limit/--offset, or read a smaller view",
    "stale_generation": "the archive advanced under the read; re-run to read the current generation",
    "invalid_request": "check the option values named above against `--help`",
}


def read_failure_exit_code(exc: BaseException) -> int:
    """Return the exit status of a failed CLI read.

    Every typed read failure used to become a ``click.UsageError``, which exits
    2 -- the *empty* status (:data:`EMPTY_EXIT_CODE`) -- and prints Click's
    usage banner. A dropped daemon connection, a read that hit its deadline and
    a cancelled read were therefore indistinguishable by exit status from
    "matched nothing", and three of them framed a transport failure as a syntax
    mistake (polylogue-jtrtj). ``exit_for`` is the outcome authority for a read
    that *produced* an envelope; this is its counterpart for one that did not,
    and neither indexes ``OUTCOME_EXIT_CODES`` by name.
    """

    from polylogue.cli.operation_kernel import OperationCancelledError

    if isinstance(exc, OperationCancelledError):
        return CANCELLED_EXIT_CODE
    return FAILED_READ_EXIT_CODE


def read_failure_message(exc: BaseException) -> str:
    """Render a failed read as one operator-facing line plus its remedy.

    The call id is included whenever the transport reported one: a daemon-side
    failure is diagnosed from the daemon's log by that id, and omitting it left
    the operator with a message they could not correlate.
    """

    from polylogue.cli.operation_kernel import OperationCancelledError, OperationFailedError

    detail = str(getattr(exc, "detail", None) or exc)
    code = "cancelled" if isinstance(exc, OperationCancelledError) else str(getattr(exc, "code", "") or "")
    parts = [detail]
    request_id = getattr(exc, "request_id", None)
    if request_id:
        parts.append(f"call {request_id}")
    if isinstance(exc, OperationFailedError):
        deadline_ms = exc.data.get("deadline_ms")
        if isinstance(deadline_ms, int):
            parts.append(f"deadline {deadline_ms} ms")
    line = "; ".join(parts)
    remedy = _READ_FAILURE_REMEDIES.get(code)
    return f"{line}\nRemedy: {remedy}" if remedy else line


def exit_for_read_failure(exc: BaseException) -> NoReturn:
    """Emit a failed read's refusal and leave with its own status.

    The single terminal for a read that produced no envelope. Machine callers
    still receive the structured error document: ``machine_main`` re-raises a
    bare ``SystemExit`` unchanged, so emitting it here is what keeps
    ``--format json`` parseable on a transport failure.
    """

    import sys

    from polylogue.cli.shared.machine_errors import error_runtime, extract_command, wants_json

    code = read_failure_exit_code(exc)
    message = read_failure_message(exc)
    argv = list(sys.argv[1:])
    if wants_json(argv):
        error_runtime(message, command=extract_command(argv), exception_type=type(exc).__qualname__).emit(
            exit_code=code
        )
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(code) from exc


def envelope_outcome(envelope: Mapping[str, object]) -> OutcomeEnvelope:
    """Read the terminal outcome an envelope carries, defaulting to empty.

    A supplied outcome is preserved verbatim; only an envelope carrying none at
    all (a locally assembled refusal page) falls back to the zero-row decision.
    """

    raw = envelope.get("outcome")
    return OutcomeEnvelope.model_validate(raw) if raw is not None else decide_outcome(matched=0)


def emit_empty_page(
    envelope: dict[str, object],
    *,
    output_format: str,
    typo_hint: str | None = None,
    diagnostics: QueryMissDiagnosticsPayload | Mapping[str, object] | None = None,
) -> NoReturn:
    """Emit the canonical no-results response and exit on the envelope's outcome.

    The ``empty`` state exits 2, which distinguishes "the query ran and matched
    nothing" from a successful read with results (0) and from an error (1), so
    callers can branch on an empty result set. Machine formats still receive a
    parseable empty envelope; text surfaces get the human-readable message.

    ``diagnostics`` (polylogue-jnj.12) carries the shared miss-diagnosis
    envelope -- either the typed payload or an already-serialized mapping
    forwarded verbatim from a daemon response. Machine formats embed it in the
    empty envelope; text surfaces render a "Why this may have missed:" block,
    verbosity controlled upstream by the caller's ``--why`` request.
    """

    convergence_warning = convergence_warning_line()
    diagnostics_payload = diagnostics_dict(diagnostics)
    # Preserve the operation boundary's decision when a daemon supplied one.
    # In particular, a degraded zero-row answer must not be translated into
    # the local ``empty`` state merely because this adapter has no rows to
    # render.  Local builders still decide ``empty`` when no outcome exists.
    outcome = envelope_outcome(envelope)
    empty = {**envelope, "items": [], "total": 0, "outcome": outcome.to_dict()}
    if convergence_warning is not None:
        empty["archive_converging"] = True
        empty["convergence_warning"] = convergence_warning
    if diagnostics_payload is not None:
        empty["diagnostics"] = diagnostics_payload
    if output_format == "json":
        click.echo(json.dumps(empty, indent=2, sort_keys=True))
    elif output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(empty, sort_keys=False, allow_unicode=True), nl=False)
    elif output_format in {"ndjson", "csv"}:
        pass  # no rows to emit
    else:
        if convergence_warning is not None:
            click.echo(convergence_warning)
        click.echo("No sessions matched.")
        if typo_hint is not None:
            click.echo(typo_hint)
        if diagnostics_payload is not None:
            print_diagnostics_lines(diagnostics_payload)
    exit_for(outcome)


def emit_no_results(
    env: AppEnv,
    *,
    selection: SessionQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
    output_format: str = "text",
    message: str | None = None,
    hint: str | None = None,
    exit_code: int | None = EMPTY_EXIT_CODE,
) -> None:
    """Render a canonical no-results message for the spec-driven verb surfaces.

    ``exit_code`` is the empty-outcome status by default -- derived from
    :func:`outcome_exit_code`, never named -- and ``None`` means "render only,
    the caller owns the exit".
    """
    from polylogue.archive.query.fields import describe_spec_selection_fields
    from polylogue.cli.shared.machine_errors import error_no_results

    filters = describe_spec_selection_fields(selection) if selection is not None else []
    resolved_message = message or ("No sessions matched filters." if filters else "No sessions matched.")
    if output_format == "json":
        error_no_results(
            resolved_message,
            filters=filters or None,
            diagnostics=diagnostics.to_dict() if diagnostics is not None else None,
        ).emit(exit_code=exit_code or EMPTY_EXIT_CODE)

    warning = convergence_warning_line()
    if filters and message is None:
        if warning is not None:
            env.ui.console.print(warning)
        env.ui.console.print("No sessions matched filters:")
        for item in filters:
            env.ui.console.print(f"  {item}")
        env.ui.console.print(hint or "Hint: try broadening your filters or use `read --all` to browse")
    else:
        if warning is not None:
            env.ui.console.print(warning)
        env.ui.console.print(resolved_message)

    typo_hint = _maybe_subcommand_typo_hint(selection)
    if typo_hint is not None:
        env.ui.console.print(typo_hint)

    if diagnostics is not None and diagnostics.reasons:
        env.ui.console.print("Why this may have missed:")
        for line in diagnostics.human_reason_lines():
            env.ui.console.print(f"  - {line}" if not line.startswith("  ") else f"    {line.strip()}")

    if exit_code is not None:
        raise SystemExit(exit_code)


__all__ = [
    "CANCELLED_EXIT_CODE",
    "EMPTY_EXIT_CODE",
    "FAILED_READ_EXIT_CODE",
    "exit_for_read_failure",
    "read_failure_exit_code",
    "read_failure_message",
    "convergence_warning_line",
    "diagnostics_dict",
    "emit_empty_page",
    "emit_no_results",
    "envelope_outcome",
    "exit_for",
    "maybe_subcommand_typo_hint",
    "print_diagnostics_lines",
]
