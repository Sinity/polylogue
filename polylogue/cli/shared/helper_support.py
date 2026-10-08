"""Shared CLI helper primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config

if TYPE_CHECKING:
    from polylogue.cli.operation_kernel import OperationIndeterminateError


class DaemonRequiredError(click.ClickException):
    """A CLI route refused because no resident daemon is serving the archive.

    A plain :class:`click.ClickException` is rendered by ``machine_main`` as
    ``runtime_error``, which is also what a corrupt tier and an unexpected
    exception produce. That flattening meant a ``--format json`` client had no
    way to recognise "start the daemon", even though the terminal format has
    printed that remedy for as long as the refusal has existed
    (polylogue-re6s3 AC4).

    The terminal message always names ``polylogued run`` verbatim, and carries
    the offending archive so an operator with several archives starts the right
    one. ``operation`` and ``archive_root`` also travel as machine fields, so
    the client does not have to parse them back out of prose.
    """

    #: Consumed by ``polylogue.cli.machine_main`` and matched against
    #: ``machine_errors.DAEMON_REQUIRED``.
    code = "daemon_required"

    def __init__(
        self,
        detail: str,
        *,
        operation: str | None = None,
        archive_root: object = None,
    ) -> None:
        self.operation = operation
        self.archive_root = None if archive_root is None else str(archive_root)
        super().__init__(detail)


class OperationIndeterminateRefusal(click.ClickException):
    """A write the daemon accepted whose outcome never came back.

    ``recovery`` is the operation's declared recovery
    (:class:`~polylogue.operations.daemon_protocol.DaemonRecovery`) and travels
    to machine callers as ``details.recovery`` beside the unresolved
    ``request_id``, so a client settles the write without parsing prose.
    """

    #: Matched against ``machine_errors.OPERATION_INDETERMINATE``.
    code = "operation_indeterminate"

    def __init__(self, detail: str, *, operation: str, recovery: str, request_id: str | None) -> None:
        self.operation = operation
        self.recovery = recovery
        self.request_id = request_id
        super().__init__(detail)


class MutationPartiallyAppliedRefusal(click.ClickException):
    """A batched write that committed some parts before a later part stopped it.

    Never an ordinary refusal: part of the effect is durable, so a client that
    retried the selection would act on a changed archive. The applied counts
    travel to machine callers as typed ``details`` beside the stop reason.
    """

    #: Matched against ``machine_errors.MUTATION_PARTIALLY_APPLIED``.
    code = "mutation_partially_applied"

    def __init__(
        self,
        detail: str,
        *,
        operation: str,
        completed_chunks: int,
        affected_count: int,
        not_attempted: tuple[int, ...],
        not_attempted_count: int | None,
        stop_reason: str | None,
    ) -> None:
        self.operation = operation
        self.completed_chunks = completed_chunks
        self.affected_count = affected_count
        self.not_attempted = not_attempted
        self.not_attempted_count = not_attempted_count
        self.stop_reason = stop_reason
        super().__init__(detail)


def partially_applied_refusal(exc: Exception, operation: str) -> MutationPartiallyAppliedRefusal | None:
    """Return the typed partial-application refusal a failed batch result carries, if any.

    The daemon's batch state reports ``effect`` and the committed part and
    row counts; a failed or cancelled batch with a committed effect applied
    part of its selection.
    """
    from polylogue.cli.operation_kernel import OperationFailedError

    if not isinstance(exc, OperationFailedError):
        return None
    data = exc.data
    completed = data.get("completed_chunks")
    affected = data.get("affected_count")
    completed_chunks = completed if isinstance(completed, int) and not isinstance(completed, bool) else 0
    affected_count = affected if isinstance(affected, int) and not isinstance(affected, bool) else 0
    if data.get("effect") != "committed" and not completed_chunks and not affected_count:
        return None
    raw_not_attempted = data.get("not_attempted")
    not_attempted = (
        tuple(item for item in raw_not_attempted if isinstance(item, int))
        if isinstance(raw_not_attempted, list)
        else ()
    )
    raw_count = data.get("not_attempted_count")
    not_attempted_count = raw_count if type(raw_count) is int and raw_count >= len(not_attempted) else None
    count_text = str(not_attempted_count) if not_attempted_count is not None else "unknown"
    stop_reason = data.get("stop_reason")
    return MutationPartiallyAppliedRefusal(
        f"{operation} partially applied ({exc.code}): {completed_chunks} part(s) committed, "
        f"{affected_count} row(s) affected; {count_text} part(s) not attempted",
        operation=operation,
        completed_chunks=completed_chunks,
        affected_count=affected_count,
        not_attempted=not_attempted,
        not_attempted_count=not_attempted_count,
        stop_reason=str(stop_reason) if stop_reason is not None else None,
    )


def mutation_refusal(exc: Exception, operation: str) -> click.ClickException:
    """Translate one typed operation failure into the CLI's refusal voice.

    Seven copies of this translator existed -- ``archive_query``, ``excise``,
    ``reset``, ``maintenance/_raw_identity``, ``maintenance/_blob_gc``,
    ``maintenance/_blob_integrity`` and ``maintenance/_blob_publications`` --
    and six of them spelled the daemon-absent case as a bare
    ``click.ClickException`` reading "daemon is unavailable; it must execute
    <operation>". That message states the obstacle and withholds the remedy,
    and being untyped it reached ``--format json`` as ``runtime_error``. Both
    defects were per-copy, so fixing one left five (polylogue-re6s3 AC4).

    One function means the refusal is one fact: every route names
    ``polylogued run`` and every route is typed.
    """
    from polylogue.cli.operation_kernel import (
        OperationFailedError,
        OperationIndeterminateError,
        OperationUnavailableError,
    )

    if isinstance(exc, OperationIndeterminateError):
        # Deliberately NOT a daemon-required refusal: the daemon accepted the
        # request, so the write may have happened. Telling the operator to
        # start a daemon and retry would invite a duplicate mutation.
        return indeterminate_refusal(exc, operation)
    if isinstance(exc, OperationUnavailableError):
        return DaemonRequiredError(
            f"daemon is unavailable; it must execute {operation}. Start one with `polylogued run` "
            "(and drop --no-daemon / POLYLOGUE_NO_DAEMON if either is set).",
            operation=operation,
        )
    partial = partially_applied_refusal(exc, operation)
    if partial is not None:
        return partial
    if isinstance(exc, OperationFailedError):
        if exc.code == "selection_empty":
            from polylogue.cli.verb_cardinality import EmptyCardinalityError

            return EmptyCardinalityError(str(exc.detail))
        if exc.code == "selection_ambiguous":
            from polylogue.cli.verb_cardinality import ambiguous_resident_selection

            sample = exc.data.get("session_ids_sample")
            candidates = tuple(value for value in sample if isinstance(value, str)) if isinstance(sample, list) else ()
            return ambiguous_resident_selection(operation, candidates)
        return click.ClickException(f"daemon refused {operation} ({exc.code}): {exc.detail}")
    return click.ClickException(f"{operation} failed: {exc}")


def indeterminate_refusal(exc: OperationIndeterminateError, operation: str) -> OperationIndeterminateRefusal:
    """Name the declared way to settle a write whose outcome never came back.

    The recovery is the operation's own declaration
    (:attr:`~polylogue.operations.daemon_protocol.DaemonOperationSpec.recovery`),
    so a request the daemon recorded durably is settled by replaying its id and
    any other write is never resent blindly.
    """
    from polylogue.operations.daemon_protocol import DaemonRecovery, daemon_operation_spec

    spec = daemon_operation_spec(operation)
    recovery = spec.recovery if spec is not None else DaemonRecovery.RECONCILE
    request = f"request {exc.request_id}" if exc.request_id else "the request"
    if recovery is DaemonRecovery.AWAIT_REQUEST:
        remedy = f"the daemon recorded {request}; resubmitting its id replays the outcome without executing again"
    else:
        remedy = (
            f"do not retry {request}: the daemon settles the attempt from durable evidence, "
            "so read the target's state before issuing a new request"
        )
    return OperationIndeterminateRefusal(
        f"{operation} outcome is indeterminate after the daemon accepted the request; {remedy}",
        operation=operation,
        recovery=recovery.value,
        request_id=exc.request_id,
    )


def fail(command: str, message: str) -> NoReturn:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(f"{command}: {message}")


def load_effective_config(env: AppEnv) -> Config:
    """Return the effective runtime configuration."""
    return env.config


__all__ = [
    "DaemonRequiredError",
    "MutationPartiallyAppliedRefusal",
    "OperationIndeterminateRefusal",
    "fail",
    "partially_applied_refusal",
    "indeterminate_refusal",
    "load_effective_config",
    "mutation_refusal",
]
