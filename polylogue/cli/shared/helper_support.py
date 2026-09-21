"""Shared CLI helper primitives."""

from __future__ import annotations

from typing import NoReturn

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config


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
        return click.ClickException(
            f"{operation} outcome is indeterminate after the daemon accepted the request; "
            "do not retry offline, inspect daemon audit state before retrying"
        )
    if isinstance(exc, OperationUnavailableError):
        return DaemonRequiredError(
            f"daemon is unavailable; it must execute {operation}. Start one with `polylogued run` "
            "(and drop --no-daemon / POLYLOGUE_NO_DAEMON if either is set).",
            operation=operation,
        )
    if isinstance(exc, OperationFailedError):
        return click.ClickException(f"daemon refused {operation} ({exc.code}): {exc.detail}")
    return click.ClickException(f"{operation} failed: {exc}")


def fail(command: str, message: str) -> NoReturn:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(f"{command}: {message}")


def load_effective_config(env: AppEnv) -> Config:
    """Return the effective runtime configuration."""
    return env.config


__all__ = ["DaemonRequiredError", "fail", "load_effective_config", "mutation_refusal"]
