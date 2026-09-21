"""One owner for "run a declared read, and say which executor answered".

Every CLI read route reaches a declared operation through the operation
kernel, so *which* executor served it is a transport fact the result's own
authority reports rather than a branch the adapter took.  Naming that fact in
one place is what keeps the ``served-by:`` marker honest across routes: a
route that rendered ``daemon`` for a read the in-process executor answered
would be making a claim the result does not support.

Import-light on purpose.  ``cli/archive_query.py`` is the query adapter and
carries the whole query grammar with it; a read view that only needs to
dispatch and name its executor must not pay that import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.cli.operation_kernel import OperationRequest
    from polylogue.config import Config

__all__ = ["ServedBy", "daemon_route_disabled", "dispatch_read"]


@dataclass(frozen=True, slots=True)
class ServedBy:
    """Which executor answered, as the result's own authority reports it."""

    identity: str
    elapsed_ms: int | None

    def line(self) -> str:
        if self.elapsed_ms is None:
            return self.identity
        transport = "uds, " if self.identity == "daemon" else ""
        return f"{self.identity} ({transport}{self.elapsed_ms}ms)"


def dispatch_read(
    config: Config,
    request: OperationRequest,
    *,
    daemon_disabled: bool = False,
) -> tuple[dict[str, object], ServedBy]:
    """Run one declared read and return its result body plus the daemon timing.

    Every root-query capability goes through here, so "which executor
    answered" is a transport fact recorded in the envelope rather than a
    semantic fork in the adapter.
    """

    from polylogue.cli.operation_kernel import OperationEnvelopeError, dispatch

    result = dispatch(config, request, daemon_disabled=daemon_disabled)
    if not isinstance(result.value, dict):
        raise OperationEnvelopeError(f"{request.operation} returned a non-object result")
    timing = result.envelope.get("timing") if result.envelope is not None else None
    elapsed_ms = timing.get("elapsed_ms") if isinstance(timing, Mapping) else None
    # The executor is named by the result's own authority, not by which branch
    # of this adapter ran: a rendered "daemon" provenance for a read the
    # in-process executor answered would be a claim the result does not
    # support.
    identity = str(result.authority.get("server_identity") or result.authority.get("mode") or "unknown")
    return dict(result.value), ServedBy(identity, elapsed_ms if isinstance(elapsed_ms, int) else None)


def daemon_route_disabled(*, flag: bool = False) -> bool:
    """Whether this invocation has opted out of the daemon read route.

    ``--no-daemon`` is a root CLI flag, but the opt-out is also configurable,
    so the decision is read from both here rather than restated per route: a
    route that only honoured the flag would still reach a running daemon for
    an operator who turned the client off in configuration.
    """

    if flag:
        return True
    from polylogue.config import load_polylogue_config

    settings = load_polylogue_config()
    if settings.no_daemon:
        return True
    return settings.daemon_client_mode == "off"
