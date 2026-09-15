"""Liveness probe: is a resident daemon actually serving this archive?

The single subtlety here is why the answer is read off the result's
*authority* rather than off the call succeeding.  The declared ``status``
operation is the one read a probe can afford, but it declares
``DaemonFallback.DIRECT_READ``: with no daemon running at all, the dispatch
still succeeds by executing the read in this process.  Reporting that as
"a daemon is reachable" would be a lie, and strictly worse than the HTTP
probe this replaced, which at least failed honestly.

``OperationResult.authority["mode"]`` is the discriminator: ``"daemon"``
when a daemon served the operation, ``"direct"`` when the local reader did
(it is stamped from ``OperationContext.serving_identity``).  Do not
"simplify" this into `try: dispatch(); return True`.
"""

from __future__ import annotations

from typing import Any


def daemon_serving_probe(config: Any) -> tuple[bool, str | None]:
    """Return ``(served_by_daemon, failure_reason)`` without ever raising.

    A probe that explodes is worse than one that says "no", so every failure
    — unavailable daemon, refused operation, unreadable archive — becomes a
    reason string with ``False``.
    """
    try:
        from polylogue.cli.operation_kernel import configured_read_operation

        result = configured_read_operation(config, "status", {})
        mode = result.authority.get("mode")
        if mode == "daemon":
            return True, None
        return False, f"served directly by this process (authority mode {mode!r}); no daemon answered"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


__all__ = ["daemon_serving_probe"]
