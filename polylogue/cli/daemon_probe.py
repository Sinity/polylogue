"""Liveness probe: is a resident daemon actually serving this archive?

The single subtlety here is why the answer is read off the result's
*authority* rather than off the call succeeding.  The declared ``status``
operation is the one read a probe can afford, but it declares
The result's authority is the discriminator, so a successful call proves the
resident daemon answered. Do not simplify this into `try: dispatch(); return True`.
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
        return False, f"unexpected authority mode {mode!r}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


__all__ = ["daemon_serving_probe"]
