"""Shared loopback-host detection.

Single source of truth used by the daemon HTTP API and the browser-capture
receiver to decide whether a bind/client/Origin host is loopback. Aligned
with RFC 5735: the entire ``127.0.0.0/8`` block is loopback for IPv4, plus
``::1`` for IPv6 and the ``localhost`` name.
"""

from __future__ import annotations

import ipaddress

LOOPBACK_HOST_NAMES: frozenset[str] = frozenset({"localhost"})


def is_loopback_host(host: str) -> bool:
    """Return True if ``host`` is a loopback bind address or name.

    Accepts the literal name ``localhost``; otherwise defers to
    :class:`ipaddress.ip_address` so the full ``127.0.0.0/8`` block and
    ``::1`` resolve correctly. Empty or malformed inputs return False.
    """
    if not host:
        return False
    if host in LOOPBACK_HOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


LOOPBACK_ORIGIN_PREFIXES: tuple[str, ...] = (
    "http://localhost:",
    "https://localhost:",
    "http://[::1]:",
    "https://[::1]:",
)


def is_loopback_origin(origin: str) -> bool:
    """Return True if a browser ``Origin`` header points at a loopback host.

    Covers the full ``127.0.0.0/8`` block as well as ``::1`` (bracketed per
    RFC 3986) and the ``localhost`` name. Both ``http`` and ``https`` are
    accepted; anything else returns False.
    """
    if not origin:
        return False
    for scheme in ("http://", "https://"):
        if not origin.startswith(scheme):
            continue
        rest = origin[len(scheme) :]
        host = rest.split("/", 1)[0]
        if host.startswith("["):
            end = host.find("]")
            if end == -1:
                return False
            return is_loopback_host(host[1:end])
        host = host.split(":", 1)[0]
        return is_loopback_host(host)
    return False


__all__ = ["LOOPBACK_HOST_NAMES", "LOOPBACK_ORIGIN_PREFIXES", "is_loopback_host", "is_loopback_origin"]
