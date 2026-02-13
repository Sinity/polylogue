"""Sync event types and handlers.

Decouples event dispatch (notifications, webhooks, shell commands) from
the CLI layer so that MCP servers, daemons, and batch jobs can also
react to sync results.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

from polylogue.lib.log import get_logger

if TYPE_CHECKING:
    from polylogue.storage.store import RunResult

LOGGER = get_logger(__name__)

# Pattern for detecting shell metacharacters that could enable command injection.
# Note: bare ``$VAR`` is harmless with ``shell=False`` (passed literally), but
# command substitution ``$(...)`` and backticks are dangerous even conceptually.
# We reject ``$(`` specifically rather than all ``$`` to allow env var references
# in commands that are informational (the var won't expand with shell=False,
# but users may use wrapper scripts that read env vars).
_UNSAFE_PATTERN = re.compile(r'[;&|`(){}[\]<>!\\]|\$\(')


@dataclass
class SyncEvent:
    """Event emitted after a sync completes."""

    new_conversations: int
    run_result: RunResult


class SyncEventHandler(Protocol):
    """Protocol for objects that react to sync events."""

    def on_sync(self, event: SyncEvent) -> None: ...


class CompositeSyncHandler:
    """Dispatches sync events to multiple handlers in order."""

    __slots__ = ("_handlers",)

    def __init__(self, handlers: list[SyncEventHandler]) -> None:
        self._handlers = handlers

    def on_sync(self, event: SyncEvent) -> None:
        for handler in self._handlers:
            try:
                handler.on_sync(event)
            except Exception:
                LOGGER.exception("Sync handler %s failed", type(handler).__name__)


class NotificationHandler:
    """Desktop notification via ``notify-send`` (or equivalent)."""

    def on_sync(self, event: SyncEvent) -> None:
        if event.new_conversations <= 0:
            return
        try:
            import subprocess

            subprocess.run(
                ["notify-send", "Polylogue", f"Synced {event.new_conversations} new conversation(s)"],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            pass  # notify-send not available


def _validate_webhook_url(url: str) -> None:
    """Validate a webhook URL for SSRF protection.

    Rejects:
    - Non-HTTP(S) schemes
    - Private/loopback/link-local IP addresses
    - Cloud metadata endpoints (169.254.169.254)

    Raises:
        ValueError: If URL targets an unsafe destination
    """
    parsed = urlparse(url)

    # Only allow http and https schemes
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Webhook URL must use http or https scheme, got: {parsed.scheme!r}")

    if not parsed.hostname:
        raise ValueError("Webhook URL must have a hostname")

    hostname = parsed.hostname

    # Resolve hostname to IP addresses and check each one
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve webhook hostname {hostname!r}: {exc}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                f"Webhook URL resolves to private/reserved address {ip} "
                f"(hostname: {hostname!r}). This is blocked for SSRF protection."
            )


class WebhookHandler:
    """POST to webhook URL on sync events.

    Validates the target URL at construction time to prevent SSRF attacks
    against internal/private network endpoints.
    """

    __slots__ = ("_url",)

    def __init__(self, url: str) -> None:
        _validate_webhook_url(url)
        self._url = url

    def on_sync(self, event: SyncEvent) -> None:
        if event.new_conversations <= 0:
            return
        try:
            import json
            import urllib.request

            # Re-validate at dispatch time (DNS could change)
            _validate_webhook_url(self._url)

            data = json.dumps({"event": "sync", "new_conversations": event.new_conversations}).encode()
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)  # noqa: S310
        except ValueError as exc:
            LOGGER.warning("Webhook blocked for %s: %s", self._url, exc)
        except Exception as exc:
            LOGGER.warning("Webhook failed for %s: %s", self._url, exc)


def _validate_exec_command(command: str) -> list[str]:
    """Validate and parse an exec command string into a safe argv list.

    Rejects commands containing shell metacharacters that could enable
    injection attacks. Returns the parsed argument list for use with
    ``subprocess.run(shell=False)``.

    Args:
        command: Command string to validate and parse

    Returns:
        List of command arguments suitable for subprocess.run()

    Raises:
        ValueError: If command is empty or contains unsafe metacharacters
    """
    if not command or not command.strip():
        raise ValueError("Exec command cannot be empty")

    if _UNSAFE_PATTERN.search(command):
        raise ValueError(
            f"Exec command contains unsafe shell metacharacters: {command!r}. "
            "Use a simple command without shell operators like ;, &, |, $, backticks, etc."
        )

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Cannot parse exec command {command!r}: {exc}") from exc

    if not argv:
        raise ValueError("Exec command parsed to empty argument list")

    return argv


class ExecHandler:
    """Run a command on sync events.

    The command is validated at construction time to reject shell
    metacharacters, then parsed via ``shlex.split()`` and run with
    ``shell=False`` to prevent command injection.
    """

    __slots__ = ("_argv",)

    def __init__(self, command: str) -> None:
        self._argv = _validate_exec_command(command)

    def on_sync(self, event: SyncEvent) -> None:
        if event.new_conversations <= 0:
            return
        import os
        import subprocess

        env = os.environ.copy()
        env["POLYLOGUE_NEW_COUNT"] = str(event.new_conversations)
        subprocess.run(self._argv, env=env, check=False)


__all__ = [
    "SyncEvent",
    "SyncEventHandler",
    "CompositeSyncHandler",
    "NotificationHandler",
    "WebhookHandler",
    "ExecHandler",
    "_validate_webhook_url",
    "_validate_exec_command",
]
