"""Sync event types and handlers.

Decouples event dispatch (notifications, webhooks, shell commands) from
the CLI layer so that MCP servers, daemons, and batch jobs can also
react to sync results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from polylogue.storage.store import RunResult

LOGGER = logging.getLogger(__name__)


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


class WebhookHandler:
    """POST to webhook URL on sync events."""

    __slots__ = ("_url",)

    def __init__(self, url: str) -> None:
        self._url = url

    def on_sync(self, event: SyncEvent) -> None:
        if event.new_conversations <= 0:
            return
        try:
            import json
            import urllib.request

            data = json.dumps({"event": "sync", "new_conversations": event.new_conversations}).encode()
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            LOGGER.warning("Webhook failed for %s: %s", self._url, exc)


class ExecHandler:
    """Run shell command on sync events."""

    __slots__ = ("_command",)

    def __init__(self, command: str) -> None:
        self._command = command

    def on_sync(self, event: SyncEvent) -> None:
        if event.new_conversations <= 0:
            return
        import os
        import subprocess

        env = os.environ.copy()
        env["POLYLOGUE_NEW_COUNT"] = str(event.new_conversations)
        subprocess.run(self._command, shell=True, env=env, check=False)


__all__ = [
    "SyncEvent",
    "SyncEventHandler",
    "CompositeSyncHandler",
    "NotificationHandler",
    "WebhookHandler",
    "ExecHandler",
]
