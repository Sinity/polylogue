"""Shared daemon status payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig, receiver_status_payload
from polylogue.core.json import JSONDocument, json_document
from polylogue.sources.live import WatchSource
from polylogue.sources.live.watcher import default_sources


def live_source_status_payload(sources: tuple[WatchSource, ...]) -> JSONDocument:
    """Return status for configured live-ingest roots."""
    items = [
        {
            "name": source.name,
            "root": str(source.root),
            "exists": source.exists(),
        }
        for source in sources
    ]
    existing = sum(1 for item in items if item["exists"])
    return json_document(
        {
            "ok": True,
            "source_count": len(items),
            "existing_source_count": existing,
            "sources": items,
        }
    )


def browser_capture_status_payload(spool_path: Path | None = None) -> JSONDocument:
    """Return status for the browser-capture receiver component."""
    config = BrowserCaptureReceiverConfig.default()
    if spool_path is not None:
        config = BrowserCaptureReceiverConfig(spool_path=spool_path, allowed_origins=config.allowed_origins)
    return json_document(receiver_status_payload(config))


def daemon_status_payload(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_spool_path: Path | None = None,
) -> JSONDocument:
    """Return the local daemon component status payload."""
    watch_sources = sources if sources is not None else default_sources()
    return json_document(
        {
            "ok": True,
            "daemon": "polylogued",
            "checked_at": datetime.now(UTC).isoformat(),
            "live": live_source_status_payload(watch_sources),
            "browser_capture": browser_capture_status_payload(browser_capture_spool_path),
        }
    )


__all__ = [
    "browser_capture_status_payload",
    "daemon_status_payload",
    "live_source_status_payload",
]
