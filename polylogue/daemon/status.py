"""Shared daemon status payloads."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig, receiver_status_payload
from polylogue.core.json import JSONDocument, json_document
from polylogue.paths import blob_store_root, db_path
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
    cfg_default = BrowserCaptureReceiverConfig.default()
    if spool_path is not None:
        config = BrowserCaptureReceiverConfig(
            spool_path=spool_path,
            allowed_origins=cfg_default.allowed_origins,
            allow_remote=cfg_default.allow_remote,
            auth_token=cfg_default.auth_token,
        )
    else:
        config = cfg_default
    return json_document(receiver_status_payload(config))


def _db_size_info() -> dict[str, object]:
    dbf = db_path()
    info: dict[str, object] = {"db_path": str(dbf)}
    if dbf.exists():
        info["db_size_bytes"] = dbf.stat().st_size
        wal = dbf.with_suffix(".db-wal")
        if wal.exists():
            info["wal_size_bytes"] = wal.stat().st_size
        try:
            st = os.statvfs(str(dbf.parent))
            info["disk_free_bytes"] = st.f_frsize * st.f_bavail
        except OSError:
            pass
    return info


def _blob_size_info() -> int:
    root = blob_store_root()
    if not root.exists():
        return 0
    total = 0
    try:
        for entry in root.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def _fts_readiness_info() -> dict[str, bool]:
    """Check FTS readiness without importing heavy modules."""
    try:
        from polylogue.config import Config
        from polylogue.paths import archive_root, render_root
        from polylogue.readiness import get_readiness

        cfg = Config(archive_root=archive_root(), render_root=render_root(), sources=[])
        report = get_readiness(cfg, deep=False, probe_only=False)
        counts = report.counts()
        return {
            "messages_ready": counts.ok > 0,
            "action_events_ready": counts.ok > 0,
        }
    except Exception:
        return {"messages_ready": False, "action_events_ready": False}


def _insight_freshness_info() -> dict[str, object]:
    """Check insight materialization status."""
    try:
        import asyncio

        from polylogue.api import Polylogue

        async def _check() -> dict[str, object]:
            async with Polylogue() as p:
                status = await p.get_session_insight_status()
                return {
                    "sessions_with_profiles": getattr(status, "profiled_count", 0),
                    "total_sessions": getattr(status, "total_count", 0),
                }

        return asyncio.run(_check())
    except Exception:
        return {"sessions_with_profiles": 0, "total_sessions": 0}


def daemon_status_payload(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_spool_path: Path | None = None,
) -> JSONDocument:
    """Return the local daemon component status payload."""
    watch_sources = sources if sources is not None else default_sources()

    last_ingestion = None
    try:
        from polylogue.daemon.events import get_last_ingestion_batch

        last = get_last_ingestion_batch()
        if last:
            last_ingestion = {
                "ts": last.get("ts"),
                "payload": last.get("payload"),
            }
    except Exception:
        pass

    db_info = _db_size_info()
    blob_size = _blob_size_info()
    fts = _fts_readiness_info()

    return json_document(
        {
            "ok": True,
            "daemon": "polylogued",
            "checked_at": datetime.now(UTC).isoformat(),
            "live": live_source_status_payload(watch_sources),
            "browser_capture": browser_capture_status_payload(browser_capture_spool_path),
            "db_path": db_info.get("db_path"),
            "db_size_bytes": db_info.get("db_size_bytes", 0),
            "wal_size_bytes": db_info.get("wal_size_bytes", 0),
            "blob_dir_size_bytes": blob_size,
            "disk_free_bytes": db_info.get("disk_free_bytes", 0),
            "quick_check_result": "unknown",
            "quick_check_age_s": None,
            "watcher_roots": [str(s.root) for s in watch_sources],
            "browser_capture_active": True,
            "operations": [],
            "last_ingestion_batch": last_ingestion,
            "fts_readiness": fts,
        }
    )


def format_daemon_status_lines(payload: JSONDocument) -> list[str]:
    """Render daemon component status as plain text lines."""
    lines = ["Polylogue daemon"]
    live = payload.get("live")
    if isinstance(live, dict):
        lines.append(f"Live sources: {live.get('existing_source_count', 0)}/{live.get('source_count', 0)} available")
        sources = live.get("sources", [])
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    state = "available" if source.get("exists") else "missing"
                    lines.append(f"  {source.get('name')}: {source.get('root')} ({state})")
    browser_capture = payload.get("browser_capture")
    if isinstance(browser_capture, dict):
        lines.append(f"Browser capture spool: {browser_capture.get('spool_path')}")
        origins = browser_capture.get("allowed_origins", [])
        origin_text = ", ".join(str(item) for item in origins) if isinstance(origins, list) else str(origins)
        lines.append(f"Browser capture origins: {origin_text}")
    return lines


__all__ = [
    "browser_capture_status_payload",
    "daemon_status_payload",
    "format_daemon_status_lines",
    "live_source_status_payload",
]
