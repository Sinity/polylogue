"""Shared daemon status payloads."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig, receiver_status_payload
from polylogue.core.json import JSONDocument, json_document
from polylogue.paths import blob_store_root, db_path
from polylogue.sources.live import WatchSource
from polylogue.sources.live.watcher import default_sources

# ---------------------------------------------------------------------------
# Typed sub-models
# ---------------------------------------------------------------------------


class ComponentState(BaseModel):
    watcher: str = "stopped"
    api: str = "stopped"
    browser_capture: str = "stopped"


class SourceLagItem(BaseModel):
    name: str
    root: str
    exists: bool
    file_count: int = 0


class IngestionThroughput(BaseModel):
    messages_per_second: float = 0.0
    files_per_second: float = 0.0


class FTSReadiness(BaseModel):
    messages_ready: bool = False
    action_events_ready: bool = False


class InsightFreshness(BaseModel):
    sessions_with_profiles: int = 0
    total_sessions: int = 0


# ---------------------------------------------------------------------------
# DaemonStatus — typed model consumed by all surfaces
# ---------------------------------------------------------------------------


class DaemonStatus(BaseModel):
    """Typed daemon status consumed by CLI, TUI, web, browser extension, MCP."""

    daemon_liveness: bool = False
    component_state: ComponentState = Field(default_factory=ComponentState)
    source_lag: list[SourceLagItem] = Field(default_factory=list)
    failing_files: list[str] = Field(default_factory=list)
    current_operations: list[dict[str, object]] = Field(default_factory=list)
    reset_queue: list[dict[str, object]] = Field(default_factory=list)
    ingestion_throughput: IngestionThroughput = Field(default_factory=IngestionThroughput)
    db_size_bytes: int = 0
    wal_size_bytes: int = 0
    blob_dir_size_bytes: int = 0
    disk_free_bytes: int = 0
    fts_readiness: FTSReadiness = Field(default_factory=FTSReadiness)
    insight_freshness: InsightFreshness = Field(default_factory=InsightFreshness)
    browser_capture_active: bool = False
    checked_at: str = ""


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


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


def _safe_int(value: object) -> int:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return int(value)
    return 0


def _check_daemon_liveness() -> bool:
    """Check whether the daemon process is running via pidfile."""
    try:
        from polylogue.paths import archive_root

        pidfile = Path(archive_root()) / "daemon.pid"
        if not pidfile.exists():
            return False
        pid = int(pidfile.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def build_daemon_status(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_spool_path: Path | None = None,
) -> DaemonStatus:
    """Build a typed DaemonStatus from durable component state."""
    watch_sources = sources if sources is not None else default_sources()
    db_info = _db_size_info()
    fts = _fts_readiness_info()
    freshness = _insight_freshness_info()

    return DaemonStatus(
        daemon_liveness=_check_daemon_liveness(),
        component_state=ComponentState(
            watcher="running" if watch_sources else "stopped",
            api="running",
            browser_capture="running" if browser_capture_spool_path else "stopped",
        ),
        source_lag=[SourceLagItem(name=s.name, root=str(s.root), exists=s.exists()) for s in watch_sources],
        db_size_bytes=_safe_int(db_info.get("db_size_bytes", 0)),
        wal_size_bytes=_safe_int(db_info.get("wal_size_bytes", 0)),
        blob_dir_size_bytes=_blob_size_info(),
        disk_free_bytes=_safe_int(db_info.get("disk_free_bytes", 0)),
        fts_readiness=FTSReadiness(
            messages_ready=fts.get("messages_ready", False),
            action_events_ready=fts.get("action_events_ready", False),
        ),
        insight_freshness=InsightFreshness(
            sessions_with_profiles=_safe_int(freshness.get("sessions_with_profiles", 0)),
            total_sessions=_safe_int(freshness.get("total_sessions", 0)),
        ),
        browser_capture_active=browser_capture_spool_path is not None,
        checked_at=datetime.now(UTC).isoformat(),
    )


def daemon_status_payload(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_spool_path: Path | None = None,
) -> JSONDocument:
    """Return the local daemon component status payload (backward-compat dict)."""
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

    status = build_daemon_status(
        sources=sources,
        browser_capture_spool_path=browser_capture_spool_path,
    )

    db_info = _db_size_info()
    blob_size = _blob_size_info()
    fts = _fts_readiness_info()

    return json_document(
        {
            "ok": True,
            "daemon": "polylogued",
            "daemon_liveness": status.daemon_liveness,
            "checked_at": status.checked_at,
            "component_state": status.component_state.model_dump(),
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
            "browser_capture_active": status.browser_capture_active,
            "operations": [],
            "last_ingestion_batch": last_ingestion,
            "fts_readiness": fts,
        }
    )


def format_daemon_status_lines(payload: JSONDocument) -> list[str]:
    """Render daemon component status as plain text lines."""
    lines = ["Polylogue daemon"]
    if payload.get("daemon_liveness"):
        lines.append("  Status: running")
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
    "DaemonStatus",
    "build_daemon_status",
    "browser_capture_status_payload",
    "daemon_status_payload",
    "format_daemon_status_lines",
    "live_source_status_payload",
]
