from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .storage.db import open_connection
from .ingestion.drive_client import default_credentials_path, default_token_path
from .storage.index import index_status

LOGGER = logging.getLogger(__name__)
HEALTH_TTL_SECONDS = 600


@dataclass
class HealthCheck:
    name: str
    status: str
    detail: str


def _cache_path(archive_root: Path) -> Path:
    return archive_root / "health.json"


def _load_cached(archive_root: Path) -> dict | None:
    """Load health check cache from disk.

    Returns the cached dict if present and valid, or None if:
    - Cache file doesn't exist (cache miss)
    - Cache file is corrupted (invalid JSON)
    - Cache file contains non-dict JSON
    - File can't be read (permissions, encoding, etc.)

    All errors are logged as warnings to aid debugging.
    """
    path = _cache_path(archive_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Health cache corrupted (invalid JSON) at %s: %s", path, exc)
        return None
    except UnicodeDecodeError as exc:
        LOGGER.warning("Health cache has encoding error at %s: %s", path, exc)
        return None
    except PermissionError as exc:
        LOGGER.warning("Cannot read health cache (permission denied) at %s: %s", path, exc)
        return None
    except OSError as exc:
        LOGGER.warning("Error reading health cache at %s: %s", path, exc)
        return None

    # Validate it's actually a dict, not just valid JSON
    if not isinstance(payload, dict):
        LOGGER.warning("Health cache contains non-dict JSON (got %s) at %s, treating as cache miss",
                      type(payload).__name__, path)
        return None

    return payload


def _write_cache(archive_root: Path, payload: dict) -> None:
    """Write health check cache to disk.

    Creates parent directories as needed.
    Logs a warning on failure but does not raise (cache write is not critical).
    """
    path = _cache_path(archive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Failed to write health cache to %s: %s", path, exc)


def run_health(config: Config) -> dict:
    checks: list[HealthCheck] = []

    checks.append(HealthCheck("config", "ok", f"Loaded {config.path}"))

    if config.archive_root.exists():
        checks.append(HealthCheck("archive_root", "ok", str(config.archive_root)))
    else:
        checks.append(HealthCheck("archive_root", "warning", f"Missing {config.archive_root}"))

    if config.render_root.exists():
        checks.append(HealthCheck("render_root", "ok", str(config.render_root)))
    else:
        checks.append(HealthCheck("render_root", "warning", f"Missing {config.render_root}"))

    try:
        with open_connection(None):
            checks.append(HealthCheck("database", "ok", "DB reachable"))
    except Exception as exc:
        checks.append(HealthCheck("database", "error", f"DB error: {exc}"))

    idx = index_status()
    if idx["exists"]:
        checks.append(HealthCheck("index", "ok", f"messages indexed: {idx['count']}"))
    else:
        checks.append(HealthCheck("index", "warning", "index not built"))

    for source in config.sources:
        if source.folder:
            cred_path = (
                Path(os.environ.get("POLYLOGUE_CREDENTIAL_PATH", "")).expanduser()
                if os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
                else default_credentials_path()
            )
            token_path = (
                Path(os.environ.get("POLYLOGUE_TOKEN_PATH", "")).expanduser()
                if os.environ.get("POLYLOGUE_TOKEN_PATH")
                else default_token_path()
            )
            cred_status = "ok" if cred_path.exists() else "warning"
            token_status = "ok" if token_path.exists() else "warning"
            checks.append(
                HealthCheck(
                    f"source:{source.name}",
                    cred_status,
                    f"drive folder '{source.folder}' credentials: {cred_path}",
                )
            )
            checks.append(
                HealthCheck(
                    f"source:{source.name}:token",
                    token_status,
                    f"drive token: {token_path}",
                )
            )
        else:
            if source.path and source.path.exists():
                checks.append(HealthCheck(f"source:{source.name}", "ok", str(source.path)))
            else:
                checks.append(HealthCheck(f"source:{source.name}", "warning", f"missing path: {source.path}"))

    payload = {
        "timestamp": int(time.time()),
        "checks": [check.__dict__ for check in checks],
    }
    _write_cache(config.archive_root, payload)
    return payload


def get_health(config: Config) -> dict:
    cached = _load_cached(config.archive_root)
    now = int(time.time())
    if cached:
        timestamp = cached.get("timestamp")
        if isinstance(timestamp, int) and (now - timestamp) < HEALTH_TTL_SECONDS:
            cached["cached"] = True
            cached["age_seconds"] = now - timestamp
            return cached
    payload = run_health(config)
    payload["cached"] = False
    payload["age_seconds"] = 0
    return payload


def cached_health_summary(archive_root: Path) -> str:
    cached = _load_cached(archive_root)
    if not cached:
        return "not run"
    timestamp = cached.get("timestamp")
    if not isinstance(timestamp, int):
        return "unknown"
    age = int(time.time()) - timestamp
    checks = cached.get("checks")
    if isinstance(checks, list):
        counts: dict[str, int] = {}
        for check in checks:
            if not isinstance(check, dict):
                continue
            status = check.get("status")
            if isinstance(status, str):
                counts[status] = counts.get(status, 0) + 1
        if counts:
            parts = []
            for key in ("ok", "warning", "error"):
                if key in counts:
                    parts.append(f"{key}={counts[key]}")
            extras = [f"{key}={value}" for key, value in counts.items() if key not in {"ok", "warning", "error"}]
            summary = ", ".join(parts + extras)
            return f"cached {age}s ago ({summary})"
    return f"cached {age}s ago"


__all__ = ["get_health", "run_health", "HealthCheck", "HEALTH_TTL_SECONDS", "cached_health_summary"]
