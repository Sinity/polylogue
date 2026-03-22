"""Runtime/environment health checks."""

from __future__ import annotations

import os

from .config import Config
from .health_models import HealthCheck, HealthReport, VerifyStatus
from .storage.backends.connection import connection_context, open_connection


def run_runtime_health(config: Config) -> HealthReport:
    """Run runtime environment health checks."""
    checks: list[HealthCheck] = []

    db = config.db_path
    if db.exists():
        try:
            with open(db, "a"):
                pass
            checks.append(HealthCheck("db_writable", VerifyStatus.OK, summary=f"Writable: {db}"))
        except OSError as exc:
            checks.append(HealthCheck("db_writable", VerifyStatus.ERROR, summary=f"Not writable: {exc}"))
    else:
        parent = db.parent
        if parent.exists():
            writable = os.access(parent, os.W_OK)
            if writable:
                checks.append(
                    HealthCheck(
                        "db_writable",
                        VerifyStatus.OK,
                        summary=f"Parent writable, DB will be created: {db}",
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        "db_writable",
                        VerifyStatus.ERROR,
                        summary=f"Parent not writable: {parent}",
                    )
                )
        else:
            checks.append(
                HealthCheck("db_writable", VerifyStatus.WARNING, summary=f"Parent missing: {parent}")
            )

    try:
        from polylogue.storage.backends.schema import SCHEMA_VERSION

        with open_connection(None) as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current == SCHEMA_VERSION:
                checks.append(
                    HealthCheck("schema_version", VerifyStatus.OK, summary=f"v{current} (current)")
                )
            elif current == 0:
                checks.append(
                    HealthCheck("schema_version", VerifyStatus.WARNING, summary="Uninitialized (v0)")
                )
            else:
                checks.append(
                    HealthCheck(
                        "schema_version",
                        VerifyStatus.ERROR,
                        summary=f"v{current} (expected v{SCHEMA_VERSION})",
                    )
                )
    except Exception as exc:
        checks.append(HealthCheck("schema_version", VerifyStatus.ERROR, summary=f"Cannot check: {exc}"))

    try:
        with connection_context(None) as conn:
            fts = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if fts:
                conn.execute("SELECT * FROM messages_fts LIMIT 0")
                checks.append(
                    HealthCheck(
                        "fts_tables",
                        VerifyStatus.OK,
                        summary="FTS5 table present and queryable",
                    )
                )
            else:
                checks.append(
                    HealthCheck("fts_tables", VerifyStatus.WARNING, summary="FTS5 table not found")
                )
    except Exception as exc:
        checks.append(HealthCheck("fts_tables", VerifyStatus.ERROR, summary=f"FTS check failed: {exc}"))

    try:
        import sqlite_vec  # noqa: F401

        checks.append(
            HealthCheck("sqlite_vec", VerifyStatus.OK, summary="sqlite-vec extension available")
        )
    except ImportError:
        checks.append(
            HealthCheck(
                "sqlite_vec",
                VerifyStatus.WARNING,
                summary="sqlite-vec not installed (vector search unavailable)",
            )
        )

    for label, path in [("archive_root", config.archive_root), ("render_root", config.render_root)]:
        if path.exists():
            writable = os.access(path, os.W_OK)
            status = VerifyStatus.OK if writable else VerifyStatus.ERROR
            detail = f"Writable: {path}" if writable else f"Not writable: {path}"
        else:
            parent = path.parent
            if parent.exists() and os.access(parent, os.W_OK):
                status = VerifyStatus.OK
                detail = f"Will be created: {path}"
            else:
                status = VerifyStatus.WARNING
                detail = f"Missing and parent not writable: {path}"
        checks.append(HealthCheck(f"{label}_writable", status, summary=detail))

    from polylogue.paths import config_home

    cfg_home = config_home()
    if cfg_home.exists():
        checks.append(HealthCheck("config_path", VerifyStatus.OK, summary=str(cfg_home)))
    else:
        checks.append(
            HealthCheck("config_path", VerifyStatus.OK, summary=f"Not yet created: {cfg_home}")
        )

    drive_sources = [source for source in config.sources if source.is_drive]
    if drive_sources and config.drive_config:
        from polylogue.sources.drive_client import default_credentials_path, default_token_path

        cred = default_credentials_path(config.drive_config)
        token = default_token_path(config.drive_config)
        if cred.exists():
            checks.append(HealthCheck("drive_credentials", VerifyStatus.OK, summary=str(cred)))
        else:
            checks.append(
                HealthCheck("drive_credentials", VerifyStatus.WARNING, summary=f"Missing: {cred}")
            )
        if token.exists():
            checks.append(HealthCheck("drive_token", VerifyStatus.OK, summary=str(token)))
        else:
            checks.append(
                HealthCheck(
                    "drive_token",
                    VerifyStatus.WARNING,
                    summary=f"Missing (auth required): {token}",
                )
            )

    import shutil
    import sys

    term = os.environ.get("TERM", "unknown")
    cols, rows = shutil.get_terminal_size()
    is_tty = sys.stdout.isatty()
    force_plain = os.environ.get("POLYLOGUE_FORCE_PLAIN", "")

    term_detail = f"TERM={term}, {cols}x{rows}, tty={is_tty}"
    if force_plain:
        term_detail += ", POLYLOGUE_FORCE_PLAIN=1"
    checks.append(HealthCheck("terminal", VerifyStatus.OK, summary=term_detail))

    try:
        import rich  # noqa: F401

        rich_ok = True
    except ImportError:
        rich_ok = False
    try:
        import textual  # noqa: F401

        textual_ok = True
    except ImportError:
        textual_ok = False
    checks.append(
        HealthCheck(
            "ui_libraries",
            VerifyStatus.OK if rich_ok else VerifyStatus.WARNING,
            summary=f"Rich={'yes' if rich_ok else 'no'}, Textual={'yes' if textual_ok else 'no'}",
        )
    )

    vhs_available = shutil.which("vhs") is not None
    checks.append(
        HealthCheck(
            "vhs",
            VerifyStatus.OK if vhs_available else VerifyStatus.WARNING,
            summary="VHS available" if vhs_available else "VHS not found (showcase capture unavailable)",
        )
    )

    return HealthReport(checks=checks)
