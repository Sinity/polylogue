"""First-run diagnostics for ``polylogue ops status``.

Probes archive/daemon/source state and returns a single ``StatusDiagnostic``
that surface layers (``status``, ``tutorial``) can render or marshal to JSON.

Every probe is defensive: a failing probe degrades into the next kind in the
ordered chain rather than raising. The output is the actionable text the
operator should see — never a traceback. (#1263)
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DiagnosticKind = Literal[
    "no_archive",
    "schema_mismatch",
    "locked_db",
    "stale_pidfile",
    "no_sources",
    "no_daemon",
    "missing_optional_dep",
    "unknown_db_error",
    "healthy",
]


@dataclass(frozen=True, slots=True)
class StatusDiagnostic:
    """Outcome of a first-run probe sweep.

    ``kind`` chooses the headline shape. ``headline`` is short and
    user-readable; ``detail`` is one or more sentences with the
    actionable next step. ``next_action`` is the single shell command
    we recommend (suitable for JSON ``next_action`` field).
    """

    kind: DiagnosticKind
    headline: str
    detail: str
    next_action: str


def diagnose_first_run(daemon_alive: bool) -> StatusDiagnostic:
    """Return a single ``StatusDiagnostic`` for the current archive state.

    ``daemon_alive`` is the upstream observation — the caller already
    tried to contact the daemon HTTP endpoint and is telling us whether
    that succeeded. We never call into the daemon ourselves.

    Order of checks (first match wins):

    1. archive db missing → ``no_archive``
    2. db open fails with "locked" → ``locked_db``
    3. db schema version outside ``{0, SCHEMA_VERSION}`` → ``schema_mismatch``
    4. db open fails for any other reason → ``unknown_db_error``
    5. stale pidfile present → ``stale_pidfile``
    6. daemon down + config has empty ``roots`` → ``no_sources``
    7. daemon down + sources configured → ``no_daemon``
    8. sqlite_vec import unavailable → ``missing_optional_dep``
    9. otherwise → ``healthy``
    """
    from polylogue.cli.commands.init import starter_config_path
    from polylogue.paths import archive_root, db_path

    db = _active_archive_db(db_path(), archive_root())
    if db is None:
        config_path = starter_config_path()
        if not config_path.exists():
            return StatusDiagnostic(
                kind="no_archive",
                headline="No archive found.",
                detail=(
                    "Run `polylogue init` to detect chat sources and write a "
                    "starter config, then `polylogued run` to start ingestion."
                ),
                next_action="polylogue init && polylogued run",
            )
        return StatusDiagnostic(
            kind="no_archive",
            headline="No archive found.",
            detail="Start the daemon with `polylogued run` to begin ingestion.",
            next_action="polylogued run",
        )

    schema_diag = _probe_schema(db)
    if schema_diag is not None:
        return schema_diag

    pidfile_diag = _probe_stale_pidfile(daemon_alive)
    if pidfile_diag is not None:
        return pidfile_diag

    if not daemon_alive:
        sources_diag = _probe_no_sources()
        if sources_diag is not None:
            return sources_diag
        return StatusDiagnostic(
            kind="no_daemon",
            headline="Daemon not running.",
            detail=("The archive looks healthy but the daemon is down. Start it with `polylogued run`."),
            next_action="polylogued run",
        )

    dep_diag = _probe_missing_optional_dep()
    if dep_diag is not None:
        return dep_diag

    return StatusDiagnostic(
        kind="healthy",
        headline=f"Archive at {archive_root()} is healthy.",
        detail="",
        next_action="",
    )


def _active_archive_db(_db_anchor: Path, root: Path) -> Path | None:
    """Return the archive DB file that should drive first-run diagnostics."""
    archive_db = root / "index.db"
    if archive_db.exists():
        return archive_db
    return None


def _probe_schema(db: Path) -> StatusDiagnostic | None:
    """Open the database and classify schema/locking state.

    Returns ``None`` if the database opens cleanly at the expected
    version. Returns an appropriate diagnostic otherwise. Never raises.
    """
    try:
        from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
        from polylogue.storage.sqlite.schema_bootstrap import schema_version_mismatch_message
    except Exception as exc:  # pragma: no cover - import-time misconfiguration
        return StatusDiagnostic(
            kind="unknown_db_error",
            headline="Archive database could not be inspected.",
            detail=f"Internal error loading schema metadata: {exc}",
            next_action="polylogue ops doctor",
        )

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.5)
        row = conn.execute("PRAGMA user_version").fetchone()
        current_version = int(row[0]) if row else 0
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            return StatusDiagnostic(
                kind="locked_db",
                headline="Archive database is locked.",
                detail=(
                    "Another process (likely the daemon) holds the database. "
                    "Stop or wait for it, then retry. If no daemon is running, "
                    "remove a stale lock with `polylogue ops doctor --repair`."
                ),
                next_action="polylogue ops doctor --repair",
            )
        return StatusDiagnostic(
            kind="unknown_db_error",
            headline="Archive database could not be opened.",
            detail=f"SQLite reported: {exc}. Try `polylogue ops doctor` to diagnose.",
            next_action="polylogue ops doctor",
        )
    except Exception as exc:
        return StatusDiagnostic(
            kind="unknown_db_error",
            headline="Archive database could not be opened.",
            detail=f"Unexpected error: {exc}. Try `polylogue ops doctor` to diagnose.",
            next_action="polylogue ops doctor",
        )
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()

    expected_version = INDEX_SCHEMA_VERSION
    if current_version != 0 and current_version != expected_version:
        detail = (
            f"Archive index database version {current_version} is not the expected "
            f"version {INDEX_SCHEMA_VERSION}. Recreate index.db from source.db."
            if db.name == "index.db"
            else schema_version_mismatch_message(current_version)
        )
        return StatusDiagnostic(
            kind="schema_mismatch",
            headline=f"Schema version {current_version} is not runtime {expected_version}.",
            detail=detail,
            next_action="polylogue ops reset --index && polylogued run",
        )
    return None


def _probe_stale_pidfile(daemon_alive: bool) -> StatusDiagnostic | None:
    """Detect a pidfile pointing at a dead or non-polylogued process.

    Mirrors ``polylogue.daemon.cli._verify_pidfile``: a pidfile is valid
    only when ``/proc/<pid>/cmdline`` contains ``polylogued``. If the
    daemon HTTP probe already succeeded we trust that signal and skip
    the pidfile check.
    """
    if daemon_alive:
        return None
    from polylogue.paths import archive_root

    pidfile = archive_root() / "daemon.pid"
    if not pidfile.exists():
        return None
    try:
        old_pid = int(pidfile.read_text().strip())
    except (ValueError, OSError):
        return StatusDiagnostic(
            kind="stale_pidfile",
            headline="Stale daemon pidfile detected.",
            detail=(
                f"`{pidfile}` exists but its contents are unreadable. "
                "Remove it and start the daemon with `polylogued run`."
            ),
            next_action="polylogued run",
        )

    alive = False
    try:
        os.kill(old_pid, 0)
        alive = True
    except OSError:
        alive = False

    is_polylogued = False
    if alive:
        try:
            cmdline = Path(f"/proc/{old_pid}/cmdline").read_bytes()
            is_polylogued = b"polylogued" in cmdline
        except OSError:
            is_polylogued = False

    if alive and is_polylogued:
        # A polylogued is alive but our HTTP probe failed — likely
        # bind issue or unreachable URL, not a stale pidfile.
        return None

    return StatusDiagnostic(
        kind="stale_pidfile",
        headline="Stale daemon pidfile detected.",
        detail=(
            f"`{pidfile}` references PID {old_pid} which is not a running "
            "polylogued process. Remove the pidfile and run `polylogued run` again."
        ),
        next_action="polylogued run",
    )


def _probe_no_sources() -> StatusDiagnostic | None:
    """Return a ``no_sources`` diagnostic when the config has empty roots.

    The check is best-effort: we only flag the user when we can prove
    there is no source configured. If reading the config raises we
    decline to comment so we don't shadow a genuine daemon-down issue.
    """
    from polylogue.cli.commands.init import starter_config_path

    config_path = starter_config_path()
    if not config_path.exists():
        return StatusDiagnostic(
            kind="no_sources",
            headline="No configured chat sources.",
            detail=(
                f"No `{config_path}` found. Run `polylogue init` to detect chat sources and write a starter config."
            ),
            next_action="polylogue init",
        )
    try:
        body = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if _config_has_empty_roots(body):
        return StatusDiagnostic(
            kind="no_sources",
            headline="No configured chat sources.",
            detail=(
                f"`{config_path}` has `roots = []`. Edit the file to add "
                "source paths, or re-run `polylogue init --force` after "
                "installing a chat tool."
            ),
            next_action="polylogue init --force",
        )
    return None


def _config_has_empty_roots(toml_body: str) -> bool:
    """Lightweight check for ``roots = []`` in the [sources] section.

    Avoids importing a TOML parser at status-probe time. We accept some
    false negatives (commented-out or alternate quoting) in exchange for
    zero exception surface — the worst case is we fail to surface
    ``no_sources`` and the operator gets ``no_daemon`` instead.
    """
    for raw_line in toml_body.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        normalized = stripped.replace(" ", "")
        if normalized.startswith("roots=[]"):
            return True
    return False


def _probe_missing_optional_dep() -> StatusDiagnostic | None:
    """Flag a missing ``sqlite_vec`` install (gates semantic search).

    Only reported when the rest of the system is healthy — the absence
    is not load-bearing but it limits ``polylogue --similar`` and
    ``--semantic``.
    """
    if importlib.util.find_spec("sqlite_vec") is not None:
        return None
    return StatusDiagnostic(
        kind="missing_optional_dep",
        headline="Optional dependency `sqlite_vec` is not installed.",
        detail=(
            "Semantic search (`polylogue --similar`, `polylogue --semantic`) "
            "needs the `sqlite-vec` package. Install it with "
            "`pip install sqlite-vec` (or via your devshell) to enable it. "
            "Lexical search continues to work."
        ),
        next_action="pip install sqlite-vec",
    )


def diagnostic_payload(diag: StatusDiagnostic) -> dict[str, str]:
    """Return a stable JSON-serializable view of a diagnostic."""
    return {
        "kind": diag.kind,
        "headline": diag.headline,
        "detail": diag.detail,
        "next_action": diag.next_action,
    }


__all__ = [
    "DiagnosticKind",
    "StatusDiagnostic",
    "diagnose_first_run",
    "diagnostic_payload",
]
