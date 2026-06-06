"""Reset command for clearing database and state.

Supports identity-preserving soft-delete: sessions can be tombstoned
rather than hard-deleted, preserving user metadata across reset cycles.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import click

from polylogue.archive.query.path_prefix import escaped_sql_path_prefix_patterns
from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.paths import (
    archive_root,
    blob_store_root,
    cache_home,
    data_home,
    drive_cache_path,
    drive_token_path,
    state_home,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_suppression

_ARCHIVE_DATABASES = (
    ("source database", "source.db"),
    ("index database", "index.db"),
    ("embeddings database", "embeddings.db"),
    ("user database", "user.db"),
    ("ops database", "ops.db"),
)


def _archive_root() -> Path:
    return archive_root()


def _index_db_path() -> Path:
    return _archive_root() / "index.db"


def _source_db_path() -> Path:
    return _archive_root() / "source.db"


def _user_db_path() -> Path:
    return _archive_root() / "user.db"


def _archive_database_targets() -> list[tuple[str, Path]]:
    root = _archive_root()
    targets: list[tuple[str, Path]] = []
    for name, filename in _ARCHIVE_DATABASES:
        path = root / filename
        if path.exists():
            targets.append((name, path))
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(f"{path.name}{suffix}")
            if sidecar.exists():
                targets.append((f"{name} {suffix}", sidecar))
    return targets


def _resolve_archive_session_ids(tokens: list[str]) -> list[str]:
    """Resolve exact or prefix archive session tokens.

    If the archive tier is absent, exact tokens are still valid tombstone
    targets because suppressions live in the user tier and survive re-ingest.
    """

    unique_tokens = list(dict.fromkeys(tokens))
    archive_db = _index_db_path()
    if not archive_db.exists():
        return unique_tokens

    conn = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True)
    try:
        resolved: list[str] = []
        for token in unique_tokens:
            exact = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (token,)).fetchone()
            if exact is not None:
                resolved.append(str(exact[0]))
                continue
            rows = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE session_id LIKE ?
                ORDER BY session_id
                LIMIT 2
                """,
                (f"{token}%",),
            ).fetchall()
            if not rows:
                resolved.append(token)
                continue
            if len(rows) > 1:
                raise click.ClickException(f"session id prefix {token!r} is ambiguous")
            resolved.append(str(rows[0][0]))
        return list(dict.fromkeys(resolved))
    finally:
        conn.close()


def _delete_archive_sessions(session_ids: list[str]) -> int:
    archive_db = _index_db_path()
    if not session_ids or not archive_db.exists():
        return 0
    conn = sqlite3.connect(archive_db)
    conn.execute("PRAGMA foreign_keys = ON")
    deleted = 0
    try:
        with conn:
            for session_id in session_ids:
                cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                deleted += max(int(cursor.rowcount), 0)
        return deleted
    finally:
        conn.close()


def _suppress_archive_sessions(session_ids: list[str], *, reason: str) -> int:
    if not session_ids:
        return 0
    user_db = _user_db_path()
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    try:
        with conn:
            for session_id in session_ids:
                upsert_suppression(conn, session_id=session_id, reason=reason, mode="hide")
        return len(session_ids)
    finally:
        conn.close()


def _tombstone_archive_sessions(session_ids: list[str], *, reason: str) -> tuple[int, int]:
    resolved = _resolve_archive_session_ids(session_ids)
    suppressed = _suppress_archive_sessions(resolved, reason=reason)
    deleted = _delete_archive_sessions(resolved)
    return suppressed, deleted


def _archive_session_ids_from_source(source_path: Path) -> list[str]:
    index_db = _index_db_path()
    source_db = _source_db_path()
    if not index_db.exists() or not source_db.exists():
        return []
    exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(source_path)
    conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    try:
        conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
        rows = conn.execute(
            """
            SELECT s.session_id
            FROM sessions s
            JOIN source.raw_sessions r ON r.raw_id = s.raw_id
            WHERE REPLACE(r.source_path, char(92), '/') = ?
               OR REPLACE(r.source_path, char(92), '/') LIKE ? ESCAPE '\\'
            ORDER BY s.session_id
            """,
            (exact_prefix, child_prefix),
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


@click.command("reset")
@click.option("--database", is_flag=True, help="Delete the SQLite database")
@click.option("--blob", is_flag=True, help="Delete the content-addressed blob store")
@click.option("--assets", is_flag=True, help="Delete archived assets/attachments")
@click.option("--cache", is_flag=True, help="Delete search indexes, schemas, and cache")
@click.option("--auth", is_flag=True, help="Delete Google Drive OAuth tokens")
@click.option("--all", "reset_all", is_flag=True, help="Reset everything")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--session", "conv_id", default=None, help="Tombstone a specific session by ID")
@click.option(
    "--source",
    "source_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Tombstone all sessions from a source path",
)
@click.pass_obj
def reset_command(
    env: AppEnv,
    database: bool,
    blob: bool,
    assets: bool,
    cache: bool,
    auth: bool,
    reset_all: bool,
    yes: bool,
    conv_id: str | None,
    source_path: Path | None,
) -> None:
    """Reset database, blob store, assets, cache, or auth state.

    By default, requires explicit flags to specify what to reset.
    Use --all to reset everything.

    \b
    Identity-preserving reset:
      --session ID  Tombstone a specific session (preserves user metadata)
      --source PATH      Tombstone all sessions from a source path
    """
    if reset_all:
        database = blob = assets = cache = auth = True

    # Identity-preserving soft-delete paths. Archive rows are
    # rebuildable and user suppressions as the durable tombstone.
    if conv_id:
        suppressed, deleted = _tombstone_archive_sessions([conv_id], reason="reset --session")
        env.ui.console.print(
            f"Tombstoned session {conv_id}: {suppressed} suppression(s), {deleted} archive row(s) deleted."
        )
        return

    if source_path:
        session_ids = _archive_session_ids_from_source(source_path)
        if not session_ids:
            env.ui.console.print(f"No sessions found for source {source_path}.")
            return
        suppressed, deleted = _tombstone_archive_sessions(session_ids, reason=f"reset --source {source_path}")
        env.ui.console.print(
            f"Tombstoned {len(session_ids)} session(s) from {source_path}: "
            f"{suppressed} suppression(s), {deleted} archive row(s) deleted."
        )
        return

    if not (database or blob or assets or cache or auth):
        fail(
            "reset",
            "Specify at least one target (e.g., --database, --assets, --cache, --auth) or use --all",
        )

    targets = []
    if database:
        targets.extend(_archive_database_targets())
    if blob:
        _blob_root = blob_store_root()
        if _blob_root.exists():
            targets.append(("blob store", _blob_root))
    if assets:
        assets_dir = data_home() / "assets"
        if assets_dir.exists():
            targets.append(("assets", assets_dir))
    if cache:
        if cache_home().exists():
            targets.append(("cache/indexes", cache_home()))
        schemas_dir = data_home() / "schemas"
        if schemas_dir.exists():
            targets.append(("inferred schemas", schemas_dir))
        _drive_cache = drive_cache_path()
        if _drive_cache.exists():
            targets.append(("drive cache", _drive_cache))
    if auth and drive_token_path().exists():
        targets.append(("OAuth token", drive_token_path()))
    if reset_all:
        last_source = state_home() / "last-source.json"
        if last_source.exists():
            targets.append(("last-source state", last_source))

    if not targets:
        env.ui.console.print("Nothing to reset (no files exist for selected targets).")
        return

    # Show what will be deleted
    lines = [f"  {name}: {path}" for name, path in targets]
    env.ui.summary("Will delete", lines)

    # Confirm unless --yes
    if not yes:
        if env.ui.plain:
            env.ui.console.print("Use --yes to confirm deletion.")
            return
        if not env.ui.confirm("Delete these files/directories?", default=False):
            env.ui.console.print("Reset cancelled.")
            return

    # Perform deletion
    deleted = 0
    for name, path in targets:
        try:
            if path.is_file():
                path.unlink()
                deleted += 1
                env.ui.console.print(f"  Deleted {name}: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                deleted += 1
                env.ui.console.print(f"  Deleted {name}: {path}")
        except OSError as exc:
            env.ui.console.print(f"  Failed to delete {name}: {exc}")

    env.ui.console.print(f"\nReset complete: {deleted} item(s) deleted.")


__all__ = ["reset_command"]
