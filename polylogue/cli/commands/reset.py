"""Reset command for clearing database and state.

Supports identity-preserving soft-delete: conversations can be tombstoned
rather than hard-deleted, preserving user metadata across reset cycles.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path

import click

from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.paths import (
    blob_store_root,
    cache_home,
    data_home,
    db_path,
    drive_cache_path,
    drive_token_path,
    render_root,
    state_home,
)
from polylogue.storage.sqlite.connection_profile import open_connection

# Schema for the tombstone/trash table.
_TOMBSTONE_DDL = """
CREATE TABLE IF NOT EXISTS conversation_trash (
    conversation_id TEXT PRIMARY KEY,
    tombstoned_at TEXT NOT NULL,
    identity_key TEXT,
    payload_json TEXT
);
"""


def _ensure_trash_table(db: Path) -> None:
    """Ensure the conversation_trash table exists in the archive database."""
    if not db.exists():
        return
    conn = open_connection(db)
    try:
        conn.execute(_TOMBSTONE_DDL)
        conn.commit()
    finally:
        conn.close()


def _tombstone_conversations(db: Path, conversation_ids: list[str]) -> int:
    """Soft-delete conversations by recording trash entries.

    Records tombstone metadata in the ``conversation_trash`` table and
    removes message content. The conversation row itself is preserved so
    that re-imported content can be correlated via ``identity_ledger``.
    """
    if not conversation_ids:
        return 0
    _ensure_trash_table(db)
    conn = open_connection(db)
    try:
        from datetime import UTC, datetime

        ts = datetime.now(UTC).isoformat()
        count = 0
        c = conn.cursor()
        for conv_id in conversation_ids:
            # Look up the identity ledger key to preserve it in the tombstone
            identity_key = None
            try:
                row = c.execute(
                    "SELECT provider || ':' || source || ':' || source_path || ':' || provider_conversation_id || ':' || raw_hash "
                    "FROM identity_ledger WHERE current_conversation_id = ? LIMIT 1",
                    (conv_id,),
                ).fetchone()
                if row:
                    identity_key = row[0]
            except Exception:
                pass
            # Record the tombstone
            c.execute(
                "INSERT OR IGNORE INTO conversation_trash (conversation_id, tombstoned_at, identity_key) VALUES (?, ?, ?)",
                (conv_id, ts, identity_key),
            )
            # Delete associated messages (hard-delete — content lives in blob store)
            c.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            # Remove from FTS
            with contextlib.suppress(Exception):
                c.execute(
                    "DELETE FROM messages_fts WHERE rowid IN (SELECT rowid FROM messages WHERE conversation_id = ?)",
                    (conv_id,),
                )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


@click.command("reset")
@click.option("--database", is_flag=True, help="Delete the SQLite database")
@click.option("--blob", is_flag=True, help="Delete the content-addressed blob store")
@click.option("--assets", is_flag=True, help="Delete archived assets/attachments")
@click.option("--render", is_flag=True, help="Delete rendered conversations (Markdown/HTML)")
@click.option("--cache", is_flag=True, help="Delete search indexes, schemas, and cache")
@click.option("--auth", is_flag=True, help="Delete Google Drive OAuth tokens")
@click.option("--all", "reset_all", is_flag=True, help="Reset everything")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--conversation", "conv_id", default=None, help="Tombstone a specific conversation by ID")
@click.option(
    "--source",
    "source_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Tombstone all conversations from a source path",
)
@click.pass_obj
def reset_command(
    env: AppEnv,
    database: bool,
    blob: bool,
    assets: bool,
    render: bool,
    cache: bool,
    auth: bool,
    reset_all: bool,
    yes: bool,
    conv_id: str | None,
    source_path: Path | None,
) -> None:
    """Reset database, blob store, assets, rendered outputs, or auth state.

    By default, requires explicit flags to specify what to reset.
    Use --all to reset everything.

    \b
    Identity-preserving reset:
      --conversation ID  Tombstone a specific conversation (preserves user metadata)
      --source PATH      Tombstone all conversations from a source path
    """
    if reset_all:
        database = blob = assets = render = cache = auth = True

    # Identity-preserving soft-delete paths
    _db = db_path()
    if conv_id:
        count = _tombstone_conversations(_db, [conv_id])
        env.ui.console.print(f"Tombstoned conversation {conv_id}: {count} row(s) affected.")
        return

    if source_path:
        if not _db.exists():
            env.ui.console.print("Database does not exist; nothing to tombstone.")
            return
        conn = open_connection(_db)
        try:
            rows = conn.execute(
                "SELECT id FROM conversations WHERE source_path LIKE ?",
                (f"{source_path}%",),
            ).fetchall()
            conv_ids = [r[0] for r in rows]
        finally:
            conn.close()
        if not conv_ids:
            env.ui.console.print(f"No conversations found for source {source_path}.")
            return
        count = _tombstone_conversations(_db, conv_ids)
        env.ui.console.print(f"Tombstoned {len(conv_ids)} conversation(s) from {source_path}: {count} row(s) affected.")
        return

    if not (database or blob or assets or render or cache or auth):
        fail(
            "reset",
            "Specify at least one target (e.g., --database, --assets, --render, --cache, --auth) or use --all",
        )

    targets = []
    if database and _db.exists():
        targets.append(("database", _db))
        # Also clean up WAL/SHM files alongside the database.
        for suffix in (".db-wal", ".db-shm"):
            wal_path = _db.with_suffix(suffix)
            if wal_path.exists():
                targets.append((f"database {suffix}", wal_path))
    if blob:
        _blob_root = blob_store_root()
        if _blob_root.exists():
            targets.append(("blob store", _blob_root))
    if assets:
        assets_dir = data_home() / "assets"
        if assets_dir.exists():
            targets.append(("assets", assets_dir))
    if render and render_root().exists():
        targets.append(("render results", render_root()))
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
