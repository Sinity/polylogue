"""Canonical neutral generation whose only stale fact is its runtime identity."""

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def make_empty_managed_index(root: Path, *, stale: bool = True) -> Path:
    initialize_active_archive_root(root)
    return promote_empty_managed_index(root, stale=stale)


def promote_empty_managed_index(root: Path, *, stale: bool = True) -> Path:
    store = IndexGenerationStore.for_archive_root(root)
    parent = store.create(owner_id="synthetic-parent", source_snapshot="empty")
    store.promote(parent)
    path = Path(parent.index_path)
    if stale:
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("UPDATE schema_identity SET identity='synthetic-parent-runtime' WHERE tier='index'")
    return path


def apply_empty_index_transition(root: Path) -> str | None:
    from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
    from polylogue.operations.empty_index_startup import replace_empty_managed_index_on_startup
    from polylogue.operations.reset_safety import archive_tiers_closed
    from polylogue.storage.sqlite.write_lease import write_lease

    owner = acquire_durable_archive_ownership(root, owner_id="synthetic-startup")
    try:
        with archive_tiers_closed(root):
            return replace_empty_managed_index_on_startup(
                root,
                archive_owner=owner,
                write_admission=lambda actor: write_lease(actor, archive_root=root),
            )
    finally:
        owner.release()


def mutate_fixture_database(path: Path, sql: str, parameters: tuple[object, ...] = ()) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(sql, parameters)
