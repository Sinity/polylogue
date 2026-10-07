"""Synthetic purchased vectors and actual sealed backup/startup owners."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Never

import pytest

from polylogue.archive.message.roles import Role
from polylogue.config import load_polylogue_config
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.operations.embedding_lifecycle import ensure_embedding_lifecycle_startup
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement, write_lease
from tests.infra.live_ingest import write_index_session

BACKUP_TEXT = "This authored prose message is embedded once and reused under a fresh archive root."


class SyntheticVectorProvider:
    dimension = 1024

    def __init__(self) -> None:
        # Bind after pytest has installed the isolated configuration.
        self.model = load_polylogue_config().embedding_model
        self.calls: list[list[str]] = []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "document"
        self.calls.append(list(texts))
        return [[0.5] * self.dimension for _ in texts]

    def query(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []

    def query_by_session(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []

    def scoped_query(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("document-only fixture does not perform scoped retrieval")

    async def read_similarity(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("this fixture does not perform retained-session reads")


def write_embedding_session(root: Path, *, native_id: str, message_native_id: str, text: str = BACKUP_TEXT) -> str:
    with ArchiveStore(root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id=message_native_id,
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )


def connect_vector_fixture(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
    return conn


class NoCallVectorProvider(SyntheticVectorProvider):
    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> Never:
        raise AssertionError("matching backup vector must not dispatch provider acquisition")


def snapshot_embedding_backup(old_root: Path, backup: Path) -> None:
    from polylogue.storage.backup_package import _backup_sqlite

    with arm_write_lease_enforcement(), write_lease("fixture.embedding-backup", archive_root=old_root):
        _backup_sqlite(old_root / "embeddings.db", backup, archive_root_path=old_root)
    assert not backup.with_name(backup.name + "-wal").exists()
    assert not backup.with_name(backup.name + "-shm").exists()


def startup_restored_embedding_backup(backup: Path, fresh_root: Path) -> None:
    fresh_root.mkdir()
    shutil.copyfile(backup, fresh_root / "embeddings.db")
    assert {path.name for path in fresh_root.iterdir()} == {"embeddings.db"}
    with arm_write_lease_enforcement(), write_lease("startup.embedding-backup", archive_root=fresh_root):
        initialize_active_archive_root(fresh_root)
        ensure_embedding_lifecycle_startup(fresh_root)


def embedding_vector_rows(path: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    with closing(connect_vector_fixture(path)) as conn:
        return (
            conn.execute("SELECT * FROM message_embeddings_meta ORDER BY vector_derivation_hash").fetchall(),
            conn.execute(
                "SELECT vector_derivation_hash, embedding, model FROM message_embeddings ORDER BY vector_derivation_hash"
            ).fetchall(),
        )


_RESTART_PROGRAM = """
import sys
from pathlib import Path
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.operations.embedding_lifecycle import ensure_embedding_lifecycle_startup
from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement, write_lease
root = Path(sys.argv[1])
with arm_write_lease_enforcement(), write_lease("startup.embedding-backup", archive_root=root):
    initialize_active_archive_root(root)
    ensure_embedding_lifecycle_startup(root)
"""


def restart_restored_embedding_backup(root: Path) -> None:
    """Cold process admission cannot use the first constructor's process cache."""
    subprocess.run([sys.executable, "-c", _RESTART_PROGRAM, str(root)], check=True)
