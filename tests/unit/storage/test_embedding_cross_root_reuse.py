"""A vector embedded under one archive root is a cache hit under a fresh root.

The reindex must not re-embed messages whose model and input text are
unchanged. Vector rows are content-addressed by the provider request, so
carrying the vector tables from an old root into a fresh archive (a fresh
index with new message identities, and a recipe whose labels may have
changed) leaves every unchanged message fully embedded.

Anti-vacuity: folding any recipe label, the index schema version, or message
identity into ``vector_derivation_hash`` makes the fresh root select the
session as pending and spend a provider call; the assertions below fail on
either.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
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
from polylogue.storage.embeddings.identity import EMBEDDING_INPUT_SCHEMA_VERSION, EmbeddingRecipe
from polylogue.storage.embeddings.materialization import (
    count_archive_embedding_session_state,
    embed_archive_session_sync,
    select_pending_archive_session_window,
)
from polylogue.storage.embeddings.preflight import read_embedding_work_counts
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement, write_lease
from tests.infra.live_ingest import write_index_session

_TEXT = "This authored prose message is embedded once and reused under a fresh archive root."


class _CountingFakeVectorProvider:
    dimension = 1024

    def __init__(self) -> None:
        # Bind after pytest has installed the isolated configuration.
        self.model = load_polylogue_config().embedding_model
        self.calls: list[list[str]] = []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "document"
        self.calls.append(list(texts))
        return [[0.5] * self.dimension for _ in texts]

    def upsert(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("archive materialization must use the archive embedding route")

    def query(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []

    def query_by_session(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []

    def scoped_query(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("document-only fixture does not perform scoped retrieval")

    async def read_similarity(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("this fixture does not perform retained-session reads")


def _write_session(root: Path, *, native_id: str, message_native_id: str, text: str = _TEXT) -> str:
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


def _connect_vec(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
    return conn


class _NoCallVectorProvider(_CountingFakeVectorProvider):
    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> Never:
        raise AssertionError("matching backup vector must not dispatch provider acquisition")


def _snapshot_backup(old_root: Path, backup: Path) -> None:
    from polylogue.operations.archive_backup import _backup_sqlite

    with arm_write_lease_enforcement(), write_lease("fixture.embedding-backup", archive_root=old_root):
        _backup_sqlite(old_root / "embeddings.db", backup, archive_root_path=old_root)
    assert not backup.with_name(backup.name + "-wal").exists()
    assert not backup.with_name(backup.name + "-shm").exists()


def _startup_restored_backup(backup: Path, fresh_root: Path) -> None:
    fresh_root.mkdir()
    shutil.copyfile(backup, fresh_root / "embeddings.db")
    assert {path.name for path in fresh_root.iterdir()} == {"embeddings.db"}
    with arm_write_lease_enforcement(), write_lease("startup.embedding-backup", archive_root=fresh_root):
        initialize_active_archive_root(fresh_root)
        ensure_embedding_lifecycle_startup(fresh_root)


def _vector_rows(path: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    with closing(_connect_vec(path)) as conn:
        return (
            conn.execute("SELECT * FROM message_embeddings_meta ORDER BY vector_derivation_hash").fetchall(),
            conn.execute(
                "SELECT vector_derivation_hash, embedding, model FROM message_embeddings ORDER BY vector_derivation_hash"
            ).fetchall(),
        )


def test_vector_written_under_one_root_is_a_hit_under_a_fresh_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_root = tmp_path / "old"
    old_session = _write_session(old_root, native_id="old-native", message_native_id="m-old")
    initialize_archive_database(old_root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    _connect_vec(old_root / "embeddings.db").close()

    provider = _CountingFakeVectorProvider()
    assert embed_archive_session_sync(old_root / "index.db", provider, old_session).status == "embedded"
    assert len(provider.calls) == 1

    # Fresh root: different session and message identity for the same text,
    # and a recipe whose input-schema label has moved on since the old root.
    fresh_root = tmp_path / "fresh"
    backup = tmp_path / "sealed-embeddings.db"
    _snapshot_backup(old_root, backup)
    backup_hash = hashlib.sha256(backup.read_bytes()).digest()
    original_rows = _vector_rows(backup)
    _startup_restored_backup(backup, fresh_root)
    assert (fresh_root / "embeddings.db").is_symlink()
    adopted = (fresh_root / "embeddings.db").resolve()
    assert adopted.is_relative_to(fresh_root)
    assert adopted.stat().st_ino != backup.stat().st_ino
    metadata = json.loads((adopted.parent / "generation.json").read_text())
    assert metadata["archive_root"] == str(fresh_root)
    assert metadata["database_path"] == str(adopted)
    assert metadata["physical_root"] == str(adopted.parent)
    assert metadata["sealed"] is True
    assert _vector_rows(adopted) == original_rows
    # Restart with fresh owner objects through the same actual startup boundary.
    with arm_write_lease_enforcement(), write_lease("restart.embedding-backup", archive_root=fresh_root):
        initialize_active_archive_root(fresh_root)
        ensure_embedding_lifecycle_startup(fresh_root)
    fresh_session = _write_session(fresh_root, native_id="fresh-native", message_native_id="m-fresh")
    assert fresh_session != old_session
    monkeypatch.setattr(
        "polylogue.storage.embeddings.identity.EMBEDDING_INPUT_SCHEMA_VERSION",
        EMBEDDING_INPUT_SCHEMA_VERSION + "-relabelled",
    )

    with closing(_connect_vec(fresh_root / "index.db")) as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(fresh_root / "embeddings.db"),))
        pending_before = select_pending_archive_session_window(
            conn, status_table="embeddings.embedding_status", session_ids=[fresh_session]
        )
    # Exact purchased outputs are available independently of occurrence binding.
    assert pending_before == []
    assert read_embedding_work_counts(fresh_root / "index.db") == (1, 0, 0, 1)

    # Publishing the missing occurrence binding spends no provider call.
    outcome = embed_archive_session_sync(fresh_root / "index.db", _NoCallVectorProvider(), fresh_session)
    assert outcome.status == "embedded"
    assert len(provider.calls) == 1, "fresh root re-embedded unchanged text"

    with closing(_connect_vec(fresh_root / "embeddings.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1
    with closing(_connect_vec(fresh_root / "index.db")) as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(fresh_root / "embeddings.db"),))
        state = count_archive_embedding_session_state(conn, status_table="embeddings.embedding_status")
    assert state.embedded_sessions == 1
    assert state.pending_sessions == 0
    assert _vector_rows(fresh_root / "embeddings.db") == original_rows
    assert hashlib.sha256(backup.read_bytes()).digest() == backup_hash
    with closing(_connect_vec(fresh_root / "embeddings.db")) as conn:
        refs = conn.execute("SELECT message_id, session_id FROM message_embedding_refs ORDER BY message_id").fetchall()
        assert {row[1] for row in refs} == {old_session, fresh_session}
        assert len({row[0] for row in refs}) == 2


def test_ref_only_write_refuses_a_missing_vector(tmp_path: Path) -> None:
    """Reuse never fabricates a vector: an empty embedding needs its address present."""
    from polylogue.storage.sqlite.archive_tiers.embedding_write import ArchiveEmbeddingWrite, upsert_message_embeddings

    embeddings_db = tmp_path / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    with closing(_connect_vec(embeddings_db)) as conn:
        with pytest.raises(ValueError, match="ref-only"):
            upsert_message_embeddings(
                conn,
                [
                    ArchiveEmbeddingWrite(
                        message_id="codex-session:s:m",
                        session_id="codex-session:s",
                        origin="codex-session",
                        embedding=[],
                        model=load_polylogue_config().embedding_model,
                        embedded_at_ms=0,
                        vector_derivation_hash=b"\x01" * 32,
                    )
                ],
            )
        assert conn.execute("SELECT COUNT(*) FROM message_embedding_refs").fetchone()[0] == 0


@pytest.mark.parametrize("difference", ["content", "model"])
def test_restored_backup_mismatch_is_a_real_compute_miss(tmp_path: Path, difference: str) -> None:
    old = tmp_path / "old"
    session = _write_session(old, native_id="old", message_native_id="old-message")
    provider = _CountingFakeVectorProvider()
    assert embed_archive_session_sync(old / "index.db", provider, session).status == "embedded"
    backup = tmp_path / "sealed.db"
    _snapshot_backup(old, backup)
    original_hash = hashlib.sha256(backup.read_bytes()).digest()
    fresh = tmp_path / "fresh"
    _startup_restored_backup(backup, fresh)
    fresh_session = _write_session(
        fresh,
        native_id="new",
        message_native_id="new-message",
        text=_TEXT + " Changed content." if difference == "content" else _TEXT,
    )
    recipe = EmbeddingRecipe.current(
        model="voyage-3" if difference == "model" else provider.model, dimensions=provider.dimension
    )
    assert read_embedding_work_counts(fresh / "index.db", recipe=recipe) == (1, 1, 1, 0)
    assert len(provider.calls) == 1
    assert hashlib.sha256(backup.read_bytes()).digest() == original_hash
    with closing(_connect_vec(fresh / "embeddings.db")) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM message_embedding_refs WHERE session_id=?", (fresh_session,)
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("damage", ["malformed", "version", "missing-vector-table"])
def test_unsupported_selected_backup_is_refused_by_actual_startup(tmp_path: Path, damage: str) -> None:
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.embeddings.generations import EmbeddingGenerationError
    from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION

    old = tmp_path / "old"
    session = _write_session(old, native_id="old", message_native_id="old-message")
    provider = _CountingFakeVectorProvider()
    assert embed_archive_session_sync(old / "index.db", provider, session).status == "embedded"
    backup = tmp_path / "selected.db"
    _snapshot_backup(old, backup)
    if damage == "malformed":
        backup.write_bytes(b"neutral malformed selected backup")
    else:
        with closing(_connect_vec(backup)) as connection:
            if damage == "version":
                connection.execute(f"PRAGMA user_version={EMBEDDINGS_SCHEMA_VERSION + 1}")
            else:
                connection.execute("DROP TABLE message_embeddings")
            connection.commit()
    original_hash = hashlib.sha256(backup.read_bytes()).digest()
    with pytest.raises((SchemaSkew, EmbeddingGenerationError, sqlite3.DatabaseError)):
        _startup_restored_backup(backup, tmp_path / "fresh")
    assert hashlib.sha256(backup.read_bytes()).digest() == original_hash
    assert len(provider.calls) == 1
