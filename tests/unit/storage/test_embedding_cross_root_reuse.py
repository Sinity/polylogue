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
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.config import load_polylogue_config
from polylogue.storage.embeddings.identity import EMBEDDING_INPUT_SCHEMA_VERSION, EmbeddingRecipe
from polylogue.storage.embeddings.materialization import (
    count_archive_embedding_session_state,
    embed_archive_session_sync,
    select_pending_archive_session_window,
)
from polylogue.storage.embeddings.preflight import read_embedding_work_counts
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.embedding_backup_fixture import (
    BACKUP_TEXT,
    NoCallVectorProvider,
    SyntheticVectorProvider,
    connect_vector_fixture,
    embedding_vector_rows,
    restart_restored_embedding_backup,
    snapshot_embedding_backup,
    startup_restored_embedding_backup,
    write_embedding_session,
)


def test_vector_written_under_one_root_is_a_hit_under_a_fresh_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_root = tmp_path / "old"
    old_session = write_embedding_session(old_root, native_id="old-native", message_native_id="m-old")
    initialize_archive_database(old_root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    connect_vector_fixture(old_root / "embeddings.db").close()

    provider = SyntheticVectorProvider()
    assert embed_archive_session_sync(old_root / "index.db", provider, old_session).status == "embedded"
    assert len(provider.calls) == 1

    # Fresh root: different session and message identity for the same text,
    # and a recipe whose input-schema label has moved on since the old root.
    fresh_root = tmp_path / "fresh"
    backup = tmp_path / "sealed-embeddings.db"
    snapshot_embedding_backup(old_root, backup)
    backup_hash = hashlib.sha256(backup.read_bytes()).digest()
    original_rows = embedding_vector_rows(backup)
    startup_restored_embedding_backup(backup, fresh_root)
    assert (fresh_root / "embeddings.db").is_symlink()
    adopted = (fresh_root / "embeddings.db").resolve()
    assert adopted.is_relative_to(fresh_root)
    assert adopted.stat().st_ino != backup.stat().st_ino
    metadata = json.loads((adopted.parent / "generation.json").read_text())
    assert metadata["archive_root"] == str(fresh_root)
    assert metadata["database_path"] == str(adopted)
    assert metadata["physical_root"] == str(adopted.parent)
    assert metadata["sealed"] is True
    original_metadata = json.loads(((old_root / "embeddings.db").resolve().parent / "generation.json").read_text())
    assert metadata["membership_digest"] == original_metadata["membership_digest"]
    with closing(connect_vector_fixture(fresh_root / "index.db")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert embedding_vector_rows(adopted) == original_rows
    # A cold process has neither the original handles nor bootstrap memo.
    restart_restored_embedding_backup(fresh_root)
    assert (fresh_root / "embeddings.db").resolve() == adopted
    assert json.loads((adopted.parent / "generation.json").read_text()) == metadata
    fresh_session = write_embedding_session(fresh_root, native_id="fresh-native", message_native_id="m-fresh")
    assert fresh_session != old_session
    monkeypatch.setattr(
        "polylogue.storage.embeddings.identity.EMBEDDING_INPUT_SCHEMA_VERSION",
        EMBEDDING_INPUT_SCHEMA_VERSION + "-relabelled",
    )

    with closing(connect_vector_fixture(fresh_root / "index.db")) as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(fresh_root / "embeddings.db"),))
        pending_before = select_pending_archive_session_window(
            conn, status_table="embeddings.embedding_status", session_ids=[fresh_session]
        )
    # Exact purchased outputs are available independently of occurrence binding.
    assert pending_before == []
    assert read_embedding_work_counts(fresh_root / "index.db") == (1, 0, 0, 1)

    # Publishing the missing occurrence binding spends no provider call.
    outcome = embed_archive_session_sync(fresh_root / "index.db", NoCallVectorProvider(), fresh_session)
    assert outcome.status == "embedded"
    assert len(provider.calls) == 1, "fresh root re-embedded unchanged text"

    with closing(connect_vector_fixture(fresh_root / "embeddings.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1
    with closing(connect_vector_fixture(fresh_root / "index.db")) as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(fresh_root / "embeddings.db"),))
        state = count_archive_embedding_session_state(conn, status_table="embeddings.embedding_status")
    assert state.embedded_sessions == 1
    assert state.pending_sessions == 0
    assert embedding_vector_rows(fresh_root / "embeddings.db") == original_rows
    assert hashlib.sha256(backup.read_bytes()).digest() == backup_hash
    with closing(connect_vector_fixture(fresh_root / "embeddings.db")) as conn:
        refs = conn.execute("SELECT message_id, session_id FROM message_embedding_refs ORDER BY message_id").fetchall()
        assert {row[1] for row in refs} == {old_session, fresh_session}
        assert len({row[0] for row in refs}) == 2


def test_ref_only_write_refuses_a_missing_vector(tmp_path: Path) -> None:
    """Reuse never fabricates a vector: an empty embedding needs its address present."""
    from polylogue.storage.sqlite.archive_tiers.embedding_write import ArchiveEmbeddingWrite, upsert_message_embeddings

    embeddings_db = tmp_path / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    with closing(connect_vector_fixture(embeddings_db)) as conn:
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
    session = write_embedding_session(old, native_id="old", message_native_id="old-message")
    provider = SyntheticVectorProvider()
    assert embed_archive_session_sync(old / "index.db", provider, session).status == "embedded"
    backup = tmp_path / "sealed.db"
    snapshot_embedding_backup(old, backup)
    original_hash = hashlib.sha256(backup.read_bytes()).digest()
    fresh = tmp_path / "fresh"
    startup_restored_embedding_backup(backup, fresh)
    fresh_session = write_embedding_session(
        fresh,
        native_id="new",
        message_native_id="new-message",
        text=BACKUP_TEXT + " Changed content." if difference == "content" else BACKUP_TEXT,
    )
    recipe = EmbeddingRecipe.current(
        model="voyage-3" if difference == "model" else provider.model, dimensions=provider.dimension
    )
    assert read_embedding_work_counts(fresh / "index.db", recipe=recipe) == (1, 1, 1, 0)
    assert len(provider.calls) == 1
    assert hashlib.sha256(backup.read_bytes()).digest() == original_hash
    with closing(connect_vector_fixture(fresh / "embeddings.db")) as connection:
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
    session = write_embedding_session(old, native_id="old", message_native_id="old-message")
    provider = SyntheticVectorProvider()
    assert embed_archive_session_sync(old / "index.db", provider, session).status == "embedded"
    backup = tmp_path / "selected.db"
    snapshot_embedding_backup(old, backup)
    if damage == "malformed":
        backup.write_bytes(b"neutral malformed selected backup")
    else:
        with closing(connect_vector_fixture(backup)) as connection:
            if damage == "version":
                connection.execute(f"PRAGMA user_version={EMBEDDINGS_SCHEMA_VERSION + 1}")
            else:
                connection.execute("DROP TABLE message_embeddings")
            connection.commit()
    original_hash = hashlib.sha256(backup.read_bytes()).digest()
    with pytest.raises((SchemaSkew, EmbeddingGenerationError, sqlite3.DatabaseError, RuntimeError)):
        startup_restored_embedding_backup(backup, tmp_path / "fresh")
    assert hashlib.sha256(backup.read_bytes()).digest() == original_hash
    assert len(provider.calls) == 1
