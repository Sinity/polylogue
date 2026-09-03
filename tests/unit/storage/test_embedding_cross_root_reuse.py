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

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.config import load_polylogue_config
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
from polylogue.storage.embeddings.identity import EMBEDDING_INPUT_SCHEMA_VERSION
from polylogue.storage.embeddings.materialization import (
    count_archive_embedding_session_state,
    embed_archive_session_sync,
    select_pending_archive_session_window,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_TEXT = "This authored prose message is embedded once and reused under a fresh archive root."
_CONFIGURED_EMBEDDING_MODEL = load_polylogue_config().embedding_model


class _CountingFakeVectorProvider:
    model = _CONFIGURED_EMBEDDING_MODEL
    dimension = 1024

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "document"
        self.calls.append(list(texts))
        return [[0.5] * self.dimension for _ in texts]


def _write_session(root: Path, *, native_id: str, message_native_id: str) -> str:
    with ArchiveStore(root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id=message_native_id,
                        role=Role.USER,
                        text=_TEXT,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=_TEXT)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            )
        )


def _connect_vec(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
    return conn


def _copy_vector_tables(source: Path, destination: Path) -> None:
    """Carry the content-addressed vector tables (not refs/status/ledger) across roots."""
    with _connect_vec(destination) as conn:
        conn.execute("ATTACH DATABASE ? AS old", (str(source),))
        conn.execute("INSERT INTO message_embeddings_meta SELECT * FROM old.message_embeddings_meta")
        conn.execute(
            "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) "
            "SELECT vector_derivation_hash, embedding, model FROM old.message_embeddings"
        )
        conn.commit()


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
    fresh_session = _write_session(fresh_root, native_id="fresh-native", message_native_id="m-fresh")
    assert fresh_session != old_session
    initialize_archive_database(fresh_root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    _copy_vector_tables(old_root / "embeddings.db", fresh_root / "embeddings.db")
    monkeypatch.setattr(
        "polylogue.storage.embeddings.identity.EMBEDDING_INPUT_SCHEMA_VERSION",
        EMBEDDING_INPUT_SCHEMA_VERSION + "-relabelled",
    )

    with _connect_vec(fresh_root / "index.db") as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(fresh_root / "embeddings.db"),))
        pending_before = select_pending_archive_session_window(
            conn, status_table="embeddings.embedding_status", session_ids=[fresh_session]
        )
    # The session ledger is per-root, so the fresh session is selected once...
    assert pending_before == [fresh_session]

    # ...but converging it finds every vector already present and spends no provider call.
    outcome = embed_archive_session_sync(fresh_root / "index.db", provider, fresh_session)
    assert outcome.status == "embedded"
    assert len(provider.calls) == 1, "fresh root re-embedded unchanged text"

    with _connect_vec(fresh_root / "embeddings.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1
    with _connect_vec(fresh_root / "index.db") as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(fresh_root / "embeddings.db"),))
        state = count_archive_embedding_session_state(conn, status_table="embeddings.embedding_status")
    assert state.embedded_sessions == 1
    assert state.pending_sessions == 0


def test_ref_only_write_refuses_a_missing_vector(tmp_path: Path) -> None:
    """Reuse never fabricates a vector: an empty embedding needs its address present."""
    from polylogue.storage.sqlite.archive_tiers.embedding_write import ArchiveEmbeddingWrite, upsert_message_embeddings

    embeddings_db = tmp_path / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    with _connect_vec(embeddings_db) as conn:
        with pytest.raises(ValueError, match="ref-only"):
            upsert_message_embeddings(
                conn,
                [
                    ArchiveEmbeddingWrite(
                        message_id="codex-session:s:m",
                        session_id="codex-session:s",
                        origin="codex-session",
                        embedding=[],
                        model=_CONFIGURED_EMBEDDING_MODEL,
                        embedded_at_ms=0,
                        vector_derivation_hash=b"\x01" * 32,
                    )
                ],
            )
        assert conn.execute("SELECT COUNT(*) FROM message_embedding_refs").fetchone()[0] == 0
