"""Rebuild-survival proof for content-addressed embeddings (polylogue-q88p).

Operator ruling 2026-07-20: vectors must be about content, not transient index
identity. Before this change, embedding freshness was gated on
``messages.content_hash``, which bakes in ``session_id``/``position``/
``variant_index`` -- an index rebuild or lineage-normalization shift that
renumbers a message (same text, different position/native id) silently
invalidated its vector, forcing a wasted re-embed through the paid Voyage API
(the 04kl 777K-vector incident).

``embedding_input_hash = H(model, embedder input text)`` excludes every
identity field by construction. This test proves the end-to-end consequence:
embed a session for real (through the archive materialization route, not a
unit-level hash comparison), then simulate an index rebuild that reassigns
the message's provider-native id (message_id shifts) while its text is
byte-identical, and assert the session is still classified as fully embedded
-- zero vectors invalidated, zero re-embed API calls needed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
from polylogue.storage.embeddings.materialization import (
    count_archive_embedding_session_state,
    embed_archive_session_sync,
    select_pending_archive_session_window,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_TEXT = "This authored prose message must survive an identity-only rebuild unscathed."


class _CountingFakeVectorProvider:
    """Stub embedder that records every call -- proves API cost was or wasn't spent."""

    model = "voyage-4"
    dimension = 1024

    def __init__(self) -> None:
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


def _write_session(root: Path, *, native_id: str, message_native_id: str, text: str) -> str:
    with ArchiveStore(root) as archive:
        return archive.write_parsed(
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
            )
        )


def _connect_vec(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
    return conn


def test_identity_shifting_rebuild_does_not_invalidate_vector(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id = _write_session(root, native_id="rebuild-survival", message_native_id="m1", text=_TEXT)
    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    _connect_vec(embeddings_db).close()

    provider = _CountingFakeVectorProvider()
    outcome = embed_archive_session_sync(index_db, provider, session_id)
    assert outcome.status == "embedded"
    assert len(provider.calls) == 1, "first embed must spend exactly one provider call"

    with _connect_vec(embeddings_db) as conn:
        meta_rows_before = conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0]
        vector_rows_before = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
    assert meta_rows_before == 1
    assert vector_rows_before == 1

    with _connect_vec(index_db) as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        state_before = count_archive_embedding_session_state(conn, status_table="embeddings.embedding_status")
    assert state_before.embedded_sessions == 1
    assert state_before.pending_sessions == 0

    # Simulate an index rebuild that reassigns the message's provider-native
    # id -- e.g. lineage normalization recomputing which prefix is shared, or
    # a provider regenerating tool/message ids on re-export. The archive
    # write path treats this as a full-replace of the session (same native
    # session id, differing message identity) because the OLD
    # identity-contaminated content_hash changes even though the text is
    # byte-identical.
    replaced_session_id = _write_session(
        root, native_id="rebuild-survival", message_native_id="m1-renumbered-by-rebuild", text=_TEXT
    )
    assert replaced_session_id == session_id

    with sqlite3.connect(index_db) as conn:
        rebuilt_message_id = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    assert "m1-renumbered-by-rebuild" in str(rebuilt_message_id)

    # The embeddings tier was never touched by the rebuild (it is a
    # separately rebuildable, source-of-truth-independent tier) -- confirm
    # the vector/meta rows are still exactly what the first embed wrote.
    with _connect_vec(embeddings_db) as conn:
        meta_rows_after = conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0]
        vector_rows_after = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
    assert meta_rows_after == meta_rows_before
    assert vector_rows_after == vector_rows_before

    # The load-bearing assertion: the session-level freshness predicate,
    # which gates every real selection surface (daemon catch-up, manual
    # backfill, convergence), must classify the rebuilt session as still
    # fully embedded -- zero pending, zero vectors invalidated -- WITHOUT
    # embed_archive_session_sync ever being called again.
    with _connect_vec(index_db) as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        state_after = count_archive_embedding_session_state(conn, status_table="embeddings.embedding_status")
        pending_window = select_pending_archive_session_window(
            conn, status_table="embeddings.embedding_status", session_ids=[session_id]
        )

    assert state_after.pending_sessions == 0, "identity-only rebuild must not mark the session pending"
    assert state_after.embedded_sessions == 1
    assert pending_window == [], "the rebuilt session must not appear in the pending-embedding selection window"

    # Never call embed_archive_session_sync again -- if it were called and
    # actually re-embedded, that would itself prove the bug is back (a
    # wasted provider call for unchanged text). We assert the *selection*
    # layer already excludes it, which is what protects production from
    # ever making that wasted call.
    assert len(provider.calls) == 1, "no additional provider calls were spent surviving the rebuild"
