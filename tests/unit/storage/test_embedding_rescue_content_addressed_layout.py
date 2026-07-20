"""04kl rescue lands vectors directly into the v4 content-addressed layout.

Operator ordering (polylogue-q88p, 2026-07-20): design the content-addressed
keying first, then run the one-time 04kl rescue directly INTO it -- migrate
once, not twice. The retired evidence file predates this bead and is
structurally v3 (vectors keyed by ``message_id``, meta bound to the OLD
identity-contaminated ``content_hash``); this test proves
:func:`polylogue.storage.embeddings.rescue.execute_embedding_rescue`
recomputes ``embedding_input_hash`` for each matched message from its
*current* embedder input text and writes the rescued vector under that new
key, plus a ``message_embedding_refs`` row -- never resurrecting the old
message_id-keyed shape.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
from polylogue.storage.embeddings.identity import EmbeddingRecipe, embedding_input_hash
from polylogue.storage.embeddings.rescue import execute_embedding_rescue, plan_embedding_rescue
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_TEXT = "Retired-tier evidence prose that must be rescued into the new content-addressed layout."
_MODEL = "voyage-4"
_DIMENSION = 1024

# The v3 retired-tier shape (pre-polylogue-q88p): vectors keyed by
# message_id, meta bound to the OLD identity-contaminated content_hash.
# Deliberately hand-declared here (not imported from production) because the
# whole point of this fixture is to be the OLD, no-longer-current schema --
# it must not silently track future changes to the live DDL.
_RETIRED_DDL = f"""
CREATE VIRTUAL TABLE message_embeddings USING vec0(
    message_id TEXT PRIMARY KEY,
    embedding float[{_DIMENSION}],
    +session_id TEXT,
    +origin TEXT
);

CREATE TABLE message_embeddings_meta (
    message_id      TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    dimension       INTEGER NOT NULL,
    content_hash    BLOB NOT NULL,
    embedded_at_ms  INTEGER,
    needs_reindex   INTEGER NOT NULL DEFAULT 0,
    recipe_hash     BLOB,
    derivation_key  BLOB,
    generation      INTEGER NOT NULL DEFAULT 0
);
"""


def _connect_vec(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
    return conn


def _write_live_session(root: Path, *, native_id: str, text: str) -> str:
    with ArchiveStore(root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            )
        )


def _build_retired_fixture(path: Path, *, message_id: str, session_id: str, content_hash: bytes) -> list[float]:
    """Build a v3-shaped retired embeddings.db and return the stored vector."""
    conn = _connect_vec(path)
    try:
        conn.executescript(_RETIRED_DDL)
        vector = [0.42] * _DIMENSION
        conn.execute(
            "INSERT INTO message_embeddings (message_id, embedding, session_id, origin) VALUES (?, ?, ?, ?)",
            (message_id, struct.pack(f"<{_DIMENSION}f", *vector), session_id, "codex-session"),
        )
        conn.execute(
            """
            INSERT INTO message_embeddings_meta (
                message_id, model, dimension, content_hash, embedded_at_ms, needs_reindex
            ) VALUES (?, ?, ?, ?, ?, 0)
            """,
            (message_id, _MODEL, _DIMENSION, content_hash, 1_700_000_000_000),
        )
        conn.commit()
    finally:
        conn.close()
    return vector


def test_rescue_lands_vectors_into_content_addressed_layout(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    session_id = _write_live_session(archive_root, native_id="rescue-into-v4", text=_TEXT)
    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"

    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT message_id, content_hash FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
    message_id, content_hash = str(row[0]), bytes(row[1])

    retired_db = tmp_path / "embeddings.db.v2-retired-fixture"
    retired_vector = _build_retired_fixture(
        retired_db, message_id=message_id, session_id=session_id, content_hash=content_hash
    )

    recipe = EmbeddingRecipe.current(model=_MODEL, dimensions=_DIMENSION)

    plan = plan_embedding_rescue(index_db, retired_db, recipe=recipe)
    assert plan.counts.fully_rescuable_sessions == 1
    assert plan.counts.rescuable_messages == 1
    assert plan.counts.skipped_missing == 0
    assert plan.counts.skipped_hash_mismatch == 0
    assert plan.counts.skipped_model_mismatch == 0

    report = execute_embedding_rescue(
        index_db,
        retired_db,
        embeddings_db,
        recipe=recipe,
        mutation_authority="offline-exclusive",
    )
    assert report.rescued_sessions == 1
    assert report.rescued_messages == 1
    assert report.ok, "sampled byte-identity verification must pass"
    assert report.verified_sample_total >= 1
    assert report.verified_sample_ok == report.verified_sample_total

    expected_hash = embedding_input_hash(model=_MODEL, input_text=_TEXT)

    with _connect_vec(embeddings_db) as conn:
        # The v4 layout: vectors keyed by embedding_input_hash, never by
        # message_id -- assert the schema shape directly, not just counts.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(message_embeddings_meta)").fetchall()}
        assert "message_id" not in columns, "meta must not be message_id-keyed in the rescued layout"
        assert "embedding_input_hash" in columns

        meta_row = conn.execute(
            "SELECT model, dimension FROM message_embeddings_meta WHERE embedding_input_hash = ?",
            (expected_hash,),
        ).fetchone()
        assert meta_row is not None, "rescued vector must be keyed by the recomputed embedding_input_hash"
        assert meta_row[0] == _MODEL
        assert meta_row[1] == _DIMENSION

        ref_row = conn.execute(
            "SELECT session_id, origin, embedding_input_hash FROM message_embedding_refs WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        assert ref_row is not None, "rescue must write a message_embedding_refs row for the rescued message"
        assert ref_row[0] == session_id
        assert bytes(ref_row[2]) == expected_hash

        vector_row = conn.execute(
            "SELECT embedding FROM message_embeddings WHERE embedding_input_hash = ?",
            (expected_hash.hex(),),
        ).fetchone()
        assert vector_row is not None
        rescued_vector = list(struct.unpack(f"<{_DIMENSION}f", vector_row[0]))
        assert rescued_vector == pytest.approx(retired_vector), "rescued bytes must match the retired vector exactly"

    # Idempotent: a second rescue pass over an already-rescued archive is a
    # pure no-op (the session already reads as fresh).
    second_report = execute_embedding_rescue(
        index_db,
        retired_db,
        embeddings_db,
        recipe=recipe,
        mutation_authority="offline-exclusive",
    )
    assert second_report.rescued_sessions == 0
    assert second_report.skipped_already_fresh_sessions == 1
