"""Tests for standalone/off-mode local excision (polylogue-27m).

Anti-vacuity: ``test_reingest_does_not_resurrect_excised_content`` exercises
the real acquire-time write chokepoint
(``write_source_raw_session``/``ContentExcisedError``) that
``polylogue.pipeline.services.archive_ingest.parse_sources_archive`` relies
on for every ordinary re-ingest; removing the gate in
``write_source_raw_session`` (or reverting the ``write_pair`` skip-not-abort
handling) makes it fail. ``test_apply_removes_rows_from_every_tier``
exercises the real cross-tier DELETE statements against real archive-tier
schemas (not a toy replica); commenting out any one tier's delete makes the
corresponding assertion fail.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind
from polylogue.security.excision import (
    apply_session_excision,
    plan_session_excision,
    resolve_session_excision_target,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ContentExcisedError,
    deterministic_blob_hash,
    is_blob_hash_excised,
    write_source_raw_session,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_session(
    archive_root: Path,
    *,
    native_id: str,
    payload: bytes = b'{"native_id": "x"}',
    with_message: bool = True,
    with_block: bool = True,
    with_embedding: bool = False,
) -> str:
    """Seed a minimal but real session spanning source.db + index.db (+ optionally embeddings.db)."""

    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    source_conn = sqlite3.connect(source_db)
    source_conn.execute("PRAGMA foreign_keys = ON")
    try:
        raw_id = write_source_raw_session(
            source_conn,
            origin="codex-session",
            source_path=f"/fake/{native_id}.jsonl",
            source_index=0,
            payload=payload,
            acquired_at_ms=1_000,
            native_id=native_id,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_conn = sqlite3.connect(index_db)
    index_conn.execute("PRAGMA foreign_keys = ON")
    try:
        index_conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms)
            VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)
            """,
            (native_id, raw_id, f"Session {native_id}"),
        )
        session_id = index_conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[
            0
        ]
        message_id: object | None = None
        if with_message:
            index_conn.execute(
                "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
                "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
                (session_id,),
            )
            message_id = index_conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            if with_block:
                index_conn.execute(
                    "INSERT INTO blocks (message_id, session_id, position, block_type, text) "
                    "VALUES (?, ?, 0, 'text', 'hello secret')",
                    (message_id, session_id),
                )
        index_conn.commit()
    finally:
        index_conn.close()

    if with_embedding and with_message:
        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

        embeddings_db = archive_root / "embeddings.db"
        initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
        emb_conn = sqlite3.connect(embeddings_db)
        try:
            try_load_sqlite_vec(emb_conn)
            emb_conn.execute(
                "INSERT INTO message_embeddings (message_id, embedding, session_id, origin) "
                "VALUES (?, ?, ?, 'codex-session')",
                (message_id, b"\x00\x00\x80\x3f" * 1024, session_id),
            )
            emb_conn.execute(
                "INSERT INTO message_embeddings_meta (message_id, model, dimension, content_hash) "
                "VALUES (?, 'test-model', 1024, zeroblob(32))",
                (message_id,),
            )
            emb_conn.execute(
                "INSERT INTO embedding_status (session_id, message_count_embedded) VALUES (?, 1)",
                (session_id,),
            )
            emb_conn.commit()
        finally:
            emb_conn.close()

    return str(session_id)


class TestPlanSessionExcision:
    def test_not_found_for_unknown_session(self, tmp_path: Path) -> None:
        plan = plan_session_excision(tmp_path, "codex-session:does-not-exist")
        assert plan.found is False

    def test_counts_every_tier(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="plan-1", with_embedding=True)
        plan = plan_session_excision(tmp_path, session_id)
        assert plan.found is True
        assert plan.source_raw_rows == 1
        assert plan.source_blob_refs == 1
        assert plan.index_sessions == 1
        assert plan.index_messages == 1
        assert plan.index_blocks == 1
        assert plan.embeddings_vectors == 1

    def test_dry_run_does_not_mutate(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="plan-2")
        plan_session_excision(tmp_path, session_id)
        # Session must still be readable after a plan-only call.
        target = resolve_session_excision_target(tmp_path, session_id)
        assert target.found is True


class TestApplySessionExcision:
    def test_not_found_returns_found_false(self, tmp_path: Path) -> None:
        receipt = apply_session_excision(tmp_path, "codex-session:nope", reason="r", actor="user:local")
        assert receipt.found is False

    def test_apply_removes_rows_from_every_tier(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-1", with_embedding=True)

        receipt = apply_session_excision(tmp_path, session_id, reason="contained a secret", actor="user:local")
        assert receipt.found is True
        assert receipt.counts["index_sessions"] == 1
        assert receipt.counts["index_messages"] == 1
        assert receipt.counts["index_blocks"] == 1
        assert receipt.counts["source_raw_rows"] == 1
        assert receipt.counts["source_blob_refs"] == 1
        assert receipt.counts["embeddings_vectors"] == 1
        assert len(receipt.removed_blob_hashes) == 1

        index_conn = sqlite3.connect(tmp_path / "index.db")
        try:
            assert (
                index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
                == 0
            )
            assert (
                index_conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
                == 0
            )
            assert (
                index_conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
            )
        finally:
            index_conn.close()

        source_conn = sqlite3.connect(tmp_path / "source.db")
        try:
            assert source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
            assert source_conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0] == 0
        finally:
            source_conn.close()

        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec as _load_vec

        emb_conn = sqlite3.connect(tmp_path / "embeddings.db")
        try:
            _load_vec(emb_conn)
            assert emb_conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 0
            assert (
                emb_conn.execute(
                    "SELECT COUNT(*) FROM embedding_status WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                == 0
            )
        finally:
            emb_conn.close()

    def test_apply_writes_durable_audit_receipt(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-2")
        receipt = apply_session_excision(tmp_path, session_id, reason="pii leak", actor="user:audit")

        user_conn = sqlite3.connect(tmp_path / "user.db")
        try:
            row = user_conn.execute(
                "SELECT kind, target_ref, author_ref, author_kind FROM assertions WHERE assertion_id = ?",
                (receipt.receipt_assertion_id,),
            ).fetchone()
        finally:
            user_conn.close()
        assert row is not None
        assert row[0] == AssertionKind.EXCISION_RECORD.value
        assert row[1] == f"session:{session_id}"
        assert row[3] == "user"

    def test_apply_removes_content_bearing_assertions_targeting_the_session(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-3")
        user_db = tmp_path / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

                upsert_assertion(
                    conn,
                    assertion_id="assertion-note:pre-existing",
                    target_ref=f"session:{session_id}",
                    kind=AssertionKind.NOTE,
                    body_text="quoting the secret span here",
                    author_ref="user:local",
                    author_kind="user",
                    now_ms=1,
                )
        finally:
            conn.close()

        apply_session_excision(tmp_path, session_id, reason="r", actor="user:local")

        conn = sqlite3.connect(user_db)
        try:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE assertion_id = ?", ("assertion-note:pre-existing",)
            ).fetchone()[0]
        finally:
            conn.close()
        assert remaining == 0

    def test_apply_is_idempotent(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-4")
        first = apply_session_excision(tmp_path, session_id, reason="r", actor="user:local")
        assert first.found is True
        second = apply_session_excision(tmp_path, session_id, reason="r-again", actor="user:local")
        assert second.found is False  # already gone; nothing left to touch

    def test_reingest_does_not_resurrect_excised_content(self, tmp_path: Path) -> None:
        payload = b'{"native_id": "resurrect-me", "secret": "sk-ant-abc123"}'
        session_id = _seed_session(tmp_path, native_id="resurrect-me", payload=payload)

        apply_session_excision(tmp_path, session_id, reason="secret leak", actor="user:local")

        source_conn = sqlite3.connect(tmp_path / "source.db")
        source_conn.execute("PRAGMA foreign_keys = ON")
        try:
            assert is_blob_hash_excised(source_conn, deterministic_blob_hash(payload)) is True
            with pytest.raises(ContentExcisedError):
                write_source_raw_session(
                    source_conn,
                    origin="codex-session",
                    source_path="/fake/resurrect-me.jsonl",
                    source_index=0,
                    payload=payload,  # identical bytes: an ordinary re-ingest of the SAME file
                    acquired_at_ms=9_999,
                    native_id="resurrect-me",
                )
            # No row was resurrected by the refused write.
            assert source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        finally:
            source_conn.close()

    def test_reingest_batch_skips_excised_file_without_aborting(self, tmp_path: Path) -> None:
        """The batch orchestration layer must skip-not-abort on ContentExcisedError.

        Uses the shared synthetic-corpus generator (real provider-shaped
        files, the same fixture machinery as
        ``tests/unit/pipeline/test_archive_ingest_commit_batching.py``)
        rather than a hand-rolled JSONL literal, so this exercises the real
        parser/detector path, not a guessed shape.
        """
        import asyncio

        from polylogue.config import Source
        from polylogue.pipeline.services.archive_ingest import parse_sources_archive
        from polylogue.scenarios import build_default_corpus_specs
        from polylogue.schemas.synthetic import SyntheticCorpus
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier as _Tier

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_archive_database(archive_root / "source.db", _Tier.SOURCE)
        initialize_archive_database(archive_root / "index.db", _Tier.INDEX)

        specs = build_default_corpus_specs(providers=["codex"], count=1, messages_min=2, messages_max=3, seed=11)
        corpus_dir = tmp_path / "corpus"
        written = SyntheticCorpus.write_spec_artifacts(specs[0], corpus_dir, prefix="corpus")
        sources = [Source(name="codex", path=file_path) for file_path in written.files]
        assert sources

        # First ingest establishes the raw row + session normally.
        result_first = asyncio.run(parse_sources_archive(archive_root, sources))
        assert result_first.excised_skips == 0
        assert result_first.counts["sessions"] >= 1

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            row = index_conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()
        finally:
            index_conn.close()
        assert row is not None
        session_id = str(row[0])
        receipt = apply_session_excision(archive_root, session_id, reason="test", actor="user:local")
        assert receipt.found is True

        # Re-ingest the SAME unmodified file: must skip (not raise/abort).
        result_second = asyncio.run(parse_sources_archive(archive_root, sources))
        assert result_second.excised_skips >= 1

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            remaining = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 0
