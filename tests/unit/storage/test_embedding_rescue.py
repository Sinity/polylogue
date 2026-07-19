"""Focused tests for polylogue-04kl: retired-tier embedding vector rescue.

Builds a synthetic "post-incident" scenario directly: a minimal ``index.db``
fixture (mirroring ``test_embedding_orphan_reconcile.py``'s pattern) plus a
retired ``embeddings.db``-shaped source file and a fresh target
``embeddings.db``, then asserts :func:`plan_embedding_rescue` classifies
sessions/messages correctly and :func:`execute_embedding_rescue` only ever
mutates *fully rescuable* sessions, is idempotent on rerun, and its sampled
byte-identity verification actually catches a corrupted copy.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import NamedTuple

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.rescue import (
    execute_embedding_rescue,
    plan_embedding_rescue,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    ArchiveEmbeddingWrite,
    upsert_message_embeddings,
)
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_MODEL = "voyage-4"
_RECIPE = EmbeddingRecipe.current(model=_MODEL, dimensions=EMBEDDING_DIMENSION)

_INDEX_DDL = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,
    title TEXT,
    sort_key_ms INTEGER DEFAULT 0,
    authored_user_message_count INTEGER NOT NULL DEFAULT 1,
    assistant_message_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT 'authored prose long enough for embedding',
    role TEXT NOT NULL DEFAULT 'user',
    message_type TEXT NOT NULL DEFAULT 'message',
    material_origin TEXT NOT NULL DEFAULT 'human_authored',
    word_count INTEGER NOT NULL DEFAULT 8,
    content_hash BLOB NOT NULL
);
"""


def _content_hash(seed: str) -> bytes:
    return (seed.encode("utf-8") * 32)[:32]


def _connect_index(path: Path, *, sessions: list[str], messages: dict[str, list[tuple[str, bytes]]]) -> None:
    """Build a minimal index.db: sessions + messages with explicit content hashes."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_INDEX_DDL)
        conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
        for session_id in sessions:
            conn.execute(
                "INSERT INTO sessions (session_id, origin) VALUES (?, ?)",
                (session_id, "codex-session"),
            )
        for session_id, entries in messages.items():
            for message_id, content_hash in entries:
                conn.execute(
                    "INSERT INTO messages (message_id, session_id, content_hash) VALUES (?, ?, ?)",
                    (message_id, session_id, content_hash),
                )
        conn.commit()
    finally:
        conn.close()


def _connect_embeddings_tier(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    return conn


def _vector_for(seed: int) -> list[float]:
    return [float(seed) + i * 0.001 for i in range(EMBEDDING_DIMENSION)]


def _write_retired(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    session_id: str,
    content_hash: bytes,
    model: str = _MODEL,
    seed: int = 1,
) -> None:
    upsert_message_embeddings(
        conn,
        [
            ArchiveEmbeddingWrite(
                message_id=message_id,
                session_id=session_id,
                origin=Origin.CODEX_SESSION,
                embedding=_vector_for(seed),
                model=model,
                embedded_at_ms=1_700_000_000_000,
                content_hash=content_hash,
            )
        ],
    )


def _target_embeddings_path(tmp_path: Path) -> Path:
    return tmp_path / "embeddings.db"


def _connect_target_for_read(path: Path) -> sqlite3.Connection:
    """Plain connection to the target tier with sqlite-vec loaded for reads."""
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip("sqlite-vec extension is unavailable")
        raise AssertionError("unreachable") from error
    return conn


class TestPlanEmbeddingRescue:
    def test_classifies_matched_missing_hash_mismatch_model_mismatch(self, tmp_path: Path) -> None:
        # Session "full": both messages have exact retired matches -> fully rescuable.
        full = "codex-session:full"
        full_m1, full_m2 = f"{full}:m1", f"{full}:m2"
        hash_m1, hash_m2 = _content_hash("m1"), _content_hash("m2")

        # Session "partial": one matched, one missing from the retired tier.
        partial = "codex-session:partial"
        partial_m1, partial_m2 = f"{partial}:m1", f"{partial}:m2"
        hash_p1, hash_p2 = _content_hash("p1"), _content_hash("p2")

        # Session "stale": one message whose retired content_hash no longer matches.
        stale = "codex-session:stale"
        stale_m1 = f"{stale}:m1"
        hash_s1_now = _content_hash("s1-now")
        hash_s1_retired = _content_hash("s1-retired")

        # Session "wrong-model": retired vector was embedded under a different model.
        wrong_model = "codex-session:wrong-model"
        wrong_model_m1 = f"{wrong_model}:m1"
        hash_w1 = _content_hash("w1")

        _connect_index(
            tmp_path / "index.db",
            sessions=[full, partial, stale, wrong_model],
            messages={
                full: [(full_m1, hash_m1), (full_m2, hash_m2)],
                partial: [(partial_m1, hash_p1), (partial_m2, hash_p2)],
                stale: [(stale_m1, hash_s1_now)],
                wrong_model: [(wrong_model_m1, hash_w1)],
            },
        )

        retired_path = tmp_path / "retired-embeddings.db"
        retired_conn = _connect_embeddings_tier(retired_path)
        _write_retired(retired_conn, message_id=full_m1, session_id=full, content_hash=hash_m1, seed=1)
        _write_retired(retired_conn, message_id=full_m2, session_id=full, content_hash=hash_m2, seed=2)
        _write_retired(retired_conn, message_id=partial_m1, session_id=partial, content_hash=hash_p1, seed=3)
        # partial_m2 intentionally absent from the retired tier.
        _write_retired(retired_conn, message_id=stale_m1, session_id=stale, content_hash=hash_s1_retired, seed=4)
        _write_retired(
            retired_conn,
            message_id=wrong_model_m1,
            session_id=wrong_model,
            content_hash=hash_w1,
            model="voyage-3",
            seed=5,
        )
        retired_conn.close()

        report = plan_embedding_rescue(tmp_path / "index.db", retired_path, recipe=_RECIPE)

        assert report.counts.eligible_sessions == 4
        assert report.counts.fully_rescuable_sessions == 1
        assert report.counts.rescuable_messages == 2
        assert report.counts.partial_sessions == 1
        assert report.counts.partial_matched_messages == 1
        assert report.counts.skipped_missing == 1
        assert report.counts.skipped_hash_mismatch == 1
        assert report.counts.skipped_model_mismatch == 1
        assert report.model == _MODEL

        statuses = {sample.status for sample in report.samples}
        assert statuses == {"missing", "hash_mismatch", "model_mismatch"}

    def test_plan_never_mutates_either_database(self, tmp_path: Path) -> None:
        session_id = "codex-session:s1"
        m1 = f"{session_id}:m1"
        content_hash = _content_hash("m1")
        _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: [(m1, content_hash)]})
        retired_path = tmp_path / "retired-embeddings.db"
        retired_conn = _connect_embeddings_tier(retired_path)
        _write_retired(retired_conn, message_id=m1, session_id=session_id, content_hash=content_hash, seed=1)
        retired_conn.close()

        before_index = (tmp_path / "index.db").stat().st_mtime_ns
        before_retired = retired_path.stat().st_mtime_ns

        report_a = plan_embedding_rescue(tmp_path / "index.db", retired_path, recipe=_RECIPE)
        report_b = plan_embedding_rescue(tmp_path / "index.db", retired_path, recipe=_RECIPE)

        assert report_a.to_dict() == report_b.to_dict()
        assert (tmp_path / "index.db").stat().st_mtime_ns == before_index
        assert retired_path.stat().st_mtime_ns == before_retired
        assert not (tmp_path / "embeddings.db").exists()


class _SeededFullAndPartial(NamedTuple):
    full: str
    full_messages: tuple[str, str]
    partial: str
    retired_path: Path


class TestExecuteEmbeddingRescue:
    def _seed_full_and_partial(self, tmp_path: Path) -> _SeededFullAndPartial:
        full = "codex-session:full"
        full_m1, full_m2 = f"{full}:m1", f"{full}:m2"
        hash_m1, hash_m2 = _content_hash("m1"), _content_hash("m2")

        partial = "codex-session:partial"
        partial_m1, partial_m2 = f"{partial}:m1", f"{partial}:m2"
        hash_p1, hash_p2 = _content_hash("p1"), _content_hash("p2")

        _connect_index(
            tmp_path / "index.db",
            sessions=[full, partial],
            messages={
                full: [(full_m1, hash_m1), (full_m2, hash_m2)],
                partial: [(partial_m1, hash_p1), (partial_m2, hash_p2)],
            },
        )
        retired_path = tmp_path / "retired-embeddings.db"
        retired_conn = _connect_embeddings_tier(retired_path)
        _write_retired(retired_conn, message_id=full_m1, session_id=full, content_hash=hash_m1, seed=1)
        _write_retired(retired_conn, message_id=full_m2, session_id=full, content_hash=hash_m2, seed=2)
        _write_retired(retired_conn, message_id=partial_m1, session_id=partial, content_hash=hash_p1, seed=3)
        retired_conn.close()
        return _SeededFullAndPartial(
            full=full,
            full_messages=(full_m1, full_m2),
            partial=partial,
            retired_path=retired_path,
        )

    def test_rescues_only_the_fully_matched_session(self, tmp_path: Path) -> None:
        seed = self._seed_full_and_partial(tmp_path)
        target = _target_embeddings_path(tmp_path)

        report = execute_embedding_rescue(
            tmp_path / "index.db",
            seed.retired_path,
            target,
            mutation_authority="offline-exclusive",
            now_ms=1_800_000_000_000,
        )

        assert report.rescued_sessions == 1
        assert report.rescued_messages == 2
        assert report.counts.fully_rescuable_sessions == 1
        assert report.counts.partial_sessions == 1
        assert report.verified_sample_total >= 1
        assert report.verified_sample_ok == report.verified_sample_total
        assert report.ok is True

        conn = _connect_target_for_read(target)
        try:
            full_m1, full_m2 = seed.full_messages
            for message_id in (full_m1, full_m2):
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM message_embeddings WHERE message_id = ?", (message_id,)
                    ).fetchone()[0]
                    == 1
                )
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (message_id,)
                    ).fetchone()[0]
                    == 1
                )
            # The partial session must never be written -- no vectors at all.
            partial_row_count = conn.execute(
                "SELECT COUNT(*) FROM message_embeddings WHERE session_id = ?", (seed.partial,)
            ).fetchone()[0]
            assert partial_row_count == 0

            status = conn.execute(
                "SELECT message_count_embedded, needs_reindex, error_message FROM embedding_status WHERE session_id = ?",
                (seed.full,),
            ).fetchone()
            assert status == (2, 0, None)

            derivation = conn.execute(
                "SELECT attempt_state, message_count FROM embedding_derivation_state WHERE session_id = ?",
                (seed.full,),
            ).fetchone()
            assert derivation == ("succeeded", 2)
        finally:
            conn.close()

    def test_rescue_is_idempotent_on_rerun(self, tmp_path: Path) -> None:
        seed = self._seed_full_and_partial(tmp_path)
        target = _target_embeddings_path(tmp_path)

        first = execute_embedding_rescue(
            tmp_path / "index.db",
            seed.retired_path,
            target,
            mutation_authority="offline-exclusive",
            now_ms=1_800_000_000_000,
        )
        assert first.rescued_sessions == 1

        second = execute_embedding_rescue(
            tmp_path / "index.db",
            seed.retired_path,
            target,
            mutation_authority="offline-exclusive",
            now_ms=1_800_000_100_000,
        )

        assert second.rescued_sessions == 0
        assert second.skipped_already_fresh_sessions == 1
        assert second.more_pending is False

        conn = _connect_target_for_read(target)
        try:
            full_m1, _full_m2 = seed.full_messages
            row = conn.execute(
                "SELECT embedded_at_ms FROM message_embeddings_meta WHERE message_id = ?", (full_m1,)
            ).fetchone()
            # Second run must not have rewritten the already-rescued vector.
            assert row[0] == 1_800_000_000_000
        finally:
            conn.close()

    def test_limit_bounds_sessions_rescued_and_reports_more_pending(self, tmp_path: Path) -> None:
        full_a, full_b = "codex-session:full-a", "codex-session:full-b"
        m_a, m_b = f"{full_a}:m1", f"{full_b}:m1"
        hash_a, hash_b = _content_hash("a"), _content_hash("b")

        _connect_index(
            tmp_path / "index.db",
            sessions=[full_a, full_b],
            messages={full_a: [(m_a, hash_a)], full_b: [(m_b, hash_b)]},
        )
        retired_path = tmp_path / "retired-embeddings.db"
        retired_conn = _connect_embeddings_tier(retired_path)
        _write_retired(retired_conn, message_id=m_a, session_id=full_a, content_hash=hash_a, seed=1)
        _write_retired(retired_conn, message_id=m_b, session_id=full_b, content_hash=hash_b, seed=2)
        retired_conn.close()

        target = _target_embeddings_path(tmp_path)
        report = execute_embedding_rescue(
            tmp_path / "index.db",
            retired_path,
            target,
            limit=1,
            mutation_authority="offline-exclusive",
            now_ms=1_800_000_000_000,
        )

        assert report.rescued_sessions == 1
        assert report.more_pending is True

    def test_apply_requires_explicit_mutation_authority(self, tmp_path: Path) -> None:
        seed = self._seed_full_and_partial(tmp_path)
        with pytest.raises(RuntimeError, match="offline-exclusive"):
            execute_embedding_rescue(tmp_path / "index.db", seed.retired_path)

    def test_sample_verification_catches_a_corrupted_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anti-vacuity: a transit-corrupted vector must fail the sample check.

        Simulates a bug in the copy path (not a corrupt source) by
        monkeypatching the retired-vector reader to hand back a mutated
        vector for the single message written, then asserting the sample
        verification -- which re-reads both the just-written target row and
        the untouched retired source row -- reports the mismatch instead of
        silently passing.
        """
        session_id = "codex-session:solo"
        message_id = f"{session_id}:m1"
        content_hash = _content_hash("solo")

        _connect_index(
            tmp_path / "index.db",
            sessions=[session_id],
            messages={session_id: [(message_id, content_hash)]},
        )
        retired_path = tmp_path / "retired-embeddings.db"
        retired_conn = _connect_embeddings_tier(retired_path)
        _write_retired(retired_conn, message_id=message_id, session_id=session_id, content_hash=content_hash, seed=7)
        retired_conn.close()

        import polylogue.storage.embeddings.rescue as rescue_module

        real_reader = rescue_module._read_retired_vector

        def _corrupting_reader(conn: sqlite3.Connection, target_message_id: str) -> list[float] | None:
            vector = real_reader(conn, target_message_id)
            if vector is None:
                return None
            corrupted = list(vector)
            corrupted[0] = corrupted[0] + 999.0
            return corrupted

        monkeypatch.setattr(rescue_module, "_read_retired_vector", _corrupting_reader)

        target = _target_embeddings_path(tmp_path)
        report = execute_embedding_rescue(
            tmp_path / "index.db",
            retired_path,
            target,
            mutation_authority="offline-exclusive",
            now_ms=1_800_000_000_000,
        )

        assert report.rescued_sessions == 1
        assert report.verified_sample_total >= 1
        assert report.verified_sample_ok < report.verified_sample_total
        assert report.ok is False


def test_vector_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    """Guards the float32-blob roundtrip claim the rescue writer relies on.

    ``struct.unpack`` into Python floats (doubles) and re-``struct.pack``
    back to float32 must reproduce the exact original bytes -- this is what
    lets the writer accept ``list[float]`` without special-casing raw bytes
    while still promising byte-identical rescued vectors.
    """
    original = struct.pack(f"<{EMBEDDING_DIMENSION}f", *(0.1 * i for i in range(EMBEDDING_DIMENSION)))
    floats = list(struct.unpack(f"<{EMBEDDING_DIMENSION}f", original))
    roundtripped = struct.pack(f"<{EMBEDDING_DIMENSION}f", *floats)
    assert roundtripped == original
