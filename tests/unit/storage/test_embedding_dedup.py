"""Content-addressed dedup proof for the embeddings tier (polylogue-q88p).

Fork/resume/auto-compaction replays and genuinely coincidental identical
prose both produce messages with byte-identical embedder input text across
different sessions. Content-addressing means that text is embedded exactly
once regardless of how many messages/sessions reference it -- a real API
cost saving in a lineage-heavy archive, and the same mechanism the 04kl
rescue relies on when landing vectors into the new layout.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
from polylogue.storage.embeddings.materialization import embed_archive_session_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_SHARED_TEXT = "This exact prose appears verbatim in two unrelated sessions."


class _FakeVectorProvider:
    model = "voyage-4"
    dimension = 1024

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "document"
        self.calls.append(list(texts))
        return [[0.25] * self.dimension for _ in texts]

    def upsert(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("archive materialization must use the archive embedding route")

    def query(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []

    def query_by_session(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


def _write_session(root: Path, *, native_id: str, text: str, origin: Provider = Provider.CODEX) -> str:
    with ArchiveStore(root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=origin,
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


def _connect_vec(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
    return conn


def test_identical_content_across_two_sessions_embeds_once(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_a = _write_session(root, native_id="dedup-a", text=_SHARED_TEXT)
    session_b = _write_session(root, native_id="dedup-b", text=_SHARED_TEXT)
    assert session_a != session_b

    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    _connect_vec(embeddings_db).close()

    provider = _FakeVectorProvider()
    outcome_a = embed_archive_session_sync(index_db, provider, session_a)
    outcome_b = embed_archive_session_sync(index_db, provider, session_b)
    assert outcome_a.status == "embedded"
    assert outcome_b.status == "embedded"
    # Each session's embed pass still calls the provider once (per-session
    # materialization does not consult prior vectors before calling out) --
    # the dedup win is in STORAGE, not in avoiding this particular pass's
    # API call. The real cost saving is a subsequent session that reaches
    # the freshness predicate already-fresh (see
    # test_embedding_rebuild_survival.py), or the rescue path skipping the
    # API entirely (see test_embedding_rescue_content_addressed_layout.py).
    assert len(provider.calls) == 2

    with _connect_vec(embeddings_db) as conn:
        vector_rows = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
        meta_rows = conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0]
        ref_rows = conn.execute("SELECT COUNT(*) FROM message_embedding_refs").fetchone()[0]
        ref_session_ids = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT session_id FROM message_embedding_refs").fetchall()
        }
        distinct_hashes = {
            str(row[0]) for row in conn.execute("SELECT embedding_input_hash FROM message_embedding_refs").fetchall()
        }

    assert vector_rows == 1, "identical text across two sessions must produce exactly one stored vector"
    assert meta_rows == 1, "identical text across two sessions must produce exactly one meta row"
    assert ref_rows == 2, "each message still gets its own message_id -> hash ref"
    assert ref_session_ids == {session_a, session_b}
    assert len(distinct_hashes) == 1, "both refs must point at the same content-addressed hash"


def test_dedup_first_write_wins_the_stored_vector_bytes(tmp_path: Path) -> None:
    """A second write for an already-known hash never overwrites the stored vector.

    Anti-vacuity: if the write path re-inserted on every embed pass instead
    of checking meta-row presence first, this test's second (different-
    valued) provider response would clobber the first, and the assertion
    below would fail.
    """
    root = tmp_path / "archive"
    session_a = _write_session(root, native_id="dedup-first-write-a", text=_SHARED_TEXT)
    session_b = _write_session(root, native_id="dedup-first-write-b", text=_SHARED_TEXT)

    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    _connect_vec(embeddings_db).close()

    class _DistinctValueProvider(_FakeVectorProvider):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            self.calls.append(list(texts))
            return [[self.value] * self.dimension for _ in texts]

    first_provider = _DistinctValueProvider(0.11)
    outcome_a = embed_archive_session_sync(index_db, first_provider, session_a)
    assert outcome_a.status == "embedded"

    second_provider = _DistinctValueProvider(0.99)
    outcome_b = embed_archive_session_sync(index_db, second_provider, session_b)
    assert outcome_b.status == "embedded"

    with _connect_vec(embeddings_db) as conn:
        row = conn.execute("SELECT embedding FROM message_embeddings").fetchone()
    import struct

    stored = struct.unpack("<1024f", row[0])
    assert stored[0] == pytest.approx(0.11), "the first write's vector must survive, not the second"
