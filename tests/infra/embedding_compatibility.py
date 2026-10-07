"""Synthetic archive and protocol fixtures for retained embedding compatibility."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Never

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import seeds_off_event_loop
from tests.infra.live_ingest import write_index_session

_PAYLOAD = json.loads((Path(__file__).parents[1] / "fixtures" / "embedding_compatibility.json").read_text())
_TEXT = _PAYLOAD["retained"]
_NEW_TEXT = _PAYLOAD["missing"]


@seeds_off_event_loop
def _session(root: Path, *, extra: bool = False) -> tuple[str, tuple[str, ...]]:
    messages = [
        ParsedMessage(
            provider_message_id=f"m{i}",
            role=Role.USER,
            text=text,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            material_origin=MaterialOrigin.HUMAN_AUTHORED,
        )
        for i, text in enumerate([_TEXT, _TEXT] + ([_NEW_TEXT] if extra else []))
    ]
    with ArchiveStore(root) as store:
        sid = write_index_session(
            store, ParsedSession(source_name=Provider.CODEX, provider_session_id="compatibility", messages=messages)
        )
    with closing(sqlite3.connect(root / "index.db")) as conn:
        ids = tuple(
            str(r[0])
            for r in conn.execute("SELECT message_id FROM messages WHERE session_id = ? ORDER BY message_id", (sid,))
        )
    return sid, ids


class _Documents:
    dimension = 1024

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[tuple[str, ...]] = []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "document"
        self.calls.append(tuple(texts))
        return [[0.1] * self.dimension for _ in texts]

    def query(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        raise AssertionError("document protocol fixture does not query")

    def query_by_session(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        raise AssertionError("document protocol fixture does not query")

    def scoped_query(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("document-only fixture does not perform scoped retrieval")

    async def read_similarity(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("document protocol fixture does not query")


def _rows(root: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    with closing(sqlite3.connect(root / "embeddings.db")) as conn:
        return (
            conn.execute("SELECT * FROM message_embeddings_meta ORDER BY vector_derivation_hash").fetchall(),
            conn.execute("SELECT * FROM message_embedding_refs ORDER BY message_id").fetchall(),
        )


def output_rows(root: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    with closing(sqlite3.connect(root / "embeddings.db")) as conn:
        assert try_load_sqlite_vec(conn)[0]
        return (
            conn.execute("SELECT * FROM message_embeddings_meta ORDER BY vector_derivation_hash").fetchall(),
            conn.execute(
                "SELECT vector_derivation_hash, embedding, model FROM message_embeddings ORDER BY vector_derivation_hash"
            ).fetchall(),
        )


def add_settled_sessions(root: Path, *, count: int) -> None:
    """Create synthetic current occurrences sharing the already-purchased output."""
    with ArchiveStore(root) as store:
        for ordinal in range(count):
            write_index_session(
                store,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=f"settled-{ordinal}",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m0",
                            role=Role.USER,
                            text=_TEXT,
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=_TEXT)],
                            material_origin=MaterialOrigin.HUMAN_AUTHORED,
                        )
                    ],
                ),
            )
    from polylogue.storage.embeddings.generations import EmbeddingGenerationStore
    from polylogue.storage.sqlite.write_lease import write_lease

    lifecycle_store = EmbeddingGenerationStore(root)
    with write_lease("fixture.settled-bindings", archive_root=root), lifecycle_store.writer_lock() as binding:
        with closing(sqlite3.connect(binding.database_path)) as conn:
            address = conn.execute("SELECT vector_derivation_hash FROM message_embeddings_meta").fetchone()[0]
            conn.execute("ATTACH DATABASE ? AS idx", (str(root / "index.db"),))
            conn.execute(
                """INSERT INTO message_embedding_refs
                (message_id, session_id, origin, message_content_hash, vector_derivation_hash, embedded_at_ms)
                SELECT m.message_id, m.session_id, s.origin, m.content_hash, ?, 0
                FROM idx.messages m JOIN idx.sessions s ON s.session_id = m.session_id
                WHERE NOT EXISTS(SELECT 1 FROM message_embedding_refs r WHERE r.message_id = m.message_id)""",
                (address,),
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        lifecycle_store.refresh_binding_contract(binding)


def clear_embedding_refs(root: Path) -> None:
    from polylogue.storage.embeddings.generations import EmbeddingGenerationStore
    from polylogue.storage.sqlite.write_lease import write_lease

    store = EmbeddingGenerationStore(root)
    with write_lease("fixture.missing-bindings", archive_root=root), store.writer_lock() as binding:
        with closing(sqlite3.connect(binding.database_path)) as conn:
            conn.execute("DELETE FROM message_embedding_refs")
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        store.refresh_binding_contract(binding)


def embedding_file_header(root: Path) -> bytes:
    """Observe the persistent SQLite journal contract without opening SQLite."""
    with (root / "embeddings.db").open("rb") as stream:
        return stream.read(20)


def assert_embedding_handles_settled(root: Path) -> None:
    """Behavior runs everywhere; physical descriptor observation requires procfs."""
    path = (root / "embeddings.db").resolve()
    assert not tuple(path.parent.glob("embeddings.db-*"))
    if Path("/proc/self/fd").is_dir():
        from tests.infra.native_sql_descriptor_probe import selected_file_descriptors

        metadata = path.stat()
        assert selected_file_descriptors((metadata.st_dev, metadata.st_ino)) == ()
