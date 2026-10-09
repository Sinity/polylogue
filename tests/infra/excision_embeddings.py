"""Synthetic paid-output inputs for the actual Native excision witness."""

from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.excision_execution import begin_excision_control
from tests.infra.storage_records import SessionBuilder


def begin_embedding_excision_control(
    root: Path, *, shared: bool, include_survivor: bool = True
) -> tuple[Any, Any, bytes]:
    """Use the existing vector producer; no provider or purchased request runs."""
    with write_lease("test.native-embedding-intent", archive_root=root):
        bootstrap_archive_root(root)
        builders = [
            SessionBuilder(root / "index.db", "selected-vector").provider("codex").add_message(text="Neutral"),
            SessionBuilder(root / "index.db", "surviving-vector").provider("codex").add_message(text="Other"),
        ]
        if not include_survivor:
            builders = builders[:1]
        for builder in builders:
            builder.save()
        vector_hash = bytes.fromhex("ab" * 32)
        with closing(
            open_isolated_write_connection(
                root / "embeddings.db", purpose="test.native-embedding-intent", archive_root=root
            )
        ) as embeddings:
            loaded, failure = try_load_sqlite_vec(embeddings)
            if not loaded:
                raise RuntimeError("actual vector fixture requires sqlite-vec") from failure
            with closing(
                open_isolated_write_connection(
                    root / "index.db", purpose="test.native-embedding-input", archive_root=root
                )
            ) as index:
                for ordinal, builder in enumerate(builders):
                    with connection_cursor(
                        index, "SELECT message_id FROM messages WHERE session_id=?", (builder.native_session_id(),)
                    ) as rows:
                        message_id = rows.fetchone()[0]
                    upsert_message_embedding(
                        embeddings,
                        message_id=message_id,
                        session_id=builder.native_session_id(),
                        origin=builder.conv.origin,
                        embedding=[0.25] * 1024,
                        model="synthetic-model",
                        embedded_at_ms=1,
                        vector_derivation_hash=vector_hash if ordinal == 0 or shared else bytes.fromhex("cd" * 32),
                    )
            embeddings.commit()
        started, args = begin_excision_control(root, builders[0].native_session_id(), reason="synthetic")
    return started, args, vector_hash


def prepare_embedding_excision_source_command(seal: Any, started: Any, args: Any) -> None:
    """Use the original closure's canonical producer before its paid child."""
    from polylogue.security.excision import _stage_excision_source_closure

    with seal.source_producer():
        _stage_excision_source_closure(
            seal,
            reason=args.reason,
            actor=args.actor,
            excised_at_ms=int(datetime.fromisoformat(started.plan.prepared_at).timestamp() * 1000),
        )


def seed_excision_session(
    archive_root: Path,
    *,
    native_id: str,
    payload: bytes = b'{"native_id": "x"}',
    with_message: bool = True,
    with_block: bool = True,
    with_embedding: bool = False,
) -> str:
    """Seed a minimal but real session spanning source.db + index.db (+ optionally embeddings.db)."""

    import sqlite3

    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        initialize_archive_database,
    )
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture

    initialize_active_archive_root(archive_root)

    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_runtime_source_fixture(source_db)
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    source_conn = sqlite3.connect(source_db)
    source_conn.execute("PRAGMA foreign_keys = ON")
    try:
        raw_id = write_source_raw_session(
            source_conn,
            origin="codex-session",
            source_path=f"/fake/{native_id}.jsonl",
            canonical_source_path=f"/fake/{native_id}.jsonl",
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
        from polylogue.storage.embeddings.identity import EmbeddingRecipe, vector_derivation_hash
        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

        embeddings_db = archive_root / "embeddings.db"
        initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
        emb_conn = sqlite3.connect(embeddings_db)
        try:
            try_load_sqlite_vec(emb_conn)
            # Content-addressed (polylogue-q88p): vectors/meta are keyed by
            # vector_derivation_hash, not message_id; message_embedding_refs
            # is the per-message mapping excision must delete from.
            input_hash = vector_derivation_hash(model="test-model", input_text=f"excision-fixture-{native_id}")
            recipe = EmbeddingRecipe.current(model="test-model", dimensions=1024)
            emb_conn.execute(
                "INSERT INTO message_embeddings (vector_derivation_hash, embedding, model) VALUES (?, ?, ?)",
                (input_hash.hex(), b"\x00\x00\x80\x3f" * 1024, "test-model"),
            )
            emb_conn.execute(
                "INSERT INTO message_embeddings_meta (vector_derivation_hash, model, dimension, recipe_hash, output_contract_hash) VALUES (?, ?, ?, ?, ?)",
                (input_hash, "test-model", 1024, recipe.recipe_hash, recipe.output_contract_hash),
            )
            emb_conn.execute(
                "INSERT INTO message_embedding_refs (message_id, session_id, origin, vector_derivation_hash) "
                "VALUES (?, ?, 'codex-session', ?)",
                (message_id, session_id, input_hash),
            )
            emb_conn.execute(
                "INSERT INTO embedding_status (session_id, message_count_embedded) VALUES (?, 1)",
                (session_id,),
            )
            emb_conn.commit()
        finally:
            emb_conn.close()

    return str(session_id)
