"""Canonical synthetic vector archives for production projection tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.enums import Origin
from polylogue.storage.embeddings.identity import vector_derivation_hash
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def seed_vector_archive(
    root: Path,
    samples: list[tuple[str, str, str, list[float]]],
    *,
    model: str = "voyage-4",
) -> dict[tuple[str, str], tuple[str, str]]:
    """Seed exact index prose and its purchased vector, returning generated identities."""
    root.mkdir(parents=True, exist_ok=True)
    identities: dict[tuple[str, str], tuple[str, str]] = {}
    with sqlite3.connect(root / "index.db") as index, sqlite3.connect(root / "embeddings.db") as vectors:
        initialize_archive_tier(index, ArchiveTier.INDEX)
        initialize_archive_tier(vectors, ArchiveTier.EMBEDDINGS)
        for native_session, native_message, text, vector in samples:
            session_id = f"codex-session:{native_session}"
            message_id = f"{session_id}:n:{native_message}"
            index.execute(
                "INSERT OR IGNORE INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
                (native_session, Origin.CODEX_SESSION.value, native_session, b"s" * 32),
            )
            index.execute(
                """INSERT INTO messages (session_id, native_id, position, role, material_origin, word_count, content_hash)
                   VALUES (?, ?, (SELECT COUNT(*) FROM messages WHERE session_id = ?),
                           'user', 'human_authored', ?, ?)""",
                (session_id, native_message, session_id, len(text.split()), b"m" * 32),
            )
            index.execute(
                """INSERT INTO blocks (session_id, message_id, position, block_type, text, content_hash)
                   VALUES (?, ?, 0, 'text', ?, ?)""",
                (session_id, message_id, text, b"b" * 32),
            )
            upsert_message_embedding(
                vectors,
                message_id=message_id,
                session_id=session_id,
                origin=Origin.CODEX_SESSION,
                embedding=vector,
                model=model,
                embedded_at_ms=1_767_225_700_000,
                vector_derivation_hash=vector_derivation_hash(model=model, input_text=text),
            )
            identities[(native_session, native_message)] = (session_id, message_id)
    return identities
