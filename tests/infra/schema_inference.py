"""Shared archive fixtures for schema-inference gate routes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def seed_schema_inference_archive(root: Path) -> Path:
    """Create one source raw whose actual external file and blob agree."""

    initialize_active_archive_root(root)
    ground_truth = root.parent / f"{root.name}-codex-ground-truth"
    ground_truth.mkdir()
    payload = b"actual external codex raw"
    source_file = ground_truth / "session.jsonl"
    source_file.write_bytes(payload)
    blob_hash, blob_size = BlobStore(root / "blob").write_from_bytes(payload)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, logical_source_key, revision_authority
            ) VALUES ('raw-1', 'codex-session', 'session', ?, ?, ?, 100,
                      'codex:session', 'byte_proven')
            """,
            (str(source_file), bytes.fromhex(blob_hash), blob_size),
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, decision, decided_at_ms
            ) VALUES ('raw-1', 'codex:session', 'session', 'rev-1', ?, 1, 'applied', 100)
            """,
            (b"m" * 32,),
        )
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count)
            VALUES ('session', 'codex-session', 'raw-1', ?, 1)
            """,
            (b"s" * 32,),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, position, role, material_origin, content_hash)
            VALUES ('codex-session:session', 0, 'user', 'human_authored', ?)
            """,
            (b"n" * 32,),
        )
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text)
            VALUES ('codex-session:session:0.0', 'codex-session:session', 0, 'text', 'hello')
            """
        )
        conn.execute("ANALYZE blocks")
        conn.execute("ANALYZE messages")
        conn.execute("ANALYZE action_pairs")
    return ground_truth
