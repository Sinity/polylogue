"""Synthetic canonical Index delegation evidence for read-product controls."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.identity import fixture_block_content_identity


def seed_delegations(archive_root: Path, *, count: int = 1, annotation_count: int = 0) -> None:
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(archive_root)
    initialize_archive_database(archive_root / "index.db", ArchiveTier.INDEX)
    for index in range(count):
        _seed_one_delegation(archive_root, "" if index == 0 else f"-{index}")

    from polylogue.core.enums import AssertionKind
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    with sqlite3.connect(archive_root / "user.db") as conn:
        for index in range(annotation_count):
            upsert_assertion(
                conn,
                assertion_id=f"neutral-label-{index}",
                scope_ref=None,
                target_ref="session:neutral-target",
                kind=AssertionKind.ANNOTATION,
                key="neutral-label",
                value={"_schema": "other@v1", "directive_mode": "direct"},
                author_ref="user:neutral",
                author_kind="user",
                evidence_refs=[],
                now_ms=1,
            )


def _seed_one_delegation(archive_root: Path, suffix: str) -> None:
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
            VALUES (?, 'claude-code-session', 'Parent', ?, 1, 2)
            """,
            (f"parent{suffix}", hashlib.sha256(f"parent{suffix}".encode()).digest()),
        )
        parent_id = conn.execute(
            "SELECT session_id FROM sessions WHERE origin = 'claude-code-session' AND native_id = ?",
            (f"parent{suffix}",),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, title, content_hash, created_at_ms, updated_at_ms, branch_type, parent_session_id
            ) VALUES (?, 'claude-code-session', 'Child', ?, 1, 2, 'subagent', ?)
            """,
            (f"child{suffix}", hashlib.sha256(f"child{suffix}".encode()).digest(), parent_id),
        )
        child_id = conn.execute(
            "SELECT session_id FROM sessions WHERE origin = 'claude-code-session' AND native_id = ?",
            (f"child{suffix}",),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash, occurred_at_ms)
            VALUES (?, 'dispatch', 0, 'assistant', 'message', ?, 1)
            """,
            (parent_id, b"m" * 32),
        )
        message_id = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND native_id = 'dispatch'", (parent_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO blocks ( message_id, session_id, position, block_type, tool_name, tool_id, semantic_type, tool_input , content_identity, content_occurrence) VALUES (?, ?, 0, 'tool_use', 'Task', 'task-1', 'subagent', '{\"prompt\":\"review\"}', ?, 0)",
            (
                message_id,
                parent_id,
                fixture_block_content_identity("tool_use", "Task", "task-1", "subagent", '{"prompt":"review"}'),
            ),
        )
        # block_id carries the message's semantic digest and occurrence; a literal
        # tool_id ("task-1") is a different value and would leave the join
        # in delegation_facts_source (index.py) unresolved.
        block_id = conn.execute(
            "SELECT block_id FROM blocks WHERE message_id = ? AND position = 0", (message_id,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO session_links (
                src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id,
                parent_tool_use_block_id, observed_at_ms
            ) VALUES (?, 'claude-code-session', ?, 'subagent', ?, ?, 1)
            """,
            (child_id, f"parent{suffix}", parent_id, block_id),
        )
