"""Backfill ``messages.message_type`` for pre-#839 rows.

PR #836 / #944 taught the new-ingest materialization path how to assign
``MessageType.CONTEXT`` / ``MessageType.PROTOCOL`` to rows whose text
carries Claude Code or Codex context/protocol markers (see
``polylogue.archive.message.artifacts``). Rows ingested before that
landed still sit as the default ``message`` type, which made
runtime fallback classification the only source of truth — exactly what
issue #839 explicitly forbids.

This module owns the rematerialization side: counting and rewriting
those legacy rows in place using the same classifier, so persisted
``message_type`` becomes the single source of truth (AC #2) and produces
the before/after artifact-class evidence (AC #6).

The split out of ``storage/repair.py`` is the one called out in
``docs/plans/file-size-budgets.yaml`` for that file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.maintenance.models import MaintenanceCategory
from polylogue.maintenance.targets import build_maintenance_target_catalog

logger = get_logger(__name__)
_TARGET_NAME = "message_type_backfill"


@dataclass
class BackfillResult:
    """Result of a single ``message_type`` backfill pass.

    Mirrors ``storage.repair.RepairResult`` so the orchestrator in
    ``storage/repair.py`` can wrap this directly without re-coercion.
    """

    name: str
    category: MaintenanceCategory
    destructive: bool
    repaired_count: int
    success: bool
    detail: str = ""


def count_unclassified_message_type_sync(conn: sqlite3.Connection) -> int:
    """Count default-``message`` rows whose text actually carries a #839
    context or protocol marker.

    The previous implementation counted every default-``message`` row
    with non-empty text. That conflates ordinary dialogue with rows the
    backfill would actually rewrite and produced wildly inflated preview
    figures (every plain user/assistant turn was counted as a candidate).

    The current implementation walks the candidate rows and applies the
    same classifier the backfill uses, returning only the rows that
    would flip to ``context`` or ``protocol``. This is the preview count
    for the #839 message_type backfill and matches ``repaired_count``
    after the backfill runs.
    """
    from polylogue.archive.message.artifacts import classify_text_message_type

    candidates = 0
    for (text,) in conn.execute(
        "SELECT text FROM messages WHERE message_type = 'message' AND text IS NOT NULL AND text != ''"
    ):
        if classify_text_message_type(text) is not None:
            candidates += 1
    return candidates


def count_messages_by_type_sync(conn: sqlite3.Connection) -> dict[str, int]:
    """Return ``{message_type: count}`` across the messages table.

    Used to produce the #839 AC #6 before/after artifact-class evidence:
    callers snapshot this before running the backfill, run it, then
    snapshot again — the diff is the count of rows reclassified from
    ``message`` into ``context`` or ``protocol``.
    """
    counts: dict[str, int] = {}
    for message_type, count in conn.execute("SELECT message_type, COUNT(*) FROM messages GROUP BY message_type"):
        counts[str(message_type)] = int(count)
    return counts


def _backfill_pass(conn: sqlite3.Connection) -> int:
    """Walk every default-``message`` row and rewrite via the classifier.

    Uses rowid-keyset pagination because in-place UPDATE shrinks the
    WHERE-matching population, which would let OFFSET-based pagination
    skip rows on each batch.

    Returns the exact number of rows the classifier reclassified, as
    reported by ``cursor.rowcount`` on each UPDATE (not
    ``conn.total_changes``, which is monotonic across the entire
    connection lifetime and over-reports across multiple batches).
    """
    from polylogue.archive.message.artifacts import classify_text_message_type
    from polylogue.archive.message.types import MessageType

    updated = 0
    batch_size = 1000
    last_rowid = 0
    while True:
        rows = conn.execute(
            "SELECT rowid, text FROM messages"
            " WHERE message_type = 'message'"
            "  AND text IS NOT NULL AND text != ''"
            "  AND rowid > ?"
            " ORDER BY rowid"
            " LIMIT ?",
            (last_rowid, batch_size),
        ).fetchall()
        if not rows:
            break

        context_ids: list[int] = []
        protocol_ids: list[int] = []
        for rowid_val, text in rows:
            last_rowid = rowid_val
            mt = classify_text_message_type(text)
            if mt == MessageType.CONTEXT:
                context_ids.append(rowid_val)
            elif mt == MessageType.PROTOCOL:
                protocol_ids.append(rowid_val)

        if context_ids:
            placeholders = ",".join("?" * len(context_ids))
            result = conn.execute(
                f"UPDATE messages SET message_type = 'context' WHERE rowid IN ({placeholders})",
                context_ids,
            )
            updated += result.rowcount
        if protocol_ids:
            placeholders = ",".join("?" * len(protocol_ids))
            result = conn.execute(
                f"UPDATE messages SET message_type = 'protocol' WHERE rowid IN ({placeholders})",
                protocol_ids,
            )
            updated += result.rowcount
    return updated


def preview_backfill(*, count: int) -> BackfillResult:
    """Preview handler for the #839 message_type backfill.

    ``count`` should be the number of rows the classifier would actually
    rewrite (matching ``classify_text_message_type``), not the larger
    population of default-``message`` rows.
    """
    spec = build_maintenance_target_catalog().resolve_name(_TARGET_NAME)
    assert spec is not None, "message_type_backfill must be registered in the catalog"
    return BackfillResult(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        repaired_count=count,
        success=True,
        detail=(
            f"Would: classify {count:,} message rows as context or protocol"
            if count
            else "No messages need context/protocol classification"
        ),
    )


def run_backfill(_config: Config, *, dry_run: bool = False) -> BackfillResult:
    """Rematerialize pre-#839 rows so persisted ``message_type`` is the
    sole source of truth for context/protocol classification.

    Iterates messages whose ``message_type`` is still the default
    ``message``, applies the same classifier the materialization runtime
    uses, and updates to ``context`` or ``protocol`` where markers
    match.

    Idempotent: rows already classified are skipped by the WHERE clause
    on the next pass.
    """
    from polylogue.storage.sqlite.connection import connection_context

    spec = build_maintenance_target_catalog().resolve_name(_TARGET_NAME)
    assert spec is not None, "message_type_backfill must be registered in the catalog"

    try:
        with connection_context(None) as conn:
            if dry_run:
                return preview_backfill(count=count_unclassified_message_type_sync(conn))

            updated = _backfill_pass(conn)
            conn.commit()
            logger.info("message_type_backfill_complete", updated=updated)
            return BackfillResult(
                name=spec.name,
                category=spec.category,
                destructive=spec.destructive,
                repaired_count=updated,
                success=True,
                detail=f"Classified {updated:,} messages as context or protocol",
            )
    except Exception as exc:
        logger.exception("message_type_backfill_failed", error=str(exc))
        return BackfillResult(
            name=spec.name,
            category=spec.category,
            destructive=spec.destructive,
            repaired_count=0,
            success=False,
            detail=f"Backfill failed: {exc}",
        )


__all__ = [
    "BackfillResult",
    "count_messages_by_type_sync",
    "count_unclassified_message_type_sync",
    "preview_backfill",
    "run_backfill",
]
