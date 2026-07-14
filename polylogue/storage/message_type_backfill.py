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
from functools import lru_cache

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


# Per-message text reconstruction. The ``messages`` table
# has no ``text`` column; message text lives in ``blocks`` (one row per
# content block). The classifier operates on the message's prose, which we
# reconstruct by concatenating the text of its blocks in position order.
# Uses the canonical message_prose_sql builder with single-newline separator
# and text-type filter to exclude thinking/tool blocks from classification.
_NEWLINE_SEP = "'\n'"  # SQL literal for single newline separator


@lru_cache(maxsize=1)
def _message_text_by_id_sql() -> str:
    """Lazily build the per-message text reconstruction query.

    ``message_prose_sql`` is imported here, not at module scope: this
    module is reachable from ``polylogue.insights.archive`` (via
    ``storage.repair``), and ``polylogue.storage.embeddings.materialization``
    -- via its package ``__init__.py``, which eagerly imports
    ``storage.embeddings.reconcile`` -- transitively re-imports
    ``insights.archive`` while it is still initializing. A module-scope
    import here is a circular import; deferring it to first call is not.
    """
    from polylogue.storage.embeddings.materialization import message_prose_sql

    return f"""
        SELECT m.message_id AS message_id,
               {message_prose_sql("m", separator=_NEWLINE_SEP, block_types=("text",))} AS text
        FROM messages m
        WHERE m.message_type = 'message'
        GROUP BY m.message_id
    """


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

    Message text is reconstructed from the ``blocks`` table because the
    ``messages`` table carries no ``text`` column.
    """
    from polylogue.archive.message.artifacts import classify_text_message_type

    candidates = 0
    for _message_id, text in conn.execute(_message_text_by_id_sql()):
        if text and classify_text_message_type(text) is not None:
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

    Message text is reconstructed from the archive ``blocks`` table (the
    archive ``messages`` table has no ``text`` column). Candidates are
    grouped per ``message_id`` and reclassified into ``context`` or
    ``protocol`` with batched UPDATEs keyed on ``message_id``.

    Returns the exact number of rows the classifier reclassified, as
    reported by ``cursor.rowcount`` on each UPDATE (not
    ``conn.total_changes``, which is monotonic across the entire
    connection lifetime and over-reports across multiple batches).
    """
    from polylogue.archive.message.artifacts import classify_text_message_type
    from polylogue.archive.message.types import MessageType

    context_ids: list[str] = []
    protocol_ids: list[str] = []
    for message_id, text in conn.execute(_message_text_by_id_sql()):
        if not text:
            continue
        mt = classify_text_message_type(text)
        if mt == MessageType.CONTEXT:
            context_ids.append(str(message_id))
        elif mt == MessageType.PROTOCOL:
            protocol_ids.append(str(message_id))

    updated = 0
    batch_size = 1000
    for target, ids in (("context", context_ids), ("protocol", protocol_ids)):
        for start in range(0, len(ids), batch_size):
            chunk = ids[start : start + batch_size]
            placeholders = ",".join("?" * len(chunk))
            result = conn.execute(
                f"UPDATE messages SET message_type = ? WHERE message_id IN ({placeholders})",
                (target, *chunk),
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
    from contextlib import closing

    from polylogue.paths import active_index_db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    spec = build_maintenance_target_catalog().resolve_name(_TARGET_NAME)
    assert spec is not None, "message_type_backfill must be registered in the catalog"

    try:
        with closing(open_connection(active_index_db_path())) as conn:
            conn.row_factory = sqlite3.Row
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
