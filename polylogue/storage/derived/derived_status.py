"""Canonical readiness/freshness snapshot for durable derived models.

Reads ``index.db`` directly for archive FTS/retrieval status and delegates
session-insight row/readiness accounting to the canonical
``session_insight_status_sync`` snapshot. This keeps derived-model readiness
from drifting away from the profile/work-event/phase materialization contract.

Concepts that are moot in the archive (#1743) degrade to an empty,
ready status rather than crashing:

* ``messages_fts`` — the search index is external-content over ``blocks``;
  message-level FTS readiness is reported off ``messages_fts``.
* ``threads_fts`` / ``session_tag_rollups`` — no dedicated FTS or rollup
  tables exist in ``index.db``; thread counts come from ``threads`` and rollups are
  moot.
* ``transcript_embeddings`` — embeddings live in a separate ``embeddings.db``
  tier, not ``index.db``; embedding stats are reported as empty/ready off the
  ``index.db`` connection (embedding readiness has its own surface).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import TypeAlias

from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage.derived.insights import build_archive_insight_statuses, pending_docs, pending_rows
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.insights.session.status import session_insight_status_sync

MetricValue: TypeAlias = int | bool
Metrics: TypeAlias = dict[str, MetricValue]
StatusMap: TypeAlias = Mapping[str, MetricValue]

_MESSAGE_FTS_TRIGGERS: tuple[str, ...] = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")


# ---------------------------------------------------------------------------
# Native table probes
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _count(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _message_fts_triggers_present(conn: sqlite3.Connection) -> bool:
    placeholders = ",".join("?" for _ in _MESSAGE_FTS_TRIGGERS)
    present = {
        str(row[0])
        for row in conn.execute(
            f"SELECT name FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})",
            _MESSAGE_FTS_TRIGGERS,
        ).fetchall()
    }
    return all(name in present for name in _MESSAGE_FTS_TRIGGERS)


# ---------------------------------------------------------------------------
# Message FTS (messages_fts over blocks)
# ---------------------------------------------------------------------------


def _message_fts_metrics(conn: sqlite3.Connection, *, verify_full: bool) -> Metrics:
    """Message-FTS readiness reported off the archive external-content ``messages_fts``.

    The archive search index indexes ``blocks`` (one row per text/tool block),
    not ``messages``. ``message_source_rows`` therefore reports the count of
    indexable blocks (those with non-empty text) and ``message_fts_rows`` the
    indexed ``messages_fts`` rows; ``blocks`` orphaned to a deleted message are
    structurally impossible under ``ON DELETE CASCADE``.
    """
    if not _table_exists(conn, "messages_fts"):
        return {
            "message_fts_exact_counts": verify_full,
            "message_source_rows": 0,
            "message_fts_rows": 0,
            "message_fts_ready": False,
        }

    triggers_present = _message_fts_triggers_present(conn)
    if verify_full:
        indexed_rows = _count(conn, "SELECT COUNT(*) FROM messages_fts")
        total_rows = _count(conn, "SELECT COUNT(*) FROM blocks WHERE NULLIF(text, '') IS NOT NULL")
        ready = triggers_present and indexed_rows == total_rows
        return {
            "message_fts_exact_counts": True,
            "message_source_rows": total_rows,
            "message_fts_rows": indexed_rows,
            "message_fts_ready": ready,
        }

    has_indexed_rows = bool(conn.execute("SELECT 1 FROM messages_fts_docsize LIMIT 1").fetchone())
    has_indexable_blocks = bool(
        conn.execute("SELECT 1 FROM blocks WHERE NULLIF(text, '') IS NOT NULL LIMIT 1").fetchone()
    )
    ready = triggers_present and (has_indexed_rows or not has_indexable_blocks)
    return {
        "message_fts_exact_counts": False,
        "message_source_rows": 0,
        "message_fts_rows": 0,
        "message_fts_ready": ready,
    }


# ---------------------------------------------------------------------------
# Session insights (session_profiles / work_events / phases / threads)
# ---------------------------------------------------------------------------


def _session_insight_metrics(session_status: SessionInsightStatusSnapshot) -> Metrics:
    return {
        "profile_rows": session_status.profile_row_count,
        "work_event_rows": session_status.work_event_inference_count,
        "work_event_fts_rows": session_status.work_event_inference_fts_count,
        "work_event_fts_duplicates": session_status.work_event_inference_fts_duplicate_count,
        "phase_rows": session_status.phase_count,
        "thread_rows": session_status.thread_count,
        "thread_fts_rows": session_status.thread_fts_count,
        "thread_fts_duplicates": session_status.thread_fts_duplicate_count,
        "total_thread_roots": session_status.root_threads,
        "tag_rollup_rows": session_status.tag_rollup_count,
        "expected_tag_rollup_rows": session_status.expected_tag_rollup_count,
        "missing_profile_rows": session_status.missing_profile_row_count,
        "stale_profile_rows": session_status.stale_profile_row_count,
        "orphan_profile_rows": session_status.orphan_profile_row_count,
        "expected_work_event_rows": session_status.expected_work_event_inference_count,
        "stale_work_event_rows": session_status.stale_work_event_inference_count,
        "orphan_work_event_rows": session_status.orphan_work_event_inference_count,
        "expected_phase_rows": session_status.expected_phase_count,
        "stale_phase_rows": session_status.stale_phase_count,
        "orphan_phase_rows": session_status.orphan_phase_count,
        "stale_thread_rows": session_status.stale_thread_count,
        "orphan_thread_rows": session_status.orphan_thread_count,
        "stale_tag_rollup_rows": session_status.stale_tag_rollup_count,
        "profile_rows_ready": session_status.profile_rows_ready,
        "work_event_rows_ready": session_status.work_event_inference_rows_ready,
        "work_event_fts_ready": session_status.work_event_inference_fts_ready,
        "phase_rows_ready": session_status.phase_rows_ready,
        "threads_ready": session_status.threads_ready,
        "thread_fts_ready": session_status.threads_fts_ready,
        "tag_rollups_ready": session_status.tag_rollups_ready,
    }


# ---------------------------------------------------------------------------
# Embeddings — live in a separate embeddings.db tier
# ---------------------------------------------------------------------------


def _embedding_metrics() -> Metrics:
    """Transcript-embedding metrics off the ``index.db`` connection.

    Embeddings persist in a separate ``embeddings.db`` tier, not ``index.db``;
    embedding readiness has its own surface (``polylogue ops embed status`` /
    ``embedding_status_payload``). From the ``index.db`` connection the model
    is reported as empty/ready so derived-status previews do not flag it as
    perpetual debt.
    """
    return {
        "embedded_sessions": 0,
        "embedded_messages": 0,
        "pending_sessions": 0,
        "stale_messages": 0,
        "missing_provenance": 0,
        "transcript_embeddings_ready": True,
    }


def _retrieval_metrics(
    metrics: StatusMap,
    *,
    session_status: SessionInsightStatusSnapshot,
) -> Metrics:
    evidence_rows = int(metrics["profile_rows"])
    expected_evidence_rows = int(metrics["profile_rows"])
    inference_rows = int(metrics["profile_rows"]) + int(metrics["work_event_fts_rows"])
    expected_inference_rows = int(metrics["profile_rows"]) + int(metrics["work_event_rows"])
    return {
        "evidence_retrieval_rows": evidence_rows,
        "expected_evidence_retrieval_rows": expected_evidence_rows,
        "evidence_retrieval_ready": True,
        "inference_retrieval_rows": inference_rows,
        "expected_inference_retrieval_rows": expected_inference_rows,
        "inference_retrieval_ready": session_status.work_event_inference_fts_ready,
        "enrichment_retrieval_rows": int(metrics["profile_rows"]),
        "expected_enrichment_retrieval_rows": int(metrics["profile_rows"]),
        "enrichment_retrieval_ready": True,
    }


def collect_derived_model_statuses_sync(
    conn: sqlite3.Connection,
    *,
    verify_full: bool = True,
) -> dict[str, DerivedModelStatus]:
    session_status = session_insight_status_sync(conn, verify_freshness=verify_full)
    metrics: Metrics = {
        "total_sessions": session_status.total_sessions,
    }
    metrics.update(_message_fts_metrics(conn, verify_full=verify_full))
    metrics.update(_session_insight_metrics(session_status))
    metrics.update(_embedding_metrics())
    metrics.update(
        _retrieval_metrics(
            metrics,
            session_status=session_status,
        )
    )

    return {
        **build_archive_insight_statuses(metrics),
        **build_retrieval_statuses(metrics),
    }


# ---------------------------------------------------------------------------
# Retrieval / embedding statuses
# ---------------------------------------------------------------------------


def build_retrieval_statuses(metrics: Metrics) -> dict[str, DerivedModelStatus]:
    return {
        "transcript_embeddings": DerivedModelStatus(
            name="transcript_embeddings",
            ready=bool(metrics["transcript_embeddings_ready"]),
            detail=(
                f"Transcript embeddings ready ({metrics['embedded_sessions']:,}/{metrics['total_sessions']:,} sessions, {metrics['embedded_messages']:,} messages)"
                if bool(metrics["transcript_embeddings_ready"])
                else (
                    f"Transcript embeddings pending ({metrics['embedded_sessions']:,}/{metrics['total_sessions']:,} sessions, "
                    f"pending {metrics['pending_sessions']:,}, stale {metrics['stale_messages']:,}, missing provenance {metrics['missing_provenance']:,})"
                )
            ),
            source_documents=int(metrics["total_sessions"]),
            materialized_documents=int(metrics["embedded_sessions"]),
            materialized_rows=int(metrics["embedded_messages"]),
            pending_documents=int(metrics["pending_sessions"]),
            stale_rows=int(metrics["stale_messages"]),
            missing_provenance_rows=int(metrics["missing_provenance"]),
        ),
        "retrieval_evidence": DerivedModelStatus(
            name="retrieval_evidence",
            ready=bool(metrics["evidence_retrieval_ready"]),
            detail=(
                f"Evidence retrieval ready ({metrics['evidence_retrieval_rows']:,}/{metrics['expected_evidence_retrieval_rows']:,} supporting rows)"
                if bool(metrics["evidence_retrieval_ready"])
                else (
                    f"Evidence retrieval pending ({metrics['evidence_retrieval_rows']:,}/{metrics['expected_evidence_retrieval_rows']:,} supporting rows)"
                )
            ),
            source_documents=int(metrics["total_sessions"]),
            materialized_documents=int(metrics["profile_rows"]),
            source_rows=int(metrics["expected_evidence_retrieval_rows"]),
            materialized_rows=int(metrics["evidence_retrieval_rows"]),
            pending_documents=pending_docs(int(metrics["total_sessions"]), int(metrics["profile_rows"])),
            pending_rows=pending_rows(
                int(metrics["expected_evidence_retrieval_rows"]), int(metrics["evidence_retrieval_rows"])
            ),
            stale_rows=0,
            orphan_rows=int(metrics["orphan_profile_rows"]),
        ),
        "retrieval_inference": DerivedModelStatus(
            name="retrieval_inference",
            ready=bool(metrics["inference_retrieval_ready"]),
            detail=(
                f"Inference retrieval ready ({metrics['inference_retrieval_rows']:,}/{metrics['expected_inference_retrieval_rows']:,} supporting rows)"
                if bool(metrics["inference_retrieval_ready"])
                else (
                    f"Inference retrieval pending ({metrics['inference_retrieval_rows']:,}/{metrics['expected_inference_retrieval_rows']:,} supporting rows; "
                    f"work_event_fts={metrics['work_event_fts_rows']:,}/{metrics['work_event_rows']:,})"
                )
            ),
            source_documents=int(metrics["profile_rows"]),
            materialized_documents=int(metrics["profile_rows"]),
            source_rows=int(metrics["expected_inference_retrieval_rows"]),
            materialized_rows=int(metrics["inference_retrieval_rows"]),
            pending_rows=pending_rows(
                int(metrics["expected_inference_retrieval_rows"]), int(metrics["inference_retrieval_rows"])
            ),
            stale_rows=(int(metrics["work_event_fts_duplicates"]) + int(metrics["stale_work_event_rows"])),
            orphan_rows=(int(metrics["orphan_profile_rows"]) + int(metrics["orphan_work_event_rows"])),
        ),
        "retrieval_enrichment": DerivedModelStatus(
            name="retrieval_enrichment",
            ready=bool(metrics["enrichment_retrieval_ready"]),
            detail=(
                f"Enrichment retrieval ready ({metrics['enrichment_retrieval_rows']:,}/{metrics['expected_enrichment_retrieval_rows']:,} supporting rows)"
                if bool(metrics["enrichment_retrieval_ready"])
                else (
                    f"Enrichment retrieval pending ({metrics['enrichment_retrieval_rows']:,}/{metrics['expected_enrichment_retrieval_rows']:,} supporting rows)"
                )
            ),
            source_rows=int(metrics["expected_enrichment_retrieval_rows"]),
            materialized_rows=int(metrics["enrichment_retrieval_rows"]),
            pending_rows=pending_rows(
                int(metrics["expected_enrichment_retrieval_rows"]), int(metrics["enrichment_retrieval_rows"])
            ),
            stale_rows=0,
        ),
    }


__all__ = ["build_retrieval_statuses", "collect_derived_model_statuses_sync", "pending_docs", "pending_rows"]
