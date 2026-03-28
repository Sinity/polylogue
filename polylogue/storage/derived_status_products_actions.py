"""Action/search derived-model status builders."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.derived_status_support import pending_docs, pending_rows
from polylogue.storage.store import ACTION_EVENT_MATERIALIZER_VERSION


def build_action_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        "messages_fts": DerivedModelStatus(
            name="messages_fts",
            ready=bool(metrics["message_fts_ready"]),
            detail=(
                f"Messages FTS ready ({metrics['message_fts_rows']:,}/{metrics['total_messages']:,} rows)"
                if bool(metrics["message_fts_ready"])
                else f"Messages FTS pending ({metrics['message_fts_rows']:,}/{metrics['total_messages']:,} rows)"
            ),
            source_rows=int(metrics["total_messages"]),
            materialized_rows=int(metrics["message_fts_rows"]),
            pending_rows=pending_rows(int(metrics["total_messages"]), int(metrics["message_fts_rows"])),
        ),
        "action_events": DerivedModelStatus(
            name="action_events",
            ready=bool(metrics["action_rows_ready"]),
            detail=(
                f"Action-event rows ready ({metrics['action_documents']:,}/{metrics['action_source_documents']:,} conversations)"
                if bool(metrics["action_rows_ready"])
                else f"Action-event rows pending ({metrics['action_documents']:,}/{metrics['action_source_documents']:,} conversations)"
            ),
            source_documents=int(metrics["action_source_documents"]),
            materialized_documents=int(metrics["action_documents"]),
            materialized_rows=int(metrics["action_rows"]),
            pending_documents=pending_docs(int(metrics["action_source_documents"]), int(metrics["action_documents"])),
            stale_rows=int(metrics["action_stale_rows"]),
            orphan_rows=int(metrics["action_orphan_rows"]),
            materializer_version=ACTION_EVENT_MATERIALIZER_VERSION,
            matches_version=bool(metrics["action_matches_version"]),
        ),
        "action_events_fts": DerivedModelStatus(
            name="action_events_fts",
            ready=bool(metrics["action_fts_ready"]),
            detail=(
                f"Action-event FTS ready ({metrics['action_fts_rows']:,}/{metrics['action_rows']:,} rows)"
                if bool(metrics["action_fts_ready"])
                else f"Action-event FTS pending ({metrics['action_fts_rows']:,}/{metrics['action_rows']:,} rows)"
            ),
            source_rows=int(metrics["action_rows"]),
            materialized_rows=int(metrics["action_fts_rows"]),
            pending_rows=pending_rows(int(metrics["action_rows"]), int(metrics["action_fts_rows"])),
            orphan_rows=int(metrics["action_orphan_rows"]),
        ),
    }


__all__ = ["build_action_statuses"]
