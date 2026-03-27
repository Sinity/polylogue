"""Action-event derived repair flows."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.maintenance_models import MaintenanceCategory

from .action_event_lifecycle import (
    action_event_read_model_status_sync,
    action_event_repair_candidates_sync,
    rebuild_action_event_read_model_sync,
    valid_action_event_source_ids_sync,
)
from .backends.connection import connection_context
from .fts_lifecycle import repair_fts_index_sync
from .repair_support import RepairResult


def repair_action_event_read_model(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        with connection_context(None) as conn:
            status = action_event_read_model_status_sync(conn)
            candidate_ids = action_event_repair_candidates_sync(conn)

            missing_conversations = max(
                0,
                int(status["valid_source_conversation_count"]) - int(status["materialized_conversation_count"]),
            )
            stale_conversations = int(status["stale_count"])
            action_fts_pending = max(0, int(status["count"]) - int(status["action_fts_count"]))
            pending = max(len(candidate_ids), missing_conversations + stale_conversations) + action_fts_pending

            if dry_run:
                return RepairResult(
                    name="action_event_read_model",
                    category=MaintenanceCategory.DERIVED_REPAIR,
                    destructive=False,
                    repaired_count=pending,
                    success=True,
                    detail=(
                        "Would: action-event read model already ready"
                        if pending == 0 and bool(status["rows_ready"]) and bool(status["action_fts_ready"])
                        else (
                            "Would: repair action-event rows for "
                            f"{len(candidate_ids):,} conversations; "
                            f"action FTS pending {action_fts_pending:,}"
                        )
                    ),
                )

            repaired = 0
            if candidate_ids:
                repaired = rebuild_action_event_read_model_sync(conn, conversation_ids=candidate_ids)

            if not bool(status["action_fts_ready"]):
                repair_targets = candidate_ids or valid_action_event_source_ids_sync(conn)
                if repair_targets:
                    repair_fts_index_sync(conn, repair_targets)

            conn.commit()
            refreshed = action_event_read_model_status_sync(conn)
            return RepairResult(
                name="action_event_read_model",
                category=MaintenanceCategory.DERIVED_REPAIR,
                destructive=False,
                repaired_count=repaired + action_fts_pending,
                success=bool(refreshed["ready"]),
                detail=(
                    "Action-event read model ready"
                    if refreshed["ready"]
                    else (
                        "Action-event read model still incomplete: "
                        f"{refreshed['materialized_conversation_count']:,}/"
                        f"{refreshed['valid_source_conversation_count']:,} conversations, "
                        f"action FTS {refreshed['action_fts_count']:,}/{refreshed['count']:,}"
                    )
                ),
            )
    except Exception as exc:
        return RepairResult(
            name="action_event_read_model",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"Failed to repair action-event read model: {exc}",
        )


def preview_action_event_read_model(*, count: int) -> RepairResult:
    return RepairResult(
        name="action_event_read_model",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=count,
        success=True,
        detail="Would: action-event read model already ready" if count == 0 else f"Would: repair action-event rows/fts for {count:,} pending items",
    )


__all__ = ["preview_action_event_read_model", "repair_action_event_read_model"]
