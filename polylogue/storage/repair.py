"""Stable maintenance repair surface."""

from __future__ import annotations

from .repair_cleanup import (
    preview_empty_conversations,
    preview_orphaned_content_blocks,
    preview_orphaned_messages,
    repair_empty_conversations,
    repair_orphaned_attachments,
    repair_orphaned_content_blocks,
    repair_orphaned_messages,
)
from .repair_control import (
    run_archive_cleanup,
    run_safe_repairs,
    run_selected_maintenance,
)
from .repair_derived import (
    preview_action_event_read_model,
    preview_dangling_fts,
    preview_session_products,
    repair_action_event_read_model,
    repair_dangling_fts,
    repair_session_products,
    repair_wal_checkpoint,
)
from .repair_support import (
    CLEANUP_TARGETS,
    MAINTENANCE_TARGET_NAMES,
    SAFE_REPAIR_TARGETS,
    RepairResult,
)

__all__ = [
    "CLEANUP_TARGETS",
    "MAINTENANCE_TARGET_NAMES",
    "RepairResult",
    "SAFE_REPAIR_TARGETS",
    "preview_action_event_read_model",
    "preview_dangling_fts",
    "preview_empty_conversations",
    "preview_orphaned_content_blocks",
    "preview_orphaned_messages",
    "preview_session_products",
    "repair_action_event_read_model",
    "repair_dangling_fts",
    "repair_empty_conversations",
    "repair_orphaned_attachments",
    "repair_orphaned_content_blocks",
    "repair_orphaned_messages",
    "repair_session_products",
    "repair_wal_checkpoint",
    "run_archive_cleanup",
    "run_safe_repairs",
    "run_selected_maintenance",
]
