"""Maintenance target selection and preview/apply orchestration."""

from __future__ import annotations

from polylogue.config import Config

from .repair_cleanup import (
    preview_empty_conversations,
    preview_orphaned_content_blocks,
    preview_orphaned_messages,
    repair_empty_conversations,
    repair_orphaned_attachments,
    repair_orphaned_content_blocks,
    repair_orphaned_messages,
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
from .repair_support import CLEANUP_TARGETS, SAFE_REPAIR_TARGETS, RepairResult


def run_safe_repairs(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(SAFE_REPAIR_TARGETS)
    results: list[RepairResult] = []
    if "session_products" in selected:
        results.append(
            preview_session_products(count=preview_counts["session_products"])
            if dry_run and "session_products" in preview_counts
            else repair_session_products(config, dry_run=dry_run)
        )
    if "action_event_read_model" in selected:
        results.append(
            preview_action_event_read_model(count=preview_counts["action_event_read_model"])
            if dry_run and "action_event_read_model" in preview_counts
            else repair_action_event_read_model(config, dry_run=dry_run)
        )
    if "dangling_fts" in selected:
        results.append(
            preview_dangling_fts(count=preview_counts["dangling_fts"])
            if dry_run and "dangling_fts" in preview_counts
            else repair_dangling_fts(config, dry_run=dry_run)
        )
    if "wal_checkpoint" in selected:
        results.append(repair_wal_checkpoint(config, dry_run=dry_run))
    return results


def run_archive_cleanup(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(CLEANUP_TARGETS)
    results: list[RepairResult] = []
    if "orphaned_messages" in selected:
        results.append(
            preview_orphaned_messages(count=preview_counts["orphaned_messages"])
            if dry_run and "orphaned_messages" in preview_counts
            else repair_orphaned_messages(config, dry_run=dry_run)
        )
    if "orphaned_content_blocks" in selected:
        results.append(
            preview_orphaned_content_blocks(count=preview_counts["orphaned_content_blocks"])
            if dry_run and "orphaned_content_blocks" in preview_counts
            else repair_orphaned_content_blocks(config, dry_run=dry_run)
        )
    if "empty_conversations" in selected:
        results.append(
            preview_empty_conversations(count=preview_counts["empty_conversations"])
            if dry_run and "empty_conversations" in preview_counts
            else repair_empty_conversations(config, dry_run=dry_run)
        )
    if "orphaned_attachments" in selected:
        results.append(repair_orphaned_attachments(config, dry_run=dry_run))
    return results


def run_selected_maintenance(
    config: Config,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool = False,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    results: list[RepairResult] = []
    repair_targets = tuple(name for name in targets if name in SAFE_REPAIR_TARGETS)
    cleanup_targets = tuple(name for name in targets if name in CLEANUP_TARGETS)
    if repair:
        results.extend(
            run_safe_repairs(
                config,
                dry_run=dry_run,
                preview_counts=preview_counts,
                targets=repair_targets,
            )
        )
    if cleanup:
        results.extend(
            run_archive_cleanup(
                config,
                dry_run=dry_run,
                preview_counts=preview_counts,
                targets=cleanup_targets,
            )
        )
    return results


__all__ = [
    "run_archive_cleanup",
    "run_safe_repairs",
    "run_selected_maintenance",
]
