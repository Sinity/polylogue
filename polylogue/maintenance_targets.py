"""Canonical maintenance-target metadata shared across doctor, repair, and scenario surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.maintenance_models import MaintenanceCategory


class MaintenanceTargetMode(str, Enum):
    """User-facing maintenance action families."""

    REPAIR = "repair"
    CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class MaintenanceTargetSpec:
    """One named maintenance target with stable execution and reporting semantics."""

    name: str
    mode: MaintenanceTargetMode
    category: MaintenanceCategory
    destructive: bool
    description: str
    include_preview_when_healthy: bool = False
    doctor_health_operation: str = ""
    doctor_repair_operation: str = ""
    include_in_archive_health: bool = False
    archive_health_unhealthy_status: OutcomeStatus | None = None
    archive_health_requires_deep: bool = False
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "category": self.category.value,
            "destructive": self.destructive,
            "description": self.description,
            "include_preview_when_healthy": self.include_preview_when_healthy,
            "doctor_health_operation": self.doctor_health_operation,
            "doctor_repair_operation": self.doctor_repair_operation,
            "include_in_archive_health": self.include_in_archive_health,
            "archive_health_unhealthy_status": (
                self.archive_health_unhealthy_status.value if self.archive_health_unhealthy_status else None
            ),
            "archive_health_requires_deep": self.archive_health_requires_deep,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class MaintenanceTargetCatalog:
    """Canonical maintenance-target registry with grouping and alias resolution helpers."""

    specs: tuple[MaintenanceTargetSpec, ...]

    def by_name(self) -> dict[str, MaintenanceTargetSpec]:
        return {spec.name: spec for spec in self.specs}

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def names_for_mode(self, mode: MaintenanceTargetMode) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs if spec.mode is mode)

    def resolve_name(self, name: str) -> MaintenanceTargetSpec | None:
        for spec in self.specs:
            if spec.name == name or name in spec.aliases:
                return spec
        return None

    def resolve(self, names: tuple[str, ...]) -> tuple[MaintenanceTargetSpec, ...]:
        resolved: list[MaintenanceTargetSpec] = []
        seen: set[str] = set()
        for name in names:
            spec = self.resolve_name(name)
            if spec is None or spec.name in seen:
                continue
            seen.add(spec.name)
            resolved.append(spec)
        return tuple(resolved)

    def preview_target_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs if spec.include_preview_when_healthy)

    def archive_health_specs(self, *, deep: bool) -> tuple[MaintenanceTargetSpec, ...]:
        return tuple(
            spec
            for spec in self.specs
            if spec.include_in_archive_health and (deep or not spec.archive_health_requires_deep)
        )

    def maintenance_targets_for_operation_names(
        self, operation_names: tuple[str, ...]
    ) -> tuple[MaintenanceTargetSpec, ...]:
        operations = set(operation_names)
        return tuple(
            spec
            for spec in self.specs
            if spec.doctor_health_operation in operations or spec.doctor_repair_operation in operations
        )

    def repair_hint(self, names: tuple[str, ...], *, include_run_all: bool = False) -> str:
        commands = [f"`polylogue doctor --repair --target {spec.name}`" for spec in self.resolve(names)]
        if include_run_all:
            commands.append("`polylogue run all`")
        if not commands:
            return "Run `polylogue doctor --repair`."
        if len(commands) == 1:
            return f"Run {commands[0]}."
        return f"Run {', '.join(commands[:-1])}, or {commands[-1]}."

    def doctor_health_operations_for_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(
            tuple(spec.doctor_health_operation for spec in self.resolve(names) if spec.doctor_health_operation)
        )

    def doctor_repair_operations_for_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(
            tuple(spec.doctor_repair_operation for spec in self.resolve(names) if spec.doctor_repair_operation)
        )

    def help_text(self) -> str:
        names = self.names()
        if not names:
            return "Limit maintenance to named targets"
        if len(names) == 1:
            return f"Limit maintenance to named target {names[0]}"
        prefix = ", ".join(names[:-1])
        return f"Limit maintenance to named targets such as {prefix}, or {names[-1]}"


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


MAINTENANCE_TARGET_SPECS: tuple[MaintenanceTargetSpec, ...] = (
    MaintenanceTargetSpec(
        name="session_products",
        mode=MaintenanceTargetMode.REPAIR,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        description="Repair or rebuild the derived session-product read models.",
        include_preview_when_healthy=True,
        doctor_health_operation="project-session-product-health",
        doctor_repair_operation="materialize-session-products",
    ),
    MaintenanceTargetSpec(
        name="action_event_read_model",
        mode=MaintenanceTargetMode.REPAIR,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        description="Repair or rebuild the action-event read model and its FTS projection.",
        include_preview_when_healthy=True,
        doctor_health_operation="project-action-event-health",
        doctor_repair_operation="materialize-action-events",
        aliases=("action_events",),
    ),
    MaintenanceTargetSpec(
        name="dangling_fts",
        mode=MaintenanceTargetMode.REPAIR,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        description="Repair lexical message FTS rows that are missing or dangling.",
        include_preview_when_healthy=True,
        doctor_repair_operation="index-message-fts",
    ),
    MaintenanceTargetSpec(
        name="wal_checkpoint",
        mode=MaintenanceTargetMode.REPAIR,
        category=MaintenanceCategory.DATABASE_MAINTENANCE,
        destructive=False,
        description="Run a SQLite WAL checkpoint/truncate maintenance pass.",
    ),
    MaintenanceTargetSpec(
        name="orphaned_messages",
        mode=MaintenanceTargetMode.CLEANUP,
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        description="Delete message rows that reference missing conversations.",
        include_in_archive_health=True,
        archive_health_unhealthy_status=OutcomeStatus.ERROR,
    ),
    MaintenanceTargetSpec(
        name="orphaned_content_blocks",
        mode=MaintenanceTargetMode.CLEANUP,
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        description="Delete content blocks that reference missing conversations or messages.",
        include_in_archive_health=True,
        archive_health_unhealthy_status=OutcomeStatus.ERROR,
        archive_health_requires_deep=True,
    ),
    MaintenanceTargetSpec(
        name="empty_conversations",
        mode=MaintenanceTargetMode.CLEANUP,
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        description="Delete conversations that no longer contain any messages.",
        include_in_archive_health=True,
        archive_health_unhealthy_status=OutcomeStatus.WARNING,
    ),
    MaintenanceTargetSpec(
        name="orphaned_attachments",
        mode=MaintenanceTargetMode.CLEANUP,
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        description="Delete orphaned attachment references and unreferenced attachment rows.",
        include_in_archive_health=True,
        archive_health_unhealthy_status=OutcomeStatus.ERROR,
    ),
)


@lru_cache(maxsize=1)
def build_maintenance_target_catalog() -> MaintenanceTargetCatalog:
    return MaintenanceTargetCatalog(specs=MAINTENANCE_TARGET_SPECS)


SAFE_REPAIR_TARGETS = build_maintenance_target_catalog().names_for_mode(MaintenanceTargetMode.REPAIR)
CLEANUP_TARGETS = build_maintenance_target_catalog().names_for_mode(MaintenanceTargetMode.CLEANUP)
MAINTENANCE_TARGET_NAMES = build_maintenance_target_catalog().names()


__all__ = [
    "CLEANUP_TARGETS",
    "MAINTENANCE_TARGET_NAMES",
    "MAINTENANCE_TARGET_SPECS",
    "SAFE_REPAIR_TARGETS",
    "MaintenanceTargetCatalog",
    "MaintenanceTargetMode",
    "MaintenanceTargetSpec",
    "build_maintenance_target_catalog",
]
