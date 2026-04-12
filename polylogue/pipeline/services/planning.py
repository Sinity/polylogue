"""Canonical ingest planning for preview and runtime orchestration."""

from __future__ import annotations

from polylogue.config import Config, Source
from polylogue.protocols import ProgressCallback
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository

from .planning_backlog import collect_parse_backlog, collect_validation_backlog
from .planning_models import IngestPlan
from .planning_runtime import build_ingest_plan


class PlanningService:
    """Build canonical preview/runtime plans from source scans and DB state."""

    def __init__(self, backend: SQLiteBackend, config: Config):
        self.backend = backend
        self.repository = ConversationRepository(backend=backend)
        self.config = config

    async def collect_validation_backlog(
        self,
        *,
        source_names: list[str] | None,
        exclude_raw_ids: list[str] | None = None,
        force_reparse: bool = False,
    ) -> list[str]:
        return await collect_validation_backlog(
            self.backend,
            source_names=source_names,
            exclude_raw_ids=exclude_raw_ids,
            force_reparse=force_reparse,
        )

    async def collect_parse_backlog(
        self,
        *,
        source_names: list[str] | None,
        exclude_raw_ids: list[str] | None = None,
        force_reparse: bool = False,
    ) -> list[str]:
        return await collect_parse_backlog(
            self.backend,
            source_names=source_names,
            exclude_raw_ids=exclude_raw_ids,
            force_reparse=force_reparse,
        )

    async def build_plan(
        self,
        *,
        sources: list[Source],
        stage: str = "all",
        stage_sequence: list[str] | tuple[str, ...] | None = None,
        ui: object | None = None,
        progress_callback: ProgressCallback | None = None,
        preview: bool = False,
        force_reparse: bool = False,
    ) -> IngestPlan:
        return await build_ingest_plan(
            self,
            sources=sources,
            stage=stage,
            stage_sequence=stage_sequence,
            ui=ui,
            progress_callback=progress_callback,
            preview=preview,
            force_reparse=force_reparse,
        )


__all__ = ["IngestPlan", "PlanningService"]
