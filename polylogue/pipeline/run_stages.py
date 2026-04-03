from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.pipeline.run_support import PARSE_STAGES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.pipeline.services.parsing import IngestResult
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


@dataclass(slots=True)
class RenderStageOutcome:
    rendered_count: int
    failures: list[dict[str, str]]
    total: int


@dataclass(slots=True)
class SchemaGenerationOutcome:
    generated: int
    failed: int


@dataclass(slots=True)
class MaterializeStageOutcome:
    item_count: int
    rebuilt: bool
    observation: dict[str, object] | None = None


@dataclass(slots=True)
class IndexStageOutcome:
    indexed: bool
    item_count: int
    error: str | None = None


async def execute_acquire_stage(
    *,
    config: Config,
    backend: SQLiteBackend,
    sources,
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
):
    from polylogue.pipeline.services.acquisition import AcquisitionService

    acquire_service = AcquisitionService(backend=backend)
    return await acquire_service.acquire_sources(
        sources,
        ui=ui,
        progress_callback=progress_callback,
        drive_config=config.drive_config,
    )


async def execute_ingest_stage(
    *,
    config: Config,
    repository: ConversationRepository,
    archive_root,
    sources,
    stage: str,
    skip_acquire: bool = False,
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IngestResult:
    from polylogue.pipeline.services.parsing import ParsingService

    parsing_service = ParsingService(
        repository=repository,
        archive_root=archive_root,
        config=config,
    )
    return await parsing_service.ingest_sources(
        sources=sources,
        stage=stage,
        ui=ui,
        progress_callback=progress_callback,
        parse_records=stage in PARSE_STAGES,
        skip_acquire=skip_acquire,
    )


async def execute_schema_generation_stage() -> SchemaGenerationOutcome:
    from polylogue.paths import data_home as _data_home
    from polylogue.paths import db_path as _db_path
    from polylogue.schemas.schema_inference import generate_all_schemas

    results = await asyncio.to_thread(
        generate_all_schemas,
        output_dir=_data_home() / "schemas",
        db_path=_db_path(),
    )
    return SchemaGenerationOutcome(
        generated=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
    )


async def execute_materialize_stage(
    *,
    stage: str,
    source_names: Sequence[str] | None,
    processed_ids: set[str],
    backend: SQLiteBackend,
    progress_callback: ProgressCallback | None = None,
) -> MaterializeStageOutcome:
    from polylogue.pipeline.services.ingest_batch import refresh_session_products_bulk
    from polylogue.storage.session_product_rebuild import rebuild_session_products_async

    if stage == "all":
        conversation_ids = sorted(processed_ids)
        if not conversation_ids:
            return MaterializeStageOutcome(item_count=0, rebuilt=False)
        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{len(conversation_ids)}")
        observation = await refresh_session_products_bulk(backend, conversation_ids)
        return MaterializeStageOutcome(
            item_count=len(conversation_ids),
            rebuilt=False,
            observation=observation,
        )

    if stage != "materialize":
        return MaterializeStageOutcome(item_count=0, rebuilt=False)

    if source_names:
        materialize_total = await backend.queries.count_conversation_ids(
            source_names=list(source_names)
        )
        if not materialize_total:
            return MaterializeStageOutcome(item_count=0, rebuilt=False)
        conversation_ids = [
            conversation_id
            async for conversation_id in backend.queries.iter_conversation_ids(
                source_names=list(source_names)
            )
        ]
        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{materialize_total}")
        observation = await refresh_session_products_bulk(backend, conversation_ids)
        return MaterializeStageOutcome(
            item_count=materialize_total,
            rebuilt=False,
            observation=observation,
        )

    if progress_callback is not None:
        progress_callback(0, desc="Materializing: rebuilding all session products")
    async with backend.connection() as conn:
        counts = await rebuild_session_products_async(conn)
        await conn.commit()
    observation = {"mode": "rebuild", **counts}
    return MaterializeStageOutcome(
        item_count=int(counts.get("profiles", 0)),
        rebuilt=True,
        observation=observation,
    )


async def execute_render_stage(
    *,
    config: Config,
    backend: SQLiteBackend,
    stage: str,
    source_names: Sequence[str] | None,
    processed_ids: set[str],
    progress_callback: ProgressCallback | None = None,
    render_format: str = "html",
) -> RenderStageOutcome:
    from polylogue.pipeline.services.rendering import RenderService
    from polylogue.rendering.renderers import create_renderer

    render_ids = None
    render_total = 0
    if stage == "render":
        render_total = await backend.queries.count_conversation_ids(
            source_names=list(source_names) if source_names is not None else None
        )
        render_ids = backend.queries.iter_conversation_ids(
            source_names=list(source_names) if source_names is not None else None
        )
    else:
        render_ids = processed_ids
        render_total = len(processed_ids)

    if not render_total:
        return RenderStageOutcome(rendered_count=0, failures=[], total=0)

    if progress_callback is not None:
        progress_callback(0, desc=f"Rendering: 0/{render_total}")
    renderer = create_renderer(
        format=render_format,
        config=config,
        backend=backend,
    )
    render_service = RenderService(
        renderer=renderer,
        render_root=config.render_root,
        backend=backend,
    )
    render_result = await render_service.render_conversations(
        render_ids,
        total=render_total,
        progress_callback=progress_callback,
    )
    return RenderStageOutcome(
        rendered_count=render_result.rendered_count,
        failures=render_result.failures,
        total=render_total,
    )


async def execute_index_stage(
    *,
    config: Config,
    stage: str,
    source_names: Sequence[str] | None,
    processed_ids: set[str],
    backend: SQLiteBackend,
    progress_callback: ProgressCallback | None = None,
) -> IndexStageOutcome:
    from polylogue.pipeline.services.indexing import IndexService

    index_service = IndexService(config=config, backend=backend)
    try:
        if stage == "parse":
            if processed_ids:
                index_kwargs = (
                    {"progress_callback": progress_callback}
                    if progress_callback is not None
                    else {}
                )
                return IndexStageOutcome(
                    indexed=await index_service.update_index(processed_ids, **index_kwargs),
                    item_count=len(processed_ids),
                )
            return IndexStageOutcome(indexed=False, item_count=0)

        if stage == "index":
            if source_names:
                total = await backend.queries.count_conversation_ids(
                    source_names=list(source_names)
                )
                index_kwargs = (
                    {"progress_callback": progress_callback}
                    if progress_callback is not None
                    else {}
                )
                success = await index_service.update_index(
                    backend.queries.iter_conversation_ids(
                        source_names=list(source_names)
                    ),
                    **index_kwargs,
                )
                return IndexStageOutcome(indexed=success, item_count=total)
            total = await backend.queries.count_conversation_ids()
            rebuild_kwargs = (
                {"progress_callback": progress_callback}
                if progress_callback is not None
                else {}
            )
            return IndexStageOutcome(
                indexed=await index_service.rebuild_index(**rebuild_kwargs),
                item_count=total,
            )

        if stage == "all":
            idx = await index_service.get_index_status()
            if not idx["exists"]:
                rebuild_kwargs = (
                    {"progress_callback": progress_callback}
                    if progress_callback is not None
                    else {}
                )
                return IndexStageOutcome(
                    indexed=await index_service.rebuild_index(**rebuild_kwargs),
                    item_count=len(processed_ids),
                )
            if processed_ids:
                index_kwargs = (
                    {"progress_callback": progress_callback}
                    if progress_callback is not None
                    else {}
                )
                return IndexStageOutcome(
                    indexed=await index_service.update_index(processed_ids, **index_kwargs),
                    item_count=len(processed_ids),
                )
        return IndexStageOutcome(indexed=False, item_count=0)
    except Exception as exc:
        return IndexStageOutcome(
            indexed=False,
            item_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "IndexStageOutcome",
    "MaterializeStageOutcome",
    "RenderStageOutcome",
    "SchemaGenerationOutcome",
    "execute_acquire_stage",
    "execute_index_stage",
    "execute_ingest_stage",
    "execute_materialize_stage",
    "execute_render_stage",
    "execute_schema_generation_stage",
]
