from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.pipeline.ingest_support import PARSE_STAGES
from polylogue.pipeline.payload_types import (
    MaterializeStageObservation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.config import Source
    from polylogue.core.protocols import ProgressCallback
    from polylogue.pipeline.services.parsing import IngestResult
    from polylogue.pipeline.stage_models import AcquireResult
    from polylogue.storage.insights.session.runtime import SessionInsightCounts
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


@dataclass(slots=True)
class SchemaGenerationOutcome:
    generated: int
    failed: int


@dataclass(slots=True)
class MaterializeStageOutcome:
    item_count: int
    rebuilt: bool
    observation: MaterializeStageObservation | None = None


@dataclass(slots=True)
class IndexStageOutcome:
    indexed: bool
    item_count: int
    error: str | None = None


@dataclass(slots=True)
class EmbedStageOutcome:
    embedded_count: int
    error_count: int
    stats_only: bool = False


def _materialize_rebuild_observation(
    *,
    mode: str,
    counts: SessionInsightCounts,
) -> MaterializeStageObservation:
    return {
        "mode": mode,
        "profiles": counts.profiles,
        "work_events": counts.work_events,
        "phases": counts.phases,
        "threads": counts.threads,
        "tag_rollups": counts.tag_rollups,
    }


async def execute_acquire_stage(
    *,
    config: Config,
    backend: SQLiteBackend,
    sources: list[Source],
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AcquireResult:
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
    repository: SessionRepository,
    archive_root: Path,
    sources: list[Source],
    stage: str,
    skip_acquire: bool = False,
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
    raw_batch_size: int = 50,
    ingest_workers: int | None = None,
    measure_ingest_result_size: bool = False,
    force_write: bool = False,
) -> IngestResult:
    del repository, archive_root, skip_acquire, ui, progress_callback, raw_batch_size, ingest_workers
    del measure_ingest_result_size, force_write
    from polylogue.api.archive import _active_archive_root
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive
    from polylogue.pipeline.services.parsing_models import IngestResult, ParseResult
    from polylogue.pipeline.stage_models import AcquireResult

    resolved_archive_root = _active_archive_root(config)
    parse_result = (
        await parse_sources_archive(resolved_archive_root, sources) if stage in PARSE_STAGES else ParseResult()
    )
    return IngestResult(
        acquire_result=AcquireResult(),
        validation_result=None,
        parse_result=parse_result,
        parse_raw_ids=[],
        diagnostics={"batch_observations": {"batches": parse_result.batch_observations}},
    )


async def execute_schema_generation_stage() -> SchemaGenerationOutcome:
    from polylogue.paths import data_home as _data_home
    from polylogue.paths import db_path as _db_path
    from polylogue.schemas.operator.schema_inference import generate_all_schemas

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
    from polylogue.pipeline.services.ingest_batch import refresh_session_insights_bulk
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_async

    if stage in {"all", "reprocess"}:
        session_ids = sorted(processed_ids)
        if not session_ids:
            return MaterializeStageOutcome(item_count=0, rebuilt=False)

        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{len(session_ids)}")

        observation = await refresh_session_insights_bulk(backend, session_ids)
        return MaterializeStageOutcome(
            item_count=len(session_ids),
            rebuilt=False,
            observation=observation,
        )

    if stage != "materialize":
        return MaterializeStageOutcome(item_count=0, rebuilt=False)

    if source_names:
        scoped_source_names = list(source_names)
        materialize_total = await backend.count_session_ids(source_names=scoped_source_names)
        if not materialize_total:
            return MaterializeStageOutcome(item_count=0, rebuilt=False)
        session_ids = [session_id async for session_id in backend.iter_session_ids(source_names=scoped_source_names)]
        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{materialize_total}")
        observation = await refresh_session_insights_bulk(backend, session_ids)
        return MaterializeStageOutcome(
            item_count=materialize_total,
            rebuilt=False,
            observation=observation,
        )

    async with backend.connection() as conn:
        total_row = await (await conn.execute("SELECT COUNT(*) FROM sessions")).fetchone()
        total_sessions = int(total_row[0]) if total_row is not None else 0
        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{total_sessions}")
        counts = await rebuild_session_insights_async(
            conn,
            progress_callback=progress_callback,
            progress_total=total_sessions,
        )
        await conn.commit()
    observation = _materialize_rebuild_observation(mode="rebuild", counts=counts)
    return MaterializeStageOutcome(
        item_count=counts.profiles,
        rebuilt=True,
        observation=observation,
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
                index_kwargs = {"progress_callback": progress_callback} if progress_callback is not None else {}
                return IndexStageOutcome(
                    indexed=await index_service.update_index(processed_ids, **index_kwargs),
                    item_count=len(processed_ids),
                )
            return IndexStageOutcome(indexed=False, item_count=0)

        if stage == "index":
            if processed_ids:
                status = await index_service.get_index_status()
                if status["exists"]:
                    index_kwargs = {"progress_callback": progress_callback} if progress_callback is not None else {}
                    return IndexStageOutcome(
                        indexed=await index_service.update_index(processed_ids, **index_kwargs),
                        item_count=len(processed_ids),
                    )
            if source_names:
                scoped_source_names = list(source_names)
                total = await backend.count_session_ids(source_names=scoped_source_names)
                index_kwargs = {"progress_callback": progress_callback} if progress_callback is not None else {}
                success = await index_service.update_index(
                    backend.iter_session_ids(source_names=scoped_source_names),
                    **index_kwargs,
                )
                return IndexStageOutcome(indexed=success, item_count=total)
            total = await backend.count_session_ids()
            rebuild_kwargs = {"progress_callback": progress_callback} if progress_callback is not None else {}
            return IndexStageOutcome(
                indexed=await index_service.rebuild_index(**rebuild_kwargs),
                item_count=total,
            )

        if stage in {"all", "reprocess"}:
            status = await index_service.get_index_status()
            if not status["exists"]:
                rebuild_kwargs = {"progress_callback": progress_callback} if progress_callback is not None else {}
                return IndexStageOutcome(
                    indexed=await index_service.rebuild_index(**rebuild_kwargs),
                    item_count=len(processed_ids),
                )
            if processed_ids:
                # The parse stage repairs FTS for changed sessions as a
                # synchronous ingest side effect. Re-running the same repair in
                # the chained index stage doubles FTS I/O on large archives.
                return IndexStageOutcome(indexed=True, item_count=0)
        return IndexStageOutcome(indexed=False, item_count=0)
    except Exception as exc:
        return IndexStageOutcome(
            indexed=False,
            item_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )


async def execute_embed_stage(
    *,
    config: Config,
    backend: SQLiteBackend,
    session_id: str | None = None,
    model: str = "voyage-4",
    rebuild: bool = False,
    stats_only: bool = False,
    json_output: bool = False,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EmbedStageOutcome:
    """Execute the embedding stage using embed_runtime/embed_stats helpers."""
    del progress_callback

    import os

    import click

    if stats_only:
        from polylogue.storage.embeddings.status_payload import embedding_status_payload

        class _StatsEnv:
            def __init__(self, cfg: Config) -> None:
                self._config = cfg

            @property
            def config(self) -> Config:
                return self._config

        payload = embedding_status_payload(_StatsEnv(config))
        if json_output:
            import json as _json

            click.echo(_json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(f"Embedding status: {payload['status']}")
            click.echo(f"  Embedded: {payload['embedded_sessions']}/{payload['total_sessions']}")
            click.echo(f"  Pending:  {payload['pending_sessions']}")
        return EmbedStageOutcome(embedded_count=0, error_count=0, stats_only=True)

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_key:
        click.echo("Error: VOYAGE_API_KEY environment variable not set", err=True)
        click.echo("Set it with: export VOYAGE_API_KEY=your-api-key", err=True)
        raise click.Abort()

    from polylogue.storage.search_providers import create_vector_provider

    vec_provider = create_vector_provider(voyage_api_key=voyage_key)
    if vec_provider is None:
        click.echo("Error: sqlite-vec not available", err=True)
        raise click.Abort()

    if model != "voyage-4":
        vec_provider.model = model

    from polylogue.storage.embeddings.materialization import (
        embed_session_sync,
        iter_pending_sessions,
    )
    from polylogue.storage.repository import SessionRepository as _Repo

    repo = _Repo(backend=backend)

    if session_id:
        outcome = embed_session_sync(repo, vec_provider, session_id, fetch_title=True)
        if outcome.status == "not_found":
            click.echo(f"Error: Session {session_id} not found", err=True)
            raise click.Abort()
        if outcome.status in {"no_messages", "no_embeddable_messages"}:
            click.echo(f"No messages to embed in {outcome.session_id}")
            return EmbedStageOutcome(embedded_count=0, error_count=0)
        if outcome.status == "error":
            click.echo(f"Error embedding {session_id}: {outcome.error}", err=True)
            raise click.Abort()
        click.echo(f"✓ Embedded {outcome.session_id[:12]}")
        return EmbedStageOutcome(embedded_count=1, error_count=0)

    pending = iter_pending_sessions(backend, rebuild=rebuild, max_sessions=limit)
    if not pending:
        click.echo("All sessions are already embedded.")
        return EmbedStageOutcome(embedded_count=0, error_count=0)

    click.echo(f"Embedding {len(pending)} sessions...")
    embedded_count = 0
    error_count = 0
    for index, item in enumerate(pending, 1):
        outcome = embed_session_sync(repo, vec_provider, item.session_id)
        if outcome.status == "embedded":
            embedded_count += 1
        elif outcome.status in {"no_messages", "no_embeddable_messages"}:
            pass
        elif outcome.status == "error":
            error_count += 1
            label = item.title or item.session_id[:12]
            click.echo(f"Warning: [{index}/{len(pending)}] {label}: {outcome.error}", err=True)
    summary = f"✓ Embedded {embedded_count} sessions"
    if error_count:
        summary += f" ({error_count} errors)"
    click.echo(summary)
    return EmbedStageOutcome(embedded_count=embedded_count, error_count=error_count)


__all__ = [
    "EmbedStageOutcome",
    "IndexStageOutcome",
    "MaterializeStageOutcome",
    "SchemaGenerationOutcome",
    "execute_acquire_stage",
    "execute_embed_stage",
    "execute_index_stage",
    "execute_ingest_stage",
    "execute_materialize_stage",
    "execute_schema_generation_stage",
]
