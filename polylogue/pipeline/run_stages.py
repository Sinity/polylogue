from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from polylogue.config import Config
from polylogue.pipeline.run_support import PARSE_STAGES
from polylogue.storage.state_views import RenderFailurePayload
from polylogue.types import SearchProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.config import Source
    from polylogue.pipeline.services.parsing import IngestResult
    from polylogue.pipeline.stage_models import AcquireResult
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


@dataclass(slots=True)
class RenderStageOutcome:
    rendered_count: int
    failures: list[RenderFailurePayload]
    total: int
    observation: dict[str, object] | None = None


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


@dataclass(slots=True)
class SiteStageOutcome:
    conversations: int
    index_pages: int
    rendered_pages: int
    error: str | None = None


@dataclass(slots=True)
class EmbedStageOutcome:
    embedded_count: int
    error_count: int
    stats_only: bool = False


class _SiteOptions(TypedDict, total=False):
    output: Path
    title: str
    search: bool
    search_provider: str | SearchProvider
    dashboard: bool


def _site_option_path(options: dict[str, object], key: str, *, default: Path) -> Path:
    value = options.get(key)
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    return default


def _site_option_str(options: dict[str, object], key: str, *, default: str) -> str:
    value = options.get(key)
    return value if isinstance(value, str) and value.strip() else default


def _site_option_bool(options: dict[str, object], key: str, *, default: bool) -> bool:
    value = options.get(key)
    return value if isinstance(value, bool) else default


def _site_option_search_provider(
    options: dict[str, object],
    *,
    default: SearchProvider,
) -> SearchProvider:
    value = options.get("search_provider")
    if isinstance(value, SearchProvider):
        return value
    if isinstance(value, str) and value.strip():
        return SearchProvider.from_string(value)
    return default


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
    repository: ConversationRepository,
    archive_root: Path,
    sources: list[Source],
    stage: str,
    skip_acquire: bool = False,
    ui: object | None = None,
    progress_callback: ProgressCallback | None = None,
    raw_batch_size: int = 50,
    ingest_workers: int | None = None,
    measure_ingest_result_size: bool = False,
) -> IngestResult:
    from polylogue.pipeline.services.parsing import ParsingService

    parsing_service = ParsingService(
        repository=repository,
        archive_root=archive_root,
        config=config,
        raw_batch_size=raw_batch_size,
        ingest_workers=ingest_workers,
        measure_ingest_result_size=measure_ingest_result_size,
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

    if stage in {"all", "reprocess"}:
        conversation_ids = sorted(processed_ids)
        if not conversation_ids:
            return MaterializeStageOutcome(item_count=0, rebuilt=False)

        status = await backend.get_session_product_status()
        total_conversations = status.total_conversations
        profile_row_count = status.profile_row_count
        use_bounded_rebuild = profile_row_count == 0 and total_conversations == len(conversation_ids)
        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{len(conversation_ids)}")

        if use_bounded_rebuild:
            async with backend.connection() as conn:
                counts = await rebuild_session_products_async(
                    conn,
                    conversation_ids=conversation_ids,
                    progress_callback=progress_callback,
                    progress_total=len(conversation_ids),
                )
                await conn.commit()
            return MaterializeStageOutcome(
                item_count=len(conversation_ids),
                rebuilt=True,
                observation={"mode": "rebuild-from-empty", **counts.to_dict()},
            )

        observation = await refresh_session_products_bulk(backend, conversation_ids)
        return MaterializeStageOutcome(
            item_count=len(conversation_ids),
            rebuilt=False,
            observation=observation,
        )

    if stage != "materialize":
        return MaterializeStageOutcome(item_count=0, rebuilt=False)

    if source_names:
        scoped_source_names = list(source_names)
        materialize_total = await backend.count_conversation_ids(source_names=scoped_source_names)
        if not materialize_total:
            return MaterializeStageOutcome(item_count=0, rebuilt=False)
        conversation_ids = [
            conversation_id async for conversation_id in backend.iter_conversation_ids(source_names=scoped_source_names)
        ]
        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{materialize_total}")
        observation = await refresh_session_products_bulk(backend, conversation_ids)
        return MaterializeStageOutcome(
            item_count=materialize_total,
            rebuilt=False,
            observation=observation,
        )

    async with backend.connection() as conn:
        total_row = await (await conn.execute("SELECT COUNT(*) FROM conversations")).fetchone()
        total_conversations = int(total_row[0]) if total_row is not None else 0
        if progress_callback is not None:
            progress_callback(0, desc=f"Materializing: 0/{total_conversations}")
        counts = await rebuild_session_products_async(
            conn,
            progress_callback=progress_callback,
            progress_total=total_conversations,
        )
        await conn.commit()
    return MaterializeStageOutcome(
        item_count=counts.profiles,
        rebuilt=True,
        observation={"mode": "rebuild", **counts.to_dict()},
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

    if stage == "render":
        scoped_source_names = list(source_names) if source_names is not None else None
        render_total = await backend.count_conversation_ids(source_names=scoped_source_names)
        render_ids: Iterable[str] | AsyncIterator[str] = backend.iter_conversation_ids(source_names=scoped_source_names)
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
    observation: dict[str, object] = {"workers": render_result.worker_count}
    if render_result.rss_start_mb is not None:
        observation["rss_start_mb"] = render_result.rss_start_mb
    if render_result.rss_end_mb is not None:
        observation["rss_end_mb"] = render_result.rss_end_mb
    if render_result.rss_start_mb is not None and render_result.rss_end_mb is not None:
        observation["rss_delta_mb"] = round(render_result.rss_end_mb - render_result.rss_start_mb, 1)
    if render_result.max_current_rss_mb is not None:
        observation["max_current_rss_mb"] = render_result.max_current_rss_mb
    return RenderStageOutcome(
        rendered_count=render_result.rendered_count,
        failures=render_result.failures,
        total=render_total,
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
            if source_names:
                scoped_source_names = list(source_names)
                total = await backend.count_conversation_ids(source_names=scoped_source_names)
                index_kwargs = {"progress_callback": progress_callback} if progress_callback is not None else {}
                success = await index_service.update_index(
                    backend.iter_conversation_ids(source_names=scoped_source_names),
                    **index_kwargs,
                )
                return IndexStageOutcome(indexed=success, item_count=total)
            total = await backend.count_conversation_ids()
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
                index_kwargs = {"progress_callback": progress_callback} if progress_callback is not None else {}
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


async def execute_site_stage(
    *,
    backend: SQLiteBackend,
    repository: ConversationRepository,
    site_options: dict[str, object] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SiteStageOutcome:
    """Build a static HTML site from the archive."""
    from polylogue.paths import data_home
    from polylogue.site.builder import SiteBuilder, SiteConfig

    opts = site_options or {}
    output_path = _site_option_path(opts, "output", default=data_home() / "site")
    config = SiteConfig(
        title=_site_option_str(opts, "title", default="Polylogue Archive"),
        enable_search=_site_option_bool(opts, "search", default=True),
        search_provider=_site_option_search_provider(opts, default=SearchProvider.PAGEFIND),
        include_dashboard=_site_option_bool(opts, "dashboard", default=True),
    )

    builder = SiteBuilder(
        output_dir=output_path,
        config=config,
        backend=backend,
        repository=repository,
        progress_callback=progress_callback,
    )

    try:
        result = await asyncio.to_thread(builder.build)
        return SiteStageOutcome(
            conversations=result.archive.total_conversations,
            index_pages=result.outputs.total_index_pages,
            rendered_pages=result.outputs.rendered_conversation_pages,
        )
    except Exception as exc:
        return SiteStageOutcome(
            conversations=0,
            index_pages=0,
            rendered_pages=0,
            error=f"{type(exc).__name__}: {exc}",
        )


async def execute_embed_stage(
    *,
    config: Config,
    backend: SQLiteBackend,
    conversation_id: str | None = None,
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
        from polylogue.cli.embed_stats import show_embedding_stats

        class _StatsEnv:
            def __init__(self, cfg: Config) -> None:
                self._config = cfg

            @property
            def config(self) -> Config:
                return self._config

        show_embedding_stats(_StatsEnv(config), json_output=json_output)
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

    from polylogue.cli.embed_runtime import embed_batch, embed_single
    from polylogue.storage.repository import ConversationRepository as _Repo

    repo = _Repo(backend=backend)

    class _PlainUI:
        plain = True

        class _Console:
            @staticmethod
            def print(*args: object, **kwargs: object) -> None:
                del kwargs
                click.echo(" ".join(str(a) for a in args))

        console = _Console()

        class _NullProgress:
            def update(self, **kwargs: object) -> None:
                del kwargs

            def advance(self, advance: float = 1) -> None:
                del advance

            def __enter__(self) -> _PlainUI._NullProgress:
                return self

            def __exit__(self, *args: object) -> None:
                del args

        @staticmethod
        def progress(description: str, total: int = 0) -> _PlainUI._NullProgress:
            del description, total
            return _PlainUI._NullProgress()

    class _EmbedEnv:
        def __init__(self, ui_obj: _PlainUI) -> None:
            self.ui = ui_obj

    env = _EmbedEnv(_PlainUI())

    if conversation_id:
        embed_single(env, repo, vec_provider, conversation_id)
        return EmbedStageOutcome(embedded_count=1, error_count=0)

    embed_batch(env, repo, vec_provider, rebuild=rebuild, limit=limit)
    return EmbedStageOutcome(embedded_count=0, error_count=0)


__all__ = [
    "EmbedStageOutcome",
    "IndexStageOutcome",
    "MaterializeStageOutcome",
    "RenderStageOutcome",
    "SchemaGenerationOutcome",
    "SiteStageOutcome",
    "execute_acquire_stage",
    "execute_embed_stage",
    "execute_index_stage",
    "execute_ingest_stage",
    "execute_materialize_stage",
    "execute_render_stage",
    "execute_schema_generation_stage",
    "execute_site_stage",
]
