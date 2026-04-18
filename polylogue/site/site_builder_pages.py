"""Page-generation helpers for the site builder."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.site.conversation_pages import write_template_stream
from polylogue.site.index_pages import generate_dashboard, generate_provider_indexes, generate_root_index
from polylogue.site.search import generate_pagefind_config, render_search_markup

if TYPE_CHECKING:
    from jinja2 import Template

    from polylogue.site.builder import SiteBuilder
    from polylogue.site.models import ArchiveIndexStats, ConversationIndex


async def write_template_stream_for_builder(
    builder: SiteBuilder,
    template: Template,
    output_path: Path,
    **context: object,
) -> None:
    del builder
    await write_template_stream(template, output_path, **context)


async def generate_root_index_for_builder(
    builder: SiteBuilder,
    archive_stats: ArchiveIndexStats,
    *,
    generated_at: str,
) -> None:
    await generate_root_index(
        output_dir=builder.output_dir,
        env=builder.env,
        config=builder.config,
        archive_stats=archive_stats,
        generated_at=generated_at,
        write_stream=builder._write_template_stream,
    )


async def generate_provider_indexes_for_builder(
    builder: SiteBuilder,
    archive_stats: ArchiveIndexStats,
    *,
    generated_at: str,
) -> int:
    async def _conversation_iter(provider: str | None) -> AsyncIterator[ConversationIndex]:
        async for conversation in builder._iter_conversation_indexes(provider=provider):
            yield conversation

    return await generate_provider_indexes(
        output_dir=builder.output_dir,
        env=builder.env,
        config=builder.config,
        archive_stats=archive_stats,
        generated_at=generated_at,
        conversation_iter_factory=_conversation_iter,
        write_stream=builder._write_template_stream,
    )


async def generate_dashboard_for_builder(
    builder: SiteBuilder,
    archive_stats: ArchiveIndexStats,
    *,
    generated_at: str,
) -> None:
    await generate_dashboard(
        output_dir=builder.output_dir,
        env=builder.env,
        config=builder.config,
        archive_stats=archive_stats,
        generated_at=generated_at,
        write_stream=builder._write_template_stream,
    )


def search_markup_for_builder(builder: SiteBuilder) -> str:
    return render_search_markup(builder.config)


def generate_pagefind_config_for_builder(builder: SiteBuilder) -> str:
    return generate_pagefind_config(builder.output_dir)


__all__ = [
    "generate_dashboard_for_builder",
    "generate_pagefind_config_for_builder",
    "generate_provider_indexes_for_builder",
    "generate_root_index_for_builder",
    "search_markup_for_builder",
    "write_template_stream_for_builder",
]
