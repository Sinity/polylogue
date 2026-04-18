"""Aggregate index and dashboard generators for static-site generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from jinja2 import Environment

from polylogue.paths import safe_path_component
from polylogue.site.models import ArchiveIndexStats, ConversationIndex, SiteConfig
from polylogue.site.search import render_search_markup


async def generate_root_index(
    *,
    output_dir: Path,
    env: Environment,
    config: SiteConfig,
    archive_stats: ArchiveIndexStats,
    generated_at: str,
    write_stream: Callable[..., Awaitable[None]],
) -> None:
    """Generate root index.html from streamed archive aggregates."""
    template = env.get_template("index.html")
    await write_stream(
        template,
        output_dir / "index.html",
        title=config.title,
        description=config.description,
        search_markup=render_search_markup(config),
        conversations=archive_stats.root_page_conversations,
        total_conversations=archive_stats.total_conversations,
        total_messages=archive_stats.total_messages,
        providers=archive_stats.provider_counts,
        provider_count=len(archive_stats.provider_counts),
        generated_at=generated_at,
    )


async def generate_provider_indexes(
    *,
    output_dir: Path,
    env: Environment,
    config: SiteConfig,
    archive_stats: ArchiveIndexStats,
    generated_at: str,
    conversation_iter_factory: Callable[[str | None], AsyncIterator[ConversationIndex]],
    write_stream: Callable[..., Awaitable[None]],
) -> int:
    """Generate provider-scoped index pages without a full shared archive list."""
    template = env.get_template("index.html")

    for provider in archive_stats.provider_order:
        provider_dir = output_dir / safe_path_component(provider, fallback="provider")
        provider_dir.mkdir(parents=True, exist_ok=True)

        await write_stream(
            template,
            provider_dir / "index.html",
            title=f"{provider} | {config.title}",
            description=f"Conversations from {provider}",
            search_markup=render_search_markup(config),
            conversations=conversation_iter_factory(provider),
            total_conversations=archive_stats.provider_counts[provider],
            total_messages=archive_stats.provider_messages[provider],
            providers={},
            provider_count=1,
            generated_at=generated_at,
        )

    return len(archive_stats.provider_order)


async def generate_dashboard(
    *,
    output_dir: Path,
    env: Environment,
    config: SiteConfig,
    archive_stats: ArchiveIndexStats,
    generated_at: str,
    write_stream: Callable[..., Awaitable[None]],
) -> None:
    """Generate statistics dashboard from archive aggregates."""
    template = env.get_template("dashboard.html")
    await write_stream(
        template,
        output_dir / "dashboard.html",
        title=config.title,
        providers=archive_stats.provider_counts,
        max_count=max(archive_stats.provider_counts.values(), default=1),
        total_conversations=archive_stats.total_conversations,
        total_messages=archive_stats.total_messages,
        provider_count=len(archive_stats.provider_counts),
        generated_at=generated_at,
    )
