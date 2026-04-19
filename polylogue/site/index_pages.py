"""Aggregate index and dashboard generators for static-site generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from jinja2 import Environment

from polylogue.site.models import ArchiveIndexStats, ConversationIndex, ProviderIndex, SiteConfig
from polylogue.site.search import render_search_markup


def _provider_max_count(providers: tuple[ProviderIndex, ...]) -> int:
    return max((provider.conversation_count for provider in providers), default=1)


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
    providers = archive_stats.provider_indexes()
    await write_stream(
        template,
        output_dir / "index.html",
        title=config.title,
        description=config.description,
        search_markup=render_search_markup(config),
        conversations=archive_stats.root_page_conversations,
        total_conversations=archive_stats.total_conversations,
        total_messages=archive_stats.total_messages,
        providers=providers,
        provider_count=len(providers),
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
    providers = archive_stats.provider_indexes()
    for provider in providers:
        provider_dir = output_dir / Path(provider.path).parent
        provider_dir.mkdir(parents=True, exist_ok=True)

        await write_stream(
            template,
            provider_dir / "index.html",
            title=f"{provider.name} | {config.title}",
            description=f"Conversations from {provider.name}",
            search_markup=render_search_markup(config),
            conversations=conversation_iter_factory(provider.name),
            total_conversations=provider.conversation_count,
            total_messages=provider.message_count,
            providers=(),
            provider_count=1,
            generated_at=generated_at,
        )

    return len(providers)


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
    providers = archive_stats.provider_indexes()
    await write_stream(
        template,
        output_dir / "dashboard.html",
        title=config.title,
        providers=providers,
        max_count=_provider_max_count(providers),
        total_conversations=archive_stats.total_conversations,
        total_messages=archive_stats.total_messages,
        provider_count=len(providers),
        generated_at=generated_at,
    )
