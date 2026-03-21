"""Static site builder for polylogue archives.

Generates a complete static HTML site with index pages, dashboard,
and client-side search support.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import IO, TYPE_CHECKING
from uuid import uuid4

from jinja2 import DictLoader, Environment, Template, select_autoescape

from polylogue.logging import get_logger
from polylogue.paths import safe_path_component
from polylogue.publication import (
    ArchivePublicationSummary,
    ArtifactProofSummary,
    OutputManifest,
    PublicationRunSummary,
    SiteOutputSummary,
    SitePublicationManifest,
)
from polylogue.rendering.core import build_rendered_message_payload
from polylogue.rendering.renderers.html import MarkdownRenderer, PygmentsHighlighter
from polylogue.storage.store import PublicationRecord
from polylogue.types import SearchProvider

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


def _format_summary_date(value: object, fmt: str, summary_id: str) -> str | None:
    if value is None:
        return None
    try:
        return value.strftime(fmt)
    except (AttributeError, ValueError) as exc:
        logger.debug("Timestamp format error for %s: %s", summary_id, exc)
        return str(value)[:10] if fmt == "%Y-%m-%d" else str(value)

# Default index page template
INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #16161a;
            --bg-elevated: #1e1e24;
            --text-primary: #f8f9fa;
            --text-secondary: #94a3b8;
            --text-muted: #6b7280;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --border: #2d2d35;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-elevated: #ffffff;
                --text-primary: #111827;
                --text-secondary: #4b5563;
                --text-muted: #9ca3af;
                --border: #e5e7eb;
            }
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 3rem 1.5rem;
        }

        header {
            margin-bottom: 3rem;
            text-align: center;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to bottom right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }

        .stat {
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .conversation-list {
            display: grid;
            gap: 1.5rem;
            list-style: none;
        }

        .conversation-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.2s;
        }

        .conversation-card:hover {
            border-color: var(--accent);
            box-shadow: 0 0 20px var(--accent-glow);
            transform: translateY(-2px);
        }

        .conversation-link {
            text-decoration: none;
            color: inherit;
        }

        .conversation-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .conversation-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
            flex-wrap: wrap;
        }

        .badge {
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
        }

        .sidebar {
            margin-bottom: 2rem;
        }

        .provider-list {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            list-style: none;
        }

        .provider-list a {
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.875rem;
            transition: all 0.2s;
        }

        .provider-list a:hover {
            border-color: var(--accent);
            color: var(--text-primary);
        }

        nav.nav-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
        }

        nav a {
            color: var(--accent);
            text-decoration: none;
        }

        .footer {
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        @media (max-width: 768px) {
            .container { padding: 2rem 1rem; }
            h1 { font-size: 2rem; }
            .stats { gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p class="subtitle">{{ description }}</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{{ total_conversations }}</div>
                    <div class="stat-label">Conversations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ total_messages }}</div>
                    <div class="stat-label">Messages</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ provider_count }}</div>
                    <div class="stat-label">Providers</div>
                </div>
            </div>
        </header>

        {{ search_markup | safe }}

        {% if providers %}
        <div class="sidebar">
            <ul class="provider-list">
                {% for provider, count in providers.items() %}
                <li><a href="{{ provider }}/index.html">{{ provider }} ({{ count }})</a></li>
                {% endfor %}
                <li><a href="dashboard.html">Dashboard</a></li>
            </ul>
        </div>
        {% endif %}

        <ul class="conversation-list">
            {% for conv in conversations %}
            <li class="conversation-card">
                <a href="{{ conv.path }}" class="conversation-link">
                    <h2 class="conversation-title">{{ conv.title or conv.id[:12] }}</h2>
                    <div class="conversation-meta">
                        <span class="badge">{{ conv.provider }}</span>
                        <span>{{ conv.message_count }} messages</span>
                        {% if conv.created_at %}
                        <span>{{ conv.created_at }}</span>
                        {% endif %}
                    </div>
                </a>
            </li>
            {% endfor %}
        </ul>

        <div class="footer">
            <p>Generated {{ generated_at }} by <a href="https://github.com/anthropics/polylogue">Polylogue</a></p>
        </div>
    </div>
</body>
</html>
"""

CONVERSATION_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} | Polylogue</title>
    <style>
        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #16161a;
            --bg-elevated: #1e1e24;
            --bg-code: #282c34;
            --text-primary: #f8f9fa;
            --text-secondary: #94a3b8;
            --text-muted: #6b7280;
            --accent: #6366f1;
            --border: #2d2d35;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-elevated: #ffffff;
                --bg-code: #f6f8fa;
                --text-primary: #111827;
                --text-secondary: #4b5563;
                --text-muted: #9ca3af;
                --border: #e5e7eb;
            }
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container { max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }

        .nav-back {
            color: var(--accent);
            text-decoration: none;
            display: inline-block;
            margin-bottom: 1.5rem;
        }

        h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }

        .conv-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .message {
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
            border-radius: 8px;
            border-left: 3px solid var(--border);
        }

        .message-user {
            background: rgba(99, 102, 241, 0.06);
            border-left-color: rgba(99, 102, 241, 0.5);
        }

        .message-assistant {
            background: rgba(16, 185, 129, 0.06);
            border-left-color: rgba(16, 185, 129, 0.5);
        }

        .message-system {
            background: rgba(245, 158, 11, 0.06);
            border-left-color: rgba(245, 158, 11, 0.5);
        }

        .message-tool {
            background: rgba(139, 92, 246, 0.06);
            border-left-color: rgba(139, 92, 246, 0.5);
            font-size: 0.85rem;
        }

        .message-role {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .message-body {
            word-break: break-word;
            line-height: 1.7;
        }

        .message-body p { margin-bottom: 0.75rem; }
        .message-body p:last-child { margin-bottom: 0; }

        .message-body ul, .message-body ol {
            margin: 0.5rem 0 0.75rem 1.5rem;
        }

        .message-body li { margin-bottom: 0.25rem; }

        .message-body blockquote {
            border-left: 3px solid var(--accent);
            padding-left: 1rem;
            margin: 0.5rem 0;
            color: var(--text-secondary);
        }

        .message-body a { color: var(--accent); }

        .message-body h1, .message-body h2, .message-body h3,
        .message-body h4, .message-body h5, .message-body h6 {
            margin: 1rem 0 0.5rem;
        }

        .message-body table {
            border-collapse: collapse;
            margin: 0.75rem 0;
            width: 100%;
        }

        .message-body th, .message-body td {
            border: 1px solid var(--border);
            padding: 0.4rem 0.75rem;
            text-align: left;
        }

        .message-body th {
            background: var(--bg-elevated);
            font-weight: 600;
        }

        pre {
            background: var(--bg-code, #282c34);
            padding: 0.75rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85em;
            margin: 0.5rem 0;
            border: 1px solid var(--border);
        }

        code {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.9em;
        }

        p code {
            background: var(--bg-elevated, #1e1e24);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
        }

        .highlight {
            background: var(--bg-code, #282c34) !important;
            border-radius: 6px;
            overflow-x: auto;
            margin: 0.5rem 0;
        }

        .highlight pre {
            margin: 0;
            padding: 0.75rem;
            background: transparent !important;
            border: none;
        }

        {{ highlight_css }}

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="../../index.html" class="nav-back">← Back to Index</a>
        <h1>{{ title }}</h1>
        <div class="conv-meta">
            <span class="badge">{{ provider }}</span>
            <span>{{ message_count }} messages</span>
            {% if updated_at %}<span>{{ updated_at }}</span>{% endif %}
        </div>

        <div data-pagefind-body>
        {% for msg in messages %}
        <div class="message message-{{ msg.role or 'unknown' }}">
            <div class="message-role">{{ msg.role or 'unknown' }}</div>
            <div class="message-body">{{ msg.html_content | safe }}</div>
        </div>
        {% endfor %}
        </div>

        <div class="footer">
            <p>Generated by <a href="https://github.com/anthropics/polylogue">Polylogue</a></p>
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard | {{ title }}</title>
    <style>
        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #16161a;
            --bg-elevated: #1e1e24;
            --text-primary: #f8f9fa;
            --text-secondary: #94a3b8;
            --text-muted: #6b7280;
            --accent: #6366f1;
            --border: #2d2d35;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-elevated: #ffffff;
                --text-primary: #111827;
                --text-secondary: #4b5563;
                --text-muted: #9ca3af;
                --border: #e5e7eb;
            }
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container { max-width: 1000px; margin: 0 auto; padding: 3rem 1.5rem; }

        h1 { font-size: 2rem; margin-bottom: 2rem; }

        .nav-back {
            color: var(--accent);
            text-decoration: none;
            display: inline-block;
            margin-bottom: 2rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--accent);
        }

        .stat-card .label {
            color: var(--text-muted);
            font-size: 0.875rem;
            text-transform: uppercase;
        }

        .provider-breakdown {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }

        .provider-breakdown h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }

        .provider-bar {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }

        .provider-name {
            width: 120px;
            font-weight: 500;
        }

        .bar-container {
            flex-grow: 1;
            height: 24px;
            background: var(--bg-elevated);
            border-radius: 4px;
            margin: 0 1rem;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            background: var(--accent);
            border-radius: 4px;
        }

        .provider-count {
            width: 80px;
            text-align: right;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="nav-back">← Back to Index</a>
        <h1>Archive Dashboard</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{{ total_conversations }}</div>
                <div class="label">Total Conversations</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ total_messages }}</div>
                <div class="label">Total Messages</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ provider_count }}</div>
                <div class="label">Providers</div>
            </div>
            {% if embedding_coverage is defined %}
            <div class="stat-card">
                <div class="value">{{ embedding_coverage }}%</div>
                <div class="label">Embedding Coverage</div>
            </div>
            {% endif %}
        </div>

        <div class="provider-breakdown">
            <h2>By Provider</h2>
            {% for provider, count in providers.items() %}
            <div class="provider-bar">
                <span class="provider-name">{{ provider }}</span>
                <div class="bar-container">
                    <div class="bar-fill" style="width: {{ (count / max_count * 100)|round }}%"></div>
                </div>
                <span class="provider-count">{{ count }}</span>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""


@dataclass
class SiteConfig:
    """Configuration for static site generation."""

    title: str = "Polylogue Archive"
    description: str = "AI conversation archive"
    enable_search: bool = True
    search_provider: SearchProvider = SearchProvider.PAGEFIND
    conversations_per_page: int = 100
    include_dashboard: bool = True


@dataclass
class ConversationIndex:
    """Indexed conversation for site generation."""

    id: str
    title: str
    provider: str
    created_at: str | None
    updated_at: str | None
    message_count: int
    preview: str
    path: str

    @classmethod
    def from_summary(cls, summary: object, message_count: int) -> ConversationIndex:
        sid = str(summary.id)
        provider = getattr(summary.provider, "value", str(summary.provider))
        return cls(
            id=sid,
            title=summary.display_title or sid[:12],
            provider=provider,
            created_at=_format_summary_date(summary.created_at, "%Y-%m-%d", sid),
            updated_at=_format_summary_date(summary.updated_at, "%Y-%m-%d %H:%M", sid),
            message_count=message_count,
            preview=summary.summary or "",
            path=f"{safe_path_component(provider, fallback='provider')}/{sid[:12]}/conversation.html",
        )


@dataclass
class ArchiveIndexStats:
    """Streaming archive aggregates used by site-generation surfaces."""

    root_page_conversations: list[ConversationIndex] = field(default_factory=list)
    provider_counts: dict[str, int] = field(default_factory=dict)
    provider_messages: dict[str, int] = field(default_factory=dict)
    provider_order: list[str] = field(default_factory=list)
    total_conversations: int = 0
    total_messages: int = 0

    def record(self, conversation: ConversationIndex, *, root_page_size: int) -> None:
        """Accumulate counters and the root index first page in scan order."""
        if len(self.root_page_conversations) < root_page_size:
            self.root_page_conversations.append(conversation)

        self.total_conversations += 1
        self.total_messages += conversation.message_count
        if conversation.provider not in self.provider_counts:
            self.provider_order.append(conversation.provider)
        self.provider_counts[conversation.provider] = (
            self.provider_counts.get(conversation.provider, 0) + 1
        )
        self.provider_messages[conversation.provider] = (
            self.provider_messages.get(conversation.provider, 0) + conversation.message_count
        )


@dataclass
class ConversationPageBuildStats:
    """Conversation-page materialization counts for one site build."""

    total: int = 0
    rendered: int = 0
    reused: int = 0
    failed: int = 0

    def record(self, status: str) -> None:
        self.total += 1
        if status == "rendered":
            self.rendered += 1
        elif status == "reused":
            self.reused += 1
        elif status == "failed":
            self.failed += 1


class SiteBuilder:
    """Build a static HTML site from a polylogue archive.

    Generates:
    - index.html: Root listing with recent conversations
    - {provider}/index.html: Per-provider index pages
    - {provider}/{id[:12]}/conversation.html: Individual conversation pages
    - dashboard.html: Archive statistics
    - search-index.json: For client-side search (lunr.js / pagefind)
    """

    SUMMARY_PAGE_SIZE = 500

    def __init__(
        self,
        output_dir: Path,
        config: SiteConfig | None = None,
        *,
        backend: SQLiteBackend | None = None,
        repository: ConversationRepository | None = None,
    ) -> None:
        """Initialize site builder.

        Args:
            output_dir: Directory for generated site
            config: Optional site configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config or SiteConfig()
        if repository is not None and backend is None:
            backend = repository.backend
        self._backend = backend
        self._repository = repository
        self._owns_storage = backend is None and repository is None

        # Set up markdown rendering with syntax highlighting
        self._highlighter = PygmentsHighlighter()
        self._md_renderer = MarkdownRenderer(self._highlighter)

        # Inject Pygments CSS into conversation template
        highlight_css = self._highlighter.get_css()
        conv_template = CONVERSATION_TEMPLATE.replace(
            "{{ highlight_css }}", highlight_css
        )

        self.env = Environment(
            loader=DictLoader({
                "index.html": INDEX_TEMPLATE,
                "dashboard.html": DASHBOARD_TEMPLATE,
            }),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
        )
        self.page_env = Environment(
            loader=DictLoader({
                "conversation.html": conv_template,
            }),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
        )

    def _open_storage(self) -> tuple[SQLiteBackend, ConversationRepository]:
        """Return the canonical storage pair used by site generation."""
        if self._backend is None:
            from polylogue.storage.backends.async_sqlite import SQLiteBackend

            self._backend = SQLiteBackend()
        if self._repository is None:
            from polylogue.storage.repository import ConversationRepository

            self._repository = ConversationRepository(backend=self._backend)
        return self._backend, self._repository

    def build(self, incremental: bool = True) -> SitePublicationManifest:
        """Build complete static site (sync entry point).

        Args:
            incremental: If True, skip conversation pages whose file is newer
                than the conversation's updated_at timestamp.

        Returns:
            Typed publication manifest for the completed site build.
        """
        return asyncio.run(self._build_async(incremental))

    async def _build_async(self, incremental: bool = True) -> SitePublicationManifest:
        """Async implementation of the site build."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        build_started = perf_counter()
        try:
            generated_display = datetime.now().strftime("%Y-%m-%d %H:%M")
            generated_at = datetime.now(timezone.utc).isoformat()
            archive_stats, conversation_pages = await self._scan_archive(
                incremental=incremental
            )

            # Generate pages (index pages always rebuilt; conversation pages respect incremental)
            await self._generate_root_index(archive_stats, generated_at=generated_display)
            provider_index_pages = await self._generate_provider_indexes(
                archive_stats,
                generated_at=generated_display,
            )

            dashboard_pages = 0
            if self.config.include_dashboard:
                await self._generate_dashboard(
                    archive_stats,
                    generated_at=generated_display,
                )
                dashboard_pages = 1

            # Generate search index
            search_status = "disabled"
            if self.config.enable_search and self.config.search_provider == "pagefind":
                search_status = await asyncio.to_thread(self._generate_pagefind_config)
            elif self.config.enable_search:
                search_status = "json_index_written"

            proof_summary = await self._artifact_proof_summary()
            latest_run = await self._latest_run_summary()
            artifact_manifest = await asyncio.to_thread(
                OutputManifest.scan,
                self.output_dir,
                include_hashes=True,
                exclude_paths={"site-manifest.json"},
            )
            duration_ms = int((perf_counter() - build_started) * 1000)
            manifest = SitePublicationManifest(
                publication_id=f"site-{uuid4().hex[:16]}",
                generated_at=generated_at,
                output_dir=str(self.output_dir),
                duration_ms=duration_ms,
                config={
                    "title": self.config.title,
                    "description": self.config.description,
                    "enable_search": self.config.enable_search,
                    "search_provider": str(self.config.search_provider),
                    "conversations_per_page": self.config.conversations_per_page,
                    "include_dashboard": self.config.include_dashboard,
                },
                archive=ArchivePublicationSummary(
                    total_conversations=archive_stats.total_conversations,
                    total_messages=archive_stats.total_messages,
                    provider_count=len(archive_stats.provider_counts),
                    provider_counts=dict(sorted(archive_stats.provider_counts.items())),
                    provider_messages=dict(sorted(archive_stats.provider_messages.items())),
                ),
                outputs=SiteOutputSummary(
                    root_index_pages=1,
                    provider_index_pages=provider_index_pages,
                    dashboard_pages=dashboard_pages,
                    total_index_pages=1 + provider_index_pages + dashboard_pages,
                    total_conversation_pages=conversation_pages.total,
                    rendered_conversation_pages=conversation_pages.rendered,
                    reused_conversation_pages=conversation_pages.reused,
                    failed_conversation_pages=conversation_pages.failed,
                    search_documents=(
                        archive_stats.total_conversations if self.config.enable_search else 0
                    ),
                    search_enabled=self.config.enable_search,
                    search_provider=(
                        str(self.config.search_provider) if self.config.enable_search else None
                    ),
                    search_status=search_status,
                    incremental=incremental,
                ),
                latest_run=latest_run,
                artifact_proof=proof_summary,
                artifacts=artifact_manifest,
            )
            manifest_path = self.output_dir / "site-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _, repository = self._open_storage()
            await repository.record_publication(
                PublicationRecord(
                    publication_id=manifest.publication_id,
                    publication_kind=manifest.publication_kind,
                    generated_at=manifest.generated_at,
                    output_dir=manifest.output_dir,
                    duration_ms=manifest.duration_ms,
                    manifest=manifest.model_dump(mode="json"),
                )
            )
            return manifest
        finally:
            if self._owns_storage and self._backend is not None:
                await self._backend.close()
                self._backend = None
                self._repository = None

    async def _iter_conversation_indexes(
        self,
        *,
        provider: str | None = None,
        backend: SQLiteBackend | None = None,
        repository: ConversationRepository | None = None,
    ) -> AsyncIterator[ConversationIndex]:
        """Yield lightweight conversation indexes in repository sort order."""
        if backend is None or repository is None:
            backend, repository = self._open_storage()

        async for summaries in repository.iter_summary_pages(
            page_size=self.SUMMARY_PAGE_SIZE,
            provider=provider,
        ):
            message_counts = await backend.queries.get_message_counts_batch(
                [str(summary.id) for summary in summaries]
            )
            for summary in summaries:
                yield ConversationIndex.from_summary(
                    summary,
                    message_counts.get(str(summary.id), 0),
                )

    def _search_document(self, conversation: ConversationIndex) -> dict[str, str]:
        """Build a JSON-search entry for a site index conversation."""
        return {
            "id": conversation.id,
            "title": conversation.title,
            "provider": conversation.provider,
            "preview": conversation.preview,
            "path": conversation.path,
        }

    async def _scan_archive(
        self,
        *,
        incremental: bool,
    ) -> tuple[ArchiveIndexStats, ConversationPageBuildStats]:
        """Scan archive summaries once and drive streaming site outputs from that pass."""
        _backend, repository = self._open_storage()
        stats = ArchiveIndexStats()
        page_stats = ConversationPageBuildStats()
        search_path = self.output_dir / "search-index.json"
        search_handle: IO[str] | None = None
        wrote_search_entry = False

        if self.config.enable_search and self.config.search_provider != "pagefind":
            search_handle = search_path.open("w", encoding="utf-8")
            search_handle.write("[")

        try:
            async for conversation in self._iter_conversation_indexes(
                backend=self._backend,
                repository=repository,
            ):
                stats.record(
                    conversation,
                    root_page_size=self.config.conversations_per_page,
                )
                if search_handle is not None:
                    if wrote_search_entry:
                        search_handle.write(",")
                    json.dump(self._search_document(conversation), search_handle)
                    wrote_search_entry = True
                page_stats.record(await self._generate_conversation_page(
                    repository,
                    conversation,
                    incremental=incremental,
                ))
        except Exception:
            if search_handle is not None:
                search_handle.close()
                search_path.unlink(missing_ok=True)
                search_handle = None
            raise
        finally:
            if search_handle is not None:
                search_handle.write("]")
                search_handle.close()

        return stats, page_stats

    async def _iter_conversation_page_messages(
        self,
        repo,
        conversation_id: str,
    ) -> AsyncIterator[dict[str, object]]:
        """Yield site message payloads lazily for a conversation page."""
        async for msg in repo.iter_messages(conversation_id):
            if not msg.text:
                continue
            yield build_rendered_message_payload(
                message_id=msg.id,
                role=msg.role or "unknown",
                text=msg.text,
                timestamp=msg.timestamp,
                parent_message_id=msg.parent_id,
                branch_index=msg.branch_index,
                render_html=self._md_renderer.render,
            )

    async def _generate_conversation_page(
        self,
        repository: ConversationRepository,
        conversation: ConversationIndex,
        *,
        incremental: bool = True,
    ) -> str:
        """Generate one conversation page, or keep an up-to-date existing one."""
        template = self.page_env.get_template("conversation.html")
        page_path = self.output_dir / conversation.path
        page_path.parent.mkdir(parents=True, exist_ok=True)

        if incremental and page_path.exists():
            if conversation.updated_at:
                try:
                    file_mtime = datetime.fromtimestamp(page_path.stat().st_mtime)
                    conv_updated = datetime.fromisoformat(conversation.updated_at)
                    if file_mtime > conv_updated:
                        return "reused"
                except (ValueError, OSError):
                    pass
            else:
                return "reused"

        try:
            await self._write_template_stream(
                template,
                page_path,
                title=conversation.title,
                provider=conversation.provider,
                message_count=conversation.message_count,
                updated_at=conversation.updated_at,
                messages=self._iter_conversation_page_messages(
                    repository,
                    conversation.id,
                ),
            )
            return "rendered"
        except Exception as exc:
            page_path.unlink(missing_ok=True)
            logger.warning(
                "Skipping conversation page %s: %s",
                conversation.id,
                exc,
            )
            return "failed"

    async def _write_template_stream(
        self,
        template: Template,
        output_path: Path,
        **context: object,
    ) -> None:
        """Render a template to disk without materializing the full output string."""
        stream = template.generate_async(**context)
        with output_path.open("w", encoding="utf-8") as handle:
            async for chunk in stream:
                handle.write(chunk)

    async def _generate_root_index(
        self,
        archive_stats: ArchiveIndexStats,
        *,
        generated_at: str,
    ) -> None:
        """Generate root index.html from streamed archive aggregates."""
        template = self.env.get_template("index.html")
        await self._write_template_stream(
            template,
            self.output_dir / "index.html",
            title=self.config.title,
            description=self.config.description,
            search_markup=self._search_markup(),
            conversations=archive_stats.root_page_conversations,
            total_conversations=archive_stats.total_conversations,
            total_messages=archive_stats.total_messages,
            providers=archive_stats.provider_counts,
            provider_count=len(archive_stats.provider_counts),
            generated_at=generated_at,
        )

    async def _generate_provider_indexes(
        self,
        archive_stats: ArchiveIndexStats,
        *,
        generated_at: str,
    ) -> int:
        """Generate provider-scoped index pages without a full shared archive list."""
        template = self.env.get_template("index.html")

        for provider in archive_stats.provider_order:
            safe_provider = safe_path_component(provider, fallback="provider")
            provider_dir = self.output_dir / safe_provider
            provider_dir.mkdir(parents=True, exist_ok=True)

            await self._write_template_stream(
                template,
                provider_dir / "index.html",
                title=f"{provider} | {self.config.title}",
                description=f"Conversations from {provider}",
                search_markup=self._search_markup(),
                conversations=self._iter_conversation_indexes(provider=provider),
                total_conversations=archive_stats.provider_counts[provider],
                total_messages=archive_stats.provider_messages[provider],
                providers={},
                provider_count=1,
                generated_at=generated_at,
            )

        return len(archive_stats.provider_order)

    async def _generate_dashboard(
        self,
        archive_stats: ArchiveIndexStats,
        *,
        generated_at: str,
    ) -> None:
        """Generate statistics dashboard from archive aggregates."""
        template = self.env.get_template("dashboard.html")
        await self._write_template_stream(
            template,
            self.output_dir / "dashboard.html",
            title=self.config.title,
            providers=archive_stats.provider_counts,
            max_count=max(archive_stats.provider_counts.values(), default=1),
            total_conversations=archive_stats.total_conversations,
            total_messages=archive_stats.total_messages,
            provider_count=len(archive_stats.provider_counts),
            generated_at=generated_at,
        )

    async def _latest_run_summary(self) -> PublicationRunSummary | None:
        """Return the latest pipeline run summary for manifest embedding."""
        backend, _repository = self._open_storage()
        record = await backend.get_latest_run()
        if record is None:
            return None
        return PublicationRunSummary(
            run_id=record.run_id,
            timestamp=record.timestamp,
            counts=record.counts,
            indexed=record.indexed,
            duration_ms=record.duration_ms,
        )

    async def _artifact_proof_summary(self) -> ArtifactProofSummary | None:
        """Return durable artifact-proof summary for manifest embedding."""
        backend, _repository = self._open_storage()

        def _load() -> ArtifactProofSummary | None:
            from polylogue.schemas.verification import prove_raw_artifact_coverage

            report = prove_raw_artifact_coverage(db_path=backend.db_path)
            return ArtifactProofSummary(
                total_records=report.total_records,
                provider_count=len(report.providers),
                contract_backed_records=report.contract_backed_records,
                unsupported_parseable_records=report.unsupported_parseable_records,
                recognized_non_parseable_records=report.recognized_non_parseable_records,
                unknown_records=report.unknown_records,
                decode_errors=report.decode_errors,
                linked_sidecars=report.linked_sidecars,
                orphan_sidecars=report.orphan_sidecars,
                subagent_streams=report.subagent_streams,
                streams_with_sidecars=report.streams_with_sidecars,
                artifact_counts=report.artifact_counts,
                clean=report.is_clean,
            )

        return await asyncio.to_thread(_load)

    def _search_markup(self) -> str:
        """Render the search UI snippet for index pages."""
        if not self.config.enable_search:
            return ""
        if self.config.search_provider == "pagefind":
            return """
        <div id="search" style="margin-bottom: 2rem;"></div>
        <link href="/_pagefind/pagefind-ui.css" rel="stylesheet">
        <script src="/_pagefind/pagefind-ui.js"></script>
        <script>
            window.addEventListener('DOMContentLoaded', function() {
                new PagefindUI({ element: "#search", showSubResults: true });
            });
        </script>
"""
        return """
        <div class="search-panel" style="margin-bottom: 2rem;">
            <input id="search-input" type="search" placeholder="Search conversations..." style="width: 100%; padding: 0.85rem 1rem; border-radius: 10px; border: 1px solid var(--border); background: var(--bg-secondary); color: var(--text-primary);" />
            <p id="search-status" style="margin-top: 0.75rem; color: var(--text-secondary);"></p>
            <ul id="search-results" class="conversation-list" style="margin-top: 1rem; display: none;"></ul>
        </div>
        <script>
            window.addEventListener('DOMContentLoaded', async function() {
                const input = document.getElementById('search-input');
                const status = document.getElementById('search-status');
                const results = document.getElementById('search-results');
                const archiveList = document.querySelector('.conversation-list');
                let docs = [];
                try {
                    const response = await fetch('/search-index.json');
                    docs = response.ok ? await response.json() : [];
                } catch (error) {
                    status.textContent = 'Search index unavailable.';
                    return;
                }

                function renderResults(query) {
                    const term = query.trim().toLowerCase();
                    if (!term) {
                        results.style.display = 'none';
                        results.innerHTML = '';
                        archiveList.style.display = '';
                        status.textContent = '';
                        return;
                    }

                    archiveList.style.display = 'none';
                    const matches = docs.filter((doc) => {
                        return [doc.title, doc.provider, doc.preview]
                            .filter(Boolean)
                            .join(' ')
                            .toLowerCase()
                            .includes(term);
                    }).slice(0, 50);

                    results.innerHTML = matches.map((doc) => `
                        <li class="conversation-card">
                            <a href="${doc.path}" class="conversation-link">
                                <h2 class="conversation-title">${doc.title}</h2>
                                <div class="conversation-meta">
                                    <span class="badge">${doc.provider}</span>
                                    <span>${doc.preview || ''}</span>
                                </div>
                            </a>
                        </li>
                    `).join('');
                    results.style.display = matches.length ? '' : 'none';
                    status.textContent = matches.length
                        ? `${matches.length} result(s)`
                        : 'No conversations matched.';
                }

                input.addEventListener('input', (event) => renderResults(event.target.value));
            });
        </script>
"""

    def _generate_pagefind_config(self) -> str:
        """Generate pagefind index.

        Writes pagefind config and attempts to run pagefind to build the
        search index. If pagefind is not installed, writes the config anyway
        so the user can run it manually.
        """
        import shutil
        import subprocess

        config = {
            "site": str(self.output_dir),
            "output_subdir": "_pagefind",
        }
        (self.output_dir / "pagefind.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        # Try to run pagefind to build the search index
        pagefind_bin = shutil.which("pagefind")
        if pagefind_bin:
            try:
                subprocess.run(
                    [pagefind_bin, "--site", str(self.output_dir)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=True,
                )
                return "built"
            except subprocess.CalledProcessError:
                return "failed"
            except FileNotFoundError:
                return "pending"
        else:
            return "pending"

__all__ = ["SiteBuilder", "SiteConfig", "ConversationIndex"]
