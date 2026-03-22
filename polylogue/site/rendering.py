"""Template and page-rendering helpers for static-site generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import DictLoader, Environment, Template, select_autoescape

from polylogue.logging import get_logger
from polylogue.paths import safe_path_component
from polylogue.rendering.core import build_rendered_message_payload
from polylogue.rendering.renderers.html import PygmentsHighlighter
from polylogue.site.models import ArchiveIndexStats, ConversationIndex, SiteConfig
from polylogue.site.search import render_search_markup

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)

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


def build_template_environments(highlighter: PygmentsHighlighter) -> tuple[Environment, Environment]:
    """Build index/dashboard and conversation template environments."""
    conv_template = CONVERSATION_TEMPLATE.replace(
        "{{ highlight_css }}",
        highlighter.get_css(),
    )
    index_env = Environment(
        loader=DictLoader({
            "index.html": INDEX_TEMPLATE,
            "dashboard.html": DASHBOARD_TEMPLATE,
        }),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=True,
    )
    page_env = Environment(
        loader=DictLoader({
            "conversation.html": conv_template,
        }),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=True,
    )
    return index_env, page_env


async def write_template_stream(
    template: Template,
    output_path: Path,
    **context: object,
) -> None:
    """Render a template to disk without materializing the full output string."""
    stream = template.generate_async(**context)
    with output_path.open("w", encoding="utf-8") as handle:
        async for chunk in stream:
            handle.write(chunk)


async def iter_conversation_page_messages(
    repository: ConversationRepository,
    conversation_id: str,
    *,
    render_html: Callable[[str], str],
) -> AsyncIterator[dict[str, object]]:
    """Yield site message payloads lazily for a conversation page."""
    async for msg in repository.iter_messages(conversation_id):
        if not msg.text:
            continue
        yield build_rendered_message_payload(
            message_id=msg.id,
            role=msg.role or "unknown",
            text=msg.text,
            timestamp=msg.timestamp,
            parent_message_id=msg.parent_id,
            branch_index=msg.branch_index,
            render_html=render_html,
        )


async def generate_conversation_page(
    *,
    output_dir: Path,
    page_env: Environment,
    repository: ConversationRepository,
    conversation: ConversationIndex,
    render_html: Callable[[str], str],
    incremental: bool = True,
) -> str:
    """Generate one conversation page, or keep an up-to-date existing one."""
    template = page_env.get_template("conversation.html")
    page_path = output_dir / conversation.path
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
        await write_template_stream(
            template,
            page_path,
            title=conversation.title,
            provider=conversation.provider,
            message_count=conversation.message_count,
            updated_at=conversation.updated_at,
            messages=iter_conversation_page_messages(
                repository,
                conversation.id,
                render_html=render_html,
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


async def generate_root_index(
    *,
    output_dir: Path,
    env: Environment,
    config: SiteConfig,
    archive_stats: ArchiveIndexStats,
    generated_at: str,
    write_stream: Callable[..., object] = write_template_stream,
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
    write_stream: Callable[..., object] = write_template_stream,
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
    write_stream: Callable[..., object] = write_template_stream,
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
