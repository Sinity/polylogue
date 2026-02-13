"""Static site builder for polylogue archives.

Generates a complete static HTML site with index pages, dashboard,
and client-side search support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import DictLoader, Environment, select_autoescape

from polylogue.lib.log import get_logger
from polylogue.paths import safe_path_component
from polylogue.rendering.renderers.html import MarkdownRenderer, PygmentsHighlighter

logger = get_logger(__name__)

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

        <div id="search" style="margin-bottom: 2rem;"></div>
        <link href="/_pagefind/pagefind-ui.css" rel="stylesheet">
        <script src="/_pagefind/pagefind-ui.js"></script>
        <script>
            window.addEventListener('DOMContentLoaded', function() {
                new PagefindUI({ element: "#search", showSubResults: true });
            });
        </script>

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
    search_provider: str = "pagefind"  # or "lunr"
    conversations_per_page: int = 100
    include_dashboard: bool = True


@dataclass
class ConversationIndex:
    """Indexed conversation for site generation."""

    id: str
    title: str
    provider: str
    source: str | None
    created_at: str | None
    updated_at: str | None
    message_count: int
    preview: str
    path: str


class SiteBuilder:
    """Build a static HTML site from a polylogue archive.

    Generates:
    - index.html: Root listing with recent conversations
    - {provider}/index.html: Per-provider index pages
    - {provider}/{id[:12]}/conversation.html: Individual conversation pages
    - dashboard.html: Archive statistics
    - search-index.json: For client-side search (lunr.js / pagefind)
    """

    def __init__(
        self,
        output_dir: Path,
        config: SiteConfig | None = None,
    ) -> None:
        """Initialize site builder.

        Args:
            output_dir: Directory for generated site
            config: Optional site configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config or SiteConfig()

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
                "conversation.html": conv_template,
            }),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def build(self, incremental: bool = True) -> dict[str, int]:
        """Build complete static site.

        Args:
            incremental: If True, skip conversation pages whose file is newer
                than the conversation's updated_at timestamp.

        Returns:
            Dict with counts: {"conversations": N, "index_pages": N}
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build conversation index
        conversations = self._build_index()

        # Generate pages (index pages always rebuilt; conversation pages respect incremental)
        self._generate_root_index(conversations)
        provider_count = self._generate_provider_indexes(conversations)
        conv_pages = self._generate_conversation_pages(conversations, incremental=incremental)

        if self.config.include_dashboard:
            self._generate_dashboard(conversations)

        # Generate search index
        if self.config.enable_search:
            self._generate_search_index(conversations)

        return {
            "conversations": len(conversations),
            "conversation_pages": conv_pages,
            "index_pages": 1 + provider_count + (1 if self.config.include_dashboard else 0),
        }

    def _build_index(self) -> list[ConversationIndex]:
        """Build index of all conversations.

        Uses lightweight summaries to avoid loading message content into memory.
        For a 4800+ conversation archive this reduces memory from ~9GB to ~100MB.
        """
        from polylogue.storage.backends.sqlite import SQLiteBackend
        from polylogue.storage.repository import ConversationRepository

        backend = SQLiteBackend()
        repo = ConversationRepository(backend=backend)

        # Use summaries (no message content) instead of full conversations
        summaries = repo.list_summaries(limit=100_000)

        # Batch-fetch message counts
        all_ids = [str(s.id) for s in summaries]
        msg_counts = backend.get_message_counts_batch(all_ids)

        conversations: list[ConversationIndex] = []

        for summary in summaries:
            sid = str(summary.id)

            # Format created_at
            created_at_str = None
            if summary.created_at:
                try:
                    created_at_str = summary.created_at.strftime("%Y-%m-%d")
                except Exception as exc:
                    logger.debug("Timestamp format error for %s: %s", sid, exc)
                    created_at_str = str(summary.created_at)[:10]

            updated_at_str = None
            if summary.updated_at:
                try:
                    updated_at_str = summary.updated_at.strftime("%Y-%m-%d %H:%M")
                except Exception as exc:
                    logger.debug("Timestamp format error for %s: %s", sid, exc)
                    updated_at_str = str(summary.updated_at)

            provider = summary.provider or "unknown"
            conversations.append(ConversationIndex(
                id=sid,
                title=summary.display_title or sid[:12],
                provider=provider,
                source=None,
                created_at=created_at_str,
                updated_at=updated_at_str,
                message_count=msg_counts.get(sid, 0),
                preview=summary.summary or "",
                path=f"{safe_path_component(provider, fallback='provider')}/{sid[:12]}/conversation.html",
            ))

        # Sort by updated_at descending
        conversations.sort(
            key=lambda c: c.updated_at or "",
            reverse=True,
        )

        return conversations

    def _generate_conversation_pages(
        self, conversations: list[ConversationIndex], *, incremental: bool = True
    ) -> int:
        """Generate individual HTML pages for each conversation.

        Streams messages from the database to avoid loading entire conversations
        into memory. For a 4800+ conversation archive, this is the most expensive
        step but produces the actual viewable conversation content.

        Args:
            conversations: List of conversations to generate pages for
            incremental: If True, skip pages whose file mtime is newer than
                the conversation's updated_at timestamp.

        Returns:
            Number of conversation pages generated
        """
        from polylogue.storage.backends.sqlite import SQLiteBackend
        from polylogue.storage.repository import ConversationRepository

        backend = SQLiteBackend()
        repo = ConversationRepository(backend=backend)

        template = self.env.get_template("conversation.html")
        generated = 0

        skipped = 0
        for conv_idx in conversations:
            # Build output path matching the link in index pages
            page_path = self.output_dir / conv_idx.path
            page_path.parent.mkdir(parents=True, exist_ok=True)

            # Incremental: skip if page exists and is newer than conversation update
            if incremental and page_path.exists():
                if conv_idx.updated_at:
                    try:
                        from datetime import datetime

                        file_mtime = datetime.fromtimestamp(page_path.stat().st_mtime)
                        conv_updated = datetime.fromisoformat(conv_idx.updated_at)
                        if file_mtime > conv_updated:
                            generated += 1
                            continue
                    except (ValueError, OSError):
                        pass  # Parse/stat error — rebuild this page
                else:
                    # No updated_at on conversation — skip if file exists
                    generated += 1
                    continue

            try:
                # Stream messages without loading full conversation
                messages = []
                for msg in repo.iter_messages(conv_idx.id, limit=500):
                    if not msg.text:
                        continue
                    text = msg.text
                    # Truncate very long messages for the static site
                    if len(text) > 5000:
                        text = text[:5000] + "\n\n[... truncated ...]"
                    html_content = self._md_renderer.render(text)
                    messages.append({
                        "role": msg.role or "unknown",
                        "text": text,
                        "html_content": html_content,
                    })

                html = template.render(
                    title=conv_idx.title,
                    provider=conv_idx.provider,
                    message_count=conv_idx.message_count,
                    updated_at=conv_idx.updated_at,
                    messages=messages,
                )

                page_path.write_text(html, encoding="utf-8")
                generated += 1
            except Exception as exc:
                logger.warning(
                    "Skipping conversation page %s: %s", conv_idx.id, exc
                )
                skipped += 1

        if skipped:
            logger.warning("Skipped %d conversation pages due to errors", skipped)

        return generated

    def _generate_root_index(self, conversations: list[ConversationIndex]) -> None:
        """Generate root index.html."""
        # Group by provider for sidebar
        by_provider: dict[str, int] = {}
        total_messages = 0
        for conv in conversations:
            by_provider[conv.provider] = by_provider.get(conv.provider, 0) + 1
            total_messages += conv.message_count

        template = self.env.get_template("index.html")
        html = template.render(
            title=self.config.title,
            description=self.config.description,
            conversations=conversations[:self.config.conversations_per_page],
            total_conversations=len(conversations),
            total_messages=total_messages,
            providers=by_provider,
            provider_count=len(by_provider),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

        (self.output_dir / "index.html").write_text(html, encoding="utf-8")

    def _generate_provider_indexes(self, conversations: list[ConversationIndex]) -> int:
        """Generate per-provider index pages.

        Returns:
            Number of provider index pages generated
        """
        # Group by provider
        by_provider: dict[str, list[ConversationIndex]] = {}
        for conv in conversations:
            by_provider.setdefault(conv.provider, []).append(conv)

        template = self.env.get_template("index.html")

        for provider, convs in by_provider.items():
            safe_provider = safe_path_component(provider, fallback="provider")
            provider_dir = self.output_dir / safe_provider
            provider_dir.mkdir(parents=True, exist_ok=True)

            total_messages = sum(c.message_count for c in convs)

            html = template.render(
                title=f"{provider} | {self.config.title}",
                description=f"Conversations from {provider}",
                conversations=convs,
                total_conversations=len(convs),
                total_messages=total_messages,
                providers={},  # Don't show sidebar on provider pages
                provider_count=1,
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )

            (provider_dir / "index.html").write_text(html, encoding="utf-8")

        return len(by_provider)

    def _generate_dashboard(self, conversations: list[ConversationIndex]) -> None:
        """Generate statistics dashboard."""
        by_provider: dict[str, int] = {}
        total_messages = 0
        for conv in conversations:
            by_provider[conv.provider] = by_provider.get(conv.provider, 0) + 1
            total_messages += conv.message_count

        max_count = max(by_provider.values()) if by_provider else 1

        template = self.env.get_template("dashboard.html")
        html = template.render(
            title=self.config.title,
            providers=by_provider,
            max_count=max_count,
            total_conversations=len(conversations),
            total_messages=total_messages,
            provider_count=len(by_provider),
        )

        (self.output_dir / "dashboard.html").write_text(html, encoding="utf-8")

    def _generate_search_index(self, conversations: list[ConversationIndex]) -> None:
        """Generate client-side search index."""
        if self.config.search_provider == "pagefind":
            self._generate_pagefind_config()
        else:
            self._generate_lunr_index(conversations)

    def _generate_pagefind_config(self) -> None:
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
                logger.info("Pagefind search index built successfully")
            except subprocess.CalledProcessError as exc:
                logger.warning("Pagefind indexing failed: %s", exc.stderr)
            except FileNotFoundError:
                logger.info("Pagefind not found — search index not built")
        else:
            logger.info(
                "Pagefind not found in PATH. Run 'pagefind --site %s' to build search index.",
                self.output_dir,
            )

    def _generate_lunr_index(self, conversations: list[ConversationIndex]) -> None:
        """Generate lunr.js search index."""
        documents = []
        for conv in conversations:
            documents.append({
                "id": conv.id,
                "title": conv.title,
                "provider": conv.provider,
                "preview": conv.preview,
                "path": conv.path,
            })

        (self.output_dir / "search-index.json").write_text(
            json.dumps(documents),
            encoding="utf-8",
        )


__all__ = ["SiteBuilder", "SiteConfig", "ConversationIndex"]
