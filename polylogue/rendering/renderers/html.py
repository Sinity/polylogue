"""Enhanced HTML renderer with Pygments syntax highlighting and modern styling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import DictLoader, Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from polylogue.render_paths import render_root
from polylogue.rendering.core import ConversationFormatter

if TYPE_CHECKING:
    pass


class PygmentsHighlighter:
    """Code highlighter using Pygments."""

    def __init__(self, style: str = "monokai") -> None:
        """Initialize highlighter with a Pygments style.

        Args:
            style: Pygments style name (default: monokai for dark theme)
        """
        self.formatter = HtmlFormatter(
            style=style,
            cssclass="highlight",
            linenos=False,
            wrapcode=True,
        )

    def get_css(self) -> str:
        """Get CSS for syntax highlighting.

        Returns:
            CSS string for Pygments highlighting
        """
        return self.formatter.get_style_defs(".highlight")

    def highlight_code(self, code: str, language: str | None = None) -> str:
        """Highlight code block.

        Args:
            code: Source code to highlight
            language: Language hint (optional)

        Returns:
            HTML with syntax highlighting
        """
        try:
            lexer = get_lexer_by_name(language, stripall=True) if language else guess_lexer(code)
        except ClassNotFound:
            # Fall back to plain text
            escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f'<pre class="highlight"><code>{escaped}</code></pre>'

        return highlight(code, lexer, self.formatter)


class MarkdownRenderer:
    """Enhanced markdown renderer with code highlighting."""

    def __init__(self, highlighter: PygmentsHighlighter | None = None) -> None:
        """Initialize markdown renderer.

        Args:
            highlighter: Optional Pygments highlighter instance
        """
        self.highlighter = highlighter or PygmentsHighlighter()
        self.md = MarkdownIt("commonmark", {"html": False, "linkify": True})
        self.md.enable("table")

    def render(self, text: str) -> str:
        """Render markdown to HTML with syntax highlighting.

        Args:
            text: Markdown text to render

        Returns:
            HTML string with syntax-highlighted code blocks
        """
        if not text:
            return ""

        # First pass: standard markdown rendering
        html = self.md.render(text)

        # Second pass: enhance code blocks with Pygments
        html = self._enhance_code_blocks(html)

        return html

    def _enhance_code_blocks(self, html: str) -> str:
        """Replace code blocks with Pygments-highlighted versions.

        Args:
            html: HTML string with code blocks

        Returns:
            HTML with enhanced code blocks
        """
        # Match ```language\ncode\n``` patterns that became <pre><code class="language-xxx">
        pattern = r'<pre><code class="language-(\w+)">(.*?)</code></pre>'

        def replace_code(match: re.Match[str]) -> str:
            language = match.group(1)
            code = match.group(2)
            # Unescape HTML entities
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return self.highlighter.highlight_code(code, language)

        html = re.sub(pattern, replace_code, html, flags=re.DOTALL)

        # Also handle code blocks without language (generic <pre><code>)
        pattern_plain = r'<pre><code>([^<]*)</code></pre>'

        def replace_plain_code(match: re.Match[str]) -> str:
            code = match.group(1)
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            # Try to guess the language
            return self.highlighter.highlight_code(code, None)

        html = re.sub(pattern_plain, replace_plain_code, html, flags=re.DOTALL)

        return html


# Default template with enhanced styling and syntax highlighting
DEFAULT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="{{ theme }}">
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
            --accent-glow: rgba(99, 102, 241, 0.4);
            --border: #2d2d35;
            --border-subtle: #1f1f23;
            --user-bg: rgba(99, 102, 241, 0.1);
            --user-border: rgba(99, 102, 241, 0.3);
            --assistant-bg: rgba(16, 185, 129, 0.1);
            --assistant-border: rgba(16, 185, 129, 0.3);
            --system-bg: rgba(245, 158, 11, 0.1);
            --system-border: rgba(245, 158, 11, 0.3);
            --tool-bg: rgba(139, 92, 246, 0.1);
            --tool-border: rgba(139, 92, 246, 0.3);
        }

        [data-theme="light"] {
            --bg-primary: #ffffff;
            --bg-secondary: #f9fafb;
            --bg-elevated: #ffffff;
            --bg-code: #f6f8fa;
            --text-primary: #111827;
            --text-secondary: #4b5563;
            --text-muted: #9ca3af;
            --border: #e5e7eb;
            --border-subtle: #f3f4f6;
            --user-bg: #e3f2fd;
            --assistant-bg: #f5f5f5;
        }

        @media (prefers-color-scheme: light) {
            :root:not([data-theme="dark"]) {
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-elevated: #ffffff;
                --bg-code: #f6f8fa;
                --text-primary: #111827;
                --text-secondary: #4b5563;
                --text-muted: #9ca3af;
                --border: #e5e7eb;
                --border-subtle: #f3f4f6;
                --user-bg: #e3f2fd;
                --assistant-bg: #f5f5f5;
            }
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.7;
            -webkit-font-smoothing: antialiased;
        }

        .nav-header {
            position: sticky;
            top: 0;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            z-index: 100;
        }

        .nav-back {
            color: var(--accent);
            text-decoration: none;
            font-weight: 500;
        }

        .nav-breadcrumb {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }

        .conversation-header {
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }

        .conversation-header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(to right, var(--text-primary), var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .conversation-meta {
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .badge {
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
        }

        .meta-item {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .messages {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .message {
            display: flex;
            gap: 1rem;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border-subtle);
            transition: border-color 0.2s;
        }

        .message:hover { border-color: var(--border); }
        .message-user { background: var(--user-bg); border-color: var(--user-border); }
        .message-assistant { background: var(--assistant-bg); border-color: var(--assistant-border); }
        .message-system { background: var(--system-bg); border-color: var(--system-border); }
        .message-tool, .message-tool_use, .message-tool_result {
            background: var(--tool-bg);
            border-color: var(--tool-border);
        }

        .message-avatar {
            flex-shrink: 0;
        }

        .avatar-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 8px;
            background: var(--bg-elevated);
            font-weight: 700;
            font-size: 0.75rem;
            color: var(--accent);
        }

        .message-user .avatar-icon {
            background: var(--accent);
            color: white;
        }

        .message-content {
            flex-grow: 1;
            min-width: 0;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }

        .role-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .timestamp {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .message-body {
            color: var(--text-primary);
            font-size: 0.95rem;
        }

        .message-body p { margin-bottom: 1rem; }
        .message-body p:last-child { margin-bottom: 0; }

        .message-body pre {
            background: var(--bg-code);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            overflow-x: auto;
            margin: 1rem 0;
        }

        .message-body code {
            font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.875em;
        }

        .message-body p code {
            background: var(--bg-elevated);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
        }

        /* Pygments highlight overrides */
        .highlight {
            background: var(--bg-code) !important;
            border-radius: 8px;
            overflow-x: auto;
            margin: 1rem 0;
        }

        .highlight pre {
            margin: 0;
            padding: 1rem;
            background: transparent !important;
            border: none;
        }

        .toc {
            position: fixed;
            right: 2rem;
            top: 6rem;
            width: 200px;
            max-height: calc(100vh - 8rem);
            overflow-y: auto;
            padding: 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 0.75rem;
            display: none;
        }

        @media (min-width: 1200px) {
            .toc { display: block; }
        }

        .toc h3 {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }

        .toc-list {
            list-style: none;
        }

        .toc-list li { margin-bottom: 0.5rem; }

        .toc-list a {
            color: var(--text-secondary);
            text-decoration: none;
            display: block;
            padding: 0.25rem 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .toc-list a:hover { color: var(--text-primary); }

        @media (max-width: 768px) {
            .container { padding: 1rem; }
            .message { flex-direction: column; gap: 0.5rem; }
            .conversation-header h1 { font-size: 1.5rem; }
        }

        {{ highlight_css }}
    </style>
</head>
<body>
    <nav class="nav-header">
        <a href="../index.html" class="nav-back">← Back</a>
        <span class="nav-breadcrumb">{{ provider }} / {{ conversation_id[:12] }}...</span>
    </nav>

    <main class="container">
        <header class="conversation-header">
            <h1>{{ title or "Untitled Conversation" }}</h1>
            <div class="conversation-meta">
                <span class="badge">{{ provider }}</span>
                <span class="meta-item">{{ message_count }} messages</span>
                {% if created_at %}
                <span class="meta-item">{{ created_at }}</span>
                {% endif %}
            </div>
        </header>

        <div class="messages">
            {% for msg in messages %}
            <article class="message message-{{ msg.role or 'unknown' }}" id="msg-{{ loop.index }}">
                <div class="message-avatar">
                    <span class="avatar-icon">{{ (msg.role or 'msg')[:2] | upper }}</span>
                </div>
                <div class="message-content">
                    <header class="message-header">
                        <span class="role-label">{{ msg.role or 'message' }}</span>
                        {% if msg.timestamp %}
                        <time class="timestamp">{{ msg.timestamp }}</time>
                        {% endif %}
                    </header>
                    <div class="message-body">
                        {{ msg.html_content | safe }}
                    </div>
                </div>
            </article>
            {% endfor %}
        </div>
    </main>

    <aside class="toc">
        <h3>Messages</h3>
        <ol class="toc-list">
            {% for msg in messages %}
            <li>
                <a href="#msg-{{ loop.index }}">
                    {{ msg.role or 'unknown' }}: {{ (msg.text or "")[:30] }}{% if msg.text|length > 30 %}...{% endif %}
                </a>
            </li>
            {% endfor %}
        </ol>
    </aside>
</body>
</html>
"""


class HTMLRenderer:
    """Enhanced HTML renderer with syntax highlighting and polished styling.

    Features:
    - Pygments syntax highlighting for code blocks
    - Dark/light theme support with prefers-color-scheme
    - Message bubbles with role-based styling
    - Table of contents sidebar with message anchors
    - Navigation breadcrumbs
    """

    def __init__(
        self,
        archive_root: Path,
        template_path: Path | None = None,
        theme: str = "dark",
    ) -> None:
        """Initialize the HTML renderer.

        Args:
            archive_root: Root directory for archived conversations
            template_path: Optional path to custom Jinja2 HTML template
            theme: Theme name ("dark" or "light"), default "dark"
        """
        self.archive_root = archive_root
        self.template_path = template_path
        self.theme = theme
        self.formatter = ConversationFormatter(archive_root)

        # Initialize Pygments highlighter
        style = "monokai" if theme == "dark" else "default"
        self.highlighter = PygmentsHighlighter(style=style)
        self.md_renderer = MarkdownRenderer(self.highlighter)

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'html'
        """
        return "html"

    def _prepare_messages(self, conversation_id: str) -> list[dict[str, str | None]]:
        """Prepare messages for template rendering with HTML content.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of message dicts with html_content field
        """
        from polylogue.storage.backends.sqlite import open_connection

        messages: list[dict[str, str | None]] = []
        with open_connection(None) as conn:
            rows = conn.execute(
                """
                SELECT message_id, role, text, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp
                """,
                (conversation_id,),
            ).fetchall()

            for row in rows:
                text = row["text"] or ""
                html_content = self.md_renderer.render(text)
                messages.append({
                    "id": row["message_id"],
                    "role": row["role"] or "message",
                    "text": text,
                    "html_content": html_content,
                    "timestamp": row["timestamp"],
                })

        return messages

    def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to enhanced HTML format.

        Args:
            conversation_id: ID of the conversation to render
            output_path: Directory where the HTML file should be written

        Returns:
            Path to the generated HTML file

        Raises:
            ValueError: If conversation not found
            IOError: If output path is invalid or write fails
        """
        # Use shared formatter to get metadata
        formatted = self.formatter.format(conversation_id)

        # Determine output path
        render_root_path = render_root(output_path, formatted.provider, conversation_id)
        render_root_path.mkdir(parents=True, exist_ok=True)

        # Save markdown file (for backward compatibility)
        md_path = render_root_path / "conversation.md"
        md_path.write_text(formatted.markdown_text, encoding="utf-8")

        # Prepare messages with HTML content
        messages = self._prepare_messages(conversation_id)

        # Set up template
        loader: FileSystemLoader | DictLoader
        if self.template_path and self.template_path.exists():
            loader = FileSystemLoader(self.template_path.parent)
            template_name = self.template_path.name
        else:
            loader = DictLoader({"conversation.html": DEFAULT_HTML_TEMPLATE})
            template_name = "conversation.html"

        env = Environment(loader=loader, autoescape=select_autoescape(["html", "xml"]))
        template = env.get_template(template_name)

        # Render HTML
        html_content = template.render(
            title=formatted.title,
            provider=formatted.provider,
            conversation_id=conversation_id,
            messages=messages,
            message_count=len(messages),
            created_at=formatted.metadata.get("created_at"),
            highlight_css=self.highlighter.get_css(),
            theme=self.theme,
        )

        html_path = render_root_path / "conversation.html"
        html_path.write_text(html_content, encoding="utf-8")

        return html_path


__all__ = ["HTMLRenderer", "PygmentsHighlighter", "MarkdownRenderer"]
