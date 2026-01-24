"""HTML renderer implementation using Jinja2."""

from __future__ import annotations

from pathlib import Path

from jinja2 import DictLoader, Environment, FileSystemLoader
from markdown_it import MarkdownIt

from polylogue.render_paths import render_root
from polylogue.rendering.core import ConversationFormatter

DEFAULT_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    body {
      font-family: system-ui, Segoe UI, Roboto, sans-serif;
      max-width: 960px;
      margin: 2rem auto;
      line-height: 1.6;
      padding: 0 1rem;
      background-color: #f9fafb;
      color: #111827;
    }
    pre {
      white-space: pre-wrap;
      background: #f3f4f6;
      padding: 1rem;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
    }
    code {
      font-family: ui-monospace, Menlo, monospace;
    }
    h1 { border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
    h2 { margin-top: 2rem; color: #374151; }
    .metadata { color: #6b7280; font-size: 0.875rem; margin-bottom: 2rem; }
    .attachment { font-size: 0.875rem; color: #2563eb; }
  </style>
</head>
<body>
  {{ body|safe }}
</body>
</html>
"""


class HTMLRenderer:
    """Renders conversations to HTML format using Jinja2 templates."""

    def __init__(self, archive_root: Path, template_path: Path | None = None):
        """Initialize the HTML renderer.

        Args:
            archive_root: Root directory for archived conversations
            template_path: Optional path to custom Jinja2 HTML template
        """
        self.archive_root = archive_root
        self.template_path = template_path
        self.formatter = ConversationFormatter(archive_root)

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'html'
        """
        return "html"

    def _render_html(self, markdown_text: str, title: str) -> str:
        """Convert markdown to HTML using Jinja2 template.

        Args:
            markdown_text: Markdown content to convert
            title: Page title

        Returns:
            Rendered HTML string
        """
        md = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table")
        body_html = md.render(markdown_text)

        loader: FileSystemLoader | DictLoader
        if self.template_path and self.template_path.exists():
            loader = FileSystemLoader(self.template_path.parent)
            template_name = self.template_path.name
        else:
            loader = DictLoader({"index.html": DEFAULT_HTML_TEMPLATE})
            template_name = "index.html"

        env = Environment(loader=loader, autoescape=True)
        template = env.get_template(template_name)
        return template.render(title=title, body=body_html)

    def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to HTML format.

        Args:
            conversation_id: ID of the conversation to render
            output_path: Directory where the HTML file should be written

        Returns:
            Path to the generated HTML file

        Raises:
            ValueError: If conversation not found
            IOError: If output path is invalid or write fails
        """
        # Use shared formatter to get markdown
        formatted = self.formatter.format(conversation_id)

        # Determine output path
        render_root_path = render_root(output_path, formatted.provider, conversation_id)
        render_root_path.mkdir(parents=True, exist_ok=True)

        # Save markdown file (for backward compatibility)
        md_path = render_root_path / "conversation.md"
        md_path.write_text(formatted.markdown_text, encoding="utf-8")

        # Render HTML
        html_path = render_root_path / "conversation.html"
        html_content = self._render_html(formatted.markdown_text, formatted.title)
        html_path.write_text(html_content, encoding="utf-8")

        return html_path


__all__ = ["HTMLRenderer"]
