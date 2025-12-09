from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import re
import math

from jinja2 import Environment, BaseLoader
from markupsafe import escape
from markdown_it import MarkdownIt

from .render import MarkdownDocument


def _human_size(num: float | int | None) -> str | None:
    if num is None or num <= 0:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = min(int(math.log(num, 1024)) if num > 0 else 0, len(units) - 1)
    value = num / (1024 ** idx)
    return f"{value:.2f} {units[idx]}"


THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "background": "#ffffff",
        "foreground": "#222222",
        "callout_border": "#3b82f6",
        "link": "#2563eb",
    },
    "dark": {
        "background": "#111827",
        "foreground": "#e5e7eb",
        "callout_border": "#60a5fa",
        "link": "#93c5fd",
    },
}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{{ title }}</title>
    <style>
      :root {
        --bg: {{ theme.background }};
        --fg: {{ theme.foreground }};
        --accent: {{ theme.link }};
        --border: {{ theme.callout_border }}44;
      }
      * { box-sizing: border-box; }
      body {
        background: var(--bg);
        color: var(--fg);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0;
        line-height: 1.6;
      }
      header {
        position: sticky;
        top: 0;
        z-index: 10;
        backdrop-filter: blur(6px);
        background: color-mix(in srgb, var(--bg) 88%, transparent);
        border-bottom: 1px solid var(--border);
        padding: 0.75rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
      }
      main {
        max-width: 980px;
        margin: 0 auto;
        padding: 1.5rem;
      }
      h1 { margin: 0; font-size: 1.5rem; }
      a { color: var(--accent); }
      .pill {
        border: 1px solid var(--border);
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        font-size: 0.9rem;
      }
      .toolbar { display: flex; gap: 0.75rem; align-items: center; }
      .search-input {
        flex: 1;
        padding: 0.45rem 0.75rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--bg) 96%, #00000008);
        color: var(--fg);
      }
      .layout { display: grid; grid-template-columns: 260px 1fr; gap: 1.25rem; align-items: start; }
      .card {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        background: color-mix(in srgb, var(--bg) 96%, #00000006);
      }
      .metadata table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 0.5rem;
      }
      .metadata th, .metadata td {
        border: 1px solid var(--border);
        padding: 0.35rem 0.5rem;
        text-align: left;
        font-size: 0.9rem;
      }
      .toc ul { list-style: none; padding-left: 0; margin: 0; }
      .toc li { margin: 0.2rem 0; }
      .toc a { text-decoration: none; }
      .attachments { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
      .attachment {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.6rem 0.75rem;
        background: color-mix(in srgb, var(--bg) 94%, #00000008);
      }
      .content blockquote {
        border-left: 4px solid {{ theme.callout_border }};
        padding-left: 1rem;
        margin-left: 0;
        background: {{ theme.callout_border }}11;
      }
      details.callout {
        margin: 1rem 0;
        border-left: 4px solid {{ theme.callout_border }};
        padding-left: 1rem;
        background: {{ theme.callout_border }}11;
      }
      details.callout summary {
        cursor: pointer;
        font-weight: 600;
        list-style: none;
      }
      details.callout summary::-webkit-details-marker { display: none; }
      details.callout summary::before {
        content: '▶';
        display: inline-block;
        margin-right: 0.5rem;
        transform: rotate(0deg);
        transition: transform 0.2s ease;
      }
      details.callout[open] summary::before { transform: rotate(90deg); }
      pre {
        background: {{ theme.foreground }}11;
        padding: 0.75rem;
        overflow-x: auto;
      }
      code { font-family: "JetBrains Mono", "Fira Code", monospace; }
      h2 { margin-top: 2rem; }
      .hidden { display: none !important; }
    </style>
  </head>
  <body>
    <header>
      <div class="pill">{{ title }}</div>
      <input class="search-input" id="client-search" type="search" placeholder="Filter text (press / to focus)" />
      <div class="toolbar"><span class="pill" id="match-count"></span></div>
    </header>
    <main>
      <div class="layout">
        <div class="card toc">
          <h3>Contents</h3>
          <ul id="toc"></ul>
          {% if attachments %}
          <h4>Attachments</h4>
          <div class="attachments">
            {% for att in attachments %}
              <div class="attachment">
                <div><a href="{{ att.link }}" target="_blank">{{ att.name }}</a></div>
                {% if att.size %}<div>{{ att.size }}</div>{% endif %}
              </div>
            {% endfor %}
          </div>
          {% endif %}
        </div>
        <div>
          <div class="card metadata">
            <table>
              <tbody>
                {% for key, value in metadata.items() %}
                <tr>
                  <th>{{ key }}</th>
                  <td>{{ value }}</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
          <div class="content" id="content">{{ body_html | safe }}</div>
        </div>
      </div>
    </main>
    <script>
      const tocEl = document.getElementById('toc');
      const headings = document.querySelectorAll('#content h1, #content h2, #content h3');
      headings.forEach(h => {
        if (!h.id) return;
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `#${h.id}`;
        a.textContent = h.textContent;
        li.appendChild(a);
        tocEl.appendChild(li);
      });
      const searchInput = document.getElementById('client-search');
      const content = document.getElementById('content');
      const blocks = Array.from(content.querySelectorAll('p, li, blockquote, details'));
      const matchCount = document.getElementById('match-count');
      function applyFilter(term) {
        const q = term.trim().toLowerCase();
        let matches = 0;
        blocks.forEach(el => {
          const text = el.textContent.toLowerCase();
          const hit = q && text.includes(q);
          el.classList.toggle('hidden', q && !hit);
          if (hit) matches += 1;
        });
        matchCount.textContent = q ? `${matches} match(es)` : '';
      }
      searchInput.addEventListener('input', (e) => applyFilter(e.target.value));
      window.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement !== searchInput) {
          e.preventDefault();
          searchInput.focus();
        }
      });
    </script>
  </body>
</html>
"""


@dataclass
class HtmlRenderOptions:
    theme: str = "light"


def _anchorize(text: str) -> str:
    """Create a URL-safe anchor id from text."""
    safe = re.sub(r"[^a-zA-Z0-9\\-_]+", "-", text).strip("-").lower()
    return safe or "section"


_MD_RENDERER = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
_JINJA_ENV = Environment(loader=BaseLoader(), autoescape=True) if Environment is not None else None
_JINJA_TEMPLATE = _JINJA_ENV.from_string(HTML_TEMPLATE) if _JINJA_ENV is not None else None


def render_html(document: MarkdownDocument, options: HtmlRenderOptions) -> str:
    theme = THEMES.get(options.theme, THEMES["light"])
    body_html = _MD_RENDERER.render(document.body)
    # Inject anchors for headings to support deep links from search/open.
    body_html = re.sub(
        r"<h([1-6])>(.*?)</h\1>",
        lambda m: f'<h{m.group(1)} id="{_anchorize(m.group(2))}">{m.group(2)}</h{m.group(1)}>',
        body_html,
    )
    body_html = _transform_callouts(body_html)
    metadata_rows = {k: v for k, v in document.metadata.items() if k != "attachments"}
    metadata_rows["attachments"] = len(document.attachments)
    attachments = [
        {
            "name": att.name,
            "link": att.link,
            "size": _human_size(att.size_bytes) if hasattr(att, "size_bytes") else None,
        }
        for att in document.attachments
    ]

    if _JINJA_TEMPLATE is None:
        rows = "".join(
            f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>" for k, v in metadata_rows.items()
        )
        return f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{document.metadata.get('title', 'Conversation')}</title>
    <style>
      body {{ background: {theme['background']}; color: {theme['foreground']}; font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 820px; line-height: 1.6; }}
      a {{ color: {theme['link']}; }}
      table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }}
      th, td {{ border: 1px solid {theme['callout_border']}44; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.9rem; }}
      blockquote {{ border-left: 4px solid {theme['callout_border']}; padding-left: 1rem; margin-left: 0; background: {theme['callout_border']}11; }}
      pre {{ background: {theme['foreground']}11; padding: 0.75rem; overflow-x: auto; }}
      code {{ font-family: "JetBrains Mono", monospace; }}
    </style>
  </head>
  <body>
    <h1>{escape(str(document.metadata.get('title', 'Conversation')))}</h1>
    <div class=\"metadata\">
      <table><tbody>{rows}</tbody></table>
    </div>
    <div class=\"content\">{body_html}</div>
  </body>
</html>
"""

    return _JINJA_TEMPLATE.render(
        title=document.metadata.get("title", "Conversation"),
        metadata=metadata_rows,
        body_html=body_html,
        theme=theme,
        attachments=attachments,
    )


def write_html(document: MarkdownDocument, path: Path, theme: str) -> None:
    options = HtmlRenderOptions(theme=theme)
    html_text = render_html(document, options)
    path.write_text(html_text, encoding="utf-8")


def _transform_callouts(html: str) -> str:
    import re

    pattern = re.compile(
        r"<blockquote>\s*<p>\[!([A-Z]+)\]([+-])\s+(.*?)</p>(.*?)</blockquote>",
        flags=re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        kind, fold, header, body = match.groups()
        open_attr = " open" if fold == "+" else ""
        summary = header.strip()
        body_html = body.strip()
        if body_html and not body_html.startswith("<"):
            body_html = f"<p>{body_html}</p>"
        parts = [
            f"<details class=\"callout\" data-kind=\"{kind.lower()}\"{open_attr}>",
            f"<summary>{summary}</summary>",
        ]
        if body_html:
            parts.append(body_html)
        parts.append("</details>")
        return "\n".join(parts)

    return pattern.sub(repl, html)
