from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

try:
    from jinja2 import Environment, BaseLoader
except ImportError:  # pragma: no cover
    Environment = None  # type: ignore
    BaseLoader = object  # type: ignore

from markdown_it import MarkdownIt

from .render import MarkdownDocument


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
      body {
        background: {{ theme.background }};
        color: {{ theme.foreground }};
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 2rem auto;
        max-width: 820px;
        line-height: 1.6;
      }
      a { color: {{ theme.link }}; }
      .metadata table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1.5rem;
      }
      .metadata th, .metadata td {
        border: 1px solid {{ theme.callout_border }}44;
        padding: 0.4rem 0.6rem;
        text-align: left;
        font-size: 0.9rem;
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
      details.callout summary::-webkit-details-marker {
        display: none;
      }
      details.callout summary::before {
        content: '▶';
        display: inline-block;
        margin-right: 0.5rem;
        transform: rotate(0deg);
        transition: transform 0.2s ease;
      }
      details.callout[open] summary::before {
        transform: rotate(90deg);
      }
      pre {
        background: {{ theme.foreground }}11;
        padding: 0.75rem;
        overflow-x: auto;
      }
      code {
        font-family: "JetBrains Mono", "Fira Code", monospace;
      }
      h2 {
        margin-top: 2rem;
      }
    </style>
  </head>
  <body>
    <h1>{{ title }}</h1>
    <div class="metadata">
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
    <div class="content">{{ body_html | safe }}</div>
  </body>
</html>
"""


@dataclass
class HtmlRenderOptions:
    theme: str = "light"


def render_html(document: MarkdownDocument, options: HtmlRenderOptions) -> str:
    theme = THEMES.get(options.theme, THEMES["light"])
    md = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    body_html = md.render(document.body)
    body_html = _transform_callouts(body_html)
    metadata_rows = {k: v for k, v in document.metadata.items() if k != "attachments"}
    metadata_rows["attachments"] = len(document.attachments)

    if Environment is None:
        rows = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in metadata_rows.items()
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
    <h1>{document.metadata.get('title', 'Conversation')}</h1>
    <div class=\"metadata\">
      <table><tbody>{rows}</tbody></table>
    </div>
    <div class=\"content\">{body_html}</div>
  </body>
</html>
"""

    env = Environment(loader=BaseLoader(), autoescape=False)
    template = env.from_string(HTML_TEMPLATE)
    return template.render(
        title=document.metadata.get("title", "Conversation"),
        metadata=metadata_rows,
        body_html=body_html,
        theme=theme,
    )


def write_html(document: MarkdownDocument, path: Path, theme: str) -> None:
    options = HtmlRenderOptions(theme=theme)
    html_text = render_html(document, options)
    path.write_text(html_text, encoding="utf-8")


def _transform_callouts(html: str) -> str:
    import re

    lines = html.splitlines()
    result: list[str] = []
    callout_re = re.compile(r"^<blockquote><p>\[!([A-Z]+)\]([+-])\s+(.*)</p>$")
    i = 0
    while i < len(lines):
        match = callout_re.match(lines[i])
        if not match:
            result.append(lines[i])
            i += 1
            continue
        kind, fold, header = match.groups()
        open_attr = " open" if fold == "+" else ""
        result.append(f"<details class=\"callout\"{open_attr}><summary>{header}</summary>")
        i += 1
        while i < len(lines) and not lines[i].startswith("</blockquote>"):
            result.append(lines[i])
            i += 1
        if i < len(lines):
            result.append("</details>")
            i += 1
    return "\n".join(result)
