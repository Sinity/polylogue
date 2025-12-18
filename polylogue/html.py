from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import re
import math
import json
import urllib.parse

from jinja2 import Environment, BaseLoader
from markupsafe import escape
from markdown_it import MarkdownIt

from .render import AttachmentInfo, MarkdownDocument


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
      .muted { opacity: 0.8; font-size: 0.85rem; }
      .toolbar { display: flex; gap: 0.75rem; align-items: center; }
      .search-input {
        flex: 1;
        padding: 0.45rem 0.75rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--bg) 96%, #00000008);
        color: var(--fg);
      }
      .sidebar-input {
        width: 100%;
        padding: 0.4rem 0.6rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--bg) 96%, #00000008);
        color: var(--fg);
        margin: 0.4rem 0 0.75rem;
      }
      .layout { display: grid; grid-template-columns: 260px 1fr; gap: 1.25rem; align-items: start; }
      .card {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        background: color-mix(in srgb, var(--bg) 96%, #00000006);
      }
      @media (max-width: 860px) {
        header { flex-wrap: wrap; }
        .layout { grid-template-columns: 1fr; }
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
      .toc a:hover { text-decoration: underline; }
      .attachments { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
      .attachment {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.6rem 0.75rem;
        background: color-mix(in srgb, var(--bg) 94%, #00000008);
      }
      .attachment .att-meta { display: flex; gap: 0.5rem; align-items: baseline; flex-wrap: wrap; }
      .attachment .att-name { font-weight: 600; }
      .attachment .att-sub { font-size: 0.85rem; opacity: 0.8; }
      .attachment img {
        width: 100%;
        max-height: 160px;
        object-fit: cover;
        border-radius: 8px;
        border: 1px solid var(--border);
        margin-top: 0.5rem;
      }
      .lightbox {
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        background: color-mix(in srgb, #000 75%, transparent);
        z-index: 30;
        padding: 1.25rem;
      }
      .lightbox[open] { display: flex; }
      .lightbox-card {
        max-width: 1100px;
        width: 100%;
        max-height: 92vh;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--bg);
        overflow: hidden;
        display: grid;
        grid-template-rows: auto 1fr;
      }
      .lightbox-header {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 0.85rem;
        border-bottom: 1px solid var(--border);
      }
      .lightbox-header a { text-decoration: none; }
      .lightbox-body {
        padding: 0.75rem;
        overflow: auto;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .lightbox-body img {
        max-width: 100%;
        max-height: 80vh;
        object-fit: contain;
        border-radius: 10px;
        border: 1px solid var(--border);
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
      .anchor-btn {
        margin-left: 0.4rem;
        font-size: 0.85rem;
        color: var(--accent);
        text-decoration: none;
        opacity: 0.7;
      }
      .anchor-btn:hover { opacity: 1; }
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
          <input class="sidebar-input" id="att-filter" type="search" placeholder="Filter attachments" />
          <div class="attachments">
            {% for att in attachments %}
              <div class="attachment" data-att="{{ (att.name ~ ' ' ~ (att.kind or '') ~ ' ' ~ (att.ext or ''))|lower }}">
                <div class="att-meta">
                  <span class="att-name">{{ att.icon }} <a href="{{ att.link }}" target="_blank">{{ att.name }}</a></span>
                  {% if att.size %}<span class="att-sub">{{ att.size }}</span>{% endif %}
                </div>
                {% if att.preview_src %}
                  <a href="{{ att.link }}" target="_blank">
                    <img class="att-preview" data-full="{{ att.link }}" data-name="{{ att.name }}" src="{{ att.preview_src }}" alt="{{ att.name }}" />
                  </a>
                {% endif %}
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
    <div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="Attachment preview">
      <div class="lightbox-card">
        <div class="lightbox-header">
          <div>
            <strong id="lightbox-title"></strong>
            <span class="muted" id="lightbox-hint"></span>
          </div>
          <div class="toolbar">
            <a id="lightbox-open" href="#" target="_blank" class="pill">open</a>
            <a href="#" id="lightbox-close" class="pill">close</a>
          </div>
        </div>
        <div class="lightbox-body">
          <img id="lightbox-img" alt="" />
        </div>
      </div>
    </div>
    <script>
      const tocEl = document.getElementById('toc');
      const headings = document.querySelectorAll('#content h1, #content h2, #content h3');
      headings.forEach(h => {
        if (!h.id) return;
        const clone = h.cloneNode(true);
        clone.querySelectorAll('.anchor-btn').forEach(n => n.remove());
        const headingText = (clone.textContent || '').trim();
        const anchor = document.createElement('a');
        anchor.href = `#${h.id}`;
        anchor.textContent = '🔗';
        anchor.className = 'anchor-btn';
        anchor.title = 'Copy link';
        anchor.addEventListener('click', evt => {
          evt.preventDefault();
          const url = `${window.location.origin}${window.location.pathname}#${h.id}`;
          navigator.clipboard?.writeText(url).catch(() => {});
          window.location.hash = h.id;
        });
        h.appendChild(anchor);
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `#${h.id}`;
        a.textContent = headingText || h.id;
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

      // Add copy-link buttons for message anchors (msg-N)
      const msgAnchors = content.querySelectorAll('a[id^="msg-"]');
      msgAnchors.forEach(anchor => {
        const id = anchor.getAttribute('id');
        if (!id) return;
        const btn = document.createElement('a');
        btn.href = `#${id}`;
        btn.textContent = '🔗';
        btn.className = 'anchor-btn';
        btn.title = 'Copy link to message';
        btn.addEventListener('click', evt => {
          evt.preventDefault();
          const url = `${window.location.origin}${window.location.pathname}#${id}`;
          navigator.clipboard?.writeText(url).catch(() => {});
          window.location.hash = id;
        });
        anchor.insertAdjacentElement('afterend', btn);
      });

      // Attachments: filter + lightbox previews.
      const attFilter = document.getElementById('att-filter');
      const attCards = Array.from(document.querySelectorAll('.attachment'));
      function applyAttFilter(term) {
        const q = (term || '').trim().toLowerCase();
        for (const card of attCards) {
          const hay = card.getAttribute('data-att') || '';
          card.classList.toggle('hidden', q && !hay.includes(q));
        }
      }
      if (attFilter) attFilter.addEventListener('input', (e) => applyAttFilter(e.target.value));

      const lightbox = document.getElementById('lightbox');
      const lightboxImg = document.getElementById('lightbox-img');
      const lightboxTitle = document.getElementById('lightbox-title');
      const lightboxOpen = document.getElementById('lightbox-open');
      const lightboxHint = document.getElementById('lightbox-hint');
      const lightboxClose = document.getElementById('lightbox-close');

      function closeLightbox() {
        if (!lightbox) return;
        lightbox.removeAttribute('open');
        if (lightboxImg) lightboxImg.src = '';
      }
      function openLightbox(src, name) {
        if (!lightbox) return;
        lightbox.setAttribute('open', 'true');
        if (lightboxImg) lightboxImg.src = src;
        if (lightboxImg) lightboxImg.alt = name || '';
        if (lightboxTitle) lightboxTitle.textContent = name || '';
        if (lightboxHint) lightboxHint.textContent = ' (Esc to close)';
        if (lightboxOpen) lightboxOpen.href = src;
      }
      document.querySelectorAll('.att-preview').forEach(img => {
        img.addEventListener('click', (e) => {
          e.preventDefault();
          const src = img.getAttribute('data-full') || img.getAttribute('src');
          const name = img.getAttribute('data-name') || '';
          openLightbox(src, name);
        });
      });
      if (lightboxClose) lightboxClose.addEventListener('click', (e) => { e.preventDefault(); closeLightbox(); });
      if (lightbox) lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
      window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });
    </script>
  </body>
</html>
"""


@dataclass
class HtmlRenderOptions:
    theme: str = "light"


def _anchorize(text: str) -> str:
    """Create a URL-safe anchor id from text."""
    safe = re.sub(r"<[^>]+>", "", text or "")
    safe = re.sub(r"[^a-zA-Z0-9\\-_]+", "-", safe).strip("-").lower()
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
    def _fmt(value: object) -> object:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, sort_keys=True, ensure_ascii=False)
            except Exception:
                return str(value)
        return value

    metadata_rows_raw = {k: _fmt(v) for k, v in document.metadata.items() if k != "attachments"}
    metadata_rows_raw["attachments"] = len(document.attachments)
    metadata_rows = {k: metadata_rows_raw[k] for k in sorted(metadata_rows_raw)}
    attachment_total_bytes = 0
    attachments_sorted = sorted(
        document.attachments,
        key=lambda att: (
            (att.name or "").lower(),
            (att.local_path.as_posix() if getattr(att, "local_path", None) else (att.link or "")).lower(),
        ),
    )
    attachments = [
        {
            "name": att.name,
            "link": urllib.parse.quote(att.local_path.as_posix()) if getattr(att, "local_path", None) else att.link,
            "size": _human_size(att.size_bytes) if hasattr(att, "size_bytes") else None,
            "preview_src": _attachment_preview_src(att),
            "ext": getattr(att.local_path, "suffix", "") if getattr(att, "local_path", None) else Path(att.link).suffix,
            "kind": "image" if _attachment_preview_src(att) else "file",
            "icon": _attachment_icon(att),
        }
        for att in attachments_sorted
    ]
    for att in attachments_sorted:
        if getattr(att, "size_bytes", None):
            try:
                attachment_total_bytes += int(att.size_bytes or 0)
            except Exception:
                pass
    if attachment_total_bytes:
        metadata_rows["attachmentBytes"] = int(attachment_total_bytes)

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


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


def _attachment_preview_src(att: AttachmentInfo) -> Optional[str]:
    local = getattr(att, "local_path", None)
    if local is None:
        return None
    suffix = local.suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        return None
    return urllib.parse.quote(local.as_posix())


def _attachment_icon(att: AttachmentInfo) -> str:
    local = getattr(att, "local_path", None)
    suffix = ""
    if local is not None:
        suffix = local.suffix.lower()
    elif isinstance(getattr(att, "link", None), str):
        suffix = Path(att.link).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "🖼"
    if suffix in {".pdf"}:
        return "📄"
    if suffix in {".md", ".txt"}:
        return "📝"
    if suffix in {".json", ".jsonl", ".yaml", ".yml"}:
        return "🧾"
    if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz"}:
        return "🗜"
    return "📎"
