from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment

from ..commands import CommandEnv
from ..db import open_connection
from ..schema import stamp_payload
from .editor import open_in_editor


@dataclass(frozen=True)
class TimelineRow:
    provider: str
    conversation_id: str
    slug: str
    title: Optional[str]
    last_updated: Optional[str]
    tokens: int
    words: int
    attachments: int
    attachment_bytes: int
    branch_count: int
    output_path: Optional[str]
    html_path: Optional[str]


_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Polylogue Timeline</title>
  <style>
    :root {
      --bg: {{ '#ffffff' if theme == 'light' else '#111827' }};
      --fg: {{ '#111827' if theme == 'light' else '#e5e7eb' }};
      --muted: {{ '#6b7280' if theme == 'light' else '#9ca3af' }};
      --border: {{ '#e5e7eb' if theme == 'light' else '#374151' }};
      --accent: {{ '#2563eb' if theme == 'light' else '#93c5fd' }};
      --row: {{ '#f9fafb' if theme == 'light' else '#0b1220' }};
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--fg); font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
    header { position: sticky; top: 0; z-index: 10; border-bottom: 1px solid var(--border); padding: 0.75rem 1rem; background: color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter: blur(6px); display: flex; gap: 0.75rem; align-items: center; }
    h1 { font-size: 1.1rem; margin: 0; }
    .pill { border: 1px solid var(--border); padding: 0.25rem 0.6rem; border-radius: 999px; color: var(--muted); font-size: 0.85rem; }
    .search { flex: 1; padding: 0.5rem 0.75rem; border-radius: 10px; border: 1px solid var(--border); background: color-mix(in srgb, var(--bg) 96%, #00000008); color: var(--fg); }
    main { padding: 1rem; max-width: 1200px; margin: 0 auto; }
    table { width: 100%; border-collapse: collapse; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    thead th { text-align: left; font-size: 0.85rem; color: var(--muted); padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--border); background: color-mix(in srgb, var(--bg) 94%, #00000006); position: sticky; top: 56px; }
    tbody td { padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border); vertical-align: top; }
    tbody tr:nth-child(even) { background: var(--row); }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .muted { color: var(--muted); font-size: 0.85rem; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .links a { margin-right: 0.6rem; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <header>
    <h1>Polylogue Timeline</h1>
    <input id="q" class="search" type="search" placeholder="Filter (provider/title/slug) — press /" />
    <span class="pill" id="count"></span>
  </header>
  <main>
    <div class="muted">Generated at {{ generated_at }} (rows={{ rows|length }})</div>
    <div style="height: 0.75rem"></div>
    <table>
      <thead>
        <tr>
          <th>Updated</th>
          <th>Provider</th>
          <th>Title</th>
          <th>Slug</th>
          <th class="num">Branches</th>
          <th class="num">Tokens</th>
          <th class="num">Words</th>
          <th class="num">Attachments</th>
          <th class="num">Attachment MiB</th>
          <th>Links</th>
        </tr>
      </thead>
      <tbody id="rows">
        {% for row in rows %}
        <tr data-filter="{{ (row.provider ~ ' ' ~ (row.title or '') ~ ' ' ~ row.slug)|lower }}">
          <td class="muted">{{ row.last_updated or '-' }}</td>
          <td>{{ row.provider }}</td>
          <td>{{ row.title or '-' }}</td>
          <td><code>{{ row.slug }}</code></td>
          <td class="num">{{ row.branch_count }}</td>
          <td class="num">{{ row.tokens }}</td>
          <td class="num">{{ row.words }}</td>
          <td class="num">{{ row.attachments }}</td>
          <td class="num">{{ '%.2f'|format((row.attachment_bytes or 0) / (1024*1024)) }}</td>
          <td class="links">
            {% if row.output_path %}<a href="{{ row.output_path }}">md</a>{% endif %}
            {% if row.html_path %}<a href="{{ row.html_path }}">html</a>{% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </main>
  <script>
    const q = document.getElementById('q');
    const count = document.getElementById('count');
    const rows = Array.from(document.querySelectorAll('#rows tr'));
    function apply() {
      const term = (q.value || '').trim().toLowerCase();
      let visible = 0;
      for (const tr of rows) {
        const hay = tr.getAttribute('data-filter') || '';
        const hit = !term || hay.includes(term);
        tr.classList.toggle('hidden', !hit);
        if (hit) visible += 1;
      }
      count.textContent = term ? `${visible} match(es)` : '';
    }
    q.addEventListener('input', apply);
    window.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== q) {
        e.preventDefault();
        q.focus();
      }
    });
  </script>
</body>
</html>
"""


def run_timeline_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    providers = _normalize_filter(getattr(args, "providers", None))
    limit = getattr(args, "limit", 500)
    limit = int(limit) if isinstance(limit, int) else 500
    if limit <= 0:
        limit = 500
    json_mode = bool(getattr(args, "json", False))
    out_path = getattr(args, "out", None)
    theme = getattr(args, "theme", "light") or "light"
    open_result = bool(getattr(args, "open", False))

    rows = _load_rows(limit=limit, providers=providers)

    if json_mode:
        payload = stamp_payload(
            {
                "generated_at": datetime.now(timezone.utc).timestamp(),
                "count": len(rows),
                "rows": [row.__dict__ for row in rows],
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if out_path is None:
        out_path = Path.cwd() / "timeline.html"
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    html = Environment(loader=BaseLoader(), autoescape=True).from_string(_TEMPLATE).render(
        rows=rows,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        theme=theme,
    )
    out.write_text(html, encoding="utf-8")
    if open_result:
        open_in_editor(out)
    else:
        env.ui.console.print(f"[green]Wrote timeline → {out}[/green]")


def _normalize_filter(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    values = {chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()}
    return values or None


def _decode_meta(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _load_rows(*, limit: int, providers: Optional[set[str]]) -> List[TimelineRow]:
    with open_connection(None) as conn:
        rows = conn.execute(
            """
            SELECT provider, conversation_id, slug, title, last_updated, metadata_json
              FROM conversations
            """
        ).fetchall()
        branch_rows = conn.execute(
            """
            SELECT provider, conversation_id, COUNT(*) AS n
              FROM branches
             GROUP BY provider, conversation_id
            """
        ).fetchall()

    branch_counts: Dict[tuple[str, str], int] = {(r["provider"], r["conversation_id"]): int(r["n"] or 0) for r in branch_rows}
    out: List[TimelineRow] = []
    for row in rows:
        provider = (row["provider"] or "").lower()
        if providers and provider not in providers:
            continue
        conversation_id = row["conversation_id"]
        slug = row["slug"]
        title = row["title"]
        last_updated = row["last_updated"]
        meta = _decode_meta(row["metadata_json"])
        tokens = int(meta.get("token_count", meta.get("tokens", 0)) or 0)
        words = int(meta.get("word_count", meta.get("words", 0)) or 0)
        attachments = int(meta.get("attachments", 0) or 0)
        attachment_bytes = int(meta.get("attachment_bytes", meta.get("attachmentBytes", 0)) or 0)
        output_path = meta.get("outputPath")
        html_path = meta.get("htmlPath")
        bc = branch_counts.get((row["provider"], conversation_id), 0)
        out.append(
            TimelineRow(
                provider=provider,
                conversation_id=conversation_id,
                slug=slug,
                title=title,
                last_updated=last_updated,
                tokens=tokens,
                words=words,
                attachments=attachments,
                attachment_bytes=attachment_bytes,
                branch_count=bc,
                output_path=str(output_path) if isinstance(output_path, str) else None,
                html_path=str(html_path) if isinstance(html_path, str) else None,
            )
        )
    out.sort(key=lambda r: (r.last_updated or "", r.provider, r.slug), reverse=True)
    return out[:limit]


__all__ = ["run_timeline_cli"]

