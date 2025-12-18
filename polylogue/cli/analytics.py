from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment

from ..commands import CommandEnv
from ..db import open_connection
from ..schema import stamp_payload
from .editor import open_in_editor


_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Polylogue Analytics</title>
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
    main { padding: 1rem; max-width: 1200px; margin: 0 auto; }
    h2 { font-size: 1rem; margin: 1.25rem 0 0.5rem; }
    .muted { color: var(--muted); font-size: 0.85rem; }
    table { width: 100%; border-collapse: collapse; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    thead th { text-align: left; font-size: 0.85rem; color: var(--muted); padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--border); background: color-mix(in srgb, var(--bg) 94%, #00000006); }
    tbody td { padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border); vertical-align: top; }
    tbody tr:nth-child(even) { background: var(--row); }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
  <header>
    <h1>Polylogue Analytics</h1>
    <span class="muted">generated {{ generated_at }}</span>
  </header>
  <main>
    <h2>Providers</h2>
    <table>
      <thead><tr>
        <th>Provider</th>
        <th class="num">Conversations</th>
        <th class="num">Branches</th>
        <th class="num">Messages</th>
        <th class="num">Attachments</th>
        <th class="num">Attachment MiB</th>
      </tr></thead>
      <tbody>
        {% for row in provider_rows %}
        <tr>
          <td>{{ row.provider }}</td>
          <td class="num">{{ row.conversations }}</td>
          <td class="num">{{ row.branches }}</td>
          <td class="num">{{ row.messages }}</td>
          <td class="num">{{ row.attachments }}</td>
          <td class="num">{{ '%.2f'|format(row.attachment_bytes / (1024*1024)) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <h2>Top Models</h2>
    <table>
      <thead><tr>
        <th>Provider</th>
        <th>Model</th>
        <th class="num">Messages</th>
        <th class="num">Tokens</th>
      </tr></thead>
      <tbody>
        {% for row in model_rows %}
        <tr>
          <td>{{ row.provider }}</td>
          <td><code>{{ row.model }}</code></td>
          <td class="num">{{ row.messages }}</td>
          <td class="num">{{ row.tokens }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <h2>Roles</h2>
    <table>
      <thead><tr>
        <th>Provider</th>
        <th>Role</th>
        <th class="num">Messages</th>
        <th class="num">Tokens</th>
      </tr></thead>
      <tbody>
        {% for row in role_rows %}
        <tr>
          <td>{{ row.provider }}</td>
          <td><code>{{ row.role }}</code></td>
          <td class="num">{{ row.messages }}</td>
          <td class="num">{{ row.tokens }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <h2>Branch Hotspots</h2>
    <div class="muted">Conversations with most branches</div>
    <div style="height: 0.5rem"></div>
    <table>
      <thead><tr>
        <th>Provider</th>
        <th>Slug</th>
        <th>Title</th>
        <th class="num">Branches</th>
      </tr></thead>
      <tbody>
        {% for row in branch_hotspots %}
        <tr>
          <td>{{ row.provider }}</td>
          <td><code>{{ row.slug }}</code></td>
          <td>{{ row.title or '-' }}</td>
          <td class="num">{{ row.branch_count }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def run_analytics_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    providers = _normalize_filter(getattr(args, "providers", None))
    model_limit = max(1, int(getattr(args, "model_limit", 25) or 25))
    hotspot_limit = max(1, int(getattr(args, "hotspot_limit", 25) or 25))
    json_mode = bool(getattr(args, "json", False))
    out_path = getattr(args, "out", None)
    theme = getattr(args, "theme", "light") or "light"
    open_result = bool(getattr(args, "open", False))

    payload = _build_payload(providers=providers, model_limit=model_limit, hotspot_limit=hotspot_limit)

    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if out_path is None:
        out_path = Path.cwd() / "analytics.html"
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    html = Environment(loader=BaseLoader(), autoescape=True).from_string(_TEMPLATE).render(
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        theme=theme,
        provider_rows=payload["providerRows"],
        model_rows=payload["modelRows"],
        role_rows=payload["roleRows"],
        branch_hotspots=payload["branchHotspots"],
    )
    out.write_text(html, encoding="utf-8")
    if open_result:
        open_in_editor(out)
    else:
        env.ui.console.print(f"[green]Wrote analytics → {out}[/green]")


def _normalize_filter(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    values = {chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()}
    return values or None


def _placeholders(n: int) -> str:
    return ",".join("?" for _ in range(n))


def _build_payload(*, providers: Optional[set[str]], model_limit: int, hotspot_limit: int) -> Dict[str, Any]:
    provider_rows: List[Dict[str, Any]] = []
    model_rows: List[Dict[str, Any]] = []
    role_rows: List[Dict[str, Any]] = []
    branch_hotspots: List[Dict[str, Any]] = []

    params: List[object] = []
    provider_where = ""
    if providers:
        provider_where = f"WHERE provider IN ({_placeholders(len(providers))})"
        params.extend(sorted(providers))

    with open_connection(None) as conn:
        conv_rows = conn.execute(
            f"SELECT provider, COUNT(*) AS n FROM conversations {provider_where} GROUP BY provider",
            params,
        ).fetchall()
        branch_rows = conn.execute(
            f"SELECT provider, COUNT(*) AS n FROM branches {provider_where} GROUP BY provider",
            params,
        ).fetchall()
        msg_rows = conn.execute(
            f"SELECT provider, COUNT(*) AS n, COALESCE(SUM(token_count), 0) AS tokens FROM messages {provider_where} GROUP BY provider",
            params,
        ).fetchall()
        att_rows = conn.execute(
            f"SELECT provider, COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes FROM attachments {provider_where} GROUP BY provider",
            params,
        ).fetchall()

        conv_counts = {row["provider"]: int(row["n"] or 0) for row in conv_rows}
        branch_counts = {row["provider"]: int(row["n"] or 0) for row in branch_rows}
        msg_counts = {row["provider"]: (int(row["n"] or 0), int(row["tokens"] or 0)) for row in msg_rows}
        att_counts = {row["provider"]: (int(row["n"] or 0), int(row["bytes"] or 0)) for row in att_rows}

        all_providers = sorted(set(conv_counts) | set(branch_counts) | set(msg_counts) | set(att_counts))
        for provider in all_providers:
            messages_n, messages_tokens = msg_counts.get(provider, (0, 0))
            attachments_n, attachments_bytes = att_counts.get(provider, (0, 0))
            provider_rows.append(
                {
                    "provider": provider,
                    "conversations": conv_counts.get(provider, 0),
                    "branches": branch_counts.get(provider, 0),
                    "messages": messages_n,
                    "messageTokens": messages_tokens,
                    "attachments": attachments_n,
                    "attachment_bytes": attachments_bytes,
                }
            )

        role_params = list(params)
        role_rows_raw = conn.execute(
            f"""
            SELECT provider, COALESCE(role, 'unknown') AS role, COUNT(*) AS n, COALESCE(SUM(token_count), 0) AS tokens
              FROM messages
              {provider_where}
             GROUP BY provider, role
             ORDER BY provider, n DESC, role
            """,
            role_params,
        ).fetchall()
        for row in role_rows_raw:
            role_rows.append({"provider": row["provider"], "role": row["role"], "messages": int(row["n"] or 0), "tokens": int(row["tokens"] or 0)})

        model_params = list(params)
        model_where = provider_where
        if model_where:
            model_where += " AND model IS NOT NULL AND model != ''"
        else:
            model_where = "WHERE model IS NOT NULL AND model != ''"
        model_rows_raw = conn.execute(
            f"""
            SELECT provider, model, COUNT(*) AS n, COALESCE(SUM(token_count), 0) AS tokens
              FROM messages
              {model_where}
             GROUP BY provider, model
             ORDER BY n DESC, tokens DESC
             LIMIT ?
            """,
            [*model_params, model_limit],
        ).fetchall()
        for row in model_rows_raw:
            model_rows.append(
                {
                    "provider": row["provider"],
                    "model": row["model"],
                    "messages": int(row["n"] or 0),
                    "tokens": int(row["tokens"] or 0),
                }
            )

        hotspot_params = list(params)
        hotspot_where = ""
        if providers:
            hotspot_where = f"WHERE b.provider IN ({_placeholders(len(providers))})"
        hotspot_rows = conn.execute(
            f"""
            SELECT b.provider, b.conversation_id, COUNT(*) AS n,
                   c.slug AS slug, c.title AS title
              FROM branches AS b
              JOIN conversations AS c
                ON c.provider = b.provider AND c.conversation_id = b.conversation_id
              {hotspot_where}
             GROUP BY b.provider, b.conversation_id
             ORDER BY n DESC, b.provider, c.slug
             LIMIT ?
            """,
            [*hotspot_params, hotspot_limit],
        ).fetchall()
        for row in hotspot_rows:
            branch_hotspots.append(
                {
                    "provider": row["provider"],
                    "conversation_id": row["conversation_id"],
                    "slug": row["slug"],
                    "title": row["title"],
                    "branch_count": int(row["n"] or 0),
                }
            )

    return stamp_payload(
        {
            "generated_at": datetime.now(timezone.utc).timestamp(),
            "provider_rows": provider_rows,
            "role_rows": role_rows,
            "model_rows": model_rows,
            "branch_hotspots": branch_hotspots,
        }
    )


__all__ = ["run_analytics_cli"]

