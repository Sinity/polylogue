"""Daemon-served SSR and bundled assets for the progressive WebUI.

The legacy reader remains in :mod:`polylogue.daemon.web_shell`.  This module
owns only the strangler mount at ``/app``: a semantic first page produced from
the shared query transaction and a manifest-governed set of local Vite assets.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from polylogue.archive.query.execution_control import classify_unit_expression_workload
from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest
from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request
from polylogue.logging import get_logger
from polylogue.surfaces.payloads import MessageQueryRowPayload, QueryUnitEnvelope

ARCHIVE_OVERVIEW_EXPRESSION = "messages where words >= 0 | sort by time desc"
ARCHIVE_OVERVIEW_LIMIT = 6
_WEBUI_ENTRY_NAME = "archive-overview"
_HASHED_ASSET_RE = re.compile(r"^[A-Za-z0-9._-]+-[A-Za-z0-9_-]{8,}\.(?:css|js)$")
logger = get_logger(__name__)


class WebUIAssetError(RuntimeError):
    """Raised when the packaged Vite manifest or one of its assets is invalid."""


@dataclass(frozen=True, slots=True)
class WebUIEntrypoint:
    """Manifest-resolved browser entry and its local stylesheets."""

    script: str
    stylesheets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WebUIAsset:
    """One manifest-governed immutable asset ready for HTTP transport."""

    name: str
    body: bytes
    content_type: str
    etag: str


class WebUIAssetBundle:
    """Read a Vite build through package resources or a test/dev override."""

    def __init__(self, root: Traversable | Path) -> None:
        self._root = root
        self._manifest = self._read_manifest()
        self._allowed_assets = self._manifest_asset_names()

    @classmethod
    def discover(cls, override_root: Path | None = None) -> WebUIAssetBundle:
        root: Traversable | Path
        if override_root is None:
            root = resources.files("polylogue.daemon").joinpath("static", "dist")
        else:
            root = override_root
        return cls(root)

    def entrypoint(self) -> WebUIEntrypoint:
        return self.entrypoint_for(_WEBUI_ENTRY_NAME)

    def entrypoint_for(self, name: str) -> WebUIEntrypoint:
        entry: dict[str, object] | None = None
        for raw in self._manifest.values():
            if not isinstance(raw, dict):
                continue
            if raw.get("isEntry") is True and raw.get("name") == name:
                entry = raw
                break
        if entry is None:
            raise WebUIAssetError(f"Vite manifest has no {name!r} entry")
        script = self._manifest_asset_name(entry.get("file"))
        stylesheets = self._entrypoint_stylesheets(entry)
        return WebUIEntrypoint(script=script, stylesheets=stylesheets)

    def _entrypoint_stylesheets(self, entry: dict[str, object]) -> tuple[str, ...]:
        """Collect CSS emitted on an entry and its Vite import graph.

        Vite is free to attach shared CSS to an imported chunk rather than the
        entry itself.  The daemon must preserve that graph detail while still
        exposing only manifest-governed, immutable assets to the WebUI.
        """
        stylesheets: list[str] = []
        seen_chunks: set[str] = set()

        def visit(chunk: dict[str, object], chunk_name: str) -> None:
            if chunk_name in seen_chunks:
                return
            seen_chunks.add(chunk_name)
            raw_imports = chunk.get("imports", [])
            if not isinstance(raw_imports, list):
                raise WebUIAssetError("Vite manifest imports field must be a list")
            for imported_name in raw_imports:
                if not isinstance(imported_name, str):
                    raise WebUIAssetError("Vite manifest import name must be a string")
                imported = self._manifest.get(imported_name)
                if not isinstance(imported, dict):
                    raise WebUIAssetError(f"Vite manifest import is missing: {imported_name}")
                visit(imported, imported_name)
            raw_css = chunk.get("css", [])
            if not isinstance(raw_css, list):
                raise WebUIAssetError("Vite manifest css field must be a list")
            for value in raw_css:
                stylesheet = self._manifest_asset_name(value)
                if stylesheet not in stylesheets:
                    stylesheets.append(stylesheet)

        visit(entry, "<entry>")
        return tuple(stylesheets)

    def read_asset(self, name: str) -> WebUIAsset:
        if name not in self._allowed_assets:
            raise FileNotFoundError(name)
        normalized = self._manifest_asset_name(name)
        resource = self._root.joinpath(normalized)
        if not resource.is_file():
            raise WebUIAssetError(f"manifest asset is missing: {normalized}")
        body = resource.read_bytes()
        content_type = "text/javascript; charset=utf-8" if normalized.endswith(".js") else "text/css; charset=utf-8"
        digest = hashlib.sha256(body).hexdigest()
        return WebUIAsset(
            name=normalized,
            body=body,
            content_type=content_type,
            etag=f'"sha256-{digest}"',
        )

    def _read_manifest(self) -> dict[str, object]:
        manifest_resource = self._root.joinpath("manifest.json")
        if not manifest_resource.is_file():
            raise WebUIAssetError("Vite manifest is missing from the packaged WebUI")
        try:
            payload = json.loads(manifest_resource.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebUIAssetError("Vite manifest is unreadable") from exc
        if not isinstance(payload, dict):
            raise WebUIAssetError("Vite manifest root must be an object")
        return payload

    def _manifest_asset_names(self) -> frozenset[str]:
        names: set[str] = set()
        for raw in self._manifest.values():
            if not isinstance(raw, dict):
                continue
            for field in ("file", "css", "assets"):
                value = raw.get(field)
                if isinstance(value, str):
                    names.add(self._manifest_asset_name(value))
                elif isinstance(value, list):
                    names.update(self._manifest_asset_name(item) for item in value)
        return frozenset(names)

    @staticmethod
    def _manifest_asset_name(value: object) -> str:
        if not isinstance(value, str):
            raise WebUIAssetError("Vite manifest asset name must be a string")
        path = PurePosixPath(value)
        if len(path.parts) != 1 or path.name != value or value in {".", ".."}:
            raise WebUIAssetError("Vite assets must be emitted as single path segments")
        if not _HASHED_ASSET_RE.fullmatch(value):
            raise WebUIAssetError(f"Vite asset is not content-hashed: {value}")
        return value


def load_archive_overview_page(archive_root: Path) -> QueryUnitEnvelope:
    """Read the first overview page through the shared bounded transaction."""

    unit_request = query_unit_request(
        expression=ARCHIVE_OVERVIEW_EXPRESSION,
        limit=ARCHIVE_OVERVIEW_LIMIT,
        offset=0,
    )
    transaction = QueryTransaction(
        archive_root,
        QueryTransactionRequest(
            operation="query_units",
            arguments={
                "expression": unit_request.expression,
                "session_filters": dict(unit_request.session_filters or {}),
            },
            page_size=unit_request.limit,
            offset=unit_request.offset,
        ),
        workload_class=classify_unit_expression_workload(unit_request.expression),
    )
    payload = transaction.run_sync(
        lambda archive: query_unit_envelope(
            archive,
            unit_request,
            execution_context=transaction.context,
            transaction_request=transaction.request,
        )
    )
    if not isinstance(payload, QueryUnitEnvelope) or payload.unit != "message":
        raise RuntimeError("archive overview query returned an unexpected envelope")
    return payload


def render_archive_overview_page(
    bundle: WebUIAssetBundle,
    page: QueryUnitEnvelope | None,
    *,
    notice: str | None = None,
) -> str:
    """Render semantic archive HTML plus a small progressive-island seam."""

    entry = bundle.entrypoint()
    rows = _message_rows(page)
    continuation = page.continuation if page is not None else None
    query_ref = page.query_ref if page is not None else None
    result_ref = page.result_ref if page is not None else None
    bootstrap = _json_script(
        {
            "continuation": continuation,
            "query_ref": query_ref,
            "result_ref": result_ref,
        }
    )
    stylesheet_links = "\n".join(
        f'    <link rel="stylesheet" href="/app/assets/{html.escape(name, quote=True)}">' for name in entry.stylesheets
    )
    if rows:
        rendered_rows = "\n".join(_render_message_row(row) for row in rows)
    else:
        rendered_rows = (
            '          <li class="activity-row"><p class="activity-row__preview">'
            "No indexed message activity is available.</p></li>"
        )
    status_text = notice or (
        f"Showing {len(rows)} bounded archive records. Additional pages replay the opaque continuation."
    )
    disabled = ' disabled=""' if continuation is None else ""
    button_text = "All activity loaded" if continuation is None else "Load more activity"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>Archive overview · Polylogue</title>
{stylesheet_links}
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to archive overview</a>
    {_render_site_header("/app")}
    <main id="main" class="page-shell">
      <p class="eyebrow">Daemon-served semantic HTML</p>
      <h1>Archive overview</h1>
      <p class="lede">A bounded first page is readable without JavaScript. Preact enhances only the continuation control.</p>
      <section class="activity-panel" aria-labelledby="recent-activity-title">
        <div class="activity-panel__heading">
          <h2 id="recent-activity-title">Recent archive messages</h2>
          <p>{html.escape(status_text)}</p>
        </div>
        <ol id="archive-activity-list" class="activity-list">
{rendered_rows}
        </ol>
        <div id="archive-overview-island" data-island="archive-overview">
          <button class="load-more" type="button"{disabled} aria-controls="archive-activity-more" aria-busy="false">{button_text}</button>
          <p class="island-status" role="status" aria-live="polite"></p>
          <ol id="archive-activity-more" class="activity-list activity-list--continued" aria-label="Additional archive activity"></ol>
        </div>
      </section>
    </main>
    <script id="archive-overview-bootstrap" type="application/json">{bootstrap}</script>
    <script type="module" src="/app/assets/{html.escape(entry.script, quote=True)}"></script>
  </body>
</html>
"""


SESSION_LIST_LIMIT = 20
SESSION_READ_MESSAGE_LIMIT = 30


def render_session_list_page(
    bundle: WebUIAssetBundle,
    page: Mapping[str, object] | None,
    filters: Mapping[str, str],
    *,
    notice: str | None = None,
) -> str:
    """Render the session list: origin/date/repo facets over paged session rows."""

    entry = bundle.entrypoint_for("session-list")
    stylesheet_links = "\n".join(
        f'    <link rel="stylesheet" href="/app/assets/{html.escape(name, quote=True)}">' for name in entry.stylesheets
    )
    items = page.get("items") if page is not None else None
    rows = items if isinstance(items, list) else []
    total = page.get("total") if page is not None else None
    limit = page.get("limit") if isinstance(page, Mapping) else SESSION_LIST_LIMIT
    offset = page.get("offset") if isinstance(page, Mapping) else 0
    limit_int = limit if isinstance(limit, int) else SESSION_LIST_LIMIT
    offset_int = offset if isinstance(offset, int) else 0
    has_more = isinstance(total, int) and offset_int + len(rows) < total
    rendered_rows = (
        "\n".join(_render_session_card(row) for row in rows)
        if rows
        else '          <li class="activity-row"><p class="activity-row__preview">No sessions match these filters.</p></li>'
    )
    total_text = f"{total:,} sessions" if isinstance(total, int) else "an unknown number of sessions"
    status_text = notice or f"Showing {len(rows)} of {total_text}. Additional pages replay the same facets."
    bootstrap = _json_script(
        {
            "filters": dict(filters),
            "limit": limit_int,
            "next_offset": offset_int + len(rows) if has_more else None,
        }
    )
    facet_form = _render_session_facet_form(filters)
    disabled = "" if has_more else ' disabled=""'
    button_text = "All matching sessions loaded" if not has_more else "Load more sessions"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>Sessions · Polylogue</title>
{stylesheet_links}
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to session list</a>
    {_render_site_header("/app/sessions")}
    <main id="main" class="page-shell">
      <p class="eyebrow">Daemon-served semantic HTML</p>
      <h1>Sessions</h1>
      <p class="lede">Filter the archive by origin, time window, and repo. A bounded first page is readable without JavaScript.</p>
      {facet_form}
      <section class="activity-panel" aria-labelledby="session-list-title">
        <div class="activity-panel__heading">
          <h2 id="session-list-title">Matching sessions</h2>
          <p>{html.escape(status_text)}</p>
        </div>
        <ol id="session-list" class="activity-list">
{rendered_rows}
        </ol>
        <div id="session-list-island" data-island="session-list">
          <button class="load-more" type="button"{disabled} aria-controls="session-list-more" aria-busy="false">{button_text}</button>
          <p class="island-status" role="status" aria-live="polite"></p>
          <ol id="session-list-more" class="activity-list activity-list--continued" aria-label="Additional matching sessions"></ol>
        </div>
      </section>
    </main>
    <script id="session-list-bootstrap" type="application/json">{bootstrap}</script>
    <script type="module" src="/app/assets/{html.escape(entry.script, quote=True)}"></script>
  </body>
</html>
"""


def _render_session_facet_form(filters: Mapping[str, str]) -> str:
    origin = html.escape(str(filters.get("origin") or ""), quote=True)
    since = html.escape(str(filters.get("since") or ""), quote=True)
    repo = html.escape(str(filters.get("repo") or ""), quote=True)
    return f"""      <form class="session-facets" method="get" action="/app/sessions" role="search" aria-label="Session filters">
        <div class="session-facets__field">
          <label for="facet-origin">Origin</label>
          <input id="facet-origin" name="origin" type="text" value="{origin}" placeholder="e.g. codex-session" autocomplete="off">
        </div>
        <div class="session-facets__field">
          <label for="facet-since">Since</label>
          <input id="facet-since" name="since" type="text" value="{since}" placeholder="e.g. 7d or 2026-07-01" autocomplete="off">
        </div>
        <div class="session-facets__field">
          <label for="facet-repo">Repo</label>
          <input id="facet-repo" name="repo" type="text" value="{repo}" placeholder="e.g. polylogue" autocomplete="off">
        </div>
        <button type="submit">Apply filters</button>
      </form>"""


def _render_session_card(raw: object) -> str:
    row = raw if isinstance(raw, Mapping) else {}
    session_id = str(row.get("id") or "")
    title = row.get("title") or session_id or "[untitled session]"
    origin = str(row.get("origin") or "unknown-export")
    date = row.get("date") or row.get("created_at")
    time_markup = (
        "<span>Time unavailable</span>"
        if not date
        else f'<time datetime="{html.escape(str(date), quote=True)}">{html.escape(str(date))}</time>'
    )
    message_count = row.get("message_count")
    word_count = row.get("word_count")
    repo = row.get("repo")
    detail_bits = [
        f"{message_count:,} messages" if isinstance(message_count, int) else "unknown message count",
    ]
    if isinstance(word_count, int):
        detail_bits.append(f"{word_count:,} words")
    if repo:
        detail_bits.append(html.escape(str(repo)))
    else:
        detail_bits.append("repo unknown")
    return f"""          <li class="activity-row" data-session-id="{html.escape(session_id, quote=True)}">
            <div class="activity-row__meta"><span class="activity-row__origin">{html.escape(origin)}</span>{time_markup}</div>
            <h3><a href="/app/sessions/{quote(session_id, safe="")}">{html.escape(str(title))}</a></h3>
            <p class="activity-row__detail">{" · ".join(detail_bits)}</p>
          </li>"""


def render_session_read_page(
    bundle: WebUIAssetBundle,
    session_id: str,
    session: Mapping[str, object] | None,
    *,
    notice: str | None = None,
) -> str:
    """Render the session read shell: header, lineage banner, message flow skeleton."""

    entry = bundle.entrypoint_for("session-read")
    stylesheet_links = "\n".join(
        f'    <link rel="stylesheet" href="/app/assets/{html.escape(name, quote=True)}">' for name in entry.stylesheets
    )
    if session is None:
        body = f"<p>{html.escape(notice or 'This session could not be found or the archive is unavailable.')}</p>"
        bootstrap = _json_script({"session_id": session_id, "messages": [], "next_offset": None})
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>Session not found · Polylogue</title>
{stylesheet_links}
  </head>
  <body>
    {_render_site_header("/app/sessions")}
    <main id="main" class="page-shell">
      <p class="eyebrow">Daemon-served semantic HTML</p>
      <h1>Session unavailable</h1>
      {body}
    </main>
    <script id="session-read-bootstrap" type="application/json">{bootstrap}</script>
    <script type="module" src="/app/assets/{html.escape(entry.script, quote=True)}"></script>
  </body>
</html>
"""
    title = session.get("title") or session.get("display_title") or session_id
    origin = str(session.get("origin") or "unknown-export")
    all_messages = session.get("messages")
    messages = all_messages if isinstance(all_messages, list) else []
    total_messages = len(messages)
    first_page = messages[:SESSION_READ_MESSAGE_LIMIT]
    has_more = total_messages > len(first_page)
    lineage_banner = _render_lineage_banner(session)
    rendered_messages = (
        "\n".join(_render_message_flow_item(m) for m in first_page)
        if first_page
        else '        <li class="message-flow__item"><p>No indexed messages are available for this session.</p></li>'
    )
    bootstrap = _json_script(
        {
            "session_id": session_id,
            "next_offset": len(first_page) if has_more else None,
        }
    )
    header_detail = " · ".join(
        bit
        for bit in (
            f"{session.get('message_count'):,} messages" if isinstance(session.get("message_count"), int) else None,
            f"{session.get('word_count'):,} words" if isinstance(session.get("word_count"), int) else None,
            str(session.get("repo")) if session.get("repo") else None,
        )
        if bit
    )
    disabled = "" if has_more else ' disabled=""'
    button_text = "All messages loaded" if not has_more else "Load more messages"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>{html.escape(str(title))} · Polylogue</title>
{stylesheet_links}
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to session transcript</a>
    {_render_site_header("/app/sessions")}
    <main id="main" class="page-shell">
      <p class="eyebrow">{html.escape(origin)}</p>
      <h1>{html.escape(str(title))}</h1>
      <p class="lede">{html.escape(header_detail) or "No structural detail is available for this session."}</p>
      {lineage_banner}
      <section class="message-flow-panel" aria-labelledby="message-flow-title">
        <div class="activity-panel__heading">
          <h2 id="message-flow-title">Message flow</h2>
          <p>{html.escape(notice or f"Showing {len(first_page)} of {total_messages:,} messages.")}</p>
        </div>
        <ol id="message-flow" class="message-flow">
{rendered_messages}
        </ol>
        <div id="session-read-island" data-island="session-read">
          <button class="load-more" type="button"{disabled} aria-controls="message-flow-more" aria-busy="false">{button_text}</button>
          <p class="island-status" role="status" aria-live="polite"></p>
          <ol id="message-flow-more" class="message-flow message-flow--continued" aria-label="Additional messages"></ol>
        </div>
      </section>
    </main>
    <script id="session-read-bootstrap" type="application/json">{bootstrap}</script>
    <script type="module" src="/app/assets/{html.escape(entry.script, quote=True)}"></script>
  </body>
</html>
"""


def _render_lineage_banner(session: Mapping[str, object]) -> str:
    """Prefer the registry's evidence-bearing lineage card; fall back to the plain banner.

    A ``kind: "lineage"`` card (from the session-level ``semantic_entries``)
    carries relation/resolution/cycle-detection evidence a bare
    ``parent_id``/``branch_type`` pair does not; render it when materialized.
    """

    session_entries_raw = session.get("semantic_entries")
    session_entries = session_entries_raw if isinstance(session_entries_raw, list) else []
    for entry in session_entries:
        if not isinstance(entry, Mapping) or entry.get("entry_type") != "card":
            continue
        card = entry.get("card")
        if isinstance(card, Mapping) and card.get("kind") == "lineage":
            return f'      <div class="lineage-banner">{_render_semantic_card(card)}</div>'
    parent_id = session.get("parent_id")
    if not parent_id:
        return ""
    branch_type = str(session.get("branch_type") or "unknown")
    parent_href = f"/app/sessions/{quote(str(parent_id), safe='')}"
    return (
        '      <p class="lineage-banner" data-branch-type="'
        + html.escape(branch_type, quote=True)
        + '">This session composes a parent prefix (<strong>'
        + html.escape(branch_type)
        + '</strong>) from <a href="'
        + parent_href
        + '">'
        + html.escape(str(parent_id))
        + "</a>.</p>"
    )


#: Display label per SemanticCardKind, matching the legacy web shell's SEM_CARD_LABEL
#: (polylogue/daemon/web_shell_semantic_cards.py) so both surfaces read the same vocabulary.
SEMANTIC_CARD_LABELS: Mapping[str, str] = {
    "shell": "shell",
    "file_read": "read",
    "file_edit": "edit",
    "search": "search",
    "web": "web",
    "task": "task",
    "mcp": "mcp",
    "lineage": "lineage",
    "attachment": "attachment",
    "fallback": "tool",
}


def _render_message_flow_item(raw: object) -> str:
    message = raw if isinstance(raw, Mapping) else {}
    if message.get("semantic_card_suppressed"):
        return ""
    message_id = str(message.get("id") or "")
    role = str(message.get("role") or "unknown")
    material_origin = str(message.get("material_origin") or "unknown")
    timestamp = message.get("timestamp")
    time_markup = (
        f'<time datetime="{html.escape(str(timestamp), quote=True)}">{html.escape(str(timestamp))}</time>'
        if timestamp
        else "<span>Time unavailable</span>"
    )
    entries_raw = message.get("semantic_entries")
    entries = entries_raw if isinstance(entries_raw, list) and entries_raw else None
    semantic_body = "".join(_render_semantic_entry(entry) for entry in entries) if entries is not None else ""
    # A nonempty semantic_entries list whose entries are all malformed or of an
    # unsupported entry_type must still fall back to honest plain text rather
    # than rendering a blank message.
    body = semantic_body or _render_message_flow_fallback_body(message)
    return f"""        <li class="message-flow__item" id="msg-{html.escape(message_id, quote=True)}" data-role="{html.escape(role, quote=True)}" data-material-origin="{html.escape(material_origin, quote=True)}">
          <div class="message-flow__meta">
            <span class="message-flow__role">{html.escape(role)}</span>
            <span class="message-flow__material-origin">{html.escape(material_origin)}</span>
            {time_markup}
          </div>
{body}
        </li>"""


def _render_message_flow_fallback_body(message: Mapping[str, object]) -> str:
    """Render the honest role/material_origin/text placeholder.

    Used only when the archive has no materialized ``semantic_entries`` for
    this message (older un-reindexed rows) - evidence is never dropped, it
    just isn't card-classified yet.
    """
    text = message.get("text")
    preview = _compact_preview(str(text or ""), limit=600) or "[empty message]"
    flags: list[str] = []
    if message.get("has_tool_use"):
        flags.append('<span class="message-flag" data-flag="tool-use">tool use</span>')
    if message.get("has_thinking"):
        flags.append('<span class="message-flag" data-flag="thinking">thinking</span>')
    if message.get("has_paste_evidence"):
        flags.append('<span class="message-flag" data-flag="paste">paste</span>')
    flags_markup = " ".join(flags)
    return (
        f'          <p class="message-flow__text">{html.escape(preview)}</p>\n'
        f'          <div class="message-flow__flags">{flags_markup}</div>'
    )


def _render_semantic_entry(raw: object) -> str:
    entry = raw if isinstance(raw, Mapping) else {}
    entry_type = entry.get("entry_type")
    if entry_type == "card" and isinstance(entry.get("card"), Mapping):
        return _render_semantic_card(entry["card"])
    if entry_type == "prose" and isinstance(entry.get("prose"), Mapping):
        return _render_semantic_prose(entry["prose"])
    if entry_type == "notice" and isinstance(entry.get("notice"), Mapping):
        return _render_semantic_notice(entry["notice"])
    return ""


def _render_card_field_value(value: str) -> str:
    """Render a card field value, hyperlinking the registry's ``session:``/``message:`` ref convention.

    ``_build_lineage_card``/``_build_task_card`` (polylogue/rendering/semantic_cards.py)
    encode cross-session and cross-message references as plain ``session:<id>``/
    ``message:<id>`` field values rather than a separate typed ref field. Linking
    them generically here (not just for the lineage card) means any card family
    that adopts the same convention - e.g. a future task/delegation child-session
    reference - gets working navigation for free.
    """

    if value.startswith("session:") and len(value) > len("session:"):
        session_ref = value[len("session:") :]
        return f'<a class="card__field-value" href="/app/sessions/{quote(session_ref, safe="")}"><code>{html.escape(value)}</code></a>'
    if value.startswith("message:") and len(value) > len("message:"):
        message_ref = value[len("message:") :]
        return f'<a class="card__field-value" href="#msg-{quote(message_ref, safe="")}"><code>{html.escape(value)}</code></a>'
    return f'<code class="card__field-value">{html.escape(value)}</code>'


def _render_semantic_card(card: Mapping[str, object]) -> str:
    kind = str(card.get("kind") or "fallback")
    label = SEMANTIC_CARD_LABELS.get(kind, kind)
    title = str(card.get("title") or "")
    fields_raw = card.get("fields")
    fields = fields_raw if isinstance(fields_raw, list) else []
    summary = card.get("summary")
    has_summary_field = any(isinstance(f, Mapping) and f.get("value") == summary for f in fields)
    summary_markup = (
        f'<p class="card__summary">{html.escape(str(summary))}</p>' if summary and not has_summary_field else ""
    )
    fields_markup = "".join(
        f'<div class="card__field"><span class="card__field-label">{html.escape(str(f.get("label") or ""))}</span>'
        f"{_render_card_field_value(str(f.get('value') or ''))}</div>"
        for f in fields
        if isinstance(f, Mapping)
    )
    previews_raw = card.get("previews")
    previews_markup = (
        "".join(_render_semantic_preview(p) for p in previews_raw if isinstance(p, Mapping))
        if isinstance(previews_raw, list)
        else ""
    )
    caveats_raw = card.get("caveats")
    caveats_markup = (
        '<ul class="card__caveats">' + "".join(f"<li>{html.escape(str(c))}</li>" for c in caveats_raw) + "</ul>"
        if isinstance(caveats_raw, list) and caveats_raw
        else ""
    )
    source_markup = _render_semantic_source(card.get("source"))
    outcome_markup = _render_semantic_outcome(card.get("outcome"))
    return (
        f'<div class="card card--{html.escape(kind, quote=True)}" data-card-kind="{html.escape(kind, quote=True)}">'
        f'<div class="card__header"><span class="card__kind">{html.escape(label)}</span>'
        f'<span class="card__title">{html.escape(title)}</span>{outcome_markup}</div>'
        f"{summary_markup}{fields_markup}{previews_markup}{caveats_markup}{source_markup}</div>"
    )


def _render_semantic_outcome(raw: object) -> str:
    if not isinstance(raw, Mapping):
        return ""
    state = str(raw.get("state") or "unknown")
    label = {"succeeded": "ok", "failed": "FAILED"}.get(state, "unknown")
    detail_bits = []
    is_error = raw.get("is_error")
    if isinstance(is_error, bool):
        detail_bits.append(f"is_error={is_error}")
    exit_code = raw.get("exit_code")
    if isinstance(exit_code, int):
        detail_bits.append(f"exit {exit_code}")
    title = f"{label} ({', '.join(detail_bits)})" if detail_bits else label
    return (
        f'<span class="card__outcome" data-outcome-state="{html.escape(state, quote=True)}" '
        f'title="{html.escape(title, quote=True)}">{html.escape(label)}</span>'
    )


def _render_semantic_preview(preview: Mapping[str, object]) -> str:
    line_count = preview.get("line_count")
    lines = line_count if isinstance(line_count, int) else 0
    meta_bits = [f"{lines} line{'' if lines == 1 else 's'}"]
    omitted_lines = preview.get("omitted_lines")
    if isinstance(omitted_lines, int) and omitted_lines:
        meta_bits.append(f"{omitted_lines} omitted")
    omitted_chars = preview.get("omitted_characters")
    if isinstance(omitted_chars, int) and omitted_chars:
        meta_bits.append(f"{omitted_chars} chars omitted")
    replacements = preview.get("encoding_replacements")
    if isinstance(replacements, int) and replacements:
        meta_bits.append(f"{replacements} replacements")
    kind = str(preview.get("kind") or "preview")
    text = str(preview.get("text") or "")
    body = _render_diff_lines(text) if kind == "diff" else html.escape(text)
    return (
        f'<details class="card__preview" data-preview-kind="{html.escape(kind, quote=True)}">'
        f"<summary>{html.escape(kind)} · {html.escape(', '.join(meta_bits))}</summary>"
        f"<pre>{body}</pre></details>"
    )


def _render_diff_lines(text: str) -> str:
    rendered_lines = []
    for line in text.split("\n"):
        css_class = "diff-ctx"
        if line.startswith("+") and not line.startswith("+++"):
            css_class = "diff-add"
        elif line.startswith("-") and not line.startswith("---"):
            css_class = "diff-del"
        elif line.startswith("@"):
            css_class = "diff-hunk"
        rendered_lines.append(f'<span class="{css_class}">{html.escape(line)}</span>')
    return "\n".join(rendered_lines)


_SOURCE_FIELD_ORDER = (
    "session_id",
    "provider_family",
    "origin",
    "message_id",
    "block_id",
    "block_index",
    "tool_name",
    "tool_id",
    "attachment_id",
    "material_origin",
    "occurred_at",
    "duration_ms",
    "parent_message_id",
    "variant_index",
    "is_active_path",
    "is_active_leaf",
    "inherited_prefix",
    "result_message_id",
    "result_block_id",
    "result_block_index",
    "result_duration_ms",
    "result_material_origin",
    "result_inherited_prefix",
)


def _render_semantic_source(raw: object) -> str:
    source = raw if isinstance(raw, Mapping) else {}
    known = set(_SOURCE_FIELD_ORDER)
    bits = [f"{key}={source[key]}" for key in _SOURCE_FIELD_ORDER if source.get(key) is not None]
    # SemanticCardSource is a loose passthrough (polylogue/rendering/semantic_card_models.py);
    # a registry-added field not yet in _SOURCE_FIELD_ORDER must still surface as
    # evidence rather than silently disappearing from the rendered card.
    bits.extend(
        f"{key}={value}"
        for key, value in source.items()
        if key not in known and value is not None and isinstance(value, str | int | float | bool)
    )
    if not bits:
        return ""
    return f'<details class="card__source"><summary>evidence</summary><code>{html.escape(chr(10).join(bits))}</code></details>'


def _render_semantic_prose(prose: Mapping[str, object]) -> str:
    block_type = str(prose.get("block_type") or "")
    text = str(prose.get("text") or "")
    meta_bits = [bit for bit in (block_type, prose.get("language")) if bit]
    material_origin = prose.get("material_origin")
    if material_origin:
        meta_bits.append(f"material:{material_origin}")
    meta_markup = (
        '<div class="prose__meta">'
        + "".join(f'<span class="chip">{html.escape(str(bit))}</span>' for bit in meta_bits)
        + "</div>"
        if meta_bits
        else ""
    )
    text_markup = f'<p class="prose__text">{html.escape(text)}</p>' if text else ""
    return f'<div class="prose" data-semantic-block-type="{html.escape(block_type, quote=True)}">{meta_markup}{text_markup}</div>'


def _render_semantic_notice(notice: Mapping[str, object]) -> str:
    count = notice.get("count")
    count_int = count if isinstance(count, int) else 0
    kind = str(notice.get("kind") or "notice")
    label = "thinking absent" if kind == "empty_thinking" else kind
    return (
        f'<div class="notice" data-notice-kind="{html.escape(kind, quote=True)}">'
        f"<strong>{html.escape(label)}</strong> · {count_int} typed block{'' if count_int == 1 else 's'}</div>"
    )


def _render_site_header(current_path: str) -> str:
    def link(href: str, label: str) -> str:
        current = ' aria-current="page"' if href == current_path else ""
        return f'<a href="{href}"{current}>{label}</a>'

    return (
        '<header class="site-header"><p class="eyebrow">Polylogue</p><nav aria-label="WebUI views">'
        + link("/app", "Archive overview")
        + link("/app/sessions", "Sessions")
        + link("/app/observability", "Observability")
        + link("/", "Legacy reader")
        + "</nav></header>"
    )


async def build_observability_payload(
    operations: object,
    status: Mapping[str, object],
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Return descriptor-owned insight panels and a status-snapshot adapter.

    The browser receives display fields from the insight registry, not a copy
    of its accessors or readiness rules.  A failed descriptor is rendered in
    place and cannot suppress healthy sibling panels.
    """
    from polylogue.insights.archive import ArchiveInsightUnavailableError
    from polylogue.insights.registry import INSIGHT_REGISTRY, fetch_insights_async

    panels: list[dict[str, object]] = []
    active_registry: Mapping[str, Any] = INSIGHT_REGISTRY if registry is None else registry
    for name, descriptor in sorted(active_registry.items()):
        fields = tuple(descriptor.fields)
        query_kwargs: dict[str, object] = {}
        if descriptor.query_model is not None and "limit" in descriptor.query_model.model_fields:
            query_kwargs["limit"] = min(12, descriptor.mcp_default_limit)
        panel: dict[str, object] = {
            "name": name,
            "display_name": descriptor.display_name,
            "json_key": descriptor.json_key,
            "fields": [field.label for field in fields],
            "readiness": _insight_readiness(status, descriptor.readiness_exempt),
            "state": "available",
            "items": [],
            "error": None,
        }
        try:
            items = await fetch_insights_async(descriptor, operations, **query_kwargs)
        except ArchiveInsightUnavailableError as exc:
            panel["state"] = "unavailable"
            panel["error"] = str(exc)
        except Exception as exc:
            logger.warning("webui insight panel degraded: %s", name, exc_info=exc)
            panel["state"] = "degraded"
            panel["error"] = str(exc)
        else:
            try:
                rendered_items: list[dict[str, object]] = []
                for item in items:
                    plain_fields: list[dict[str, str]] = []
                    for field in fields:
                        try:
                            value = field.accessor(item)
                        except (AttributeError, KeyError, TypeError):
                            value = "-"
                        plain_fields.append({"label": field.label, "value": str(value)})
                    item_json = item.model_dump(mode="json")
                    provenance = item_json.get("provenance")
                    if provenance is None:
                        provenance = {
                            key: item_json[key]
                            for key in ("materializer_version", "materialized_at", "evidence_refs")
                            if key in item_json
                        }
                    rendered_items.append({"fields": plain_fields, "json": item_json, "provenance": provenance})
            except Exception as exc:
                logger.warning("webui insight projection degraded: %s", name, exc_info=exc)
                panel["state"] = "degraded"
                panel["error"] = str(exc)
            else:
                panel["items"] = rendered_items
                if not rendered_items:
                    panel["state"] = "empty"
        panels.append(panel)
    return {"contract_version": 1, "status": _status_panel_payload(status), "insights": panels}


def _insight_readiness(status: Mapping[str, object], exempt: bool) -> dict[str, object]:
    if exempt:
        return {"state": "not_required", "required": False, "reason": "descriptor is readiness-exempt"}
    components = status.get("component_readiness")
    component = (
        components.get("session_profiles", components.get("insight_freshness"))
        if isinstance(components, Mapping)
        else None
    )
    if isinstance(component, Mapping):
        return {
            "state": str(component.get("state") or "unknown"),
            "required": True,
            "reason": component.get("reason") or component.get("detail") or component.get("summary"),
        }
    return {"state": "unknown", "required": True, "reason": "daemon did not provide insight readiness"}


def _status_panel_payload(status: Mapping[str, object]) -> dict[str, object]:
    """Adapt today's readiness map to the StatusComponentSnapshot protocol."""
    snapshot = status.get("status_snapshot")
    snapshot_payload = dict(snapshot) if isinstance(snapshot, Mapping) else {"state": "unavailable"}
    supplied = status.get("status_components")
    if isinstance(supplied, list):
        return {"adapter": "status-component-snapshot", "snapshot": snapshot_payload, "components": supplied}
    legacy = status.get("component_readiness")
    components: list[dict[str, object]] = []
    if isinstance(legacy, Mapping):
        for name, value in sorted(legacy.items()):
            record = value if isinstance(value, Mapping) else {}
            legacy_state = str(record.get("state") or "unknown")
            state = {
                "ready": "fresh",
                "stale": "stale",
                "running": "refreshing",
                "pending": "refreshing",
                "timed_out": "timed_out",
                "blocked": "degraded",
                "degraded": "degraded",
            }.get(legacy_state, "unavailable")
            components.append(
                {
                    "name": str(name),
                    "state": state,
                    "detail": record.get("reason") or record.get("detail") or record.get("summary"),
                    "last_good": record.get("last_good"),
                    "age_s": snapshot_payload.get("age_s"),
                }
            )
    return {"adapter": "legacy-component-readiness", "snapshot": snapshot_payload, "components": components}


def render_observability_page(
    bundle: WebUIAssetBundle,
    payload: Mapping[str, object],
    *,
    notice: str | None = None,
) -> str:
    """Render observability semantics before the Preact island starts."""
    entry = bundle.entrypoint_for("observability")
    stylesheet_links = "\n".join(
        f'    <link rel="stylesheet" href="/app/assets/{html.escape(name, quote=True)}">' for name in entry.stylesheets
    )
    status = payload.get("status")
    status_payload = status if isinstance(status, Mapping) else {}
    components = status_payload.get("components")
    insights = payload.get("insights")
    component_rows = components if isinstance(components, list) else []
    insight_panels = insights if isinstance(insights, list) else []
    rendered_components = "\n".join(_render_status_component(component) for component in component_rows)
    rendered_insights = "\n".join(_render_insight_panel(panel) for panel in insight_panels)
    summary = (
        notice
        or "Status and insight evidence are projected by the daemon; unavailable and degraded states remain visible."
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>Observability · Polylogue</title>
{stylesheet_links}
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to observability</a>
    {_render_site_header("/app/observability")}
    <main id="main" class="page-shell">
      <p class="eyebrow">Daemon-projected evidence</p><h1>Archive observability</h1><p class="lede">{html.escape(summary)}</p>
      <div id="observability-island" data-island="observability">
        <section class="observability-panel" aria-labelledby="status-title"><h2 id="status-title">Component status</h2><ul class="status-grid">{rendered_components}</ul></section>
        <section class="observability-panel" aria-labelledby="freshness-title"><h2 id="freshness-title">Named-source freshness</h2><p>Inspect one exact source after web credentials are established. The ladder reports source, cursor, raw, parse, index, FTS, and insight evidence without an archive-wide scan.</p><form class="source-lookup"><label for="source-path">Exact source path</label><input id="source-path" name="source" type="text" autocomplete="off"><button type="submit">Inspect source</button></form><div data-source-freshness></div></section>
        <section class="observability-panel" aria-labelledby="insights-title"><h2 id="insights-title">Insights</h2><div class="insight-grid">{rendered_insights}</div></section>
      </div>
    </main>
    <script id="observability-bootstrap" type="application/json">{_json_script(payload)}</script>
    <script type="module" src="/app/assets/{html.escape(entry.script, quote=True)}"></script>
  </body>
</html>
"""


def _render_status_component(raw: object) -> str:
    component = raw if isinstance(raw, Mapping) else {}
    name = html.escape(str(component.get("name") or "unknown component"))
    state = html.escape(str(component.get("state") or "unavailable"))
    detail = component.get("detail")
    detail_markup = f"<p>{html.escape(str(detail))}</p>" if detail else ""
    return f'<li class="status-card" data-status-state="{state}"><h3>{name}</h3><p class="state-label">{state}</p>{detail_markup}</li>'


def _render_insight_panel(raw: object) -> str:
    panel = raw if isinstance(raw, Mapping) else {}
    title = html.escape(str(panel.get("display_name") or panel.get("name") or "Unnamed insight"))
    state = html.escape(str(panel.get("state") or "unavailable"))
    items = panel.get("items")
    rows = items if isinstance(items, list) else []
    error = panel.get("error")
    if error:
        body = f"<p>{html.escape(str(error))}</p>"
    elif not rows:
        body = "<p>No materialized rows are available for this bounded view.</p>"
    else:
        rendered_rows: list[str] = []
        for item in rows:
            item_record = item if isinstance(item, Mapping) else {}
            fields = item_record.get("fields")
            pairs = fields if isinstance(fields, list) else []
            values = " ".join(
                f"<span><strong>{html.escape(str(pair.get('label') or 'value'))}</strong> {html.escape(str(pair.get('value') or '-'))}</span>"
                for pair in pairs
                if isinstance(pair, Mapping)
            )
            rendered_rows.append(f"<li>{values}</li>")
        body = f"<ul>{''.join(rendered_rows)}</ul>"
    return f'<article class="insight-card" data-insight-state="{state}"><h3>{title}</h3><p class="state-label">{state}</p>{body}</article>'


def render_webui_asset_error(detail: str) -> str:
    """Return a dependency-free semantic error page when the build is absent."""

    return f"""<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>WebUI unavailable · Polylogue</title></head>
  <body><main><h1>WebUI unavailable</h1><p>{html.escape(detail)}</p><p><a href="/">Open the legacy reader</a></p></main></body>
</html>
"""


def _message_rows(page: QueryUnitEnvelope | None) -> tuple[MessageQueryRowPayload, ...]:
    if page is None:
        return ()
    rows: list[MessageQueryRowPayload] = []
    for item in page.items:
        if isinstance(item, MessageQueryRowPayload):
            rows.append(item)
    return tuple(rows)


def _render_message_row(row: MessageQueryRowPayload) -> str:
    session_id = str(row.session_id)
    title = row.title or session_id
    preview = _compact_preview(row.text)
    if not preview:
        preview = "[empty message]"
    occurred_at = _format_timestamp(row.occurred_at_ms)
    time_markup = (
        "<span>Time unavailable</span>"
        if occurred_at is None
        else f'<time datetime="{occurred_at[0]}">{occurred_at[1]}</time>'
    )
    return f"""          <li class="activity-row" data-message-id="{html.escape(str(row.message_id), quote=True)}">
            <div class="activity-row__meta"><span class="activity-row__origin">{html.escape(row.origin)}</span>{time_markup}</div>
            <h3><a href="/app/sessions/{quote(session_id, safe="")}#msg-{quote(str(row.message_id), safe="")}">{html.escape(title)}</a></h3>
            <p class="activity-row__preview">{html.escape(preview)}</p>
            <p class="activity-row__detail">{html.escape(row.role)} · {row.word_count:,} words</p>
          </li>"""


def _compact_preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _format_timestamp(value: int | None) -> tuple[str, str] | None:
    if value is None:
        return None
    timestamp = datetime.fromtimestamp(value / 1000, tz=UTC)
    return timestamp.isoformat(), timestamp.strftime("%b %d, %Y · %H:%M UTC")


def _json_script(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


__all__ = [
    "ARCHIVE_OVERVIEW_EXPRESSION",
    "ARCHIVE_OVERVIEW_LIMIT",
    "SESSION_LIST_LIMIT",
    "SESSION_READ_MESSAGE_LIMIT",
    "WebUIAsset",
    "WebUIAssetBundle",
    "WebUIAssetError",
    "WebUIEntrypoint",
    "build_observability_payload",
    "load_archive_overview_page",
    "render_archive_overview_page",
    "render_observability_page",
    "render_session_list_page",
    "render_session_read_page",
    "render_webui_asset_error",
]
