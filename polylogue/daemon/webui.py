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
    <header class="site-header">
      <p class="eyebrow">Polylogue</p>
      <nav aria-label="Reader versions">
        <a href="/app" aria-current="page">Archive overview</a>
        <a href="/">Legacy reader</a>
      </nav>
    </header>
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
    <header class="site-header"><p class="eyebrow">Polylogue</p><nav aria-label="WebUI views"><a href="/app">Archive overview</a><a href="/app/observability" aria-current="page">Observability</a><a href="/">Legacy reader</a></nav></header>
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
            <h3><a href="/s/{quote(session_id, safe="")}">{html.escape(title)}</a></h3>
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
    "WebUIAsset",
    "WebUIAssetBundle",
    "WebUIAssetError",
    "WebUIEntrypoint",
    "build_observability_payload",
    "load_archive_overview_page",
    "render_archive_overview_page",
    "render_observability_page",
    "render_webui_asset_error",
]
