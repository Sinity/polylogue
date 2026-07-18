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
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from polylogue.archive.query.execution_control import classify_unit_expression_workload
from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest
from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request
from polylogue.surfaces.payloads import MessageQueryRowPayload, QueryUnitEnvelope

ARCHIVE_OVERVIEW_EXPRESSION = "messages where words >= 0 | sort by time desc"
ARCHIVE_OVERVIEW_LIMIT = 6
_WEBUI_ENTRY_NAME = "archive-overview"
_HASHED_ASSET_RE = re.compile(r"^[A-Za-z0-9._-]+-[A-Za-z0-9_-]{8,}\.(?:css|js)$")


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
        entry: dict[str, object] | None = None
        for raw in self._manifest.values():
            if not isinstance(raw, dict):
                continue
            if raw.get("isEntry") is True and raw.get("name") == _WEBUI_ENTRY_NAME:
                entry = raw
                break
        if entry is None:
            raise WebUIAssetError(f"Vite manifest has no {_WEBUI_ENTRY_NAME!r} entry")
        script = self._manifest_asset_name(entry.get("file"))
        raw_css = entry.get("css", [])
        if not isinstance(raw_css, list):
            raise WebUIAssetError("Vite entry css field must be a list")
        stylesheets = tuple(self._manifest_asset_name(value) for value in raw_css)
        return WebUIEntrypoint(script=script, stylesheets=stylesheets)

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
    "load_archive_overview_page",
    "render_archive_overview_page",
    "render_webui_asset_error",
]
