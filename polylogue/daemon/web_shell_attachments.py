"""MK3 reader attachment product surface for the daemon web shell (#1199).

Owns:

- attachment-card CSS for inline rendering in message bodies, the
  inspector tab listing, and the standalone library page;
- attachment-aware JS helpers used by ``web_shell_reader.py``
  (``_polyRenderAttachmentCards``, ``_polyAttachmentInspectorHtml``);
- the ``/a`` attachment-library page HTML and bootstrap JS, which
  fetches ``/api/attachments`` and renders a paginated, filterable list
  of all attachments across the archive;
- envelope helpers (``attachment_to_envelope``, ``classify_state``,
  ``build_library_payload``) consumed by ``daemon/http.py``.

The substrate exposes ``attachments`` per message via the
``session_from_records`` hydrator. This slice surfaces typed attachment
fields as an envelope plus an MK3 state token derived purely from substrate
fields.

State derivation (from ``docs/design/mk3/docs/08-state-matrix.md:89``):

- ``available`` — has a blob path and is within the size budget
- ``missing-blob`` — no on-disk path (substrate did not materialize
  the blob, or it was pruned)
- ``unsupported-kind`` — mime type is on the explicit unsupported list
  (executables, archives we won't render inline)
- ``too-large`` — size exceeds the preview budget; preview suppressed
  but metadata still visible
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Preview budget for inline attachment cards. Anything larger renders
# as ``too-large`` (metadata still shown, body suppressed). Tuned to
# match the message-card layout — 8 MiB is what fits in a viewport
# without forcing the inspector to scroll past the message body.
PREVIEW_SIZE_BUDGET = 8 * 1024 * 1024

# Mime types we explicitly will not render inline. Everything else
# falls through to ``available`` (renderers can still gate on mime
# server-side later without a state change).
UNSUPPORTED_MIME_PREFIXES = (
    "application/x-executable",
    "application/x-msdownload",
    "application/x-msdos-program",
    "application/x-sharedlib",
)
UNSUPPORTED_MIME_EXACT = frozenset(
    {
        "application/x-tar",
        "application/zip",
        "application/x-7z-compressed",
        "application/x-rar-compressed",
    }
)


def _attachment_name(attachment: Any) -> str:
    """Best-effort display name for an attachment envelope/record.

    Looks at ``name`` (domain model), then falls back to the attachment id.
    """

    if attachment is None:
        return ""
    name = getattr(attachment, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    aid = getattr(attachment, "id", None) or getattr(attachment, "attachment_id", None)
    return str(aid or "")


def classify_state(
    *,
    path: str | None,
    size_bytes: int | None,
    mime_type: str | None,
) -> str:
    """Return the MK3 attachment state token for an attachment.

    Missing-blob is observable independently of mime/size; mime check
    precedes size so an unsupported binary doesn't masquerade as a
    too-large preview.
    """

    if not path:
        return "missing-blob"
    if isinstance(mime_type, str):
        mime_lower = mime_type.lower()
        if mime_lower in UNSUPPORTED_MIME_EXACT:
            return "unsupported-kind"
        if any(mime_lower.startswith(prefix) for prefix in UNSUPPORTED_MIME_PREFIXES):
            return "unsupported-kind"
    if isinstance(size_bytes, int) and size_bytes > PREVIEW_SIZE_BUDGET:
        return "too-large"
    return "available"


def attachment_to_envelope(
    attachment: Any,
    *,
    session_id: str,
    message_id: str | None = None,
) -> dict[str, object]:
    """Build the per-attachment envelope shape returned by the daemon.

    The envelope intentionally does not include the blob bytes — the
    inline card renders metadata + state only, and the operator opens
    the source artifact through the existing ``/api/raw_artifacts/``
    surface (or, in a later slice, a dedicated blob fetch route).
    """

    aid = str(getattr(attachment, "id", None) or getattr(attachment, "attachment_id", "") or "")
    mime_type = getattr(attachment, "mime_type", None)
    size_bytes = getattr(attachment, "size_bytes", None)
    path = getattr(attachment, "path", None)
    state = classify_state(
        path=path,
        size_bytes=size_bytes,
        mime_type=mime_type,
    )
    return {
        "attachment_id": aid,
        "session_id": session_id,
        "message_id": str(message_id) if message_id is not None else None,
        "name": _attachment_name(attachment),
        "mime_type": mime_type if isinstance(mime_type, str) else None,
        "size_bytes": int(size_bytes) if isinstance(size_bytes, int) else None,
        "path": path if isinstance(path, str) else None,
        "state": state,
    }


@dataclass(frozen=True)
class LibraryEntry:
    """One row in the attachment-library listing.

    Carries the envelope plus the session context needed for the
    anchor link back to the source message in the reader.
    """

    envelope: dict[str, object]
    session_title: str
    origin: str | None
    message_anchor: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            **self.envelope,
            "session_title": self.session_title,
            "origin": self.origin,
            "message_anchor": self.message_anchor,
        }


def build_library_payload(
    entries: Iterable[LibraryEntry],
    *,
    total: int,
) -> dict[str, object]:
    """Wrap a sequence of library entries in the envelope shape consumed
    by the ``/a`` page bootstrap. Grouping by session happens
    client-side so the server contract stays a flat list."""

    items = [entry.to_dict() for entry in entries]
    return {"items": items, "total": total}


# ---------------------------------------------------------------------------
# Client-side: attachment CSS, JS helpers, library page
# ---------------------------------------------------------------------------

ATTACHMENT_CSS = r"""
.msg-attachments {
  display: flex; flex-wrap: wrap; gap: 6px;
  margin: 6px 0 4px 0;
}
.msg-attachment {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 8px;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--panel-subtle);
  color: var(--text); font-family: var(--font-mono); font-size: 11px;
  max-width: 100%;
}
.msg-attachment .att-name {
  color: var(--accent); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; max-width: 280px;
}
.msg-attachment .att-meta { color: var(--text-dim); font-size: 10px; }
.msg-attachment .att-state {
  text-transform: lowercase; font-size: 10px;
  padding: 0 4px; border-radius: 2px;
  border: 1px solid var(--border);
  color: var(--text-muted);
}
.msg-attachment.state-available .att-state { color: var(--ok); border-color: var(--ok); }
.msg-attachment.state-missing-blob .att-state { color: var(--err); border-color: var(--err); }
.msg-attachment.state-unsupported-kind .att-state,
.msg-attachment.state-too-large .att-state { color: var(--warn); border-color: var(--warn); }
.msg-attachment.state-missing-blob { opacity: 0.7; }

.att-inspector-list {
  display: flex; flex-direction: column; gap: 4px;
}
.att-inspector-row {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 4px 6px;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--panel-subtle);
}
.att-inspector-row .att-name {
  color: var(--accent); font-family: var(--font-mono); font-size: 11px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  flex: 1; min-width: 0;
}
.att-inspector-row .att-meta { color: var(--text-dim); font-size: 10px; }
"""

# JS injected into the reader bootstrap. Exports:
#
#   - ``_polyFormatBytes(size)`` — humanize a byte count for cards.
#   - ``_polyAttachmentCardHtml(att)`` — render a single inline card.
#   - ``_polyRenderAttachmentCards(m)`` — render the attachment strip
#     above the message body (called from ``renderMessageBlocks`` via
#     the optional ``_polyAttachmentStripHtml`` hook).
#   - ``_polyAttachmentInspectorHtml(c)`` — render the Attachments
#     inspector tab body for a selected session.
#   - ``initAttachmentLibrary()`` — bootstrap for the ``/a`` route.
ATTACHMENT_JS = r"""
// MK3 reader attachment product surface (#1199).

var _POLY_ATTACHMENT_STATE_LABEL = {
  'available': 'ok',
  'missing-blob': 'missing',
  'unsupported-kind': 'unsupported',
  'too-large': 'too large'
};

function _polyFormatBytes(size) {
  if (size == null || isNaN(size)) return '';
  if (size < 1024) return size + ' B';
  if (size < 1024 * 1024) return (size / 1024).toFixed(1) + ' KiB';
  if (size < 1024 * 1024 * 1024) return (size / (1024 * 1024)).toFixed(1) + ' MiB';
  return (size / (1024 * 1024 * 1024)).toFixed(2) + ' GiB';
}

function _polyAttachmentStateLabel(state) {
  return _POLY_ATTACHMENT_STATE_LABEL[state] || (state || 'unknown');
}

function _polyAttachmentCardHtml(att) {
  if (!att) return '';
  var state = att.state || 'unknown';
  var name = att.name || att.attachment_id || 'attachment';
  var meta = '';
  if (att.mime_type) meta += esc(att.mime_type);
  var sizeStr = _polyFormatBytes(att.size_bytes);
  if (sizeStr) meta += (meta ? ' \u00B7 ' : '') + sizeStr;
  return ''
    + '<span class="msg-attachment state-' + esc(state) + '" '
    +   'data-attachment-id="' + escAttr(att.attachment_id || '') + '" '
    +   'title="' + escAttr(name + ' (' + state + ')') + '">'
    +   '<span class="att-name">' + esc(name) + '</span>'
    +   (meta ? '<span class="att-meta">' + meta + '</span>' : '')
    +   '<span class="att-state">' + esc(_polyAttachmentStateLabel(state)) + '</span>'
    + '</span>';
}

function _polyRenderAttachmentCards(m) {
  // Returns the attachment strip HTML for one message, or '' if the
  // message has no attachments. The renderer calls this from
  // renderMessageBlocks via the ``_polyAttachmentStripHtml`` hook so
  // the strip lands above the message body, regardless of paste /
  // tool / thinking fold variant.
  if (!m || !Array.isArray(m.attachments) || !m.attachments.length) return '';
  var cards = m.attachments.map(_polyAttachmentCardHtml).join('');
  return '<div class="msg-attachments">' + cards + '</div>';
}

// Exposed under the public hook name renderMessageBlocks looks up.
function _polyAttachmentStripHtml(m) {
  return _polyRenderAttachmentCards(m);
}

function _polyAttachmentInspectorHtml(c) {
  if (!c || !Array.isArray(c.attachments)) return '';
  var atts = c.attachments;
  if (!atts.length) {
    return '<div class="inspector-empty">No attachments in this session.</div>';
  }
  var rows = atts.map(function(att) {
    var name = att.name || att.attachment_id || 'attachment';
    var sizeStr = _polyFormatBytes(att.size_bytes);
    var pieces = [];
    if (att.mime_type) pieces.push(att.mime_type);
    if (sizeStr) pieces.push(sizeStr);
    pieces.push(_polyAttachmentStateLabel(att.state || 'unknown'));
    var meta = pieces.join(' \u00B7 ');
    var anchorHref = '';
    if (att.message_id) {
      anchorHref = '#message-' + encodeURIComponent(att.message_id);
    }
    var nameHtml = anchorHref
      ? '<a class="att-name" href="' + anchorHref + '" '
        + 'onclick="event.preventDefault();jumpToAnchor(\'message-' + escAttr(String(att.message_id)) + '\')">'
        + esc(name) + '</a>'
      : '<span class="att-name">' + esc(name) + '</span>';
    return ''
      + '<div class="att-inspector-row state-' + esc(att.state || 'unknown') + '" '
      +      'data-attachment-id="' + escAttr(att.attachment_id || '') + '">'
      +   nameHtml
      +   '<span class="att-meta">' + esc(meta) + '</span>'
      + '</div>';
  }).join('');
  return '<div class="att-inspector-list">' + rows + '</div>';
}

function renderInspectorAttachments(el, c) {
  el.innerHTML = _polyAttachmentInspectorHtml(c);
}

// ---------------------------------------------------------------------
// Attachment library bootstrap (served as a separate page at /a).
// ---------------------------------------------------------------------

var _POLY_ATTACHMENT_LIBRARY_STATE = {
  items: [],
  filterMime: '',
  filterState: '',
  filterSession: ''
};

function _polyAttachmentLibraryFilters() {
  var s = _POLY_ATTACHMENT_LIBRARY_STATE;
  return s.items.filter(function(it) {
    if (s.filterMime && (it.mime_type || '').indexOf(s.filterMime) === -1) return false;
    if (s.filterState && it.state !== s.filterState) return false;
    if (s.filterSession && it.session_id !== s.filterSession) return false;
    return true;
  });
}

function _polyAttachmentLibraryRender() {
  var listEl = document.getElementById('att-list');
  var emptyEl = document.getElementById('att-empty');
  if (!listEl || !emptyEl) return;
  var items = _polyAttachmentLibraryFilters();
  if (!items.length) {
    listEl.innerHTML = '';
    emptyEl.style.display = 'block';
    emptyEl.textContent = _POLY_ATTACHMENT_LIBRARY_STATE.items.length
      ? 'No attachments match the current filters.'
      : 'No attachments in the archive.';
    return;
  }
  emptyEl.style.display = 'none';
  // Group by session.
  var groups = {};
  var order = [];
  items.forEach(function(item) {
    var cid = item.session_id;
    if (!groups[cid]) {
      groups[cid] = {title: item.session_title, origin: item.origin, items: []};
      order.push(cid);
    }
    groups[cid].items.push(item);
  });
  var html = order.map(function(cid) {
    var g = groups[cid];
    var rows = g.items.map(function(it) {
      var href = '/c/' + encodeURIComponent(cid);
      if (it.message_anchor) href += '#' + encodeURIComponent(it.message_anchor);
      var sizeStr = _polyFormatBytes(it.size_bytes);
      var meta = [];
      if (it.mime_type) meta.push(it.mime_type);
      if (sizeStr) meta.push(sizeStr);
      var name = (it.name || it.attachment_id || 'attachment')
        .replace(/[<&]/g, function(c) { return c === '<' ? '&lt;' : '&amp;'; });
      return ''
        + '<a class="att-row state-' + (it.state || 'unknown') + '" href="' + href + '">'
        +   '<span class="att-name">' + name + '</span>'
        +   '<span class="att-meta">' + meta.join(' \u00B7 ') + '</span>'
        +   '<span class="att-state">' + _polyAttachmentStateLabel(it.state || 'unknown') + '</span>'
        + '</a>';
    }).join('');
    return ''
      + '<div class="att-group">'
      +   '<div class="att-group-title">'
      +     '<a href="/c/' + encodeURIComponent(cid) + '">'
      +     (g.title || cid).replace(/[<&]/g, function(c) { return c === '<' ? '&lt;' : '&amp;'; })
      +     '</a>'
      +     '<span class="att-origin">' + (g.origin || '') + '</span>'
      +   '</div>'
      +   rows
      + '</div>';
  }).join('');
  listEl.innerHTML = html;
}

function _polyAttachmentLibraryWireFilters() {
  var mime = document.getElementById('att-filter-mime');
  var state = document.getElementById('att-filter-state');
  var conv = document.getElementById('att-filter-session');
  if (mime) mime.addEventListener('input', function(e) {
    _POLY_ATTACHMENT_LIBRARY_STATE.filterMime = e.target.value.trim();
    _polyAttachmentLibraryRender();
  });
  if (state) state.addEventListener('change', function(e) {
    _POLY_ATTACHMENT_LIBRARY_STATE.filterState = e.target.value;
    _polyAttachmentLibraryRender();
  });
  if (conv) conv.addEventListener('input', function(e) {
    _POLY_ATTACHMENT_LIBRARY_STATE.filterSession = e.target.value.trim();
    _polyAttachmentLibraryRender();
  });
}

function initAttachmentLibrary() {
  var listEl = document.getElementById('att-list');
  var emptyEl = document.getElementById('att-empty');
  if (!listEl || !emptyEl) return;
  emptyEl.style.display = 'block';
  emptyEl.textContent = 'Loading attachments\u2026';
  _polyAttachmentLibraryWireFilters();
  fetch('/api/attachments?limit=500').then(function(r) {
    return r.json();
  }).then(function(payload) {
    _POLY_ATTACHMENT_LIBRARY_STATE.items = (payload && payload.items) || [];
    _polyAttachmentLibraryRender();
  }).catch(function(err) {
    emptyEl.textContent = 'Failed to load attachment library: ' + (err && err.message ? err.message : err);
  });
}
"""

ATTACHMENT_LIBRARY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polylogue \u00B7 Attachment Library</title>
<style>
:root {
  --bg: #15171a; --panel: #1d2025; --panel-subtle: #23262c;
  --border: #2c2f36; --text: #d6d9de; --text-muted: #8b919b;
  --text-dim: #5a5f68; --accent: #5fb3b3; --ok: #6ec27a;
  --warn: #d3a04a; --err: #d05c5c; --radius: 4px;
  --font-mono: ui-monospace,Menlo,Consolas,monospace;
  --font-ui: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--font-ui); font-size: 13px; }
header { padding: 12px 24px; border-bottom: 1px solid var(--border);
  display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
header h1 { margin: 0; font-size: 16px; font-weight: 600; }
header a { color: var(--accent); text-decoration: none; }
header a:hover { text-decoration: underline; }
#att-filters { display: flex; gap: 10px; align-items: center; padding: 8px 24px;
  border-bottom: 1px solid var(--border); background: var(--panel); flex-wrap: wrap; }
#att-filters label { color: var(--text-muted); font-size: 11px; font-family: var(--font-mono); }
#att-filters input, #att-filters select {
  background: var(--panel-subtle); color: var(--text);
  border: 1px solid var(--border); border-radius: 3px;
  padding: 3px 6px; font-family: var(--font-mono); font-size: 12px;
}
main { padding: 16px 24px; max-width: 1100px; }
#att-empty { color: var(--text-muted); padding: 24px 0; }
.att-group { margin-bottom: 24px; }
.att-group-title { display: flex; align-items: baseline; gap: 10px;
  padding: 6px 8px; border-bottom: 1px solid var(--border); font-weight: 600; }
.att-group-title a { color: var(--text); text-decoration: none; }
.att-group-title a:hover { color: var(--accent); }
.att-origin { color: var(--text-dim); font-weight: 400; font-size: 11px; font-family: var(--font-mono); }
.att-row { display: grid; grid-template-columns: 1fr 1fr 100px;
  gap: 12px; padding: 6px 8px; align-items: center;
  border-bottom: 1px solid var(--panel-subtle);
  color: var(--text); text-decoration: none; }
.att-row:hover { background: var(--panel-subtle); }
.att-row .att-name { color: var(--accent); font-family: var(--font-mono); font-size: 12px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.att-row .att-meta { color: var(--text-dim); font-size: 11px; font-family: var(--font-mono); }
.att-row .att-state { text-align: right; font-size: 10px;
  padding: 2px 6px; border-radius: 2px; border: 1px solid var(--border);
  color: var(--text-muted); justify-self: end; }
.att-row.state-available .att-state { color: var(--ok); border-color: var(--ok); }
.att-row.state-missing-blob .att-state { color: var(--err); border-color: var(--err); }
.att-row.state-unsupported-kind .att-state,
.att-row.state-too-large .att-state { color: var(--warn); border-color: var(--warn); }
.att-row.state-missing-blob { opacity: 0.7; }
</style>
</head>
<body>
<header>
  <h1>Attachment library</h1>
  <a href="/">\u2190 Back to reader</a>
</header>
<div id="att-filters">
  <label for="att-filter-mime">mime</label>
  <input id="att-filter-mime" type="text" placeholder="image/, application/pdf, ...">
  <label for="att-filter-state">state</label>
  <select id="att-filter-state">
    <option value="">(any)</option>
    <option value="available">available</option>
    <option value="missing-blob">missing-blob</option>
    <option value="unsupported-kind">unsupported-kind</option>
    <option value="too-large">too-large</option>
  </select>
  <label for="att-filter-session">session</label>
  <input id="att-filter-session" type="text" placeholder="session id">
</div>
<main>
  <div id="att-empty"></div>
  <div id="att-list"></div>
</main>
<script>
function esc(s){return (s==null?'':String(s)).replace(/[&<>"']/g,function(c){return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c];});}
function escAttr(s){return esc(s);}
__ATTACHMENT_LIBRARY_JS__
initAttachmentLibrary();
</script>
</body>
</html>
"""


def render_attachment_library_page() -> str:
    """Return the standalone HTML served at ``/a``."""

    return ATTACHMENT_LIBRARY_HTML.replace("__ATTACHMENT_LIBRARY_JS__", ATTACHMENT_JS)


__all__ = [
    "ATTACHMENT_CSS",
    "ATTACHMENT_JS",
    "LibraryEntry",
    "PREVIEW_SIZE_BUDGET",
    "UNSUPPORTED_MIME_EXACT",
    "UNSUPPORTED_MIME_PREFIXES",
    "attachment_to_envelope",
    "build_library_payload",
    "classify_state",
    "render_attachment_library_page",
]
