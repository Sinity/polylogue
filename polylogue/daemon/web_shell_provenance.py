"""Per-session provenance JS fragment for the web reader (#1125).

Kept as a sibling of :mod:`polylogue.daemon.web_shell` so the main
single-page shell stays under its file-size budget. The fragment is
interpolated into ``WEB_SHELL_HTML`` at module import time. It owns:

- the inspector "Raw" tab rendering for the provenance panel,
- the ``/api/sessions/{id}/provenance`` fetch call,
- the explicit opt-in click handler for the bounded raw payload preview.

The raw payload preview is server-bounded: the JS shows the operator
the server-declared cap and never tries to request a wider window.
"""

from __future__ import annotations

PROVENANCE_JS = r"""
function renderInspectorRaw(el, c) {
  // Per-session provenance panel (#1125). The metadata block is
  // always rendered. The raw payload preview is opt-in — the operator
  // must click "Load raw preview" before any bytes from the source
  // artifact are fetched. The preview is bounded server-side.
  var html = '<div class="inspector-section"><h4>Provenance</h4>';
  html += '<div class="inspector-field"><span class="label">Origin</span><span class="value">' + esc(c.origin || '-') + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Branch</span><span class="value">' + esc(c.branch_type || 'main') + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Parent</span><span class="value">' + esc(c.parent_id || '-') + '</span></div>';
  html += '<div id="provenance-area"><div class="inspector-empty" style="padding-top:8px">Loading provenance...</div></div>';
  html += '</div><div class="inspector-section"><h4>Raw Artifacts</h4>';
  html += '<button style="background:var(--panel-elevated);border:1px solid var(--border);color:var(--accent);padding:4px 10px;border-radius:3px;cursor:pointer;font-size:var(--small)" onclick="loadRawData()">Load artifact list</button>';
  html += '<div id="raw-data-area"></div></div>';
  el.innerHTML = html;
  loadProvenance(c.id);
}

async function loadProvenance(id) {
  var area = document.getElementById('provenance-area');
  if (!area) return;
  try {
    var data = await fetchJSON('/api/sessions/' + encodeURIComponent(id) + '/provenance');
    state.provenance = data;
    renderProvenancePanel(data);
  } catch(e) {
    area.innerHTML = '<div class="inspector-empty" style="padding-top:8px">Provenance unavailable</div>';
  }
}

function renderProvenancePanel(data) {
  var area = document.getElementById('provenance-area');
  if (!area || !data) return;
  var html = '';
  var quarantineClass = data.quarantined ? 'q-unresolved' : 'q-canonical';
  var quarantineLabel = data.quarantined ? ('quarantined: ' + (data.quarantine_reason || 'unknown')) : 'ok';
  html += '<div style="margin-top:6px"><span class="chip ' + quarantineClass + '">' + esc(quarantineLabel) + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Content hash</span><span class="value">' + esc((data.content_hash || '-').substring(0, 16) + (data.content_hash && data.content_hash.length > 16 ? '\u2026' : '')) + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Raw id</span><span class="value">' + esc((data.raw_id || '-').substring(0, 16) + (data.raw_id && data.raw_id.length > 16 ? '\u2026' : '')) + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Source</span><span class="value">' + esc(data.source_path_display || data.source_name || '-') + '</span></div>';
  if (data.source_path_contains_home) {
    html += '<div class="inspector-field"><span class="label">Path</span><span class="value">home prefix sanitized</span></div>';
  }
  html += '<div class="inspector-field"><span class="label">Acquired</span><span class="value">' + esc(data.acquired_at ? new Date(data.acquired_at).toLocaleString() : '-') + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Parsed</span><span class="value">' + esc(data.parsed_at ? new Date(data.parsed_at).toLocaleString() : '-') + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Validation</span><span class="value">' + esc(data.validation_status || '-') + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Blob size</span><span class="value">' + (data.blob_size_bytes != null ? data.blob_size_bytes.toLocaleString() + ' B' : '-') + '</span></div>';
  if (data.parse_error) {
    html += '<div class="inspector-field"><span class="label">Parse error</span><span class="value">' + esc(String(data.parse_error)) + '</span></div>';
  }
  if (data.validation_error) {
    html += '<div class="inspector-field"><span class="label">Validation error</span><span class="value">' + esc(String(data.validation_error)) + '</span></div>';
  }
  // Raw preview is opt-in. The button reveals it only on explicit click,
  // and the server enforces the size cap regardless.
  html += '<div style="margin-top:8px">'
    + '<button style="background:var(--panel-elevated);border:1px solid var(--border);color:var(--accent);padding:4px 10px;border-radius:3px;cursor:pointer;font-size:var(--small)" onclick="loadProvenanceRaw()">Load raw preview ('
    + (data.raw_preview_cap_bytes != null ? data.raw_preview_cap_bytes.toLocaleString() + ' B max' : 'bounded')
    + ')</button>'
    + '<div id="provenance-preview-area"></div></div>';
  area.innerHTML = html;
}

async function loadProvenanceRaw() {
  if (!state.selected) return;
  var id = state.selected.id;
  var area = document.getElementById('provenance-preview-area');
  if (!area) return;
  area.innerHTML = '<div style="color:var(--text-dim);font-size:var(--small);padding:8px 0">Loading raw preview...</div>';
  try {
    var data = await fetchJSON('/api/sessions/' + encodeURIComponent(id) + '/provenance?include_raw=1');
    var preview = data.raw_preview || {};
    if (!preview.available) {
      area.innerHTML = '<div class="inspector-empty" style="padding-top:8px">Raw preview unavailable: ' + esc(preview.reason || 'unknown') + '</div>';
      return;
    }
    var truncatedHint = preview.truncated ? ' (truncated, ' + preview.bytes_returned.toLocaleString() + ' of ' + (preview.total_size_bytes != null ? preview.total_size_bytes.toLocaleString() : '?') + ' B)' : ' (' + preview.bytes_returned.toLocaleString() + ' B)';
    var body = preview.encoding === 'utf-8' ? (preview.text || '') : (preview.base64 || '');
    var head = '<div style="font-size:10px;color:var(--text-dim);margin-top:6px">encoding=' + esc(preview.encoding || 'binary') + truncatedHint + '</div>';
    area.innerHTML = head + '<div class="raw-block">' + esc(body) + '</div>';
  } catch(e) {
    area.innerHTML = '<div class="inspector-empty" style="padding-top:8px">Failed to load raw preview</div>';
  }
}
"""

__all__ = ["PROVENANCE_JS"]
