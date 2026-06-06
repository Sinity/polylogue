"""Bulk-operations assets for the daemon-served reader shell (#1119).

The bulk surface is composed client-side over existing daemon endpoints:
``/api/user/marks`` for tag mutations and ``/api/sessions/{id}`` for
export. Delete and re-embed are exposed only through a preview overlay —
the daemon currently has no DELETE-session or re-embed mutation
route, so confirming the preview records ``no_endpoint`` skip entries in
the per-session status envelope rather than appearing to mutate
state. This keeps the AC contract (preview gate + typed envelope) honest
while the substrate routes catch up.

Kept as a standalone module so ``web_shell.py`` stays under the
file-size budget defined in ``docs/plans/file-size-budgets.yaml``; the
same split pattern is used for the workspace mode in
``web_shell_workspace.py``.
"""

from __future__ import annotations

BULK_CSS = r"""
.conv-item .conv-row { display: flex; align-items: flex-start; gap: 6px; }
.conv-item .conv-row .conv-body { flex: 1; min-width: 0; }
.conv-item .bulk-check { margin-top: 2px; flex-shrink: 0; cursor: pointer; accent-color: var(--accent); }
.conv-item.bulk-selected { background: var(--accent-bg); }

/* Bulk operations toolbar — appears in the sidebar when one or more
   sessions are checkbox-selected. Drives bulk tag (marks),
   markdown export (client-side), and preview-only delete/re-embed
   per #1119. The toolbar is intentionally inert until selection > 0. */
#bulk-toolbar { display: none; flex-direction: column; gap: 6px; padding: 8px 10px;
  border-bottom: 1px solid var(--border); background: var(--panel-elevated); }
#bulk-toolbar.visible { display: flex; }
#bulk-toolbar .bulk-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
#bulk-toolbar .bulk-count { color: var(--accent); font-size: var(--small); font-weight: 600; }
#bulk-toolbar .bulk-link { color: var(--text-muted); font-size: var(--small); cursor: pointer;
  background: none; border: none; padding: 0; text-decoration: underline; }
#bulk-toolbar .bulk-link:hover { color: var(--text); }
#bulk-toolbar .bulk-btn { background: var(--panel-subtle); border: 1px solid var(--border);
  color: var(--text); padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: var(--small); font-family: var(--font-ui); }
#bulk-toolbar .bulk-btn:hover { border-color: var(--accent-soft); color: var(--accent); }
#bulk-toolbar .bulk-btn.danger { color: var(--err); }
#bulk-toolbar .bulk-btn.danger:hover { border-color: var(--err); background: var(--err-bg); }
#bulk-status { font-size: var(--small); color: var(--text-muted); line-height: 1.4; }
#bulk-status .ok { color: var(--ok); }
#bulk-status .err { color: var(--err); }
#bulk-status .skip { color: var(--warn); }
#bulk-preview { display: none; position: fixed; inset: 0; background: rgba(7,11,16,0.85); z-index: 110;
  align-items: center; justify-content: center; }
#bulk-preview.visible { display: flex; }
#bulk-preview-panel { background: var(--panel-elevated); border: 1px solid var(--border-strong); border-radius: 8px;
  padding: 20px; max-width: 520px; width: 90%; max-height: 80vh; overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
#bulk-preview-panel h3 { font-size: 15px; margin-bottom: 10px; color: var(--accent); }
#bulk-preview-panel p { font-size: var(--small); color: var(--text-muted); margin-bottom: 10px; line-height: 1.5; }
#bulk-preview-panel ul { list-style: none; margin: 8px 0; padding: 0; max-height: 240px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: 3px; }
#bulk-preview-panel ul li { padding: 4px 8px; font-size: var(--small); border-bottom: 1px solid var(--border);
  color: var(--text); font-family: var(--font-mono); }
#bulk-preview-panel .actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 12px; }
"""

BULK_TOOLBAR_HTML = r"""
    <div id="bulk-toolbar" aria-label="Bulk operations toolbar">
      <div class="bulk-row">
        <span class="bulk-count" id="bulk-count">0 selected</span>
        <span style="flex:1"></span>
        <button class="bulk-link" id="bulk-select-all" type="button">Select all</button>
        <button class="bulk-link" id="bulk-clear" type="button">Clear</button>
      </div>
      <div class="bulk-row">
        <button class="bulk-btn" data-bulk-action="tag-star" type="button" title="Star all selected">Star</button>
        <button class="bulk-btn" data-bulk-action="tag-pin" type="button" title="Pin all selected">Pin</button>
        <button class="bulk-btn" data-bulk-action="tag-archive" type="button" title="Archive all selected">Archive</button>
        <button class="bulk-btn" data-bulk-action="export" type="button" title="Export selected as JSON">Export</button>
        <button class="bulk-btn danger" data-bulk-action="delete-preview" type="button" title="Preview delete (dry run)">Delete...</button>
        <button class="bulk-btn" data-bulk-action="reembed-preview" type="button" title="Preview re-embed (dry run)">Re-embed...</button>
      </div>
      <div id="bulk-status" aria-live="polite"></div>
    </div>
"""

BULK_PREVIEW_HTML = r"""
<div id="bulk-preview" role="dialog" aria-label="Bulk operation preview">
  <div id="bulk-preview-panel">
    <h3 id="bulk-preview-title">Preview</h3>
    <p id="bulk-preview-body">Confirm the destructive operation below.</p>
    <ul id="bulk-preview-list"></ul>
    <div class="actions">
      <button class="bulk-btn" type="button" onclick="closeBulkPreview()">Cancel</button>
      <button class="bulk-btn danger" id="bulk-preview-confirm" type="button">Confirm</button>
    </div>
  </div>
</div>
"""

BULK_JS = r"""
// --- Bulk selection (#1119) ------------------------------------------------
// The bulk surface is purely client-side composition over existing daemon
// endpoints. Tag operations issue per-session POSTs to /api/user/marks
// (the centralized tag mutation contract). Export streams session
// detail through GET /api/sessions/{id} and downloads a single JSON
// bundle. Delete and re-embed are exposed only through a preview overlay —
// the daemon currently has no DELETE-session or re-embed route, so the
// confirm button records a per-session "skipped: no_endpoint" entry
// rather than silently appearing to mutate state. This keeps the AC
// contract (preview gate + typed envelope) honest.

function isBulkSelected(id) { return !!state.bulkSelection[id]; }
function bulkSelectedIds() { return Object.keys(state.bulkSelection); }
function bulkSelectedCount() { return bulkSelectedIds().length; }

function setBulkSelected(id, enabled) {
  if (enabled) state.bulkSelection[id] = true;
  else delete state.bulkSelection[id];
}

function toggleBulkSelected(id) { setBulkSelected(id, !isBulkSelected(id)); }

function clearBulkSelection() {
  state.bulkSelection = {};
  state.lastBulkResult = null;
  renderSessions();
}

function selectAllVisible() {
  (state.sessions || []).forEach(function(c) { state.bulkSelection[c.id] = true; });
  renderSessions();
}

function renderBulkToolbar() {
  var bar = document.getElementById('bulk-toolbar');
  if (!bar) return;
  var count = bulkSelectedCount();
  var countEl = document.getElementById('bulk-count');
  if (countEl) countEl.textContent = count + ' selected';
  if (count > 0) bar.classList.add('visible'); else bar.classList.remove('visible');
  renderBulkStatus();
}

function renderBulkStatus() {
  var el = document.getElementById('bulk-status');
  if (!el) return;
  var result = state.lastBulkResult;
  if (!result) { el.innerHTML = ''; return; }
  var parts = [];
  parts.push('<span class="ok">' + result.succeeded.length + ' ok</span>');
  parts.push('<span class="err">' + result.failed.length + ' failed</span>');
  parts.push('<span class="skip">' + result.skipped.length + ' skipped</span>');
  var prefix = result.dryRun ? 'dry-run ' : '';
  var html = prefix + esc(result.action) + ': ' + parts.join(' \u00b7 ');
  if (result.failed.length || result.skipped.length) {
    var details = result.failed.concat(result.skipped).slice(0, 3).map(function(r) {
      return esc(r.id) + ' (' + esc(r.reason) + ')';
    });
    html += '<br><span class="err" style="font-family:var(--font-mono);font-size:10px">' + details.join(', ') + '</span>';
  }
  el.innerHTML = html;
}

async function bulkApplyMark(markType) {
  var ids = bulkSelectedIds();
  if (!ids.length) return;
  var envelope = {action: 'tag-' + markType, dryRun: false, succeeded: [], failed: [], skipped: []};
  for (var i = 0; i < ids.length; i++) {
    var id = ids[i];
    if (hasMark(id, markType)) {
      envelope.skipped.push({id: id, reason: 'already_tagged'});
      continue;
    }
    try {
      await sendJSON('/api/user/marks', 'POST', {session_id: id, mark_type: markType});
      setMarkLocal(id, markType, true);
      envelope.succeeded.push(id);
    } catch(e) {
      envelope.failed.push({id: id, reason: 'request_failed'});
    }
  }
  state.lastBulkResult = envelope;
  renderSessions();
  renderInspector();
}

async function bulkExport() {
  var ids = bulkSelectedIds();
  if (!ids.length) return;
  var envelope = {action: 'export', dryRun: false, succeeded: [], failed: [], skipped: []};
  var bundle = {exported_at: new Date().toISOString(), sessions: []};
  for (var i = 0; i < ids.length; i++) {
    var id = ids[i];
    try {
      var data = await fetchJSON('/api/sessions/' + encodeURIComponent(id));
      bundle.sessions.push(data);
      envelope.succeeded.push(id);
    } catch(e) {
      envelope.failed.push({id: id, reason: 'fetch_failed'});
    }
  }
  state.lastBulkResult = envelope;
  renderBulkStatus();
  if (envelope.succeeded.length > 0) {
    var blob = new Blob([JSON.stringify(bundle, null, 2)], {type: 'application/json'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'polylogue-bulk-export-' + Date.now() + '.json';
    document.body.appendChild(a);
    a.click();
    setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
  }
}

function openBulkPreview(kind) {
  var ids = bulkSelectedIds();
  if (!ids.length) return;
  state.bulkPending = kind;
  var titleEl = document.getElementById('bulk-preview-title');
  var bodyEl = document.getElementById('bulk-preview-body');
  var listEl = document.getElementById('bulk-preview-list');
  var labels = {
    'delete': {title: 'Preview: Delete', body: 'Delete is gated. Confirming will record a dry-run envelope with one entry per session — no daemon delete endpoint is wired yet, so every session will be skipped with reason no_endpoint.'},
    'reembed': {title: 'Preview: Re-embed', body: 'Re-embed is gated. Confirming will record a dry-run envelope with one entry per session — no daemon re-embed endpoint is wired yet, so every session will be skipped with reason no_endpoint.'}
  };
  var spec = labels[kind] || labels['delete'];
  titleEl.textContent = spec.title;
  bodyEl.textContent = spec.body;
  listEl.innerHTML = ids.slice(0, 200).map(function(id) {
    return '<li>' + esc(id) + '</li>';
  }).join('');
  document.getElementById('bulk-preview').classList.add('visible');
}

function closeBulkPreview() {
  state.bulkPending = null;
  document.getElementById('bulk-preview').classList.remove('visible');
}

function confirmBulkPreview() {
  var kind = state.bulkPending;
  if (!kind) { closeBulkPreview(); return; }
  var ids = bulkSelectedIds();
  var envelope = {action: kind, dryRun: true, succeeded: [], failed: [], skipped: []};
  ids.forEach(function(id) { envelope.skipped.push({id: id, reason: 'no_endpoint'}); });
  state.lastBulkResult = envelope;
  closeBulkPreview();
  renderBulkStatus();
}

function attachBulkHandlers() {
  document.getElementById('bulk-toolbar').addEventListener('click', function(e) {
    var btn = e.target.closest('[data-bulk-action]');
    if (!btn) return;
    var action = btn.dataset.bulkAction;
    if (action === 'tag-star') bulkApplyMark('star');
    else if (action === 'tag-pin') bulkApplyMark('pin');
    else if (action === 'tag-archive') bulkApplyMark('archive');
    else if (action === 'export') bulkExport();
    else if (action === 'delete-preview') openBulkPreview('delete');
    else if (action === 'reembed-preview') openBulkPreview('reembed');
  });
  document.getElementById('bulk-select-all').addEventListener('click', selectAllVisible);
  document.getElementById('bulk-clear').addEventListener('click', clearBulkSelection);
  document.getElementById('bulk-preview-confirm').addEventListener('click', confirmBulkPreview);
  document.getElementById('bulk-preview').addEventListener('click', function(e) {
    if (e.target.id === 'bulk-preview') closeBulkPreview();
  });
  document.getElementById('conv-list').addEventListener('change', function(e) {
    var cb = e.target.closest('.bulk-check');
    if (!cb) return;
    setBulkSelected(cb.dataset.bulkId, cb.checked);
    var item = cb.closest('.conv-item');
    if (item) item.classList.toggle('bulk-selected', cb.checked);
    renderBulkToolbar();
  });
}
"""
