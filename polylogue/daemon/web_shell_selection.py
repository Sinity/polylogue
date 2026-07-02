"""Selection-operations assets for the daemon-served reader shell (#1119).

The selection surface is composed client-side over existing daemon endpoints:
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

SELECTION_CSS = r"""
.conv-item .conv-row { display: flex; align-items: flex-start; gap: 6px; }
.conv-item .conv-row .conv-body { flex: 1; min-width: 0; }
.conv-item .selection-check { margin-top: 2px; flex-shrink: 0; cursor: pointer; accent-color: var(--accent); }
.conv-item.selection-selected { background: var(--accent-bg); }

/* Selection operations toolbar — appears in the sidebar when one or more
   sessions are checkbox-selected. Drives selection tag (marks),
   markdown export (client-side), and preview-only delete/re-embed
   per #1119. The toolbar is intentionally inert until selection > 0. */
#selection-toolbar { display: none; flex-direction: column; gap: 6px; padding: 8px 10px;
  border-bottom: 1px solid var(--border); background: var(--panel-elevated); }
#selection-toolbar.visible { display: flex; }
#selection-toolbar .selection-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
#selection-toolbar .selection-count { color: var(--accent); font-size: var(--small); font-weight: 600; }
#selection-toolbar .selection-link { color: var(--text-muted); font-size: var(--small); cursor: pointer;
  background: none; border: none; padding: 0; text-decoration: underline; }
#selection-toolbar .selection-link:hover { color: var(--text); }
#selection-toolbar .selection-btn { background: var(--panel-subtle); border: 1px solid var(--border);
  color: var(--text); padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: var(--small); font-family: var(--font-ui); }
#selection-toolbar .selection-btn:hover { border-color: var(--accent-soft); color: var(--accent); }
#selection-toolbar .selection-btn.danger { color: var(--err); }
#selection-toolbar .selection-btn.danger:hover { border-color: var(--err); background: var(--err-bg); }
#selection-status { font-size: var(--small); color: var(--text-muted); line-height: 1.4; }
#selection-status .ok { color: var(--ok); }
#selection-status .err { color: var(--err); }
#selection-status .skip { color: var(--warn); }
#selection-preview { display: none; position: fixed; inset: 0; background: rgba(7,11,16,0.85); z-index: 110;
  align-items: center; justify-content: center; }
#selection-preview.visible { display: flex; }
#selection-preview-panel { background: var(--panel-elevated); border: 1px solid var(--border-strong); border-radius: 8px;
  padding: 20px; max-width: 520px; width: 90%; max-height: 80vh; overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
#selection-preview-panel h3 { font-size: 15px; margin-bottom: 10px; color: var(--accent); }
#selection-preview-panel p { font-size: var(--small); color: var(--text-muted); margin-bottom: 10px; line-height: 1.5; }
#selection-preview-panel ul { list-style: none; margin: 8px 0; padding: 0; max-height: 240px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: 3px; }
#selection-preview-panel ul li { padding: 4px 8px; font-size: var(--small); border-bottom: 1px solid var(--border);
  color: var(--text); font-family: var(--font-mono); }
#selection-preview-panel .actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 12px; }
"""

SELECTION_TOOLBAR_HTML = r"""
    <div id="selection-toolbar" aria-label="Selection operations toolbar">
      <div class="selection-row">
        <span class="selection-count" id="selection-count">0 selected</span>
        <span style="flex:1"></span>
        <button class="selection-link" id="selection-select-all" type="button">Select all</button>
        <button class="selection-link" id="selection-clear" type="button">Clear</button>
      </div>
      <div class="selection-row">
        <button class="selection-btn" data-selection-action="tag-star" type="button" title="Star all selected">Star</button>
        <button class="selection-btn" data-selection-action="tag-pin" type="button" title="Pin all selected">Pin</button>
        <button class="selection-btn" data-selection-action="tag-archive" type="button" title="Archive all selected">Archive</button>
        <button class="selection-btn" data-selection-action="export" type="button" title="Export selected as JSON">Export</button>
        <button class="selection-btn danger" data-selection-action="delete-preview" type="button" title="Preview delete (dry run)">Delete...</button>
        <button class="selection-btn" data-selection-action="reembed-preview" type="button" title="Preview re-embed (dry run)">Re-embed...</button>
      </div>
      <div id="selection-status" aria-live="polite"></div>
    </div>
"""

SELECTION_PREVIEW_HTML = r"""
<div id="selection-preview" role="dialog" aria-label="Selection operation preview">
  <div id="selection-preview-panel">
    <h3 id="selection-preview-title">Preview</h3>
    <p id="selection-preview-body">Confirm the destructive operation below.</p>
    <ul id="selection-preview-list"></ul>
    <div class="actions">
      <button class="selection-btn" type="button" onclick="closeSelectionPreview()">Cancel</button>
      <button class="selection-btn danger" id="selection-preview-confirm" type="button">Confirm</button>
    </div>
  </div>
</div>
"""

SELECTION_JS = r"""
// --- Selection set (#1119) ------------------------------------------------
// The selection surface is purely client-side composition over existing daemon
// endpoints. Tag operations issue per-session POSTs to /api/user/marks
// (the centralized tag mutation contract). Export streams session
// detail through GET /api/sessions/{id} and downloads a single JSON
// bundle. Delete and re-embed are exposed only through a preview overlay —
// the daemon currently has no DELETE-session or re-embed route, so the
// confirm button records a per-session "skipped: no_endpoint" entry
// rather than silently appearing to mutate state. This keeps the AC
// contract (preview gate + typed envelope) honest.

function isSelectionSelected(id) { return !!state.selectionSet[id]; }
function selectionSelectedIds() { return Object.keys(state.selectionSet); }
function selectionSelectedCount() { return selectionSelectedIds().length; }

function setSelectionSelected(id, enabled) {
  if (enabled) state.selectionSet[id] = true;
  else delete state.selectionSet[id];
}

function toggleSelectionSelected(id) { setSelectionSelected(id, !isSelectionSelected(id)); }

function clearSelection() {
  state.selectionSet = {};
  state.lastSelectionResult = null;
  renderSessions();
}

function selectAllVisible() {
  (state.sessions || []).forEach(function(c) { state.selectionSet[c.id] = true; });
  renderSessions();
}

function renderSelectionToolbar() {
  var bar = document.getElementById('selection-toolbar');
  if (!bar) return;
  var count = selectionSelectedCount();
  var countEl = document.getElementById('selection-count');
  if (countEl) countEl.textContent = count + ' selected';
  if (count > 0) bar.classList.add('visible'); else bar.classList.remove('visible');
  renderSelectionStatus();
}

function renderSelectionStatus() {
  var el = document.getElementById('selection-status');
  if (!el) return;
  var result = state.lastSelectionResult;
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

async function selectionApplyMark(markType) {
  var ids = selectionSelectedIds();
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
  state.lastSelectionResult = envelope;
  renderSessions();
  renderInspector();
}

async function selectionExport() {
  var ids = selectionSelectedIds();
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
  state.lastSelectionResult = envelope;
  renderSelectionStatus();
  if (envelope.succeeded.length > 0) {
    var blob = new Blob([JSON.stringify(bundle, null, 2)], {type: 'application/json'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'polylogue-selection-export-' + Date.now() + '.json';
    document.body.appendChild(a);
    a.click();
    setTimeout(function() { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
  }
}

function openSelectionPreview(kind) {
  var ids = selectionSelectedIds();
  if (!ids.length) return;
  state.selectionPending = kind;
  var titleEl = document.getElementById('selection-preview-title');
  var bodyEl = document.getElementById('selection-preview-body');
  var listEl = document.getElementById('selection-preview-list');
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
  document.getElementById('selection-preview').classList.add('visible');
}

function closeSelectionPreview() {
  state.selectionPending = null;
  document.getElementById('selection-preview').classList.remove('visible');
}

function confirmSelectionPreview() {
  var kind = state.selectionPending;
  if (!kind) { closeSelectionPreview(); return; }
  var ids = selectionSelectedIds();
  var envelope = {action: kind, dryRun: true, succeeded: [], failed: [], skipped: []};
  ids.forEach(function(id) { envelope.skipped.push({id: id, reason: 'no_endpoint'}); });
  state.lastSelectionResult = envelope;
  closeSelectionPreview();
  renderSelectionStatus();
}

function attachSelectionHandlers() {
  document.getElementById('selection-toolbar').addEventListener('click', function(e) {
    var btn = e.target.closest('[data-selection-action]');
    if (!btn) return;
    var action = btn.dataset.selectionAction;
    if (action === 'tag-star') selectionApplyMark('star');
    else if (action === 'tag-pin') selectionApplyMark('pin');
    else if (action === 'tag-archive') selectionApplyMark('archive');
    else if (action === 'export') selectionExport();
    else if (action === 'delete-preview') openSelectionPreview('delete');
    else if (action === 'reembed-preview') openSelectionPreview('reembed');
  });
  document.getElementById('selection-select-all').addEventListener('click', selectAllVisible);
  document.getElementById('selection-clear').addEventListener('click', clearSelection);
  document.getElementById('selection-preview-confirm').addEventListener('click', confirmSelectionPreview);
  document.getElementById('selection-preview').addEventListener('click', function(e) {
    if (e.target.id === 'selection-preview') closeSelectionPreview();
  });
  document.getElementById('conv-list').addEventListener('change', function(e) {
    var cb = e.target.closest('.selection-check');
    if (!cb) return;
    setSelectionSelected(cb.dataset.selectionId, cb.checked);
    var item = cb.closest('.conv-item');
    if (item) item.classList.toggle('selection-selected', cb.checked);
    renderSelectionToolbar();
  });
}
"""
