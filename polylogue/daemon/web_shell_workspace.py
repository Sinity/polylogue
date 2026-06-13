"""Workspace-mode assets for the daemon-served reader shell."""

from __future__ import annotations

WORKSPACE_CSS = r"""
#workspace-toolbar { display: flex; align-items: center; gap: 8px; padding: 7px 16px;
  border-bottom: 1px solid var(--border); background: var(--panel-subtle); flex-wrap: wrap; }
#workspace-toolbar .workspace-mode { display: flex; gap: 3px; }
#workspace-toolbar .mode-btn { background: var(--panel-elevated); border: 1px solid var(--border);
  color: var(--text-muted); padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: var(--small); }
#workspace-toolbar .mode-btn.active { color: var(--accent); border-color: var(--accent-soft); background: var(--accent-bg); }
#workspace-toolbar .mode-btn[disabled], #workspace-toolbar .workspace-action[disabled] {
  opacity: 0.45; cursor: not-allowed; color: var(--text-dim); }
#workspace-toolbar .mode-btn[disabled]:hover, #workspace-toolbar .workspace-action[disabled]:hover {
  background: var(--panel-elevated); border-color: var(--border); }
#workspace-toolbar .workspace-spacer { flex: 1; min-width: 12px; }
#workspace-toolbar select { background: var(--panel-elevated); color: var(--text); border: 1px solid var(--border);
  border-radius: 3px; padding: 3px 6px; font-size: var(--small); max-width: 180px; }
#workspace-toolbar .workspace-action { background: var(--panel-elevated); border: 1px solid var(--border);
  color: var(--accent); padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: var(--small); }
#workspace-toolbar .workspace-action:hover { border-color: var(--accent-soft); background: var(--accent-bg); }
#workspace-toolbar .workspace-stat { color: var(--text-muted); font-size: var(--small); }
#workspace-toolbar .workspace-stat.warn { color: var(--warn); }
#stack-view { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(280px, 1fr);
  height: 100%; overflow-x: auto; overflow-y: hidden; }
.stack-lane { border-right: 1px solid var(--border); min-width: 280px; overflow-y: auto; background: var(--bg); }
.stack-lane.missing { display: flex; align-items: center; justify-content: center; color: var(--text-dim); padding: 20px; }
.stack-lane-header { position: sticky; top: 0; z-index: 2; background: var(--bg-raised);
  border-bottom: 1px solid var(--border); padding: 8px 10px; }
.stack-lane-title { font-size: var(--base); color: var(--text); line-height: 1.3; margin-bottom: 3px; }
.stack-lane-meta { display: flex; gap: 6px; flex-wrap: wrap; color: var(--text-muted); font-size: var(--small); }
#compare-view { height: 100%; overflow-y: auto; }
.compare-header { display: grid; grid-template-columns: 1fr 1fr; position: sticky; top: 0; z-index: 2;
  background: var(--bg-raised); border-bottom: 1px solid var(--border); }
.compare-pane-title { padding: 8px 12px; border-right: 1px solid var(--border); color: var(--text); }
.compare-pair { display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid var(--border); }
.compare-pair.diff-added { border-left: 2px solid var(--ok); }
.compare-pair.diff-removed { border-left: 2px solid var(--err); }
.compare-pair.diff-changed { border-left: 2px solid var(--warn); }
.compare-pair.diff-equal { border-left: 2px solid transparent; }
.compare-cell { min-width: 0; border-right: 1px solid var(--border); background: var(--bg); }
.compare-cell.empty { display: flex; align-items: center; justify-content: center; color: var(--text-dim); min-height: 68px; }
.compare-pair-marker { grid-column: 1/-1; padding: 2px 12px; font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.6px; color: var(--text-dim); background: var(--panel-subtle); border-bottom: 1px solid var(--border); }
.compare-pair-marker.changed { color: var(--warn); }
.compare-pair-marker.added { color: var(--ok); }
.compare-pair-marker.removed { color: var(--err); }
#compare-metadata-panel { padding: 8px 16px; border-bottom: 1px solid var(--border); background: var(--panel-subtle); }
#compare-metadata-panel h4 { font-size: 11px; font-weight: 600; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }
.compare-metadata-row { display: grid; grid-template-columns: 110px 1fr 1fr 70px; gap: 8px;
  font-size: var(--small); padding: 2px 0; border-bottom: 1px solid var(--border); }
.compare-metadata-row .label { color: var(--text-muted); }
.compare-metadata-row .value { font-family: var(--font-mono); font-size: 11px; word-break: break-word; color: var(--text); }
.compare-metadata-row .value.empty { color: var(--text-dim); }
.compare-metadata-row .status { font-size: 10px; text-align: right; text-transform: uppercase; letter-spacing: 0.6px; }
.compare-metadata-row.status-equal .status { color: var(--ok); }
.compare-metadata-row.status-changed .status { color: var(--warn); }
.compare-metadata-row.status-missing .status { color: var(--err); }
#compare-metadata-empty { font-size: var(--small); color: var(--text-dim); }
.workspace-degraded { padding: 10px 16px; color: var(--warn); background: var(--warn-bg);
  border-bottom: 1px solid var(--border); font-size: var(--small); }
"""

WORKSPACE_HTML = r"""
    <div id="workspace-toolbar">
      <div class="workspace-mode" id="workspace-mode-switcher">
        <button class="mode-btn" data-mode="single" onclick="setSingleMode()">Single</button>
        <button class="mode-btn" data-mode="stack" onclick="openStackFromSelection()">Stack</button>
        <button class="mode-btn" data-mode="compare" onclick="openCompareFromSelection()">Compare</button>
      </div>
      <span class="workspace-stat" id="workspace-route-status">single</span>
      <span class="workspace-stat warn" id="workspace-degraded-count"></span>
      <span class="workspace-spacer"></span>
      <select id="workspace-saved-view-select" title="Saved views" onchange="restoreSavedView(this.value)">
        <option value="">Saved views</option>
      </select>
      <button class="workspace-action" id="workspace-save-view-btn" onclick="saveCurrentView()">Save view</button>
      <select id="workspace-restore-select" title="Restore workspace" onchange="restoreWorkspace(this.value)">
        <option value="">Restore workspace</option>
      </select>
      <button class="workspace-action" id="workspace-save-btn" onclick="saveWorkspace()">Save workspace</button>
      <button class="workspace-action" id="workspace-create-recall-pack-btn" onclick="createRecallPack()">Recall pack</button>
    </div>
"""

WORKSPACE_JS = r"""
function getWorkspaceRouteFromURL() {
  var path = window.location.pathname;
  var params = new URLSearchParams(window.location.search);
  if (path === '/w/stack') {
    var ids = (params.get('ids') || '').split(',').map(function(id) { return id.trim(); }).filter(Boolean);
    return {mode: 'stack', ids: ids, focus: params.get('focus') || ''};
  }
  if (path === '/w/compare') {
    return {mode: 'compare', left: params.get('left') || '', right: params.get('right') || '', align: params.get('align') || 'prompt'};
  }
  return null;
}

function pushSingleURL(convId) {
  if (convId) {
    var url = '/s/' + encodeURIComponent(convId);
    if (window.location.pathname + window.location.search !== url) history.pushState({}, '', url);
  } else {
    if (window.location.pathname + window.location.search !== '/') history.pushState({}, '', '/');
  }
}

function pushWorkspaceURL(route) {
  var url = '/';
  if (route.mode === 'stack') {
    var stackParams = new URLSearchParams();
    stackParams.set('ids', route.ids.join(','));
    if (route.focus) stackParams.set('focus', route.focus);
    url = '/w/stack?' + stackParams.toString();
  } else if (route.mode === 'compare') {
    var compareParams = new URLSearchParams();
    compareParams.set('left', route.left);
    compareParams.set('right', route.right);
    compareParams.set('align', route.align || 'prompt');
    url = '/w/compare?' + compareParams.toString();
  }
  if (window.location.pathname + window.location.search !== url) history.pushState({}, '', url);
}

async function loadWorkspaceRoute(route, updateURL) {
  if (!route) return;
  state.mode = route.mode;
  state.selected = null;
  state.selectedRaw = null;
  if (route.mode === 'stack') {
    if (!route.ids.length) { renderMain(); return; }
    if (updateURL !== false) pushWorkspaceURL(route);
    var params = new URLSearchParams();
    params.set('ids', route.ids.join(','));
    if (route.focus) params.set('focus', route.focus);
    state.stackPayload = await fetchJSON('/api/stack?' + params.toString());
    state.comparePayload = null;
  } else if (route.mode === 'compare') {
    if (!route.left || !route.right) { renderMain(); return; }
    if (updateURL !== false) pushWorkspaceURL(route);
    var cparams = new URLSearchParams();
    cparams.set('left', route.left);
    cparams.set('right', route.right);
    cparams.set('align', route.align || 'prompt');
    state.comparePayload = await fetchJSON('/api/compare?' + cparams.toString());
    state.stackPayload = null;
  }
  renderMain();
  renderInspector();
  renderSessions();
}

function renderWorkspaceToolbar() {
  document.querySelectorAll('#workspace-mode-switcher .mode-btn').forEach(function(btn) {
    var mode = btn.dataset.mode || 'single';
    btn.classList.toggle('active', mode === state.mode || (mode === 'single' && state.mode === 'single'));
  });
  var status = document.getElementById('workspace-route-status');
  var degraded = document.getElementById('workspace-degraded-count');
  var select = document.getElementById('workspace-restore-select');
  var count = 0;
  if (state.mode === 'stack' && state.stackPayload) count = state.stackPayload.degraded_count || 0;
  if (state.mode === 'compare' && state.comparePayload) count = state.comparePayload.degraded_count || 0;
  status.textContent = state.mode;
  if (count) {
    degraded.textContent = count + ' degraded';
    degraded.classList.add('q-partial');
  } else {
    degraded.textContent = '';
    degraded.classList.remove('q-partial');
  }
  var current = select.value;
  select.innerHTML = '<option value="">Restore workspace</option>' + (state.workspaces || []).map(function(w) {
    return '<option value="' + escAttr(w.workspace_id) + '">' + esc(w.name || w.workspace_id) + '</option>';
  }).join('');
  select.value = current;
  // Mirror the same restore-pattern for saved views so they are reachable from
  // the workspace toolbar (not only via the Notes inspector tab).
  var viewSelect = document.getElementById('workspace-saved-view-select');
  if (viewSelect) {
    var currentView = viewSelect.value;
    var views = state.savedViews || [];
    var label = views.length ? 'Saved views (' + views.length + ')' : 'No saved views';
    viewSelect.innerHTML = '<option value="">' + label + '</option>' + views.map(function(v) {
      return '<option value="' + escAttr(v.view_id) + '">' + esc(v.name || v.view_id) + '</option>';
    }).join('');
    viewSelect.value = currentView;
    if (!views.length) viewSelect.setAttribute('disabled', 'true');
    else viewSelect.removeAttribute('disabled');
  }
  // Disabled-action tooltips per MK3 "disabled actions are part of the design"
  // (docs/design/mk3/docs/11-little-details.md). Stack/Compare need 2+
  // selectable sessions; recall pack and save need at least one target.
  var stackBtn = document.querySelector('#workspace-mode-switcher .mode-btn[data-mode="stack"]');
  var compareBtn = document.querySelector('#workspace-mode-switcher .mode-btn[data-mode="compare"]');
  var saveBtn = document.getElementById('workspace-save-btn');
  var recallBtn = document.getElementById('workspace-create-recall-pack-btn');
  var available = selectedSessionIds();
  if (stackBtn) {
    if (available.length < 1) { stackBtn.setAttribute('disabled', 'true'); stackBtn.title = 'Stack needs a selected session or a non-empty list'; }
    else { stackBtn.removeAttribute('disabled'); stackBtn.title = 'Open selected sessions as a stack workspace'; }
  }
  if (compareBtn) {
    if (available.length < 2) { compareBtn.setAttribute('disabled', 'true'); compareBtn.title = 'Compare needs two sessions to align'; }
    else { compareBtn.removeAttribute('disabled'); compareBtn.title = 'Open the first two selected sessions side-by-side'; }
  }
  if (saveBtn) {
    if (available.length < 1) { saveBtn.setAttribute('disabled', 'true'); saveBtn.title = 'Save workspace needs at least one open session'; }
    else { saveBtn.removeAttribute('disabled'); saveBtn.title = 'Persist current workspace layout'; }
  }
  if (recallBtn) {
    if (available.length < 1) { recallBtn.setAttribute('disabled', 'true'); recallBtn.title = 'Recall pack needs at least one open session'; }
    else { recallBtn.removeAttribute('disabled'); recallBtn.title = 'Bundle current sessions as a recall pack'; }
  }
}

function sessionHeaderHtml(c) {
  var title = esc(c.display_title || c.title || c.id || 'Untitled');
  var html = '<div class="stack-lane-header"><div class="stack-lane-title">' + title + '</div><div class="stack-lane-meta">';
  if (c.origin) html += '<span>' + esc(c.origin) + '</span>';
  if (c.message_count !== undefined) html += '<span>' + c.message_count + ' messages</span>';
  if (c.repo) html += '<span class="chip">' + esc(c.repo.split('/').pop() || c.repo) + '</span>';
  html += '</div></div>';
  return html;
}

function renderStackWorkspace() {
  var headerEl = document.getElementById('conv-header');
  var msgEl = document.getElementById('msg-list');
  var payload = state.stackPayload;
  headerEl.innerHTML = '<h2>Stack workspace</h2><div class="conv-stats"><span id="stack-focus">focus: ' + esc(payload && payload.focus || 'none') + '</span></div>';
  if (!payload) {
    msgEl.innerHTML = '<div class="main-empty"><h3>Stack unavailable</h3></div>';
    return;
  }
  msgEl.innerHTML = '<div id="stack-view"><div id="stack-items" style="display:contents">'
    + (payload.items || []).map(function(item) {
      if (item.status !== 'resolved' || !item.session) {
        return '<div class="stack-lane missing"><div><div>Missing session</div><div class="workspace-stat warn">' + esc(item.session_id || item.target_id || '') + '</div></div></div>';
      }
      var c = item.session;
      return '<section class="stack-lane" data-session-id="' + escAttr(c.id) + '">'
        + sessionHeaderHtml(c)
        + messageBlocksHtml(c.messages || [])
        + '</section>';
    }).join('')
    + '</div></div>';
}

function compareMetadataPanelHtml(metadataDiff) {
  // ``metadataDiff`` mirrors the backend ``metadata_diff`` envelope: a map of
  // field-name -> {left, right, status}. Iterates in stable backend order so
  // the panel layout is deterministic for visual smoke and snapshot tests.
  if (!metadataDiff || typeof metadataDiff !== 'object') {
    return '<div id="compare-metadata-panel"><h4>Metadata</h4>'
      + '<div id="compare-metadata-empty">No metadata to compare</div></div>';
  }
  var keys = Object.keys(metadataDiff);
  if (!keys.length) {
    return '<div id="compare-metadata-panel"><h4>Metadata</h4>'
      + '<div id="compare-metadata-empty">No metadata to compare</div></div>';
  }
  return '<div id="compare-metadata-panel"><h4>Metadata</h4>'
    + keys.map(function(field) {
      var entry = metadataDiff[field] || {};
      var status = entry.status || 'equal';
      var lDisplay = (entry.left === null || entry.left === undefined || entry.left === '')
        ? '\u2014' : (Array.isArray(entry.left) ? entry.left.join(', ') : String(entry.left));
      var rDisplay = (entry.right === null || entry.right === undefined || entry.right === '')
        ? '\u2014' : (Array.isArray(entry.right) ? entry.right.join(', ') : String(entry.right));
      var lCls = (entry.left === null || entry.left === undefined || entry.left === '') ? ' empty' : '';
      var rCls = (entry.right === null || entry.right === undefined || entry.right === '') ? ' empty' : '';
      return '<div class="compare-metadata-row status-' + esc(status) + '" data-field="' + escAttr(field) + '">'
        + '<span class="label">' + esc(field) + '</span>'
        + '<span class="value' + lCls + '">' + esc(lDisplay) + '</span>'
        + '<span class="value' + rCls + '">' + esc(rDisplay) + '</span>'
        + '<span class="status">' + esc(status) + '</span>'
        + '</div>';
    }).join('')
    + '</div>';
}

function renderCompareWorkspace() {
  var headerEl = document.getElementById('conv-header');
  var msgEl = document.getElementById('msg-list');
  var payload = state.comparePayload;
  var alignment = (payload && payload.alignment) || 'sequential';
  headerEl.innerHTML = '<h2>Compare workspace</h2><div class="conv-stats">'
    + '<span class="chip" id="compare-alignment-chip" title="Message alignment strategy">alignment: ' + esc(alignment) + '</span>'
    + '<span>align: '
    + '<select id="compare-align-select" onchange="changeCompareAlign(this.value)"><option value="prompt">prompt</option></select></span></div>';
  if (!payload) {
    msgEl.innerHTML = '<div class="main-empty"><h3>Compare unavailable</h3></div>';
    return;
  }
  var leftTitle = payload.left && payload.left.title ? payload.left.title : (payload.left && payload.left.session_id || 'left');
  var rightTitle = payload.right && payload.right.title ? payload.right.title : (payload.right && payload.right.session_id || 'right');
  var degradedSides = payload.degraded_sides || [];
  var banner = '<div id="compare-degraded-banner" style="display:none"></div>';
  if (payload.degraded_count) {
    var degradedLabel = degradedSides.length
      ? degradedSides.join(' & ') + ' side' + (degradedSides.length === 1 ? '' : 's') + ' failed to load'
      : payload.degraded_count + ' degraded side(s)';
    banner = '<div class="workspace-degraded" id="compare-degraded-banner">' + esc(degradedLabel) + '</div>';
  }
  msgEl.innerHTML = '<div id="compare-view">' + banner
    + compareMetadataPanelHtml(payload.metadata_diff)
    + '<div class="compare-header"><div class="compare-pane-title" id="compare-left-pane">' + esc(leftTitle) + '</div>'
    + '<div class="compare-pane-title" id="compare-right-pane">' + esc(rightTitle) + '</div></div>'
    + '<div id="compare-pairs">'
    + (payload.pairs || []).map(function(pair) {
      var diffStatus = pair.diff_status || 'equal';
      var marker = diffStatus !== 'equal'
        ? '<div class="compare-pair-marker ' + esc(diffStatus) + '">' + esc(diffStatus) + '</div>'
        : '';
      return '<div class="compare-pair diff-' + esc(diffStatus) + '" data-diff-status="' + esc(diffStatus) + '">'
        + marker
        + '<div class="compare-cell' + (pair.left ? '' : ' empty') + '">' + (pair.left ? messageBlocksHtml([pair.left]) : 'No message') + '</div>'
        + '<div class="compare-cell' + (pair.right ? '' : ' empty') + '">' + (pair.right ? messageBlocksHtml([pair.right]) : 'No message') + '</div>'
        + '</div>';
    }).join('')
    + '</div></div>';
}

function setSingleMode() {
  state.mode = 'single';
  state.stackPayload = null;
  state.comparePayload = null;
  pushSingleURL(state.selected ? state.selected.id : null);
  renderMain();
  renderInspector();
  renderSessions();
}

function selectedSessionIds() {
  if (state.mode === 'stack' && state.stackPayload) return (state.stackPayload.ids || []).slice();
  if (state.mode === 'compare' && state.comparePayload) {
    return [state.comparePayload.left && (state.comparePayload.left.id || state.comparePayload.left.session_id),
      state.comparePayload.right && (state.comparePayload.right.id || state.comparePayload.right.session_id)].filter(Boolean);
  }
  if (state.selected) return [state.selected.id];
  return state.sessions.slice(0, 2).map(function(c) { return c.id; });
}

function targetItemsForCurrentContext() {
  return selectedSessionIds().map(function(id) {
    return {target_type: 'session', session_id: id};
  });
}

async function openStackFromSelection() {
  var ids = selectedSessionIds();
  if (!ids.length && state.sessions.length) ids = state.sessions.slice(0, 3).map(function(c) { return c.id; });
  if (!ids.length) return;
  await loadWorkspaceRoute({mode: 'stack', ids: ids, focus: ids[0]}, true);
}

async function openParentChainAsStack(sessionId) {
  // #1203: turn topology shape into a stack workspace. Fetches the
  // parent-chain envelope (root -> target -> descendants) and routes
  // to the stack workspace with the chain pre-populated and ``focus``
  // set to the session the operator invoked the action from.
  if (!sessionId) return;
  try {
    var data = await fetchJSON('/api/sessions/' + encodeURIComponent(sessionId) + '/topology/parent-chain');
    var chain = (data && data.chain_ids) || [];
    if (!chain.length) return;
    var focus = (data && data.focus_id) || sessionId;
    await loadWorkspaceRoute({mode: 'stack', ids: chain, focus: focus}, true);
  } catch(e) {
    // Fallback: open the focused session as a single-element stack
    // so the user sees explicit feedback that the chain action fired
    // but the lineage walk failed.
    await loadWorkspaceRoute({mode: 'stack', ids: [sessionId], focus: sessionId}, true);
  }
}

async function openCompareFromSelection() {
  var ids = selectedSessionIds();
  if (ids.length < 2) ids = state.sessions.slice(0, 2).map(function(c) { return c.id; });
  if (ids.length < 2) return;
  await loadWorkspaceRoute({mode: 'compare', left: ids[0], right: ids[1], align: 'prompt'}, true);
}

async function changeCompareAlign(align) {
  if (!state.comparePayload) return;
  var left = state.comparePayload.left && (state.comparePayload.left.id || state.comparePayload.left.session_id);
  var right = state.comparePayload.right && (state.comparePayload.right.id || state.comparePayload.right.session_id);
  if (!left || !right) return;
  await loadWorkspaceRoute({mode: 'compare', left: left, right: right, align: align || 'prompt'}, true);
}

async function saveWorkspace() {
  var items = targetItemsForCurrentContext();
  if (!items.length) return;
  var name = window.prompt('Workspace name', state.mode === 'single' ? 'Reader workspace' : (state.mode + ' workspace'));
  if (!name) return;
  var id = 'workspace-' + Date.now().toString(36);
  var active = items[0];
  if (state.selected) active = {target_type: 'session', session_id: state.selected.id};
  try {
    await sendJSON('/api/user/workspaces', 'POST', {
      workspace_id: id,
      name: name,
      mode: state.mode,
      open_targets: items,
      layout: {mode: state.mode},
      active_target: active
    });
    await loadUserState();
  } catch(e) {
    state.userStateError = 'Failed to save workspace';
    renderInspector();
  }
}

async function saveCurrentView() {
  var query = {limit: state.limit, offset: 0};
  if (state.query) query.query = state.query;
  if (state.origin) query.origin = state.origin;
  var defaultName = state.query || state.origin || 'All sessions';
  var name = window.prompt('Saved view name', defaultName);
  if (!name) return;
  name = name.trim();
  if (!name) {
    state.userStateError = 'Saved view name cannot be empty';
    renderInspector();
    return;
  }
  // Naming conflict UX: storage enforces UNIQUE(name). Detect a clash
  // locally against the loaded list so the operator can choose to overwrite
  // or pick a new name before the request fires (which would otherwise hit
  // the SQLite unique constraint with a generic error).
  var existing = (state.savedViews || []).find(function(v) { return (v.name || '') === name; });
  if (existing) {
    if (!window.confirm('A saved view named "' + name + '" already exists. Overwrite it?')) return;
    try {
      await sendJSON('/api/user/saved-views/' + encodeURIComponent(existing.view_id), 'DELETE');
    } catch(e) {
      state.userStateError = 'Failed to replace existing view';
      renderInspector();
      return;
    }
  }
  try {
    await sendJSON('/api/user/saved-views', 'POST', {name: name, query: query});
    state.userStateError = '';
    await loadUserState();
  } catch(e) {
    state.userStateError = 'Failed to save view';
    renderInspector();
  }
}

async function deleteSavedView(viewId) {
  if (!viewId) return;
  var view = (state.savedViews || []).find(function(v) { return v.view_id === viewId; });
  var label = view ? (view.name || viewId) : viewId;
  if (!window.confirm('Delete saved view "' + label + '"?')) return;
  try {
    await sendJSON('/api/user/saved-views/' + encodeURIComponent(viewId), 'DELETE');
    state.userStateError = '';
    await loadUserState();
  } catch(e) {
    state.userStateError = 'Failed to delete saved view';
    renderInspector();
  }
}

function restoreSavedView(viewId) {
  if (!viewId) return;
  // Recall the saved query into the active filter chain. applySavedView lives
  // in the main shell script and reloads sessions + facets, then resets
  // the select so reselecting the same view re-applies it.
  applySavedView(viewId);
  var select = document.getElementById('workspace-saved-view-select');
  if (select) select.value = '';
}

async function restoreWorkspace(workspaceId) {
  if (!workspaceId) return;
  try {
    var workspace = await fetchJSON('/api/user/workspaces/' + encodeURIComponent(workspaceId));
    var ids = (workspace.open_targets || []).filter(function(t) {
      return t.target_type === 'session' && (t.session_id || t.target_id);
    }).map(function(t) { return t.session_id || t.target_id; });
    if (workspace.mode === 'compare' && ids.length >= 2) {
      await loadWorkspaceRoute({mode: 'compare', left: ids[0], right: ids[1], align: 'prompt'}, true);
    } else if (ids.length) {
      await loadWorkspaceRoute({mode: 'stack', ids: ids, focus: ids[0]}, true);
    }
  } catch(e) {
    state.userStateError = 'Failed to restore workspace';
    renderInspector();
  }
}

async function createRecallPack() {
  var items = targetItemsForCurrentContext();
  if (!items.length) return;
  var label = window.prompt('Recall pack label', state.mode === 'single' ? 'Reader recall pack' : (state.mode + ' recall pack'));
  if (!label) return;
  var packId = 'pack-' + Date.now().toString(36);
  try {
    await sendJSON('/api/user/recall-packs', 'POST', {
      pack_id: packId,
      label: label,
      payload: {summary: label, items: items}
    });
    await loadUserState();
  } catch(e) {
    state.userStateError = 'Failed to create recall pack';
    renderInspector();
  }
}
"""

__all__ = ["WORKSPACE_CSS", "WORKSPACE_HTML", "WORKSPACE_JS"]
