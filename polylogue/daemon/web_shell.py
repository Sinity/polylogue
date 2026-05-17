"""Polylogue MK2 web reader — single-page interactive archive cockpit."""

from __future__ import annotations

from polylogue.daemon.web_shell_bulk import (
    BULK_CSS,
    BULK_JS,
    BULK_PREVIEW_HTML,
    BULK_TOOLBAR_HTML,
)
from polylogue.daemon.web_shell_workspace import WORKSPACE_CSS, WORKSPACE_HTML, WORKSPACE_JS

WEB_SHELL_HTML = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polylogue</title>
<style>
:root {
  --bg: #070B10; --bg-raised: #0B1118; --panel: #0E151D;
  --panel-elevated: #111C26; --panel-subtle: #0A1016;
  --border: #22303D; --border-strong: #344657;
  --text: #D6E2EA; --text-muted: #8C9AA8; --text-dim: #5F6F7E;
  --accent: #5AB8D6; --accent-soft: #2B6E84; --accent-bg: #0B1F2B;
  --ok: #5FD7AE; --ok-bg: #0C2A24; --warn: #E6B450; --warn-bg: #2C220B;
  --err: #E86671; --err-bg: #2A1015; --active: #76A9FF;
  --role-user: #78B7FF; --role-assistant: #D6E2EA;
  --role-tool: #B7A6FF; --role-system: #A4B0BE;
  --role-thinking: #8F98A5;
  --provider-claude-code: #72D6A3; --provider-codex: #7EA7FF;
  --provider-chatgpt: #67D8C7; --provider-claude-ai: #D6A36B;
  --provider-gemini: #AB8FE8;
  --font-ui: Inter, ui-sans-serif, system-ui, sans-serif;
  --font-mono: JetBrains Mono, Fira Code, ui-monospace, monospace;
  --base: 13px; --small: 11px; --code: 12px; --lh: 1.45;
  --radius: 4px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: var(--font-ui); font-size: var(--base); line-height: var(--lh); overflow: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

#app { display: grid; grid-template-columns: 300px 1fr 320px; grid-template-rows: 36px 1fr 26px; height: 100vh; }

#status-strip { grid-column: 1/-1; grid-row: 1; display: flex; align-items: center; gap: 10px;
  padding: 0 12px; background: var(--bg-raised); border-bottom: 1px solid var(--border); font-size: var(--small); }
#status-strip .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
#status-strip .dot.ok { background: var(--ok); }
#status-strip .dot.warn { background: var(--warn); }
#status-strip .dot.err { background: var(--err); }
#status-strip .chip { padding: 1px 6px; border-radius: 3px; font-size: var(--small);
  background: var(--panel-elevated); border: 1px solid var(--border); color: var(--text-muted); white-space: nowrap; }
#status-strip .chip.accent { border-color: var(--accent-soft); color: var(--accent); }
#status-strip .spacer { flex: 1; }

/* MK3 data-quality chip vocabulary (docs/design/mk3/docs/11-little-details.md).
   These classes apply to any .chip and override its default muted look with
   colors keyed to the named data state. Use them so the operator can tell at
   a glance whether a value is canonical, derived, degraded, or missing. */
.chip.q-canonical { border-color: var(--ok); color: var(--ok); background: var(--ok-bg); }
.chip.q-explicit { border-color: var(--ok); color: var(--ok); background: var(--ok-bg); }
.chip.q-inferred { border-color: var(--accent-soft); color: var(--accent); background: var(--accent-bg); }
.chip.q-heuristic { border-color: var(--accent-soft); color: var(--accent); background: var(--panel-elevated); }
.chip.q-repaired { border-color: var(--accent-soft); color: var(--accent); background: var(--accent-bg); }
.chip.q-partial { border-color: var(--warn); color: var(--warn); background: var(--warn-bg); }
.chip.q-stale { border-color: var(--warn); color: var(--warn); background: var(--warn-bg); }
.chip.q-estimated { border-color: var(--warn); color: var(--warn); background: var(--panel-elevated); }
.chip.q-unresolved { border-color: var(--err); color: var(--err); background: var(--err-bg); }
.chip.q-unavailable { border-color: var(--text-dim); color: var(--text-dim); background: var(--panel-subtle); }
.chip.q-redacted { border-color: var(--text-dim); color: var(--text-dim); background: var(--panel-subtle); font-style: italic; }

#sidebar { grid-column: 1; grid-row: 2; display: flex; flex-direction: column;
  background: var(--panel); border-right: 1px solid var(--border); overflow: hidden; }
#search-box { padding: 8px 10px; border-bottom: 1px solid var(--border); display: flex; gap: 6px; }
#search-box input { flex: 1; background: var(--panel-elevated); border: 1px solid var(--border);
  color: var(--text); padding: 5px 8px; border-radius: var(--radius); font-size: var(--base); outline: none; }
#search-box input:focus { border-color: var(--accent); }
#search-box .help-btn { background: var(--panel-elevated); border: 1px solid var(--border); color: var(--text-muted);
  padding: 4px 7px; border-radius: var(--radius); cursor: pointer; font-size: var(--small); line-height: 1; }
#facet-bar { padding: 6px 10px; border-bottom: 1px solid var(--border); max-height: 140px; overflow-y: auto; }
.facet-group { margin-bottom: 4px; }
.facet-group-label { font-size: 10px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.6px; margin-bottom: 2px; }
.facet-chips { display: flex; flex-wrap: wrap; gap: 3px; }
.facet-chip { background: var(--panel-elevated); border: 1px solid var(--border);
  color: var(--text-muted); padding: 1px 7px; border-radius: 3px; cursor: pointer; font-size: var(--small); white-space: nowrap; }
.facet-chip:hover { border-color: var(--text-dim); color: var(--text); }
.facet-chip.active { background: var(--accent-bg); color: var(--accent); border-color: var(--accent-soft); }
.facet-chip .count { color: var(--text-dim); margin-left: 3px; font-size: 10px; }

#conv-list { flex: 1; overflow-y: auto; }
.conv-item { padding: 7px 10px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.1s; }
.conv-item:hover { background: var(--panel-elevated); }
.conv-item.selected { background: var(--panel-elevated); border-left: 2px solid var(--accent); padding-left: 8px; }
.conv-item .conv-title { font-size: var(--base); color: var(--text); line-height: 1.3; display: -webkit-box;
  -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.conv-item .conv-meta { display: flex; gap: 6px; align-items: center; font-size: var(--small); color: var(--text-muted); margin-top: 3px; flex-wrap: wrap; }
.conv-item .conv-meta .flag { font-size: 10px; padding: 0 4px; border-radius: 2px; background: var(--panel-subtle); }
.conv-item .conv-meta .flag.tool { color: var(--role-tool); }
.conv-item .conv-meta .flag.think { color: var(--role-thinking); }
.conv-item .conv-meta .flag.mark { color: var(--warn); border: 1px solid var(--border); }
.provider-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 3px; flex-shrink: 0; }
__BULK_CSS__
.sidebar-state { padding: 16px 12px; color: var(--text-dim); font-size: var(--small); text-align: center; line-height: 1.6; }
.sidebar-state .state-icon { font-size: 24px; margin-bottom: 6px; opacity: 0.4; }

#main { grid-column: 2; grid-row: 2; display: flex; flex-direction: column; overflow: hidden; background: var(--bg); }
#conv-header { padding: 10px 16px; border-bottom: 1px solid var(--border); background: var(--bg-raised); }
#conv-header h2 { font-size: 15px; font-weight: 500; line-height: 1.3; margin-bottom: 4px; }
#conv-header .title-row { display: flex; align-items: flex-start; gap: 10px; justify-content: space-between; }
#conv-header .title-row h2 { flex: 1; min-width: 0; }
#conv-header .mark-actions { display: flex; gap: 4px; flex-shrink: 0; }
#conv-header .mark-btn { width: 26px; height: 24px; border-radius: 3px; border: 1px solid var(--border);
  background: var(--panel-subtle); color: var(--text-muted); cursor: pointer; font-size: var(--small); }
#conv-header .mark-btn:hover { color: var(--text); border-color: var(--text-dim); }
#conv-header .mark-btn.active { color: var(--warn); border-color: var(--warn); background: var(--warn-bg); }
#conv-header .conv-stats { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; font-size: var(--small); color: var(--text-muted); }
#conv-header .conv-stats .chip { padding: 1px 6px; border-radius: 3px; font-size: var(--small);
  background: var(--panel-subtle); border: 1px solid var(--border); white-space: nowrap; }
#conv-header .conv-stats .chip.accent { border-color: var(--accent-soft); color: var(--accent); background: var(--accent-bg); }
#conv-header .conv-stats .chip.repo { font-family: var(--font-mono); font-size: 11px; }
__WORKSPACE_CSS__
#msg-list { flex: 1; overflow-y: auto; }
.msg-block { padding: 7px 16px; border-bottom: 1px solid var(--border); }
.msg-block:hover { background: var(--bg-raised); }
.msg-block .msg-header { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; font-size: var(--small); }
.msg-block .msg-role { font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.6px; }
.msg-role.user { color: var(--role-user); } .msg-role.assistant { color: var(--role-assistant); }
.msg-role.tool { color: var(--role-tool); } .msg-role.system { color: var(--role-system); }
.msg-role.thinking { color: var(--role-thinking); }
.msg-block .msg-type { color: var(--text-dim); font-size: 10px; padding: 0 4px; border-radius: 2px; background: var(--panel-subtle); }
.msg-block .msg-ts { color: var(--text-dim); font-size: 10px; margin-left: auto; }
.msg-block .msg-text { font-family: var(--font-mono); font-size: var(--code); white-space: pre-wrap; word-break: break-word;
  max-height: 500px; overflow-y: auto; }
.msg-block .msg-text.collapsed { max-height: 100px; overflow: hidden; position: relative; }
.msg-block .msg-text.collapsed::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 36px;
  background: linear-gradient(transparent, var(--bg)); }
.msg-block .msg-text.collapsed:hover::after { background: linear-gradient(transparent, var(--bg-raised)); }
.msg-block .msg-expand { color: var(--accent); cursor: pointer; font-size: var(--small); margin-top: 3px; user-select: none; }
.msg-block .msg-expand:hover { color: var(--active); }
.msg-block .msg-text:not(.collapsed) + .msg-expand { display: none; }
.tool-block { border-left: 2px solid var(--role-tool); padding-left: 12px; margin: 2px 0; }
.tool-block .tool-summary { font-size: var(--small); color: var(--text-muted); cursor: pointer; padding: 3px 0; }
.tool-block .tool-summary:hover { color: var(--text); }
.tool-block .tool-summary code { font-family: var(--font-mono); font-size: 11px; background: var(--panel-elevated);
  padding: 1px 5px; border-radius: 2px; color: var(--accent); }
.main-empty { display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; color: var(--text-dim); text-align: center; padding: 32px; }
.main-empty h3 { font-size: 15px; font-weight: 400; margin-bottom: 6px; color: var(--text-muted); }
.main-empty p { font-size: var(--small); max-width: 320px; line-height: 1.6; }
.main-empty .kbd { font-family: var(--font-mono); font-size: 11px; background: var(--panel-elevated);
  border: 1px solid var(--border); padding: 2px 6px; border-radius: 3px; margin: 0 2px; }

#inspector { grid-column: 3; grid-row: 2; background: var(--panel); border-left: 1px solid var(--border);
  overflow-y: auto; display: flex; flex-direction: column; }
#inspector-tabs { display: flex; border-bottom: 1px solid var(--border); flex-shrink: 0; }
#inspector-tabs button { flex: 1; background: none; border: none; border-bottom: 2px solid transparent;
  color: var(--text-dim); padding: 7px 8px; cursor: pointer; font-size: var(--small); font-family: var(--font-ui); }
#inspector-tabs button:hover { color: var(--text-muted); }
#inspector-tabs button.active { color: var(--accent); border-bottom-color: var(--accent); }
#inspector-content { flex: 1; overflow-y: auto; padding: 10px; }
.inspector-empty { color: var(--text-dim); font-size: var(--small); text-align: center; padding: 20px 0; }
.inspector-field { display: flex; justify-content: space-between;
  padding: 3px 0; font-size: var(--small); border-bottom: 1px solid var(--border); }
.inspector-field .label { color: var(--text-muted); flex-shrink: 0; margin-right: 8px; }
.inspector-field .value { color: var(--text); font-family: var(--font-mono); font-size: 11px; text-align: right; word-break: break-all; }
.inspector-field .value.empty { color: var(--text-dim); }
.inspector-section { margin-top: 10px; }
.inspector-section h4 { font-size: 11px; font-weight: 600; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.6px; margin-bottom: 6px; }
.user-state-row { display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 5px 0; border-bottom: 1px solid var(--border); font-size: var(--small); }
.user-state-row .label { color: var(--text-muted); }
.user-state-row .value { color: var(--text); font-family: var(--font-mono); font-size: 11px; word-break: break-word; }
.user-action { background: var(--panel-elevated); border: 1px solid var(--border); color: var(--accent);
  padding: 4px 8px; border-radius: 3px; cursor: pointer; font-size: var(--small); font-family: var(--font-ui); }
.user-action:hover { border-color: var(--accent-soft); background: var(--accent-bg); }
.saved-view-list { display: flex; flex-direction: column; gap: 4px; }
.saved-view-item { display: flex; justify-content: space-between; gap: 8px; align-items: center;
  border: 1px solid var(--border); border-radius: var(--radius); padding: 6px; background: var(--panel-subtle); }
.saved-view-item button { flex-shrink: 0; }
.annotation-list { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.annotation-item { border: 1px solid var(--border); border-radius: var(--radius); padding: 6px; background: var(--panel-subtle); }
.annotation-item .meta { color: var(--text-dim); font-family: var(--font-mono); font-size: 10px; margin-bottom: 4px; }
.annotation-item .note { color: var(--text); line-height: 1.45; white-space: pre-wrap; font-size: var(--small); }
.annotation-actions { display: flex; gap: 4px; margin-top: 6px; }
.raw-block { font-family: var(--font-mono); font-size: 10px; white-space: pre-wrap; word-break: break-all;
  background: var(--panel-subtle); border: 1px solid var(--border); padding: 8px; border-radius: var(--radius);
  max-height: 300px; overflow-y: auto; color: var(--text-muted); }

#footer { grid-column: 1/-1; grid-row: 3; display: flex; align-items: center; gap: 14px;
  padding: 0 10px; background: var(--bg-raised); border-top: 1px solid var(--border); font-size: var(--small); color: var(--text-muted); }
#footer .hint { font-size: 10px; color: var(--text-dim); }
#footer .hint kbd { font-family: var(--font-mono); font-size: 10px; background: var(--panel-elevated);
  border: 1px solid var(--border); padding: 1px 4px; border-radius: 2px; }

#help-overlay { display: none; position: fixed; inset: 0; background: rgba(7,11,16,0.85); z-index: 100;
  align-items: center; justify-content: center; }
#help-overlay.visible { display: flex; }
#help-panel { background: var(--panel-elevated); border: 1px solid var(--border-strong); border-radius: 8px;
  padding: 24px; max-width: 480px; width: 90%; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
#help-panel h3 { font-size: 16px; margin-bottom: 12px; color: var(--accent); }
#help-panel .help-grid { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; }
#help-panel .help-grid kbd { font-family: var(--font-mono); font-size: 11px; background: var(--panel-subtle);
  border: 1px solid var(--border); padding: 2px 6px; border-radius: 3px; text-align: center; color: var(--text); }
#help-panel .help-grid span { font-size: var(--small); color: var(--text-muted); }
#help-panel .help-close { margin-top: 16px; text-align: center; color: var(--text-dim);
  font-size: var(--small); cursor: pointer; }
#help-panel .help-close:hover { color: var(--text); }
</style>
</head>
<body>
<div id="app">
  <div id="status-strip">
    <span class="dot ok" id="status-dot" title="Daemon health"></span>
    <span class="chip" id="status-label">checking</span>
    <span class="chip" id="status-convs">0 convs</span>
    <span class="chip" id="status-msgs">0 msgs</span>
    <span class="chip" id="status-db">--</span>
    <span class="spacer"></span>
    <span class="chip" id="status-fts" title="FTS readiness">FTS: --</span>
    <span class="chip" id="status-insights" title="Session insight freshness" style="display:none">insights: --</span>
    <span class="chip" id="status-ingest" style="display:none">live</span>
    <span class="chip" id="status-live" title="Realtime channel">live: --</span>
  </div>
  <div id="sidebar">
    <div id="search-box">
      <input type="text" id="search" placeholder="Search conversations..." autofocus>
      <button class="help-btn" id="help-btn" title="Keyboard shortcuts (?)">?</button>
    </div>
    <div id="facet-bar"></div>
__BULK_TOOLBAR_HTML__
    <div id="conv-list"><div class="sidebar-state"><div class="state-icon">&mdash;</div>Loading...</div></div>
  </div>
  <div id="main">
    <div id="conv-header"><h2>Polylogue</h2><div class="conv-stats"></div></div>
__WORKSPACE_HTML__
    <div id="msg-list">
      <div class="main-empty">
        <h3>Select a conversation</h3>
        <p>Browse from the list or use <span class="kbd">/</span> to search. Press <span class="kbd">?</span> for shortcuts.</p>
      </div>
    </div>
  </div>
  <div id="inspector">
    <div id="inspector-tabs">
      <button class="active" data-tab="info">Info</button>
      <button data-tab="cost">Cost</button>
      <button data-tab="raw">Raw</button>
      <button data-tab="notes">Notes</button>
    </div>
    <div id="inspector-content"><div class="inspector-empty">Select a conversation to inspect</div></div>
  </div>
  <div id="footer">
    <span class="hint"><kbd>/</kbd> search</span>
    <span class="hint"><kbd>j</kbd><kbd>k</kbd> navigate</span>
    <span class="hint"><kbd>n</kbd><kbd>p</kbd> prev/next</span>
    <span class="hint"><kbd>Esc</kbd> clear</span>
    <span class="hint"><kbd>?</kbd> help</span>
    <span class="spacer" style="flex:1"></span>
    <span id="footer-result" style="font-size:10px"></span>
  </div>
</div>

<div id="help-overlay">
  <div id="help-panel">
    <h3>Keyboard Shortcuts</h3>
    <div class="help-grid">
      <kbd>/</kbd><span>Focus search</span>
      <kbd>j</kbd><span>Next conversation</span>
      <kbd>k</kbd><span>Previous conversation</span>
      <kbd>n</kbd><span>Next page</span>
      <kbd>p</kbd><span>Previous page</span>
      <kbd>Enter</kbd><span>Open selected</span>
      <kbd>Esc</kbd><span>Clear search / close</span>
      <kbd>?</kbd><span>Toggle this help</span>
    </div>
    <div class="help-close" onclick="toggleHelp()">Press <kbd>Esc</kbd> or click to close</div>
  </div>
</div>

__BULK_PREVIEW_HTML__

<script>
var API = '';
var state = {
  conversations: [], selected: null, selectedRaw: null,
  provider: '', query: '', offset: 0, limit: 100, total: 0,
  status: {}, facets: null, inspectorTab: 'info',
  marks: {}, annotations: {}, savedViews: [], workspaces: [], userStateError: '',
  mode: 'single', stackPayload: null, comparePayload: null,
  // Bulk selection state (#1119). selection is a Set-like object keyed by
  // conversation_id. lastBulkResult holds the per-conversation envelope from
  // the most recent bulk operation: {succeeded:[ids], failed:[{id,reason}],
  // skipped:[{id,reason}], dryRun:bool, action:string}.
  bulkSelection: {}, lastBulkResult: null, bulkPending: null,
  // Cost panel cache (#1122). Keyed by conversation_id; populated on demand
  // when the Cost inspector tab is opened. ``undefined`` means "not loaded
  // yet", null/{error} means "fetch failed".
  costPanels: {}
};

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function escAttr(s) { return String(s).replace(/'/g,"\\'").replace(/"/g,'&quot;'); }

async function fetchJSON(url) {
  var r = await fetch(API + url);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}
async function sendJSON(url, method, body) {
  var opts = {method: method, headers: {'Content-Type': 'application/json'}};
  if (body !== undefined) opts.body = JSON.stringify(body);
  var r = await fetch(API + url, opts);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

function markSetFor(conversationId) {
  return state.marks[conversationId] || {};
}
function hasMark(conversationId, markType) {
  return !!markSetFor(conversationId)[markType];
}
function setMarkLocal(conversationId, markType, enabled) {
  if (!state.marks[conversationId]) state.marks[conversationId] = {};
  if (enabled) state.marks[conversationId][markType] = true;
  else delete state.marks[conversationId][markType];
}
function annotationsFor(conversationId) {
  return state.annotations[conversationId] || [];
}

function getConvIdFromURL() {
  var m = window.location.pathname.match(/^\/c\/(.+)$/);
  return m ? decodeURIComponent(m[1]) : null;
}
__WORKSPACE_JS__
window.addEventListener('popstate', function() {
  var route = getWorkspaceRouteFromURL();
  if (route) { loadWorkspaceRoute(route, false); return; }
  var cid = getConvIdFromURL();
  if (cid) selectConversation(cid, false);
  else { state.mode = 'single'; state.selected = null; state.selectedRaw = null; renderMain(); renderInspector(); renderConversations(); }
});

async function loadConversations() {
  var params = new URLSearchParams();
  params.set('limit', String(state.limit));
  params.set('offset', String(state.offset));
  if (state.provider) params.set('provider', state.provider);
  if (state.query) params.set('query', state.query);
  try {
    var data = await fetchJSON('/api/conversations?' + params);
    state.conversations = data.items || [];
    state.total = data.total || 0;
    document.getElementById('footer-result').textContent =
      (state.total > 0) ? (state.total + ' results') : '';
  } catch(e) {
    state.conversations = [];
    state.total = 0;
    renderSidebarState('error', 'Failed to load conversations');
  }
  renderConversations();
}

async function loadUserState() {
  try {
    var marks = await fetchJSON('/api/user/marks');
    state.marks = {};
    (marks.items || []).forEach(function(m) {
      setMarkLocal(m.conversation_id, m.mark_type, true);
    });
    var annotations = await fetchJSON('/api/user/annotations');
    state.annotations = {};
    (annotations.items || []).forEach(function(a) {
      if (!state.annotations[a.conversation_id]) state.annotations[a.conversation_id] = [];
      state.annotations[a.conversation_id].push(a);
    });
    var savedViews = await fetchJSON('/api/user/saved-views');
    state.savedViews = savedViews.items || [];
    var workspaces = await fetchJSON('/api/user/workspaces');
    state.workspaces = workspaces.items || [];
    state.userStateError = '';
  } catch(e) {
    state.userStateError = 'User state unavailable';
  }
  renderConversations();
  renderMain();
  renderInspector();
}

async function loadConversation(id, updateURL) {
  state.mode = 'single';
  state.stackPayload = null;
  state.comparePayload = null;
  if (updateURL !== false) pushSingleURL(id);
  try {
    var data = await fetchJSON('/api/conversations/' + id);
    state.selected = data;
  } catch(e) { state.selected = null; }
  renderMain();
  renderInspector();
  renderConversations();
}

async function loadConversationRaw(id) {
  try {
    var data = await fetchJSON('/api/conversations/' + id + '/raw');
    state.selectedRaw = data;
  } catch(e) { state.selectedRaw = null; }
}

async function loadFacets() {
  var params = new URLSearchParams();
  if (state.query) params.set('query', state.query);
  if (state.provider) params.set('provider', state.provider);
  var qs = params.toString();
  try { state.facets = await fetchJSON('/api/facets' + (qs ? '?' + qs : '')); renderFacets(); } catch(e) {}
}

async function loadStatus() {
  try {
    var h = await fetchJSON('/api/health');
    var dot = document.getElementById('status-dot');
    dot.className = 'dot ' + (h.ok ? 'ok' : 'err');
    document.getElementById('status-label').textContent = h.ok ? 'healthy' : (h.quick_check || 'issues');
    var dbGB = ((h.db_size_bytes || 0) / 1073741824).toFixed(1);
    document.getElementById('status-db').textContent = 'DB: ' + dbGB + ' GB';
  } catch(e) {
    document.getElementById('status-dot').className = 'dot err';
    document.getElementById('status-label').textContent = 'offline';
  }
  try {
    var s = await fetchJSON('/api/status');
    document.getElementById('status-convs').textContent = (s.total_conversations || 0).toLocaleString() + ' convs';
    document.getElementById('status-msgs').textContent = (s.total_messages || 0).toLocaleString() + ' msgs';
    renderFtsChip(s.fts_readiness || {});
    renderInsightChip(s.insight_freshness || {});
    var ingestEl = document.getElementById('status-ingest');
    if (s.live && s.live.existing_source_count > 0) {
      ingestEl.style.display = '';
      ingestEl.textContent = 'live: ' + s.live.existing_source_count + ' srcs';
      setChipQuality(ingestEl, 'canonical');
    } else { ingestEl.style.display = 'none'; }
  } catch(e) {}
}

// Apply an MK3 data-quality class to a chip element (canonical, partial, stale,
// unavailable, etc.). Strips any prior q-* class so callers can freely flip.
function setChipQuality(el, quality) {
  if (!el) return;
  el.className = el.className.split(' ').filter(function(c) { return c.indexOf('q-') !== 0; }).join(' ');
  if (quality) el.classList.add('q-' + quality);
}

function renderFtsChip(fts) {
  var el = document.getElementById('status-fts');
  var msgReady = !!fts.messages_ready;
  var actReady = !!fts.action_events_ready;
  var label;
  var quality;
  if (msgReady && actReady) { label = 'FTS: ok'; quality = 'canonical'; }
  else if (msgReady || actReady) { label = 'FTS: partial'; quality = 'partial'; }
  else { label = 'FTS: unavailable'; quality = 'unavailable'; }
  el.textContent = label;
  setChipQuality(el, quality);
}

function renderInsightChip(freshness) {
  var el = document.getElementById('status-insights');
  var total = freshness.total_sessions || 0;
  var withProfiles = freshness.sessions_with_profiles || 0;
  if (total <= 0) { el.style.display = 'none'; return; }
  el.style.display = '';
  if (withProfiles >= total) { el.textContent = 'insights: ok'; setChipQuality(el, 'canonical'); }
  else if (withProfiles > 0) { el.textContent = 'insights: ' + withProfiles + '/' + total; setChipQuality(el, 'partial'); }
  else { el.textContent = 'insights: stale'; setChipQuality(el, 'stale'); }
}

function renderSidebarState(kind, msg) {
  var icons = {empty: '\u25a1', noresults: '\u25a2', error: '\u2715', loading: '\u2014'};
  document.getElementById('conv-list').innerHTML =
    '<div class="sidebar-state"><div class="state-icon">' + (icons[kind] || '') + '</div>' + esc(msg) + '</div>';
}

function renderConversations() {
  var el = document.getElementById('conv-list');
  var items = state.conversations;
  if (!items || !items.length) {
    // Distinguish empty archive from filtered-no-results so the operator
    // knows whether to ingest or to clear filters. Preserve filter context
    // in the empty-state message per MK3 state matrix.
    if (state.query && state.provider) {
      renderSidebarState('noresults', 'No results for query=' + state.query + ' provider=' + state.provider + '. Press Esc to clear.');
    } else if (state.query) {
      renderSidebarState('noresults', 'No results for query=' + state.query + '. Press Esc to clear.');
    } else if (state.provider) {
      renderSidebarState('noresults', 'No conversations from provider=' + state.provider + '. Press Esc to clear.');
    } else if (state.total === 0) {
      renderSidebarState('empty', 'No conversations in archive. Run `polylogued run` to ingest sources.');
    } else {
      renderSidebarState('noresults', 'No conversations on this page');
    }
    return;
  }
  el.innerHTML = items.map(function(c) {
    var sel = state.selected && state.selected.id === c.id ? ' selected' : '';
    var bulkSel = isBulkSelected(c.id) ? ' bulk-selected' : '';
    var title = esc((c.title || 'Untitled').substring(0, 100));
    var date = c.date ? new Date(c.date).toLocaleDateString() : (c.created_at ? new Date(c.created_at).toLocaleDateString() : '');
    var p = c.provider || 'unknown';
    var dotColor = 'var(--provider-' + p.replace(/_/g, '-') + ', var(--text-dim))';
    var flagsHtml = '';
    if (c.flags) {
      if (c.flags.has_tool_use) flagsHtml += '<span class="flag tool">T</span>';
      if (c.flags.has_thinking) flagsHtml += '<span class="flag think">R</span>';
      if (c.flags.has_paste) flagsHtml += '<span class="flag">P</span>';
    }
    if (hasMark(c.id, 'star')) flagsHtml += '<span class="flag mark" title="Starred">*</span>';
    if (hasMark(c.id, 'pin')) flagsHtml += '<span class="flag mark" title="Pinned">P</span>';
    if (hasMark(c.id, 'archive')) flagsHtml += '<span class="flag mark" title="Archived">A</span>';
    var repoHtml = c.repo ? '<span class="chip" style="font-size:10px;padding:0 4px">' + esc(c.repo.split('/').pop()) + '</span>' : '';
    var checked = isBulkSelected(c.id) ? ' checked' : '';
    return '<div class="conv-item' + sel + bulkSel + '" data-id="' + escAttr(c.id) + '">'
      + '<div class="conv-row">'
      + '<input type="checkbox" class="bulk-check" data-bulk-id="' + escAttr(c.id) + '" aria-label="Select conversation"' + checked + '>'
      + '<div class="conv-body" onclick="selectConversation(\'' + escAttr(c.id) + '\')">'
      + '<div class="conv-title">' + title + '</div>'
      + '<div class="conv-meta">'
      + '<span class="provider-dot" style="background:' + dotColor + '"></span>'
      + '<span>' + esc(p) + '</span>'
      + '<span>' + date + '</span>'
      + '<span>' + (c.message_count || 0) + ' msgs</span>'
      + flagsHtml + repoHtml
      + '</div></div></div></div>';
  }).join('');
  renderBulkToolbar();
}

__BULK_JS__

function renderFacets() {
  var f = state.facets;
  if (!f) { document.getElementById('facet-bar').innerHTML = ''; return; }
  var html = '';
  var providers = f.providers || {};
  var provKeys = Object.keys(providers);
  if (provKeys.length > 0) {
    html += '<div class="facet-group"><div class="facet-group-label">Providers</div><div class="facet-chips">';
    html += '<span class="facet-chip' + (!state.provider ? ' active' : '') + '" data-facet="provider" data-value="">All</span>';
    provKeys.sort(function(a,b) { return providers[b] - providers[a]; }).slice(0, 8).forEach(function(p) {
      var active = state.provider === p ? ' active' : '';
      html += '<span class="facet-chip' + active + '" data-facet="provider" data-value="' + escAttr(p) + '">'
        + esc(p) + '<span class="count">' + providers[p] + '</span></span>';
    });
    html += '</div></div>';
  }
  var tags = f.tags || {};
  var tagKeys = Object.keys(tags);
  if (tagKeys.length > 0) {
    html += '<div class="facet-group"><div class="facet-group-label">Tags</div><div class="facet-chips">';
    tagKeys.sort(function(a,b) { return tags[b] - tags[a]; }).slice(0, 10).forEach(function(t) {
      html += '<span class="facet-chip" data-facet="tag" data-value="' + escAttr(t) + '">'
        + esc(t) + '<span class="count">' + tags[t] + '</span></span>';
    });
    html += '</div></div>';
  }
  var flags = f.has_flags || {};
  if (Object.keys(flags).length > 0) {
    html += '<div class="facet-group"><div class="facet-group-label">Flags</div><div class="facet-chips">';
    Object.keys(flags).forEach(function(fl) {
      if (flags[fl] > 0) {
        html += '<span class="facet-chip" data-facet="flag" data-value="' + escAttr(fl) + '">'
          + esc(fl.replace('has_', '')) + '<span class="count">' + flags[fl] + '</span></span>';
      }
    });
    html += '</div></div>';
  }
  document.getElementById('facet-bar').innerHTML = html;
}

function renderMain() {
  renderWorkspaceToolbar();
  if (state.mode === 'stack') { renderStackWorkspace(); return; }
  if (state.mode === 'compare') { renderCompareWorkspace(); return; }
  var headerEl = document.getElementById('conv-header');
  var msgEl = document.getElementById('msg-list');
  if (!state.selected) {
    headerEl.innerHTML = '<h2>Polylogue</h2><div class="conv-stats"></div>';
    msgEl.innerHTML = '<div class="main-empty"><h3>Select a conversation</h3>'
      + '<p>Browse from the list or use <span class="kbd">/</span> to search. Press <span class="kbd">?</span> for shortcuts.</p></div>';
    return;
  }
  var c = state.selected;
  var title = esc(c.display_title || c.title || 'Untitled');
  var headerHtml = '<div class="title-row"><h2>' + title + '</h2><div class="mark-actions">'
    + markButtonHtml(c.id, 'star', '*', 'Toggle star')
    + markButtonHtml(c.id, 'pin', 'P', 'Toggle pin')
    + markButtonHtml(c.id, 'archive', 'A', 'Toggle archive')
    + '</div></div><div class="conv-stats">';
  // MK3 header chip order (docs/design/mk3/docs/11-little-details.md):
  // 1. provider/source  2. live/stale  3. repo/cwd/branch  4. counts
  // 5. cost/tokens  6. derived/insight  7. marks/tags
  // 1. provider/source
  if (c.provider) headerHtml += '<span class="chip q-canonical">' + esc(c.provider) + '</span>';
  if (c.model) headerHtml += '<span class="chip">' + esc(String(c.model)) + '</span>';
  // 2. live/stale (placeholder — wired from session provenance once #1019 surfaces it on detail)
  if (c.stale) headerHtml += '<span class="chip q-stale" title="Derived view is stale">stale</span>';
  // 3. repo/cwd/branch
  if (c.repo) headerHtml += '<span class="chip repo" title="' + escAttr(c.repo) + '">' + esc(c.repo.split('/').pop() || c.repo) + '</span>';
  if (c.cwd_display) headerHtml += '<span class="chip" title="' + escAttr(c.cwd_display) + '">' + esc(c.cwd_display.split('/').pop() || c.cwd_display) + '</span>';
  if (c.branch_type && c.branch_type !== 'main') {
    headerHtml += '<span class="chip q-inferred" title="Branch type">' + esc(c.branch_type) + '</span>';
  }
  // 4. counts
  if (c.message_count !== undefined) headerHtml += '<span>' + c.message_count + ' messages</span>';
  if (c.word_count) headerHtml += '<span>' + c.word_count.toLocaleString() + ' words</span>';
  if (c.created_at) headerHtml += '<span>' + new Date(c.created_at).toLocaleDateString() + '</span>';
  // 4b. content-shape flags
  if (c.flags) {
    if (c.flags.has_tool_use) headerHtml += '<span class="chip accent">tool use</span>';
    if (c.flags.has_thinking) headerHtml += '<span class="chip accent">thinking</span>';
    if (c.flags.has_paste) headerHtml += '<span class="chip accent">paste</span>';
  }
  // 5. cost/tokens (surface when present; estimated → q-estimated quality)
  if (c.cost_usd !== undefined && c.cost_usd !== null) {
    var costCls = c.cost_estimated ? ' q-estimated' : '';
    headerHtml += '<span class="chip' + costCls + '" title="Session cost (USD)">$' + Number(c.cost_usd).toFixed(2) + '</span>';
  }
  if (c.token_count) {
    headerHtml += '<span class="chip">' + Number(c.token_count).toLocaleString() + ' tok</span>';
  }
  // 6. derived/insight availability — if session profile is missing show unavailable chip
  if (c.insight_status === 'missing') headerHtml += '<span class="chip q-unavailable" title="Session insight not computed">no insights</span>';
  else if (c.insight_status === 'stale') headerHtml += '<span class="chip q-stale" title="Session insight is stale">insights stale</span>';
  // 7. marks/tags
  if (c.tags && c.tags.length) {
    c.tags.forEach(function(t) { headerHtml += '<span class="chip">' + esc(t) + '</span>'; });
  }
  headerHtml += '</div>';
  headerEl.innerHTML = headerHtml;

  if (!c.messages) {
    msgEl.innerHTML = '<div class="main-empty"><h3>Loading messages...</h3></div>';
    return;
  }
  if (c.messages.length === 0) {
    msgEl.innerHTML = '<div class="main-empty"><h3>No messages</h3><p>This conversation has no message content.</p></div>';
    return;
  }
  msgEl.innerHTML = messageBlocksHtml(c.messages);
}

function messageBlocksHtml(messages) {
  return (messages || []).map(function(m, idx) {
    var role = (m.role || '').toLowerCase();
    var text = m.text || '';
    var isTool = role === 'tool' || m.message_type === 'tool_use' || m.message_type === 'tool_result' || m.has_tool_use;
    var tsHtml = m.timestamp ? '<span class="msg-ts" title="' + esc(m.timestamp) + '">' + new Date(m.timestamp).toLocaleTimeString() + '</span>' : '';
    var typeTag = (m.message_type && m.message_type !== 'message') ? '<span class="msg-type">' + esc(m.message_type) + '</span>' : '';
    var textClass = 'msg-text';
    var textLen = text.length;
    var collapsed = textLen > 2000;
    if (collapsed) textClass += ' collapsed';
    var roleClass = 'msg-role ' + role;
    var blockClass = 'msg-block';
    if (isTool) blockClass += ' tool-block';
    var toolSummaryHtml = '';
    if (isTool && textLen > 200) {
      var lines = text.split('\n');
      var cmdLine = lines[0] || '';
      if (cmdLine.length > 120) cmdLine = cmdLine.substring(0, 120) + '...';
      toolSummaryHtml = '<div class="tool-summary" onclick="var t=this.parentElement.querySelector(\'.msg-text\');t.classList.toggle(\'collapsed\');this.style.display=\'none\'">'
        + esc('tool: ' + (role || 'tool')) + ' <code>' + esc(cmdLine) + '</code></div>';
    }
    return '<div class="' + blockClass + '" id="msg-' + idx + '">'
      + '<div class="msg-header"><span class="' + roleClass + '">' + esc(role || '?') + '</span>' + typeTag + tsHtml + '</div>'
      + toolSummaryHtml
      + (text ? '<div class="' + textClass + '">' + esc(text) + '</div>' : '')
      + (collapsed ? '<div class="msg-expand" onclick="var t=this.previousElementSibling;t.classList.remove(\'collapsed\');this.remove();var ts=this.parentElement.querySelector(\'.tool-summary\');if(ts)ts.style.display=\'none\'">Show all (' + textLen.toLocaleString() + ' chars)</div>' : '')
      + '</div>';
  }).join('');
}

function markButtonHtml(conversationId, markType, label, title) {
  var active = hasMark(conversationId, markType) ? ' active' : '';
  return '<button class="mark-btn' + active + '" title="' + escAttr(title) + '" onclick="toggleMark(\'' + escAttr(markType) + '\')">' + esc(label) + '</button>';
}

function renderInspector() {
  var el = document.getElementById('inspector-content');
  if (!state.selected) { el.innerHTML = '<div class="inspector-empty">Select a conversation to inspect</div>'; return; }
  var c = state.selected;
  var tab = state.inspectorTab || 'info';
  if (tab === 'info') renderInspectorInfo(el, c);
  else if (tab === 'cost') renderInspectorCost(el, c);
  else if (tab === 'raw') renderInspectorRaw(el, c);
  else if (tab === 'notes') renderInspectorNotes(el, c);
}

// --- Cost panel (#1122) --------------------------------------------------
// Loads /api/conversations/{id}/cost on demand and caches per-conversation
// in state.costPanels. Each visible number carries a confidence chip
// driven by the MK3 q-* vocabulary returned by the daemon (q-canonical /
// q-estimated / q-heuristic / q-unavailable).
async function loadCostPanel(id) {
  try {
    var data = await fetchJSON('/api/conversations/' + encodeURIComponent(id) + '/cost');
    state.costPanels[id] = data;
  } catch(e) {
    state.costPanels[id] = {error: String(e)};
  }
  if (state.selected && state.selected.id === id && state.inspectorTab === 'cost') {
    renderInspector();
  }
}

function formatUsd(value) {
  var n = Number(value || 0);
  if (n === 0) return '$0.00';
  if (n < 0.01) return '$' + n.toFixed(4);
  if (n < 1) return '$' + n.toFixed(3);
  return '$' + n.toFixed(2);
}

function costChip(label, tag) {
  return '<span class="chip ' + esc(tag) + '" title="confidence: ' + esc(tag) + '">' + esc(label) + '</span>';
}

function renderInspectorCost(el, c) {
  var cost = state.costPanels[c.id];
  if (cost === undefined) {
    el.innerHTML = '<div class="inspector-empty">Loading cost...</div>';
    loadCostPanel(c.id);
    return;
  }
  if (cost && cost.error) {
    el.innerHTML = '<div class="inspector-empty">Cost surface unavailable</div>';
    return;
  }
  var tag = cost.confidence_tag || 'q-unavailable';
  var status = cost.status || 'unavailable';
  var html = '';
  html += '<div class="inspector-section"><h4>Total</h4>';
  html += '<div class="inspector-field"><span class="label">Cost</span>'
    + '<span class="value">' + esc(formatUsd(cost.total_usd)) + ' ' + costChip(status, tag) + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Confidence</span>'
    + '<span class="value">' + esc((cost.confidence != null ? cost.confidence.toFixed(2) : '0.00')) + '</span></div>';
  if (cost.model_name) {
    html += '<div class="inspector-field"><span class="label">Model</span>'
      + '<span class="value">' + esc(cost.model_name) + '</span></div>';
  }
  if (cost.unavailable_reason) {
    html += '<div class="inspector-field"><span class="label">Reason</span>'
      + '<span class="value">' + esc(cost.unavailable_reason) + '</span></div>';
  }
  html += '</div>';

  // Basis split (#1136). Each axis is independent and never collapsed.
  var basis = cost.basis || {};
  var basisAxes = [
    ['provider_reported_usd', 'Provider-reported', 'q-canonical'],
    ['api_equivalent_usd', 'API equivalent', 'q-estimated'],
    ['subscription_equivalent_usd', 'Subscription equiv.', 'q-heuristic'],
    ['catalog_priced_usd', 'Catalog-priced', 'q-estimated'],
    ['tool_surcharge_usd', 'Tool surcharge', 'q-partial']
  ];
  var basisHasAny = basisAxes.some(function(row) { return Number(basis[row[0]] || 0) > 0; });
  if (basisHasAny) {
    html += '<div class="inspector-section"><h4>Basis split</h4>';
    basisAxes.forEach(function(row) {
      var amt = Number(basis[row[0]] || 0);
      if (amt === 0) return;
      html += '<div class="inspector-field"><span class="label">' + esc(row[1]) + '</span>'
        + '<span class="value">' + esc(formatUsd(amt)) + ' ' + costChip(row[2].replace('q-', ''), row[2]) + '</span></div>';
    });
    html += '</div>';
  }

  // Per-model breakdown (#1136). Sessions that mix models surface one row per model.
  if (cost.per_model_breakdown && cost.per_model_breakdown.length) {
    html += '<div class="inspector-section"><h4>Per-model</h4>';
    cost.per_model_breakdown.forEach(function(entry) {
      var name = entry.model_name || entry.normalized_model || 'unknown';
      html += '<div class="inspector-field"><span class="label">' + esc(name) + '</span>'
        + '<span class="value">' + esc(formatUsd(entry.total_usd)) + '</span></div>';
    });
    html += '</div>';
  }

  // Usage breakdown.
  var usage = cost.usage || {};
  if (usage.total_tokens) {
    html += '<div class="inspector-section"><h4>Tokens</h4>';
    [['input_tokens', 'Input'], ['output_tokens', 'Output'],
     ['cache_read_tokens', 'Cache read'], ['cache_write_tokens', 'Cache write'],
     ['total_tokens', 'Total']].forEach(function(row) {
      var v = Number(usage[row[0]] || 0);
      if (v === 0 && row[0] !== 'total_tokens') return;
      html += '<div class="inspector-field"><span class="label">' + esc(row[1]) + '</span>'
        + '<span class="value">' + esc(v.toLocaleString()) + '</span></div>';
    });
    html += '</div>';
  }

  if (cost.missing_reasons && cost.missing_reasons.length) {
    html += '<div class="inspector-section"><h4>Missing</h4>';
    cost.missing_reasons.forEach(function(r) {
      html += '<div class="inspector-field"><span class="label">&mdash;</span>'
        + '<span class="value">' + esc(r) + '</span></div>';
    });
    html += '</div>';
  }
  el.innerHTML = html;
}

function renderInspectorInfo(el, c) {
  var fields = [
    ['ID', c.id], ['Provider', c.provider], ['Model', c.model],
    ['Created', c.created_at ? new Date(c.created_at).toLocaleString() : ''],
    ['Updated', c.updated_at ? new Date(c.updated_at).toLocaleString() : ''],
    ['Messages', c.message_count], ['Words', (c.word_count || 0).toLocaleString()],
    ['Repo', c.repo], ['CWD', c.cwd_display], ['Branch', c.branch_type], ['Session', c.session_id]
  ];
  var html = '';
  fields.forEach(function(f) {
    var val = f[1] != null ? String(f[1]) : '';
    html += '<div class="inspector-field"><span class="label">' + esc(f[0]) + '</span>'
      + '<span class="value' + (val ? '' : ' empty') + '">' + esc(val || '\u2014') + '</span></div>';
  });
  if (c.summary) {
    html += '<div class="inspector-section"><h4>Summary</h4>'
      + '<div style="font-size:var(--small);color:var(--text-muted);line-height:1.5">' + esc(String(c.summary)) + '</div></div>';
  }
  if (c.tags && c.tags.length) {
    html += '<div class="inspector-section"><h4>Tags</h4>'
      + c.tags.map(function(t) { return '<span class="facet-chip" style="display:inline-block;margin:1px">' + esc(t) + '</span>'; }).join('')
      + '</div>';
  }
  if (c.flags) {
    html += '<div class="inspector-section"><h4>Flags</h4>' + JSON.stringify(c.flags) + '</div>';
  }
  el.innerHTML = html;
}

function renderInspectorRaw(el, c) {
  var html = '<div class="inspector-section"><h4>Provenance</h4>';
  html += '<div class="inspector-field"><span class="label">Provider</span><span class="value">' + esc(c.provider || '-') + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Branch</span><span class="value">' + esc(c.branch_type || 'main') + '</span></div>';
  html += '<div class="inspector-field"><span class="label">Parent</span><span class="value">' + esc(c.parent_id || '-') + '</span></div>';
  html += '</div><div class="inspector-section"><h4>Raw Artifacts</h4>';
  html += '<button style="background:var(--panel-elevated);border:1px solid var(--border);color:var(--accent);padding:4px 10px;border-radius:3px;cursor:pointer;font-size:var(--small)" onclick="loadRawData()">Load raw data</button>';
  html += '<div id="raw-data-area"></div></div>';
  el.innerHTML = html;
}

function renderInspectorNotes(el, c) {
  var marks = Object.keys(markSetFor(c.id));
  var querySummary = [];
  if (state.query) querySummary.push('query=' + state.query);
  if (state.provider) querySummary.push('provider=' + state.provider);
  var html = '<div class="inspector-section"><h4>Marks</h4>';
  if (marks.length) {
    html += '<div class="user-state-row"><span class="label">Active</span><span class="value">' + esc(marks.sort().join(', ')) + '</span></div>';
  } else {
    html += '<div class="inspector-empty">No marks on this conversation</div>';
  }
  html += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:8px">'
    + '<button class="user-action" onclick="toggleMark(\'star\')">Star</button>'
    + '<button class="user-action" onclick="toggleMark(\'pin\')">Pin</button>'
    + '<button class="user-action" onclick="toggleMark(\'archive\')">Archive</button>'
    + '</div></div>';
  html += '<div class="inspector-section"><h4>Saved Views</h4>'
    + '<div class="user-state-row"><span class="label">Current</span><span class="value">' + esc(querySummary.join(' / ') || 'all conversations') + '</span></div>'
    + '<button class="user-action" onclick="saveCurrentView()">Save current view</button>';
  if (state.savedViews.length) {
    html += '<div class="saved-view-list" style="margin-top:8px">';
    state.savedViews.forEach(function(v) {
      var q = v.query || {};
      var bits = [];
      if (q.query) bits.push('query=' + q.query);
      if (q.provider) bits.push('provider=' + q.provider);
      html += '<div class="saved-view-item" data-view-id="' + escAttr(v.view_id) + '"><div><div>' + esc(v.name || v.view_id) + '</div>'
        + '<div class="value">' + esc(bits.join(' / ') || 'all conversations') + '</div></div>'
        + '<div style="display:flex;gap:4px;flex-shrink:0">'
        + '<button class="user-action" onclick="applySavedView(\'' + escAttr(v.view_id) + '\')">Open</button>'
        + '<button class="user-action" title="Delete saved view" onclick="deleteSavedView(\'' + escAttr(v.view_id) + '\')">Delete</button>'
        + '</div></div>';
    });
    html += '</div>';
  } else {
    html += '<div class="inspector-empty">No saved views. Click "Save current view" to name the current filter chain.</div>';
  }
  if (state.userStateError) {
    html += '<div class="inspector-empty">' + esc(state.userStateError) + '</div>';
  }
  html += '</div><div class="inspector-section"><h4>Annotations</h4>'
    + '<button class="user-action" onclick="saveAnnotation()">Add note</button>';
  var annotations = annotationsFor(c.id);
  if (annotations.length) {
    html += '<div class="annotation-list">';
    annotations.forEach(function(a) {
      var target = a.target_type === 'message' ? ('message ' + (a.message_id || a.target_id)) : 'conversation';
      html += '<div class="annotation-item">'
        + '<div class="meta">' + esc(target) + '</div>'
        + '<div class="note">' + esc(a.note_text || '') + '</div>'
        + '<div class="annotation-actions">'
        + '<button class="user-action" onclick="editAnnotation(\'' + escAttr(a.annotation_id) + '\')">Edit</button>'
        + '<button class="user-action" onclick="deleteAnnotation(\'' + escAttr(a.annotation_id) + '\')">Delete</button>'
        + '</div></div>';
    });
    html += '</div>';
  } else {
    html += '<div class="inspector-empty">No annotations on this conversation</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

async function toggleMark(markType) {
  if (!state.selected) return;
  var id = state.selected.id;
  var enabled = hasMark(id, markType);
  try {
    if (enabled) {
      await sendJSON('/api/user/marks?conversation_id=' + encodeURIComponent(id) + '&mark_type=' + encodeURIComponent(markType), 'DELETE');
      setMarkLocal(id, markType, false);
    } else {
      await sendJSON('/api/user/marks', 'POST', {conversation_id: id, mark_type: markType});
      setMarkLocal(id, markType, true);
    }
    state.userStateError = '';
  } catch(e) {
    state.userStateError = 'Failed to update mark';
  }
  renderConversations();
  renderMain();
  renderInspector();
}

function applySavedView(viewId) {
  var view = state.savedViews.find(function(v) { return v.view_id === viewId; });
  if (!view) return;
  var query = view.query || {};
  state.query = query.query || '';
  state.provider = query.provider || '';
  state.offset = 0;
  document.getElementById('search').value = state.query;
  loadConversations();
  loadFacets();
  renderInspector();
}

function findAnnotation(annotationId) {
  var cid = state.selected ? state.selected.id : '';
  var annotations = annotationsFor(cid);
  return annotations.find(function(a) { return a.annotation_id === annotationId; }) || null;
}

async function saveAnnotation(annotationId) {
  if (!state.selected) return;
  var existing = annotationId ? findAnnotation(annotationId) : null;
  var note = window.prompt('Annotation note', existing ? existing.note_text : '');
  if (!note) return;
  var id = annotationId || ('annotation-' + Date.now().toString(36));
  try {
    await sendJSON('/api/user/annotations', 'POST', {
      annotation_id: id,
      conversation_id: state.selected.id,
      note_text: note
    });
    await loadUserState();
  } catch(e) {
    state.userStateError = 'Failed to save annotation';
    renderInspector();
  }
}

function editAnnotation(annotationId) {
  saveAnnotation(annotationId);
}

async function deleteAnnotation(annotationId) {
  if (!annotationId) return;
  try {
    await sendJSON('/api/user/annotations/' + encodeURIComponent(annotationId), 'DELETE');
    await loadUserState();
  } catch(e) {
    state.userStateError = 'Failed to delete annotation';
    renderInspector();
  }
}

async function loadRawData() {
  if (!state.selected) return;
  var id = state.selected.id;
  var area = document.getElementById('raw-data-area');
  area.innerHTML = '<div style="color:var(--text-dim);font-size:var(--small);padding:8px 0">Loading...</div>';
  try {
    await loadConversationRaw(id);
    var raw = state.selectedRaw;
    if (!raw) { area.innerHTML = '<div class="inspector-empty">No raw data available</div>'; return; }
    var html = '';
    if (raw.provider_meta && Object.keys(raw.provider_meta).length > 0) {
      html += '<div class="inspector-section"><h4>Provider Meta</h4>'
        + '<div class="raw-block">' + esc(JSON.stringify(raw.provider_meta, null, 2)) + '</div></div>';
    }
    var artifacts = raw.raw_artifacts || [];
    if (artifacts.length > 0) {
      html += '<div class="inspector-section"><h4>Raw Artifacts (' + artifacts.length + ')</h4>';
      artifacts.forEach(function(a) {
        var summary = a.source_path || a.filename || a.name || JSON.stringify(a).substring(0, 80);
        html += '<div class="raw-block" style="margin-bottom:4px">' + esc(summary) + '</div>';
      });
      html += '</div>';
    } else { html += '<div class="inspector-empty" style="padding-top:8px">No raw artifacts</div>'; }
    area.innerHTML = html;
  } catch(e) { area.innerHTML = '<div class="inspector-empty">Failed to load raw data</div>'; }
}

async function selectConversation(id, updateURL) {
  document.getElementById('msg-list').innerHTML = '<div class="main-empty"><h3>Loading...</h3></div>';
  document.getElementById('inspector-content').innerHTML = '<div class="inspector-empty">Loading...</div>';
  await loadConversation(id, updateURL);
}

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if (e.key === 'Escape') { e.target.blur(); e.preventDefault(); }
    return;
  }
  if (e.key === '/') { e.preventDefault(); document.getElementById('search').focus(); return; }
  if (e.key === '?') { e.preventDefault(); toggleHelp(); return; }
  if (e.key === 'Escape') {
    e.preventDefault();
    var help = document.getElementById('help-overlay');
    if (help.classList.contains('visible')) { toggleHelp(); return; }
    if (state.query) { state.query = ''; document.getElementById('search').value = ''; state.offset = 0; loadConversations(); return; }
    if (state.provider) { state.provider = ''; state.offset = 0; loadConversations(); renderFacets(); return; }
    return;
  }
  if (e.key === 'j' || e.key === 'k') {
    e.preventDefault();
    var items = document.querySelectorAll('.conv-item');
    if (!items.length) return;
    var sel = document.querySelector('.conv-item.selected');
    var idx = sel ? Array.from(items).indexOf(sel) : (e.key === 'j' ? -1 : items.length);
    var next = e.key === 'j' ? idx + 1 : idx - 1;
    if (next >= 0 && next < items.length) { items[next].click(); items[next].scrollIntoView({block: 'nearest'}); }
  }
  if (e.key === 'n') {
    e.preventDefault();
    if (state.offset + state.limit < state.total) { state.offset += state.limit; loadConversations(); }
  }
  if (e.key === 'p') {
    e.preventDefault();
    if (state.offset > 0) { state.offset = Math.max(0, state.offset - state.limit); loadConversations(); }
  }
});

function toggleHelp() {
  document.getElementById('help-overlay').classList.toggle('visible');
}
document.getElementById('help-overlay').addEventListener('click', toggleHelp);
document.getElementById('help-btn').addEventListener('click', function(e) { e.stopPropagation(); toggleHelp(); });

var searchTimer;
document.getElementById('search').addEventListener('input', function(e) {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(function() { state.query = e.target.value; state.offset = 0; loadConversations(); loadFacets(); }, 280);
});

document.getElementById('facet-bar').addEventListener('click', function(e) {
  var chip = e.target.closest('.facet-chip');
  if (!chip) return;
  var facet = chip.dataset.facet;
  var value = chip.dataset.value;
  if (facet === 'provider') { state.provider = value || ''; state.offset = 0; loadConversations(); renderFacets(); }
});

attachBulkHandlers();

document.getElementById('inspector-tabs').addEventListener('click', function(e) {
  if (e.target.tagName !== 'BUTTON') return;
  state.inspectorTab = e.target.dataset.tab;
  document.querySelectorAll('#inspector-tabs button').forEach(function(b) { b.classList.remove('active'); });
  e.target.classList.add('active');
  renderInspector();
});

loadConversations().then(function() {
  var route = getWorkspaceRouteFromURL();
  if (route) { loadWorkspaceRoute(route, false); return; }
  var cid = getConvIdFromURL();
  if (cid) selectConversation(cid, false);
});
loadFacets();
loadUserState();
loadStatus();

// --- Realtime channel ----------------------------------------------------
// Subscribe to /api/events (SSE) when available. On ingest/reset events the
// conversation list, facets, and status chips reload. EventSource handles
// reconnects automatically; on persistent failure we fall back to polling.
var realtime = {
  source: null,
  lastEventId: 0,
  pollTimer: null,
  refreshTimer: null,
  status: 'connecting'
};

function setLiveChip(status, lastSeen) {
  var el = document.getElementById('status-live');
  if (!el) return;
  el.className = 'chip' + (status === 'live' ? ' accent' : '');
  var label = 'live: ' + status;
  if (lastSeen) label += ' \u00b7 #' + lastSeen;
  el.textContent = label;
}

function scheduleRefresh() {
  if (realtime.refreshTimer) return;
  realtime.refreshTimer = setTimeout(function() {
    realtime.refreshTimer = null;
    loadConversations();
    loadFacets();
    loadStatus();
  }, 250);
}

function handleRealtimeEvent(payload) {
  if (!payload || typeof payload !== 'object') return;
  if (typeof payload.id === 'number') realtime.lastEventId = payload.id;
  setLiveChip('live', realtime.lastEventId);
  var kind = payload.kind || '';
  if (kind === 'ingestion_batch' || kind === 'ingest' || kind === 'reset' || kind === 'operation') {
    scheduleRefresh();
  }
}

function startPollingFallback() {
  if (realtime.pollTimer) return;
  setLiveChip('polling', realtime.lastEventId);
  realtime.pollTimer = setInterval(async function() {
    try {
      var data = await fetchJSON('/api/events?poll=1&since=' + realtime.lastEventId);
      var events = data.events || [];
      events.forEach(handleRealtimeEvent);
      if (typeof data.last_event_id === 'number') realtime.lastEventId = data.last_event_id;
      loadStatus();
    } catch(e) {
      setLiveChip('offline', realtime.lastEventId);
    }
  }, 5000);
}

function startRealtimeChannel() {
  if (typeof EventSource === 'undefined') { startPollingFallback(); return; }
  try {
    var url = '/api/events?since=' + realtime.lastEventId;
    realtime.source = new EventSource(url);
    setLiveChip('connecting', realtime.lastEventId);
    realtime.source.onopen = function() { setLiveChip('live', realtime.lastEventId); };
    var consumeMessage = function(e) {
      var data = null;
      try { data = JSON.parse(e.data); } catch(_) { return; }
      handleRealtimeEvent(data);
    };
    realtime.source.onmessage = consumeMessage;
    ['ingestion_batch', 'ingest', 'reset', 'operation'].forEach(function(kind) {
      realtime.source.addEventListener(kind, consumeMessage);
    });
    realtime.source.onerror = function() {
      setLiveChip('stale', realtime.lastEventId);
      // EventSource retries automatically; if it never reopens within 15s,
      // switch to polling fallback.
      setTimeout(function() {
        if (!realtime.source || realtime.source.readyState !== EventSource.OPEN) {
          try { realtime.source && realtime.source.close(); } catch(_) {}
          realtime.source = null;
          startPollingFallback();
        }
      }, 15000);
    };
  } catch(e) {
    startPollingFallback();
  }
}

startRealtimeChannel();
</script>
</body>
</html>""".replace("__WORKSPACE_CSS__", WORKSPACE_CSS)
    .replace("__WORKSPACE_HTML__", WORKSPACE_HTML)
    .replace("__WORKSPACE_JS__", WORKSPACE_JS)
    .replace("__BULK_CSS__", BULK_CSS)
    .replace("__BULK_TOOLBAR_HTML__", BULK_TOOLBAR_HTML)
    .replace("__BULK_PREVIEW_HTML__", BULK_PREVIEW_HTML)
    .replace("__BULK_JS__", BULK_JS)
)
