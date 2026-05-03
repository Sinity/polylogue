"""Polylogue MK2 web reader — single-page interactive archive cockpit."""

from __future__ import annotations

WEB_SHELL_HTML = r"""<!DOCTYPE html>
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
  --accent: #5AB8D6; --accent-soft: #2B6E84; --ok: #5FD7AE;
  --ok-bg: #0C2A24; --warn: #E6B450; --warn-bg: #2C220B;
  --err: #E86671; --err-bg: #2A1015; --active: #76A9FF;
  --role-user: #78B7FF; --role-assistant: #D6E2EA;
  --role-tool: #B7A6FF; --role-system: #A4B0BE;
  --role-thinking: #8F98A5;
  --provider-claude-code: #72D6A3; --provider-codex: #7EA7FF;
  --provider-chatgpt: #67D8C7; --provider-claude-ai: #D6A36B;
  --font-ui: Inter, ui-sans-serif, system-ui, sans-serif;
  --font-mono: JetBrains Mono, Fira Code, ui-monospace, monospace;
  --base: 13px; --small: 12px; --code: 12px; --lh: 1.45;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: var(--font-ui); font-size: var(--base); line-height: var(--lh); overflow: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

#app { display: grid; grid-template-columns: 320px 1fr 340px; grid-template-rows: 44px 1fr 28px; height: 100vh; }
#header { grid-column: 1/-1; grid-row: 1; display: flex; align-items: center;
  gap: 12px; padding: 0 16px; background: var(--bg-raised); border-bottom: 1px solid var(--border); }
#header h1 { font-size: 15px; font-weight: 600; color: var(--accent); letter-spacing: -0.3px; }
#header .stats { color: var(--text-muted); font-size: var(--small); }
#header .health { margin-left: auto; display: flex; align-items: center; gap: 6px; }
#header .health-dot { width: 6px; height: 6px; border-radius: 50%; }
#header .health-dot.ok { background: var(--ok); } #header .health-dot.err { background: var(--err); }

#sidebar { grid-column: 1; grid-row: 2; display: flex; flex-direction: column;
  background: var(--panel); border-right: 1px solid var(--border); overflow: hidden; }
#search-box { padding: 10px 12px; border-bottom: 1px solid var(--border); }
#search-box input { width: 100%; background: var(--panel-elevated); border: 1px solid var(--border);
  color: var(--text); padding: 6px 10px; border-radius: 4px; font-size: var(--base); outline: none; }
#search-box input:focus { border-color: var(--accent); }
#filter-bar { display: flex; gap: 2px; padding: 6px 12px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
#filter-bar button { background: var(--panel-elevated); border: 1px solid var(--border);
  color: var(--text-muted); padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: var(--small); }
#filter-bar button.active { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
#conv-list { flex: 1; overflow-y: auto; }
.conv-item { padding: 8px 12px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.1s; }
.conv-item:hover { background: var(--panel-elevated); }
.conv-item.selected { background: var(--panel-elevated); border-left: 2px solid var(--accent); }
.conv-item .conv-title { font-size: var(--base); color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-item .conv-meta { display: flex; gap: 8px; font-size: var(--small); color: var(--text-muted); margin-top: 2px; }
.provider-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 4px; }

#main { grid-column: 2; grid-row: 2; display: flex; flex-direction: column; overflow: hidden; background: var(--bg); }
#conv-header { padding: 12px 16px; border-bottom: 1px solid var(--border); background: var(--bg-raised); }
#conv-header h2 { font-size: 15px; font-weight: 500; }
#conv-header .conv-stats { font-size: var(--small); color: var(--text-muted); margin-top: 4px; }
#msg-list { flex: 1; overflow-y: auto; }
.msg-block { padding: 8px 16px; border-bottom: 1px solid var(--border); }
.msg-block .msg-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: var(--small); }
.msg-block .msg-role { font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
.msg-role.user { color: var(--role-user); } .msg-role.assistant { color: var(--role-assistant); }
.msg-role.tool { color: var(--role-tool); } .msg-role.system { color: var(--role-system); }
.msg-role.thinking { color: var(--role-thinking); }
.msg-block .msg-type { color: var(--text-dim); font-size: 11px; }
.msg-block .msg-text { font-family: var(--font-mono); font-size: var(--code); white-space: pre-wrap; word-break: break-word; max-height: 400px; overflow-y: auto; }
.msg-block .msg-text.collapsed { max-height: 120px; overflow: hidden; position: relative; }
.msg-block .msg-text.collapsed::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 40px; background: linear-gradient(transparent, var(--bg)); }
.msg-block .msg-expand { color: var(--accent); cursor: pointer; font-size: var(--small); margin-top: 4px; }
.msg-block .msg-text:not(.collapsed) + .msg-expand { display: none; }

#inspector { grid-column: 3; grid-row: 2; background: var(--panel); border-left: 1px solid var(--border); overflow-y: auto; padding: 12px; }
#inspector h3 { font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.inspector-empty { color: var(--text-dim); font-size: var(--small); }
.inspector-field { display: flex; justify-content: space-between; padding: 4px 0; font-size: var(--small); border-bottom: 1px solid var(--border); }
.inspector-field .label { color: var(--text-muted); }
.inspector-field .value { color: var(--text); font-family: var(--font-mono); font-size: var(--code); }

#statusbar { grid-column: 1/-1; grid-row: 3; display: flex; align-items: center; gap: 16px;
  padding: 0 12px; background: var(--bg-raised); border-top: 1px solid var(--border); font-size: var(--small); color: var(--text-muted); }
#statusbar span { white-space: nowrap; }
#loading { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); }
</style>
</head>
<body>
<div id="app">
  <div id="header"><h1>Polylogue</h1><span class="stats" id="header-stats"></span>
    <div class="health"><span class="health-dot ok" id="health-dot"></span><span id="health-text"></span></div>
  </div>
  <div id="sidebar">
    <div id="search-box"><input type="text" id="search" placeholder="Search conversations..." autofocus></div>
    <div id="filter-bar"></div>
    <div id="conv-list"><div id="loading">Loading...</div></div>
  </div>
  <div id="main">
    <div id="conv-header"><h2>Select a conversation</h2><div class="conv-stats"></div></div>
    <div id="msg-list"></div>
  </div>
  <div id="inspector"><h3>Inspector</h3><div class="inspector-empty">Select a conversation to view details</div></div>
  <div id="statusbar"><span id="sb-convs"></span><span id="sb-msgs"></span><span id="sb-db"></span></div>
</div>

<script>
const API = '';
let state = { conversations: [], selected: null, provider: '', query: '', stats: {} };

async function fetchJSON(url) {
  const r = await fetch(API + url);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

async function loadConversations() {
  const params = new URLSearchParams({ limit: '100' });
  if (state.provider) params.set('provider', state.provider);
  if (state.query) params.set('query', state.query);
  const data = await fetchJSON('/api/conversations?' + params);
  state.conversations = data.conversations || [];
  renderConversations();
}

async function loadConversation(id) {
  const data = await fetchJSON('/api/conversations/' + id);
  state.selected = data;
  renderMain();
  renderInspector();
}

async function loadStatus() {
  try {
    const [health, stats] = await Promise.all([
      fetchJSON('/api/health'), fetchJSON('/api/status')
    ]);
    document.getElementById('health-dot').className = 'health-dot ' + (health.ok ? 'ok' : 'err');
    document.getElementById('health-text').textContent = health.ok ? 'healthy' : health.quick_check || 'issues';
    const dbGB = (health.db_size_bytes / 1024 / 1024 / 1024).toFixed(0);
    document.getElementById('sb-db').textContent = 'DB: ' + dbGB + ' GB';
  } catch(e) {}
}

function renderConversations() {
  const el = document.getElementById('conv-list');
  if (!state.conversations.length) {
    el.innerHTML = '<div style="padding:12px;color:var(--text-dim)">No conversations found</div>';
    return;
  }
  el.innerHTML = state.conversations.map(c => {
    const sel = state.selected && state.selected.id === c.id ? ' selected' : '';
    const title = esc((c.title || 'Untitled').substring(0, 80));
    const date = c.created_at ? new Date(c.created_at).toLocaleDateString() : '';
    const p = c.provider || '';
    const color = 'var(--provider-' + p.replace(/_/g, '-') + ')';
    return '<div class="conv-item' + sel + '" data-id="' + escAttr(c.id) + '" onclick="selectConversation(\'' + escAttr(c.id) + '\')">' +
      '<div class="conv-title">' + title + '</div>' +
      '<div class="conv-meta"><span><span class="provider-dot" style="background:' + color + '"></span>' + esc(p) + '</span>' +
      '<span>' + date + '</span><span>' + (c.message_count || 0) + ' msgs</span></div></div>';
  }).join('');
}

function renderMain() {
  if (!state.selected) {
    document.getElementById('conv-header').innerHTML = '<h2>Select a conversation</h2><div class="conv-stats"></div>';
    document.getElementById('msg-list').innerHTML = '';
    return;
  }
  const c = state.selected;
  document.getElementById('conv-header').innerHTML =
    '<h2>' + esc(c.title || 'Untitled') + '</h2><div class="conv-stats">' +
    esc(c.provider || '') + ' &middot; ' + (c.message_count || 0) + ' messages &middot; ' +
    (c.word_count || 0).toLocaleString() + ' words</div>';
  if (!c.messages) {
    document.getElementById('msg-list').innerHTML = '<div id="loading">Loading messages...</div>';
    return;
  }
  document.getElementById('msg-list').innerHTML = c.messages.map(function(m) {
    var role = (m.role || '').toLowerCase();
    var text = m.text || '';
    var collapsed = text.length > 2000;
    var typeTag = m.message_type ? '<span class="msg-type">' + esc(m.message_type) + '</span>' : '';
    return '<div class="msg-block"><div class="msg-header">' +
      '<span class="msg-role ' + role + '">' + esc(role) + '</span>' + typeTag + '</div>' +
      '<div class="msg-text' + (collapsed ? ' collapsed' : '') + '">' + esc(text) + '</div>' +
      (collapsed ? '<div class="msg-expand" onclick="var t=this.previousElementSibling;t.classList.remove(\'collapsed\');this.remove()">Show all (' + text.length.toLocaleString() + ' chars)</div>' : '') +
      '</div>';
  }).join('');
}

function renderInspector() {
  var el = document.getElementById('inspector');
  if (!state.selected) {
    el.innerHTML = '<h3>Inspector</h3><div class="inspector-empty">Select a conversation to view details</div>';
    return;
  }
  var c = state.selected;
  var fields = [
    ['ID', c.id], ['Provider', c.provider],
    ['Created', c.created_at ? new Date(c.created_at).toLocaleString() : ''],
    ['Updated', c.updated_at ? new Date(c.updated_at).toLocaleString() : ''],
    ['Messages', c.message_count], ['Words', (c.word_count||0).toLocaleString()],
  ];
  el.innerHTML = '<h3>Inspector</h3>' + fields.map(function(f) {
    return '<div class="inspector-field"><span class="label">' + f[0] + '</span><span class="value">' + esc(String(f[1]||'')) + '</span></div>';
  }).join('');
}

async function selectConversation(id) {
  document.getElementById('msg-list').innerHTML = '<div id="loading">Loading...</div>';
  await loadConversation(id);
  renderConversations();
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function escAttr(s) { return String(s).replace(/'/g,"\\'").replace(/"/g,'&quot;'); }

// Filter bar
var FILTERS = ['', 'claude-code', 'codex', 'chatgpt', 'claude-ai', 'gemini'];
document.getElementById('filter-bar').innerHTML = FILTERS.map(function(f) {
  return '<button class="' + (state.provider===f?'active':'') + '" data-p="' + f + '">' + (f||'All') + '</button>';
}).join('');
document.getElementById('filter-bar').addEventListener('click', function(e) {
  if (e.target.tagName === 'BUTTON') {
    state.provider = e.target.dataset.p;
    document.querySelectorAll('#filter-bar button').forEach(function(b) { b.classList.remove('active'); });
    e.target.classList.add('active');
    loadConversations();
  }
});

// Search with debounce
var searchTimer;
document.getElementById('search').addEventListener('input', function(e) {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(function() { state.query = e.target.value; loadConversations(); }, 300);
});

// Keyboard: j/k to navigate, / to focus search
document.addEventListener('keydown', function(e) {
  if (e.key === '/' && !e.target.closest('input')) { e.preventDefault(); document.getElementById('search').focus(); }
  if ((e.key === 'j' || e.key === 'k') && !e.target.closest('input')) {
    var items = document.querySelectorAll('.conv-item');
    var sel = document.querySelector('.conv-item.selected');
    var idx = sel ? Array.from(items).indexOf(sel) : (e.key === 'j' ? -1 : items.length);
    var next = e.key === 'j' ? idx + 1 : idx - 1;
    if (next >= 0 && next < items.length) items[next].click();
  }
});

loadConversations();
loadStatus();
setInterval(loadStatus, 30000);
</script>
</body>
</html>"""
