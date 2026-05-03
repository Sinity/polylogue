"""Single-page web shell for the Polylogue daemon.

Served at ``GET /`` by the daemon API server. Vanilla JS, dark theme,
monospace font. No external dependencies.
"""

from __future__ import annotations

WEB_SHELL_HTML: str = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polylogue</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117;
    color: #c9d1d9;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  header h1 { font-size: 18px; font-weight: 600; color: #58a6ff; }
  .stat-badge {
    font-size: 12px;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px 8px;
    color: #8b949e;
  }
  .stat-badge strong { color: #c9d1d9; }
  #search-box {
    margin-left: auto;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 12px;
    color: #c9d1d9;
    font-family: inherit;
    font-size: 13px;
    width: 240px;
  }
  #search-box:focus { outline: none; border-color: #58a6ff; }
  main { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .conv-list { list-style: none; }
  .conv-item {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-bottom: 8px;
    padding: 10px 14px;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .conv-item:hover { border-color: #58a6ff; }
  .conv-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
  }
  .conv-title { font-weight: 500; color: #e6edf3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conv-meta { font-size: 12px; color: #8b949e; white-space: nowrap; }
  .conv-meta span { margin-left: 8px; }
  .conv-messages { display: none; margin-top: 10px; border-top: 1px solid #30363d; padding-top: 10px; }
  .conv-messages.open { display: block; }
  .message {
    padding: 6px 0;
    border-bottom: 1px solid #21262d;
  }
  .message:last-child { border-bottom: none; }
  .message-role {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 2px;
  }
  .message-text {
    white-space: pre-wrap;
    word-break: break-word;
    color: #c9d1d9;
    font-size: 13px;
    max-height: 300px;
    overflow-y: auto;
  }
  .loading {
    text-align: center;
    padding: 40px;
    color: #8b949e;
  }
  .error-state {
    text-align: center;
    padding: 40px;
    color: #f85149;
  }
  .empty-state {
    text-align: center;
    padding: 40px;
    color: #8b949e;
  }
  #status-bar {
    background: #161b22;
    border-top: 1px solid #30363d;
    padding: 6px 20px;
    font-size: 12px;
    color: #8b949e;
    display: flex;
    gap: 16px;
  }
  #status-bar .ok { color: #3fb950; }
  #status-bar .warn { color: #d29922; }
  #status-bar .err { color: #f85149; }
</style>
</head>
<body>
<header>
  <h1>Polylogue</h1>
  <span class="stat-badge" id="stat-convs">conversations: <strong>?</strong></span>
  <span class="stat-badge" id="stat-msgs">messages: <strong>?</strong></span>
  <span class="stat-badge" id="stat-providers">providers: <strong>?</strong></span>
  <input type="text" id="search-box" placeholder="Search conversations..." />
</header>
<main>
  <div id="conv-list" class="conv-list"></div>
</main>
<div id="status-bar"></div>
<script>
"use strict";

const API_BASE = '';

async function apiFetch(path, opts) {
  const r = await fetch(API_BASE + path, opts);
  if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  return r.json();
}

// ---- State ----
let conversations = [];
let currentQuery = '';

// ---- Render helpers ----
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderConvs(convs) {
  const list = document.getElementById('conv-list');
  list.innerHTML = '';
  if (!convs || convs.length === 0) {
    list.innerHTML = '<div class="empty-state">No conversations found.</div>';
    return;
  }
  for (const c of convs) {
    const item = document.createElement('div');
    item.className = 'conv-item';
    item.dataset.id = c.id;
    item.innerHTML = `
      <div class="conv-header">
        <span class="conv-title">${escapeHtml(c.title || '(untitled)')}</span>
        <span class="conv-meta">
          <span>${escapeHtml(c.provider || '?')}</span>
          <span>${c.message_count || 0} msgs</span>
          <span>${c.created_at ? new Date(c.created_at).toLocaleDateString() : ''}</span>
        </span>
      </div>
      <div class="conv-messages" id="msgs-${escapeHtml(c.id)}"></div>
    `;
    item.addEventListener('click', () => toggleMessages(c.id, item));
    list.appendChild(item);
  }
}

async function toggleMessages(convId, item) {
  const div = document.getElementById('msgs-' + convId);
  if (!div) return;
  if (div.classList.contains('open')) {
    div.classList.remove('open');
    div.innerHTML = '';
    return;
  }
  div.innerHTML = '<div class="loading">Loading messages...</div>';
  div.classList.add('open');
  try {
    const data = await apiFetch('/api/conversations/' + encodeURIComponent(convId) + '/messages?limit=50');
    div.innerHTML = '';
    if (!data.messages || data.messages.length === 0) {
      div.innerHTML = '<div class="empty-state">No messages.</div>';
      return;
    }
    for (const m of data.messages) {
      const msg = document.createElement('div');
      msg.className = 'message';
      const role = escapeHtml(m.role || 'unknown');
      const text = m.text || '';
      msg.innerHTML = `<div class="message-role">${role}</div><div class="message-text"></div>`;
      msg.querySelector('.message-text').textContent = text;
      div.appendChild(msg);
    }
    if (data.total > 50) {
      const more = document.createElement('div');
      more.className = 'loading';
      more.textContent = `... and ${data.total - 50} more messages`;
      div.appendChild(more);
    }
  } catch (err) {
    div.innerHTML = '<div class="error-state">Failed to load messages.</div>';
  }
}

async function loadConvs(query) {
  const list = document.getElementById('conv-list');
  list.innerHTML = '<div class="loading">Loading...</div>';
  try {
    let url = '/api/conversations?limit=50';
    if (query) url += '&query=' + encodeURIComponent(query);
    const data = await apiFetch(url);
    conversations = data.conversations || [];
    renderConvs(conversations);
  } catch (err) {
    list.innerHTML = '<div class="error-state">Failed to load conversations.</div>';
  }
}

async function loadStats() {
  try {
    const data = await apiFetch('/api/facets');
    const f = data.facets || {};
    document.getElementById('stat-convs').innerHTML = 'conversations: <strong>' + (f.total_conversations || 0) + '</strong>';
    document.getElementById('stat-msgs').innerHTML = 'messages: <strong>' + (f.total_messages || 0) + '</strong>';
    const provs = f.providers ? Object.keys(f.providers).join(', ') : 'none';
    document.getElementById('stat-providers').innerHTML = 'providers: <strong>' + escapeHtml(provs) + '</strong>';
  } catch (_) {}
}

async function loadHealth() {
  const bar = document.getElementById('status-bar');
  try {
    const h = await apiFetch('/api/health');
    const ok = h.ok;
    const db = (h.db_size_bytes || 0) / 1024 / 1024;
    bar.innerHTML = `
      <span class="${ok ? 'ok' : 'err'}">${ok ? 'healthy' : 'degraded'}</span>
      <span>db: ${db.toFixed(1)} MB</span>
      <span>wal: ${((h.wal_size_bytes || 0) / 1024).toFixed(0)} KB</span>
      <span>disk free: ${((h.disk_free_bytes || 0) / 1024 / 1024 / 1024).toFixed(1)} GB</span>
    `;
  } catch (_) {
    bar.innerHTML = '<span class="err">daemon unreachable</span>';
  }
}

// ---- Search ----
let searchTimer = null;
document.getElementById('search-box').addEventListener('input', function() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    const q = this.value.trim();
    currentQuery = q;
    loadConvs(q);
  }, 300);
});

// ---- Init ----
loadConvs('');
loadStats();
loadHealth();
setInterval(loadHealth, 30000);
</script>
</body>
</html>
"""

__all__ = ["WEB_SHELL_HTML"]
