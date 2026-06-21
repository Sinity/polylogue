"""Polylogue MK2 web reader — single-page interactive archive cockpit."""

from __future__ import annotations

from polylogue.daemon.web_shell_attachments import ATTACHMENT_CSS, ATTACHMENT_JS
from polylogue.daemon.web_shell_bulk import (
    BULK_CSS,
    BULK_JS,
    BULK_PREVIEW_HTML,
    BULK_TOOLBAR_HTML,
)
from polylogue.daemon.web_shell_lineage import LINEAGE_JS
from polylogue.daemon.web_shell_paste import PASTE_CSS, PASTE_JS
from polylogue.daemon.web_shell_provenance import PROVENANCE_JS
from polylogue.daemon.web_shell_reader import READER_CSS, READER_HELP_HTML, READER_JS
from polylogue.daemon.web_shell_realtime import REALTIME_JS
from polylogue.daemon.web_shell_similar import SIMILAR_JS
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
/* Insights browser readiness vocabulary (#1120): ready / partial (reuses warn
   palette above) / missing. q-partial is already defined for cost panels. */
.chip.q-ready { border-color: var(--ok); color: var(--ok); background: var(--ok-bg); }
.chip.q-missing { border-color: var(--text-dim); color: var(--text-dim); background: var(--panel-subtle); }
.insight-section-header { cursor: pointer; user-select: none; }
.insight-section-header .insight-toggle { display: inline-block; width: 10px; color: var(--text-dim); }
.inspector-field .value.muted { color: var(--text-dim); font-style: italic; }
.muted { color: var(--text-dim); }

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
/* #1204 — new-row / appended-message animations driven by SSE granular topics. */
@keyframes pl-row-appended { 0% { background: var(--accent-bg); } 100% { background: transparent; } }
@keyframes pl-message-appended { 0% { background: var(--accent-bg); } 100% { background: transparent; } }
.conv-item.row-appended { animation: pl-row-appended 1.8s ease-out; }
.msg-block.message-appended, [data-msg-id].message-appended { animation: pl-message-appended 1.8s ease-out; }
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
.read-profile-selector { display:flex; align-items:center; gap:8px; padding:7px 16px; border-bottom:1px solid var(--border);
  background:var(--bg-raised); color:var(--text-muted); font-size:var(--small); }
.read-profile-selector label { text-transform:uppercase; letter-spacing:0.5px; color:var(--text-dim); font-size:10px; }
.read-profile-selector select { background:var(--panel-elevated); border:1px solid var(--border); color:var(--text);
  border-radius:3px; padding:3px 6px; font-size:var(--small); }
.read-profile-selector .profile-hint { color:var(--text-dim); font-family:var(--font-mono); font-size:10px; }
.read-profile-selector.muted { color:var(--text-dim); }
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
__READER_CSS__
__PASTE_CSS__
__ATTACHMENT_CSS__
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
.annotation-composer { display: flex; flex-direction: column; gap: 6px; border: 1px solid var(--border);
  border-radius: var(--radius); padding: 6px; background: var(--panel-subtle); margin-bottom: 8px; }
.annotation-composer label { color: var(--text-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.annotation-composer select, .annotation-composer textarea { width: 100%; background: var(--panel-elevated);
  border: 1px solid var(--border); color: var(--text); border-radius: 3px; font: inherit; font-size: var(--small); }
.annotation-composer select { padding: 4px 6px; }
.annotation-composer textarea { min-height: 68px; resize: vertical; padding: 6px; line-height: 1.4; }
.annotation-composer textarea:focus, .annotation-composer select:focus { outline: none; border-color: var(--accent); }
.annotation-composer .annotation-actions { margin-top: 0; }
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
    <span class="chip" id="status-semantic" title="Semantic readiness">semantic: --</span>
    <span class="chip" id="status-insights" title="Session insight freshness" style="display:none">insights: --</span>
    <span class="chip" id="status-ingest" style="display:none">live</span>
    <span class="chip" id="status-browser-capture" title="Browser capture readiness" style="display:none">capture: --</span>
    <span class="chip" id="status-dev-loop" title="Branch-local dev loop" style="display:none">dev: --</span>
    <span class="chip" id="status-api-debug" title="Latest API request" style="display:none">api: --</span>
    <span class="chip" id="status-live" title="Realtime channel">live: --</span>
  </div>
  <div id="sidebar">
    <div id="search-box">
      <input type="text" id="search" placeholder="Search sessions..." autofocus>
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
        <h3>Select a session</h3>
        <p>Browse from the list or use <span class="kbd">/</span> to search. Press <span class="kbd">?</span> for shortcuts.</p>
      </div>
    </div>
  </div>
  <div id="inspector">
    <div id="inspector-tabs">
      <button class="active" data-tab="info">Info</button>
      <button data-tab="cost">Cost</button>
      <button data-tab="lineage">Lineage</button>
      <button data-tab="insights">Insights</button>
      <button data-tab="evidence">Evidence</button>
      <button data-tab="raw">Raw</button>
      <button data-tab="similar">Similar</button>
      <button data-tab="attachments">Attachments</button>
      <button data-tab="notes">Notes</button>
    </div>
    <div id="inspector-content"><div class="inspector-empty">Select a session to inspect</div></div>
  </div>
  <div id="footer">
    <span class="hint"><kbd>/</kbd> search</span>
    <span class="hint"><kbd>j</kbd><kbd>k</kbd> navigate</span>
    <span class="hint"><kbd>g g</kbd> top</span>
    <span class="hint"><kbd>G</kbd> bottom</span>
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
      <kbd>j</kbd><span>Next session</span>
      <kbd>k</kbd><span>Previous session</span>
      <kbd>n</kbd><span>Next page</span>
      <kbd>p</kbd><span>Previous page</span>
      <kbd>Enter</kbd><span>Open selected</span>
      <kbd>Esc</kbd><span>Clear search / close</span>
      <kbd>?</kbd><span>Toggle this help</span>
__READER_HELP_HTML__
    </div>
    <div class="help-close" onclick="toggleHelp()">Press <kbd>Esc</kbd> or click to close</div>
  </div>
</div>

__BULK_PREVIEW_HTML__

<script>
var API = '';
var state = {
  sessions: [], selected: null, selectedRaw: null,
  origin: '', query: '', offset: 0, limit: 100, total: 0,
  status: {}, facets: null, inspectorTab: 'info',
  marks: {}, annotations: {}, savedViews: [], workspaces: [], userStateError: '',
  mode: 'single', stackPayload: null, comparePayload: null,
  // Bulk selection state (#1119). selection is a Set-like object keyed by
  // session_id. lastBulkResult holds the per-session envelope from
  // the most recent bulk operation: {succeeded:[ids], failed:[{id,reason}],
  // skipped:[{id,reason}], dryRun:bool, action:string}.
  bulkSelection: {}, lastBulkResult: null, bulkPending: null,
  // Cost panel cache (#1122). Keyed by session_id; populated on demand
  // when the Cost inspector tab is opened. ``undefined`` means "not loaded
  // yet", null/{error} means "fetch failed".
  costPanels: {},
  // Per-session provenance (#1125). Loaded lazily when the Raw
  // inspector tab is opened for the selected session; raw payload
  // preview within is opt-in via explicit user click.
  provenance: null,
  // Per-session lineage envelope (#1121). Loaded lazily when the
  // Lineage inspector tab is opened. ``undefined`` means "not loaded
  // yet"; ``{error}`` means "fetch failed".
  lineage: undefined,
  // Insights browser cache (#1120). Keyed by session_id; populated
  // on demand when the Insights inspector tab is opened. Holds the
  // ``GET /api/insights/sessions/{id}`` envelope: ``{kinds: {profile,
  // timeline, phases, threads}, include, session_id, origin}``.
  insightsPanels: {},
  // Per-session collapsed-section toggles for the Insights tab.
  // Keyed as ``"<session_id>:<kind>"``; default = expanded.
  insightsCollapsed: {},
  // Per-session evidence/work-packet panel cache (#1846). Keyed by
  // session_id; populated from the new daemon recovery/assertion routes.
  // This is intentionally a read-side cache over shared DTOs, not a web-only
  // evidence model.
  evidencePanels: {},
  // Shared read-view profile inventory (#1846/#1838). Loaded from
  // /api/read-view-profiles so the shell does not grow a second profile
  // registry. selectedReadView controls the main reader pane. Supported
  // single-session profiles execute through /api/sessions/{id}/read; unsupported
  // profiles remain disabled until HTTP execution exists.
  readViewProfiles: [], selectedReadView: 'messages', readViewProfileError: '',
  readViewPayloads: {}, readViewErrors: {},
  // Per-session similarity panel cache (#1123). Keyed by
  // session_id; populated on demand when the Similar inspector
  // tab is opened. ``undefined`` means "not loaded yet"; the envelope
  // carries the explicit pipeline state under ``status``.
  similarPanels: {},
  // Latest web-shell API request metadata. This is a UI/debug aid only: it
  // records route, status, duration, request id, and a bounded response summary
  // without storing raw archive payloads.
  apiDebug: {counter: 0, last: null}
};

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function escAttr(s) { return String(s).replace(/'/g,"\\'").replace(/"/g,'&quot;'); }

function nowMs() {
  return (window.performance && performance.now) ? performance.now() : Date.now();
}

function nextApiRequestId() {
  state.apiDebug.counter += 1;
  return 'web-' + Date.now().toString(36) + '-' + state.apiDebug.counter;
}

function summarizeApiBody(text) {
  if (!text) return '';
  var compact = String(text).replace(/\s+/g, ' ').trim();
  return compact.length > 240 ? compact.slice(0, 237) + '...' : compact;
}

function rememberApiDebug(entry) {
  state.apiDebug.last = entry;
  renderApiDebugChip();
}

async function requestJSON(url, opts) {
  opts = opts || {};
  var method = opts.method || 'GET';
  var requestId = nextApiRequestId();
  var headers = Object.assign({'X-Request-ID': requestId}, opts.headers || {});
  var requestOpts = Object.assign({}, opts, {method: method, headers: headers});
  var started = nowMs();
  var response;
  try {
    response = await fetch(API + url, requestOpts);
  } catch(e) {
    var networkDuration = Math.round(nowMs() - started);
    rememberApiDebug({
      ok: false, method: method, url: url, request_id: requestId,
      status: 'network_error', duration_ms: networkDuration,
      response_summary: String(e && e.message ? e.message : e)
    });
    throw e;
  }
  var text = await response.text();
  var duration = Math.round(nowMs() - started);
  var responseRequestId = response.headers && response.headers.get ? response.headers.get('X-Request-ID') : '';
  var summary = summarizeApiBody(text);
  rememberApiDebug({
    ok: response.ok,
    method: method,
    url: url,
    request_id: requestId,
    response_request_id: responseRequestId || '',
    status: response.status,
    duration_ms: duration,
    response_summary: summary
  });
  if (!response.ok) {
    var error = new Error(String(response.status));
    error.status = response.status;
    error.request_id = requestId;
    error.response_summary = summary;
    throw error;
  }
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch(e) {
    rememberApiDebug({
      ok: false,
      method: method,
      url: url,
      request_id: requestId,
      response_request_id: responseRequestId || '',
      status: 'invalid_json',
      duration_ms: duration,
      response_summary: summary
    });
    var parseError = new Error('invalid_json');
    parseError.request_id = requestId;
    parseError.response_summary = summary;
    throw parseError;
  }
}

async function fetchJSON(url) {
  return requestJSON(url, {method: 'GET'});
}
async function sendJSON(url, method, body) {
  var opts = {method: method, headers: {'Content-Type': 'application/json'}};
  if (body !== undefined) opts.body = JSON.stringify(body);
  return requestJSON(url, opts);
}

function markSetFor(sessionId) {
  return state.marks[sessionId] || {};
}
function hasMark(sessionId, markType) {
  return !!markSetFor(sessionId)[markType];
}
function setMarkLocal(sessionId, markType, enabled) {
  if (!state.marks[sessionId]) state.marks[sessionId] = {};
  if (enabled) state.marks[sessionId][markType] = true;
  else delete state.marks[sessionId][markType];
}
function annotationsFor(sessionId) {
  return state.annotations[sessionId] || [];
}

function getSessionIdFromURL() {
  var m = window.location.pathname.match(/^\/s\/(.+)$/);
  return m ? decodeURIComponent(m[1]) : null;
}
__WORKSPACE_JS__
window.addEventListener('popstate', function() {
  var route = getWorkspaceRouteFromURL();
  if (route) { loadWorkspaceRoute(route, false); return; }
  var cid = getSessionIdFromURL();
  if (cid) selectSession(cid, false);
  else { state.mode = 'single'; state.selected = null; state.selectedRaw = null; renderMain(); renderInspector(); renderSessions(); }
});

async function loadSessions(opts) {
  var params = new URLSearchParams();
  params.set('limit', String(state.limit));
  params.set('offset', String(state.offset));
  if (state.origin) params.set('origin', state.origin);
  if (state.query) params.set('query', state.query);
  // Capture pre-load id set so we can animate newly-arrived rows after render.
  var beforeIds = {};
  (state.sessions || []).forEach(function(c) { beforeIds[c.id] = true; });
  try {
    var data = await fetchJSON('/api/sessions?' + params);
    state.sessions = data.items || [];
    state.total = data.total || 0;
    document.getElementById('footer-result').textContent =
      (state.total > 0) ? (state.total + ' results') : '';
  } catch(e) {
    state.sessions = [];
    state.total = 0;
    renderSidebarState('error', 'Failed to load sessions');
  }
  renderSessions();
  // After render, animate rows that are newly present (either flagged by
  // realtime or simply not in the previous snapshot at this offset).
  var animateIds = (opts && opts.animateNewIds) || {};
  state.sessions.forEach(function(c) {
    if (animateIds[c.id] || !beforeIds[c.id]) {
      maybeAnimateExistingRow(c.id);
    }
  });
}

async function loadReadViewProfiles() {
  try {
    var payload = await fetchJSON('/api/read-view-profiles');
    state.readViewProfiles = payload.read_views || [];
    state.readViewProfileError = '';
  } catch(e) {
    state.readViewProfiles = [];
    state.readViewProfileError = 'Read profiles unavailable';
  }
  renderMain();
}

async function loadUserState() {
  try {
    var marks = await fetchJSON('/api/user/marks');
    state.marks = {};
    (marks.items || []).forEach(function(m) {
      setMarkLocal(m.session_id, m.mark_type, true);
    });
    var annotations = await fetchJSON('/api/user/annotations');
    state.annotations = {};
    (annotations.items || []).forEach(function(a) {
      if (!state.annotations[a.session_id]) state.annotations[a.session_id] = [];
      state.annotations[a.session_id].push(a);
    });
    var savedViews = await fetchJSON('/api/user/saved-views');
    state.savedViews = savedViews.items || [];
    var workspaces = await fetchJSON('/api/user/workspaces');
    state.workspaces = workspaces.items || [];
    state.userStateError = '';
  } catch(e) {
    state.userStateError = 'User state unavailable';
  }
  renderSessions();
  renderMain();
  renderInspector();
}

async function loadSession(id, updateURL) {
  state.mode = 'single';
  state.stackPayload = null;
  state.comparePayload = null;
  if (updateURL !== false) pushSingleURL(id);
  try {
    var data = await fetchJSON('/api/sessions/' + id);
    state.selected = data;
  } catch(e) { state.selected = null; }
  renderMain();
  renderInspector();
  renderSessions();
}

async function loadSessionRaw(id) {
  try {
    var data = await fetchJSON('/api/sessions/' + encodeURIComponent(id) + '/provenance');
    state.selectedRaw = data;
  } catch(e) { state.selectedRaw = null; }
}

async function loadFacets() {
  var params = new URLSearchParams();
  if (state.query) params.set('query', state.query);
  if (state.origin) params.set('origin', state.origin);
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
    var readiness = s.component_readiness || {};
    document.getElementById('status-convs').textContent = (s.total_sessions || 0).toLocaleString() + ' convs';
    document.getElementById('status-msgs').textContent = (s.total_messages || 0).toLocaleString() + ' msgs';
    renderFtsChip(readiness.search || null, s.fts_readiness || {});
    renderSemanticChip(readiness.embeddings || null);
    renderInsightChip(readiness.session_profiles || null, s.insight_freshness || {});
    renderIngestChip(readiness.daemon_ingest || null, s.live || {});
    renderBrowserCaptureChip(readiness.browser_capture || null, s.browser_capture || {});
  } catch(e) {}
  try {
    renderDevLoopChip(await fetchJSON('/api/dev-loop'));
  } catch(e) { renderDevLoopChip(null); }
}

// Apply an MK3 data-quality class to a chip element (canonical, partial, stale,
// unavailable, etc.). Strips any prior q-* class so callers can freely flip.
function setChipQuality(el, quality) {
  if (!el) return;
  el.className = el.className.split(' ').filter(function(c) { return c.indexOf('q-') !== 0; }).join(' ');
  if (quality) el.classList.add('q-' + quality);
}

function readinessQuality(state) {
  if (state === 'ready') return 'canonical';
  if (state === 'stale') return 'stale';
  if (state === 'rebuilding' || state === 'degraded') return 'partial';
  if (state === 'poisoned') return 'redacted';
  if (state === 'missing' || state === 'blocked') return 'unavailable';
  return 'unavailable';
}

function readinessLabel(state) {
  if (state === 'ready') return 'ok';
  if (state === 'rebuilding') return 'rebuilding';
  if (state === 'stale') return 'stale';
  if (state === 'degraded') return 'degraded';
  if (state === 'missing') return 'missing';
  if (state === 'blocked') return 'blocked';
  if (state === 'poisoned') return 'poisoned';
  return 'unknown';
}

function renderComponentReadinessChip(el, label, component) {
  if (!el || !component) return false;
  var state = component.state || 'unknown';
  el.textContent = label + ': ' + readinessLabel(state);
  setChipQuality(el, readinessQuality(state));
  var caveats = component.caveats || [];
  var summary = component.summary || state;
  el.title = label + ' readiness: ' + summary + (caveats.length ? ' (' + caveats.join(', ') + ')' : '');
  return true;
}

function renderFtsChip(component, fts) {
  var el = document.getElementById('status-fts');
  if (component && component.state !== 'unknown') {
    renderComponentReadinessChip(el, 'FTS', component);
    return;
  }
  var msgReady = !!fts.messages_ready;
  var indexed = Number(fts.message_indexed_count || 0);
  var indexable = Number(fts.message_indexable_count || 0);
  var partial = !msgReady && (indexed > 0 || indexable > 0 || Object.keys(fts.surfaces || {}).length > 0);
  var label;
  var quality;
  if (msgReady) { label = 'FTS: ok'; quality = 'canonical'; }
  else if (partial) { label = 'FTS: partial'; quality = 'partial'; }
  else { label = 'FTS: unavailable'; quality = 'unavailable'; }
  el.textContent = label;
  setChipQuality(el, quality);
}

function renderSemanticChip(component) {
  var el = document.getElementById('status-semantic');
  renderComponentReadinessChip(el, 'semantic', component);
}

function renderInsightChip(component, freshness) {
  var el = document.getElementById('status-insights');
  if (component) {
    el.style.display = '';
    renderComponentReadinessChip(el, 'insights', component);
    return;
  }
  var total = freshness.total_sessions || 0;
  var withProfiles = freshness.sessions_with_profiles || 0;
  if (total <= 0) { el.style.display = 'none'; return; }
  el.style.display = '';
  if (withProfiles >= total) { el.textContent = 'insights: ok'; setChipQuality(el, 'canonical'); }
  else if (withProfiles > 0) { el.textContent = 'insights: ' + withProfiles + '/' + total; setChipQuality(el, 'partial'); }
  else { el.textContent = 'insights: stale'; setChipQuality(el, 'stale'); }
}

function renderIngestChip(component, live) {
  var el = document.getElementById('status-ingest');
  if (component) {
    el.style.display = '';
    renderComponentReadinessChip(el, 'ingest', component);
    return;
  }
  if (live && live.existing_source_count > 0) {
    el.style.display = '';
    el.textContent = 'live: ' + live.existing_source_count + ' srcs';
    setChipQuality(el, 'canonical');
  } else { el.style.display = 'none'; }
}

function renderBrowserCaptureChip(component, capture) {
  var el = document.getElementById('status-browser-capture');
  if (!el) return;
  if (!component && !capture) { el.style.display = 'none'; return; }
  el.style.display = '';
  var state = component ? (component.state || 'unknown') : 'unknown';
  var active = state === 'ready';
  var spoolReady = !!(capture && capture.spool_ready);
  var authRequired = !!(capture && capture.auth_required);
  var origins = (capture && capture.allowed_origins) || [];
  if (active && spoolReady) {
    el.textContent = 'capture: ready';
    setChipQuality(el, 'canonical');
  } else if (spoolReady) {
    el.textContent = 'capture: off';
    setChipQuality(el, 'unavailable');
  } else {
    el.textContent = 'capture: unavailable';
    setChipQuality(el, 'unavailable');
  }
  var originText = origins.length ? origins.join(', ') : 'no origins';
  el.title = 'Browser capture: ' + (active ? 'running' : 'not running')
    + '; spool ' + (spoolReady ? 'ready' : 'unavailable')
    + '; auth ' + (authRequired ? 'required' : 'not required')
    + '; origins ' + originText;
}

function renderDevLoopChip(payload) {
  var el = document.getElementById('status-dev-loop');
  if (!el) return;
  if (!payload || !payload.enabled) { el.style.display = 'none'; return; }
  el.style.display = '';
  var runId = payload.run_id || 'local';
  var label = String(runId);
  if (label.length > 14) label = label.slice(0, 11) + '...';
  el.textContent = 'dev: ' + label;
  setChipQuality(el, 'partial');
  var details = [];
  if (payload.archive_root) details.push('archive ' + payload.archive_root);
  if (payload.log_dir) details.push('logs ' + payload.log_dir);
  if (payload.api_port) details.push('api :' + payload.api_port);
  if (payload.browser_capture_port) details.push('capture :' + payload.browser_capture_port);
  el.title = 'Branch-local dev loop'
    + (payload.run_id ? ' ' + payload.run_id : '')
    + (details.length ? '; ' + details.join('; ') : '');
}

function renderApiDebugChip() {
  var el = document.getElementById('status-api-debug');
  if (!el) return;
  var entry = state.apiDebug && state.apiDebug.last;
  if (!entry) { el.style.display = 'none'; return; }
  el.style.display = '';
  var status = entry.status || 'unknown';
  if (entry.ok) {
    el.textContent = 'api: ' + entry.duration_ms + 'ms';
    setChipQuality(el, 'canonical');
  } else {
    el.textContent = 'api: ' + status;
    setChipQuality(el, 'stale');
  }
  var title = [
    entry.method + ' ' + entry.url,
    'status ' + status,
    'duration ' + entry.duration_ms + 'ms',
    'request ' + entry.request_id
  ];
  if (entry.response_request_id) title.push('response ' + entry.response_request_id);
  if (entry.response_summary) title.push('body ' + entry.response_summary);
  el.title = title.join('; ');
}

function renderSidebarState(kind, msg) {
  var icons = {empty: '\u25a1', noresults: '\u25a2', error: '\u2715', loading: '\u2014'};
  document.getElementById('conv-list').innerHTML =
    '<div class="sidebar-state"><div class="state-icon">' + (icons[kind] || '') + '</div>' + esc(msg) + '</div>';
}

function renderSessions() {
  var el = document.getElementById('conv-list');
  var items = state.sessions;
  if (!items || !items.length) {
    // Distinguish empty archive from filtered-no-results so the operator
    // knows whether to ingest or to clear filters. Preserve filter context
    // in the empty-state message per MK3 state matrix.
    if (state.query && state.origin) {
      renderSidebarState('noresults', 'No results for query=' + state.query + ' origin=' + state.origin + '. Press Esc to clear.');
    } else if (state.query) {
      renderSidebarState('noresults', 'No results for query=' + state.query + '. Press Esc to clear.');
    } else if (state.origin) {
      renderSidebarState('noresults', 'No sessions from origin=' + state.origin + '. Press Esc to clear.');
    } else if (state.total === 0) {
      renderSidebarState('empty', 'No sessions in archive. Run `polylogued run` to ingest sources.');
    } else {
      renderSidebarState('noresults', 'No sessions on this page');
    }
    return;
  }
  el.innerHTML = items.map(function(c) {
    var sel = state.selected && state.selected.id === c.id ? ' selected' : '';
    var bulkSel = isBulkSelected(c.id) ? ' bulk-selected' : '';
    var title = esc((c.title || 'Untitled').substring(0, 100));
    var date = c.created_at ? new Date(c.created_at).toLocaleDateString() : '';
    var p = c.origin || 'unknown';
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
      + '<input type="checkbox" class="bulk-check" data-bulk-id="' + escAttr(c.id) + '" aria-label="Select session"' + checked + '>'
      + '<div class="conv-body" onclick="selectSession(\'' + escAttr(c.id) + '\')">'
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
  var providers = f.origins || {};
  var provKeys = Object.keys(providers);
  if (provKeys.length > 0) {
    html += '<div class="facet-group"><div class="facet-group-label">Origins</div><div class="facet-chips">';
    html += '<span class="facet-chip' + (!state.origin ? ' active' : '') + '" data-facet="origin" data-value="">All</span>';
    provKeys.sort(function(a,b) { return providers[b] - providers[a]; }).slice(0, 8).forEach(function(p) {
      var active = state.origin === p ? ' active' : '';
      html += '<span class="facet-chip' + active + '" data-facet="origin" data-value="' + escAttr(p) + '">'
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

function renderReadViewSelector(c) {
  if (!c) return '';
  var profiles = state.readViewProfiles || [];
  if (!profiles.length) {
    var fallback = state.readViewProfileError || 'Read profiles loading';
    return '<div id="read-profile-selector" class="read-profile-selector muted">' + esc(fallback) + '</div>';
  }
  var html = '<div id="read-profile-selector" class="read-profile-selector"><label for="read-profile-select">Read view</label>'
    + '<select id="read-profile-select" onchange="applyReadViewSelection(this.value)">';
  profiles.forEach(function(profile) {
    var id = profile.view_id || profile.id;
    if (!id) return;
    var http = profile.http || {};
    var supported = http.supported === true;
    var disabled = supported ? '' : ' disabled';
    var suffix = supported ? '' : ' (pending HTTP route)';
    var selected = (state.selectedReadView === id) ? ' selected' : '';
    var title = profile.purpose || profile.description || profile.summary || '';
    html += '<option value="' + escAttr(id) + '"' + selected + disabled + ' title="' + escAttr(title) + '">'
      + esc(profile.label || id) + esc(suffix) + '</option>';
  });
  html += '</select><span class="profile-hint">from /api/read-view-profiles</span></div>';
  return html;
}

function applyReadViewSelection(viewId) {
  state.selectedReadView = viewId || 'messages';
  if (viewId === 'recovery') {
    state.inspectorTab = 'evidence';
  } else if (viewId === 'raw') {
    state.inspectorTab = 'raw';
  }
  document.querySelectorAll('#inspector-tabs button').forEach(function(b) {
    b.classList.toggle('active', b.dataset.tab === state.inspectorTab);
  });
  renderMain();
  renderInspector();
}

function readViewCacheKey(id, viewId) {
  return id + '::' + (viewId || 'messages');
}

async function loadReadViewExecution(id, viewId) {
  if (!id || !viewId || viewId === 'messages') return;
  var key = readViewCacheKey(id, viewId);
  try {
    var payload = await fetchJSON('/api/sessions/' + encodeURIComponent(id) + '/read?view=' + encodeURIComponent(viewId) + '&format=json');
    state.readViewPayloads[key] = payload;
    delete state.readViewErrors[key];
  } catch(e) {
    state.readViewErrors[key] = String(e);
  }
  if (state.selected && state.selected.id === id && state.selectedReadView === viewId) {
    renderMain();
  }
}

function renderMain() {
  renderWorkspaceToolbar();
  if (state.mode === 'stack') { renderStackWorkspace(); return; }
  if (state.mode === 'compare') { renderCompareWorkspace(); return; }
  var headerEl = document.getElementById('conv-header');
  var msgEl = document.getElementById('msg-list');
  if (!state.selected) {
    headerEl.innerHTML = '<h2>Polylogue</h2><div class="conv-stats"></div>';
    msgEl.innerHTML = '<div class="main-empty"><h3>Select a session</h3>'
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
  // 1. origin/source  2. live/stale  3. repo/cwd/branch  4. counts
  // 5. cost/tokens  6. derived/insight  7. marks/tags
  // 1. origin/source
  if (c.origin) headerHtml += '<span class="chip q-canonical">' + esc(c.origin) + '</span>';
  if (c.model) headerHtml += '<span class="chip">' + esc(String(c.model)) + '</span>';
  // 2. live/stale (placeholder — wired from session provenance once #1019 surfaces it on detail)
  if (c.stale) headerHtml += '<span class="chip q-stale" title="Derived view is stale">stale</span>';
  // 3. repo/cwd/branch
  if (c.repo) headerHtml += '<span class="chip repo" title="' + escAttr(c.repo) + '">' + esc(c.repo.split('/').pop() || c.repo) + '</span>';
  if (c.cwd_display) headerHtml += '<span class="chip" title="' + escAttr(c.cwd_display) + '">' + esc(c.cwd_display.split('/').pop() || c.cwd_display) + '</span>';
  if (c.branch_type && c.branch_type !== 'main') {
    headerHtml += '<span class="chip q-inferred" title="Branch type">' + esc(c.branch_type) + '</span>';
  }
  // 3b. Topology branch chip (#1203). The chip is rendered against the
  // lazily-loaded lineage envelope; when lineage is not yet cached we
  // trigger the fetch and fall back to a placeholder. The chip click
  // opens the Lineage inspector tab as the popover surface.
  headerHtml += renderTopologyBranchChip(c);
  headerHtml += renderOpenParentChainButton(c);
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

  var readViewSelector = renderReadViewSelector(c);
  if (state.selectedReadView && state.selectedReadView !== 'messages') {
    msgEl.innerHTML = readViewSelector + renderReadViewExecution(c, state.selectedReadView);
    return;
  }
  if (!c.messages) {
    msgEl.innerHTML = readViewSelector + '<div class="main-empty"><h3>Loading messages...</h3></div>';
    return;
  }
  if (c.messages.length === 0) {
    msgEl.innerHTML = readViewSelector + '<div class="main-empty"><h3>No messages</h3><p>This session has no message content.</p></div>';
    return;
  }
  msgEl.innerHTML = readViewSelector + messageBlocksHtml(c.messages);
}

function renderReadViewExecution(c, viewId) {
  var key = readViewCacheKey(c.id, viewId);
  if (state.readViewErrors[key]) {
    return '<div class="main-empty"><h3>Read view unavailable</h3><p>' + esc(state.readViewErrors[key]) + '</p></div>';
  }
  var envelope = state.readViewPayloads[key];
  if (envelope === undefined) {
    loadReadViewExecution(c.id, viewId);
    return '<div class="main-empty"><h3>Loading ' + esc(viewId) + '...</h3></div>';
  }
  var payload = envelope.payload || {};
  if (viewId === 'recovery') return renderRecoveryReadView(payload);
  if (viewId === 'raw') return renderRawReadView(payload);
  if (viewId === 'context') return renderContextReadView(payload);
  if (viewId === 'context-pack') return renderContextPackReadView(payload);
  if (viewId === 'neighbors') return renderNeighborsReadView(payload);
  if (viewId === 'correlation') return renderCorrelationReadView(payload);
  return '<div class="main-empty"><h3>Unsupported read view</h3><p>' + esc(viewId) + '</p></div>';
}

function renderRecoveryReadView(payload) {
  var packet = payload.work_packet || {};
  var entries = packet.entries || [];
  var html = '<div class="read-view-panel"><h3>Recovery work packet</h3>'
    + '<p class="muted">Read through /api/sessions/:id/read using the shared recovery DTO.</p>'
    + '<div class="inspector-field"><span class="label">entries</span><span class="value">' + esc(String(entries.length)) + '</span></div>'
    + '<div class="inspector-field"><span class="label">evidence refs</span><span class="value">' + esc(String((packet.evidence_refs || []).length)) + '</span></div>';
  entries.slice(0, 10).forEach(function(entry) {
    html += '<div class="annotation-item"><div class="meta">' + esc(entry.section || 'entry') + ' / ' + esc(entry.support || 'support') + '</div>'
      + '<div class="note"><strong>' + esc(entry.label || 'entry') + '</strong><br>' + esc(entry.text || '') + '</div></div>';
  });
  if (!entries.length) html += '<div class="inspector-empty">No recovery entries surfaced for this session.</div>';
  html += '</div>';
  return html;
}

function renderRawReadView(payload) {
  var artifacts = payload.raw_artifacts || [];
  var html = '<div class="read-view-panel"><h3>Raw evidence</h3>'
    + '<p class="muted">Raw bytes stay behind explicit artifact preview; this view lists sanitized artifact metadata.</p>'
    + '<div class="inspector-field"><span class="label">artifacts</span><span class="value">' + esc(String(payload.raw_artifacts_total || artifacts.length || 0)) + '</span></div>';
  artifacts.slice(0, 12).forEach(function(artifact) {
    html += '<div class="annotation-item"><div class="meta">' + esc(artifact.artifact_kind || artifact.provider || 'artifact') + '</div>'
      + '<div class="note">' + esc(artifact.raw_id || artifact.artifact_id || 'raw artifact') + '</div></div>';
  });
  if (!artifacts.length) html += '<div class="inspector-empty">No raw artifact metadata surfaced for this session.</div>';
  html += '</div>';
  return html;
}

function renderContextReadView(payload) {
  var lineage = payload.session_lineage || {};
  var related = payload.recent_related_sessions || [];
  var project = payload.project_state || {};
  var guidance = payload.guidance || {};
  var assertions = guidance.assertions || [];
  var html = '<div class="read-view-panel"><h3>Context preamble</h3>'
    + '<p class="muted">Read through /api/sessions/:id/read using the shared ContextPreamble DTO.</p>'
    + '<div class="inspector-field"><span class="label">root</span><span class="value">' + esc(lineage.logical_session_root || '-') + '</span></div>'
    + '<div class="inspector-field"><span class="label">parent</span><span class="value">' + esc(lineage.parent_session_id || '-') + '</span></div>'
    + '<div class="inspector-field"><span class="label">repo</span><span class="value">' + esc(project.repo || '-') + '</span></div>'
    + '<div class="inspector-field"><span class="label">branch</span><span class="value">' + esc(project.branch || '-') + '</span></div>';
  html += '<div class="inspector-section"><h4>Related sessions</h4>';
  related.slice(0, 8).forEach(function(session) {
    html += '<div class="annotation-item"><div class="meta">' + esc(session.terminal_state || session.origin || 'related') + '</div>'
      + '<div class="note"><strong>' + esc(session.title || 'Untitled') + '</strong><br>' + esc(session.session_id || '') + '</div></div>';
  });
  if (!related.length) html += '<div class="inspector-empty">No related sessions surfaced for this seed.</div>';
  html += '</div><div class="inspector-section"><h4>Guidance</h4>';
  assertions.slice(0, 8).forEach(function(claim) {
    html += '<div class="annotation-item"><div class="meta">' + esc(claim.kind || 'assertion') + ' / ' + esc(claim.scope_ref || '') + '</div>'
      + '<div class="note">' + esc(claim.text || '') + '</div></div>';
  });
  if (!assertions.length) html += '<div class="inspector-empty">No injectable assertion guidance for this session.</div>';
  html += '</div></div>';
  return html;
}

function renderContextPackReadView(payload) {
  var query = payload.query_context || {};
  var sessions = payload.sessions || [];
  var provenance = payload.provenance || {};
  var html = '<div class="read-view-panel"><h3>Context pack</h3>'
    + '<p class="muted">Read through /api/sessions/:id/read using the shared context-pack DTO.</p>'
    + '<div class="inspector-field"><span class="label">sessions</span><span class="value">' + esc(String(payload.total_sessions || sessions.length || 0)) + '</span></div>'
    + '<div class="inspector-field"><span class="label">messages</span><span class="value">' + esc(String(payload.total_messages || 0)) + '</span></div>'
    + '<div class="inspector-field"><span class="label">strategy</span><span class="value">' + esc(query.match_strategy || 'session-id') + '</span></div>'
    + '<div class="inspector-field"><span class="label">redacted</span><span class="value">' + esc(String(provenance.redacted !== false)) + '</span></div>';
  sessions.slice(0, 5).forEach(function(session) {
    var messages = session.messages || [];
    html += '<div class="annotation-item"><div class="meta">' + esc(session.origin || 'origin') + ' / ' + esc(session.session_id || 'session') + '</div>'
      + '<div class="note"><strong>' + esc(session.title || 'Untitled') + '</strong><br>'
      + esc(String(messages.length)) + ' messages included</div></div>';
  });
  if (!sessions.length) html += '<div class="inspector-empty">No context-pack sessions surfaced for this selection.</div>';
  html += '</div>';
  return html;
}

function renderNeighborsReadView(payload) {
  var neighbors = payload.neighbors || [];
  var html = '<div class="read-view-panel"><h3>Neighbor sessions</h3>'
    + '<p class="muted">Read through /api/sessions/:id/read using the shared neighbor-candidate DTO.</p>'
    + '<div class="inspector-field"><span class="label">candidates</span><span class="value">' + esc(String(neighbors.length)) + '</span></div>';
  neighbors.slice(0, 10).forEach(function(candidate) {
    var session = candidate.session || {};
    var reasons = candidate.reasons || [];
    var reasonText = reasons.map(function(reason) {
      return (reason.kind || 'reason') + ': ' + (reason.detail || '');
    }).join(' / ');
    html += '<div class="annotation-item"><div class="meta">rank ' + esc(String(candidate.rank || '?'))
      + ' / score ' + esc(String(candidate.score || 0)) + '</div>'
      + '<div class="note"><strong>' + esc(session.display_title || session.title || session.id || 'Untitled') + '</strong><br>'
      + esc(session.id || session.session_id || '') + '<br><span class="muted">' + esc(reasonText || 'No reasons reported') + '</span></div></div>';
  });
  if (!neighbors.length) html += '<div class="inspector-empty">No neighboring sessions surfaced for this seed.</div>';
  html += '</div>';
  return html;
}

function renderCorrelationReadView(payload) {
  var commits = payload.commits || [];
  var issues = payload.issue_refs || [];
  var prs = payload.pr_refs || [];
  var files = payload.file_paths || [];
  var html = '<div class="read-view-panel"><h3>Correlation evidence</h3>'
    + '<p class="muted">Read through /api/sessions/:id/read using the shared correlation payload.</p>'
    + '<div class="inspector-field"><span class="label">window</span><span class="value">' + esc((payload.window_start || '-') + ' -> ' + (payload.window_end || '-')) + '</span></div>'
    + '<div class="inspector-field"><span class="label">commits</span><span class="value">' + esc(String(commits.length)) + '</span></div>'
    + '<div class="inspector-field"><span class="label">issues / PRs</span><span class="value">' + esc(String(issues.length + prs.length)) + '</span></div>'
    + '<div class="inspector-field"><span class="label">files</span><span class="value">' + esc(String(files.length)) + '</span></div>';
  commits.slice(0, 8).forEach(function(commit) {
    html += '<div class="annotation-item"><div class="meta">' + esc(commit.detection_method || 'commit') + ' / confidence ' + esc(String(commit.confidence || 0)) + '</div>'
      + '<div class="note"><strong>' + esc(commit.short_sha || commit.commit_sha || 'commit') + '</strong><br>' + esc(commit.object_ref || '') + '</div></div>';
  });
  issues.concat(prs).slice(0, 8).forEach(function(ref) {
    html += '<div class="annotation-item"><div class="meta">' + esc(ref.kind || 'ref') + '</div>'
      + '<div class="note"><strong>' + esc(ref.object_ref || ref.raw_match || 'ref') + '</strong><br>' + esc(ref.url || '') + '</div></div>';
  });
  if (!commits.length && !issues.length && !prs.length && !files.length) {
    html += '<div class="inspector-empty">No commit, issue, PR, or file evidence surfaced in this session window.</div>';
  }
  html += '</div>';
  return html;
}

function messageBlocksHtml(messages) {
  // Delegates to the MK3 reader slice (#1202) which owns per-message
  // action rails, fold policies (tool / thinking / code), and the
  // keyboard-focused message card. Defined in web_shell_reader.py.
  return renderMessageBlocks(messages);
}

// --- Topology branch chip + parent-chain stack (#1203) ----------------
// The branch chip is rendered on the session header and reflects
// the resolved incoming edge kind (continuation / sidechain / fork /
// subagent). Clicking it opens the Lineage inspector tab as the
// "branch popover" — siblings, parent, and descendants are already
// projected there from the lineage envelope.
function renderTopologyBranchChip(c) {
  if (!c || !c.id) return '';
  var data = state.lineage;
  if (data === undefined) {
    // Trigger a lazy load and render a placeholder so the chip becomes
    // accurate on the next render pass without blocking initial paint.
    if (typeof loadLineage === 'function') loadLineage(c.id);
    return '<span class="chip q-unavailable" title="Loading lineage..." onclick="openLineageInspector()">branch: ?</span>';
  }
  if (data && data.error) {
    return '<span class="chip q-unresolved" title="Lineage unavailable" onclick="openLineageInspector()">branch: ?</span>';
  }
  // Find the resolved incoming edge to determine the branch kind.
  var branchKind = null;
  (data.edges || []).forEach(function(edge) {
    if (edge.resolved && edge.child_id === c.id && edge.parent_id && !branchKind) {
      branchKind = edge.kind;
    }
  });
  if (!branchKind) {
    // No resolved parent — surface root vs isolated based on node count.
    if ((data.nodes || []).length > 1) {
      return '<span class="chip q-canonical" title="Topology root — has descendants" onclick="openLineageInspector()">root</span>';
    }
    return ''; // Isolated leaf: no branch chip.
  }
  var cls;
  if (branchKind === 'subagent' || branchKind === 'sidechain') cls = 'q-heuristic';
  else if (branchKind === 'fork') cls = 'q-estimated';
  else cls = 'q-canonical';
  return '<span class="chip ' + esc(cls) + '" title="Branch type — click for siblings/parent" '
    + 'onclick="openLineageInspector()">' + esc(branchKind) + '</span>';
}

function renderOpenParentChainButton(c) {
  if (!c || !c.id) return '';
  var data = state.lineage;
  if (data === undefined || !data || data.error) {
    // The button is only meaningful when we know the chain is non-trivial.
    return '';
  }
  // Only show when there is at least one ancestor — otherwise the stack
  // would degenerate to the same session.
  var hasAncestor = (data.edges || []).some(function(edge) {
    return edge.resolved && edge.child_id === c.id && edge.parent_id;
  });
  if (!hasAncestor && (data.nodes || []).length <= 1) return '';
  return '<button class="chip accent" style="cursor:pointer;border:1px solid var(--accent-soft);background:var(--accent-bg);color:var(--accent)" '
    + 'title="Open the root \u2192 sub-agent \u2192 continuation chain as a stack workspace" '
    + 'onclick="openParentChainAsStack(\'' + escAttr(c.id) + '\')">open chain</button>';
}

function openLineageInspector() {
  state.inspectorTab = 'lineage';
  document.querySelectorAll('#inspector-tabs button').forEach(function(b) {
    b.classList.toggle('active', b.dataset.tab === 'lineage');
  });
  renderInspector();
}


function markButtonHtml(sessionId, markType, label, title) {
  var active = hasMark(sessionId, markType) ? ' active' : '';
  return '<button class="mark-btn' + active + '" title="' + escAttr(title) + '" onclick="toggleMark(\'' + escAttr(markType) + '\')">' + esc(label) + '</button>';
}

function renderInspector() {
  var el = document.getElementById('inspector-content');
  if (!state.selected) { el.innerHTML = '<div class="inspector-empty">Select a session to inspect</div>'; return; }
  var c = state.selected;
  var tab = state.inspectorTab || 'info';
  if (tab === 'info') renderInspectorInfo(el, c);
  else if (tab === 'cost') renderInspectorCost(el, c);
  else if (tab === 'lineage') renderInspectorLineage(el, c);
  else if (tab === 'insights') renderInspectorInsights(el, c);
  else if (tab === 'evidence') renderInspectorEvidence(el, c);
  else if (tab === 'raw') renderInspectorRaw(el, c);
  else if (tab === 'similar') renderInspectorSimilar(el, c);
  else if (tab === 'attachments') renderInspectorAttachments(el, c);
  else if (tab === 'notes') renderInspectorNotes(el, c);
}

// --- Insights browser (#1120) -------------------------------------------
// Loads /api/insights/sessions/{id} on demand and caches per-session
// in state.insightsPanels. Each kind (profile/timeline/phases/threads)
// surfaces a readiness chip driven by the daemon-served q-* vocabulary
// (q-ready / q-partial / q-missing). The panel never goes blank — a
// kind with no materialized row is rendered with an explicit q-missing
// chip and a "no rows recorded" body.
async function loadInsightsPanel(id) {
  try {
    var data = await fetchJSON('/api/insights/sessions/' + encodeURIComponent(id));
    state.insightsPanels[id] = data;
  } catch(e) {
    state.insightsPanels[id] = {error: String(e)};
  }
  if (state.selected && state.selected.id === id && state.inspectorTab === 'insights') {
    renderInspector();
  }
}

function insightSectionKey(convId, kind) { return convId + ':' + kind; }

function toggleInsightSection(kind) {
  if (!state.selected) return;
  var k = insightSectionKey(state.selected.id, kind);
  state.insightsCollapsed[k] = !state.insightsCollapsed[k];
  renderInspector();
}

function readinessChip(tag) {
  var label = (tag || 'q-missing').replace('q-', '');
  return '<span class="chip ' + esc(tag || 'q-missing') + '" title="readiness: ' + esc(tag || 'q-missing') + '">' + esc(label) + '</span>';
}

function insightSectionHeader(convId, kind, title, tag, count) {
  var collapsed = state.insightsCollapsed[insightSectionKey(convId, kind)];
  var arrow = collapsed ? '+' : '&minus;';
  var meta = (count != null) ? ' <span class="muted">(' + esc(String(count)) + ')</span>' : '';
  return '<h4 class="insight-section-header" onclick="toggleInsightSection(\'' + escAttr(kind) + '\')">'
    + '<span class="insight-toggle">' + arrow + '</span> '
    + esc(title) + ' ' + readinessChip(tag) + meta + '</h4>';
}

function renderInsightProfile(convId, body) {
  if (!body) body = {readiness_tag: 'q-missing', materialized: false, profile: null};
  var tag = body.readiness_tag || 'q-missing';
  var html = insightSectionHeader(convId, 'profile', 'Profile', tag, null);
  if (state.insightsCollapsed[insightSectionKey(convId, 'profile')]) return html;
  var prof = body.profile;
  if (!prof) {
    html += '<div class="inspector-field"><span class="value muted">No session profile materialized.</span></div>';
    return html;
  }
  var fields = [
    ['Messages', prof.message_count],
    ['Substantive', prof.substantive_count],
    ['Tool uses', prof.tool_use_count],
    ['Thinking', prof.thinking_count],
    ['Words', prof.word_count],
    ['Duration (ms)', prof.total_duration_ms],
    ['Wall (ms)', prof.wall_duration_ms],
    ['Tags', (prof.tags || []).join(', ')],
    ['Auto-tags', (prof.auto_tags || []).join(', ')]
  ];
  fields.forEach(function(row) {
    var v = (row[1] == null || row[1] === '') ? '\u2014' : row[1];
    html += '<div class="inspector-field"><span class="label">' + esc(row[0]) + '</span>'
      + '<span class="value">' + esc(String(v)) + '</span></div>';
  });
  return html;
}

function renderInsightTimeline(convId, body) {
  if (!body) body = {readiness_tag: 'q-missing', count: 0, events: []};
  var tag = body.readiness_tag || 'q-missing';
  var count = body.count || (body.events ? body.events.length : 0);
  var html = insightSectionHeader(convId, 'timeline', 'Work events', tag, count);
  if (state.insightsCollapsed[insightSectionKey(convId, 'timeline')]) return html;
  var events = body.events || [];
  if (!events.length) {
    html += '<div class="inspector-field"><span class="value muted">No work events recorded.</span></div>';
    return html;
  }
  events.slice(0, 50).forEach(function(ev) {
    var inf = ev.inference || {};
    var kind = inf.kind || inf.event_type || 'event';
    var summary = inf.summary || inf.label || '';
    html += '<div class="inspector-field"><span class="label">#' + esc(String(ev.event_index)) + ' '
      + esc(kind) + '</span><span class="value">' + esc(summary) + '</span></div>';
  });
  if (events.length > 50) {
    html += '<div class="inspector-field"><span class="value muted">+' + (events.length - 50) + ' more</span></div>';
  }
  return html;
}

function renderInsightPhases(convId, body) {
  if (!body) body = {readiness_tag: 'q-missing', count: 0, phases: []};
  var tag = body.readiness_tag || 'q-missing';
  var count = body.count || (body.phases ? body.phases.length : 0);
  var html = insightSectionHeader(convId, 'phases', 'Phases', tag, count);
  if (state.insightsCollapsed[insightSectionKey(convId, 'phases')]) return html;
  var phases = body.phases || [];
  if (!phases.length) {
    html += '<div class="inspector-field"><span class="value muted">No phases recorded.</span></div>';
    return html;
  }
  phases.forEach(function(ph) {
    var inf = ph.inference || {};
    var label = inf.label || inf.phase_type || ('phase ' + ph.phase_index);
    var summary = inf.summary || '';
    html += '<div class="inspector-field"><span class="label">#' + esc(String(ph.phase_index)) + ' '
      + esc(label) + '</span><span class="value">' + esc(summary) + '</span></div>';
  });
  return html;
}

function renderInsightThreads(convId, body) {
  if (!body) body = {readiness_tag: 'q-missing', count: 0, threads: []};
  var tag = body.readiness_tag || 'q-missing';
  var count = body.count || (body.threads ? body.threads.length : 0);
  var html = insightSectionHeader(convId, 'threads', 'Work threads', tag, count);
  if (state.insightsCollapsed[insightSectionKey(convId, 'threads')]) return html;
  var threads = body.threads || [];
  if (!threads.length) {
    html += '<div class="inspector-field"><span class="value muted">No thread membership recorded.</span></div>';
    return html;
  }
  threads.forEach(function(th) {
    var t = th.thread || {};
    var sessionCount = (t.session_ids || []).length || t.session_count || 0;
    var meta = (th.dominant_repo ? th.dominant_repo + ' \u00b7 ' : '')
      + sessionCount + ' sessions';
    html += '<div class="inspector-field"><span class="label">' + esc(th.thread_id.slice(0, 8)) + '</span>'
      + '<span class="value">' + esc(meta) + '</span></div>';
  });
  return html;
}

function renderInspectorInsights(el, c) {
  var data = state.insightsPanels[c.id];
  if (data === undefined) {
    el.innerHTML = '<div class="inspector-empty">Loading insights...</div>';
    loadInsightsPanel(c.id);
    return;
  }
  if (data && data.error) {
    el.innerHTML = '<div class="inspector-empty">Insights surface unavailable</div>';
    return;
  }
  var kinds = data.kinds || {};
  var html = '<div class="inspector-section">'
    + renderInsightProfile(c.id, kinds.profile)
    + '</div>'
    + '<div class="inspector-section">'
    + renderInsightTimeline(c.id, kinds.timeline)
    + '</div>'
    + '<div class="inspector-section">'
    + renderInsightPhases(c.id, kinds.phases)
    + '</div>'
    + '<div class="inspector-section">'
    + renderInsightThreads(c.id, kinds.threads)
    + '</div>';
  el.innerHTML = html;
}

// --- Cost panel (#1122) --------------------------------------------------
// Loads /api/sessions/{id}/cost on demand and caches per-session
// in state.costPanels. Each visible number carries a confidence chip
// driven by the MK3 q-* vocabulary returned by the daemon (q-canonical /
// q-estimated / q-heuristic / q-unavailable).
async function loadCostPanel(id) {
  try {
    var data = await fetchJSON('/api/sessions/' + encodeURIComponent(id) + '/cost');
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

async function loadEvidencePanel(id) {
  try {
    var recovery = await fetchJSON('/api/sessions/' + encodeURIComponent(id) + '/recovery?report=work-packet&format=json');
    var assertions = await fetchJSON('/api/assertions?target_ref=' + encodeURIComponent('session:' + id) + '&limit=20');
    state.evidencePanels[id] = {recovery: recovery, assertions: assertions};
  } catch(e) {
    state.evidencePanels[id] = {error: String(e)};
  }
  if (state.selected && state.selected.id === id && state.inspectorTab === 'evidence') {
    renderInspector();
  }
}

function renderRefList(refs) {
  if (!refs || !refs.length) return '<div class="inspector-empty">No refs surfaced</div>';
  var html = '';
  refs.slice(0, 12).forEach(function(ref) {
    var label = (typeof ref === 'string') ? ref : (ref.ref || ref.object_ref || JSON.stringify(ref));
    html += '<div class="user-state-row"><span class="label">ref</span><span class="value">' + esc(label) + '</span></div>';
  });
  if (refs.length > 12) html += '<div class="inspector-field"><span class="value muted">+' + (refs.length - 12) + ' more refs</span></div>';
  return html;
}

function renderInspectorEvidence(el, c) {
  var panel = state.evidencePanels[c.id];
  if (panel === undefined) {
    el.innerHTML = '<div class="inspector-empty">Loading evidence...</div>';
    loadEvidencePanel(c.id);
    return;
  }
  if (!panel || panel.error) {
    el.innerHTML = '<div class="inspector-empty">Evidence surface unavailable</div>';
    return;
  }
  var packet = (panel.recovery && panel.recovery.work_packet) || {};
  var assertions = (panel.assertions && panel.assertions.items) || [];
  var html = '<div class="inspector-section"><h4>Work packet</h4>'
    + '<div class="inspector-field"><span class="label">entries</span><span class="value">' + esc(String((packet.entries || []).length)) + '</span></div>'
    + '<div class="inspector-field"><span class="label">evidence refs</span><span class="value">' + esc(String((packet.evidence_refs || []).length)) + '</span></div>'
    + '</div>';
  if ((packet.entries || []).length) {
    html += '<div class="inspector-section"><h4>Packet entries</h4>';
    (packet.entries || []).slice(0, 8).forEach(function(entry) {
      html += '<div class="annotation-item"><div class="meta">' + esc(entry.section || 'entry') + ' / ' + esc(entry.support || 'support') + '</div>'
        + '<div class="note"><strong>' + esc(entry.label || 'entry') + '</strong><br>' + esc(entry.text || '') + '</div>'
        + '</div>';
    });
    html += '</div>';
  }
  html += '<div class="inspector-section"><h4>Evidence refs</h4>' + renderRefList(packet.evidence_refs || []) + '</div>';
  html += '<div class="inspector-section"><h4>Target refs</h4>' + renderRefList(packet.target_refs || []) + '</div>';
  html += '<div class="inspector-section"><h4>Assertions</h4>';
  if (assertions.length) {
    assertions.forEach(function(claim) {
      html += '<div class="annotation-item"><div class="meta">' + esc(claim.kind || 'claim') + ' / ' + esc(claim.status || '-') + '</div>'
        + '<div class="note">' + esc(claim.body_text || claim.assertion_id || '') + '</div>'
        + '<div class="meta">' + esc(claim.target_ref || '') + '</div></div>';
    });
  } else {
    html += '<div class="inspector-empty">No assertion-backed overlays for this session</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

function renderInspectorInfo(el, c) {
  var fields = [
    ['ID', c.id], ['Origin', c.origin], ['Model', c.model],
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
  // "Compare with..." entry point (#1124). Prompts for another session
  // id and opens the side-by-side compare workspace. Kept as a thin operator
  // shortcut here; richer pickers (recent sessions, lineage parent) can
  // hook into the same ``openCompareWith`` helper without changing the route.
  html += '<div class="inspector-section"><h4>Compare</h4>'
    + '<button class="user-action" onclick="openCompareWith()">Compare with\u2026</button>'
    + '</div>';
  el.innerHTML = html;
}

__PROVENANCE_JS__

__LINEAGE_JS__

__SIMILAR_JS__

async function openCompareWith() {
  if (!state.selected) return;
  var other = window.prompt('Other session id', '');
  if (!other) return;
  other = other.trim();
  if (!other) return;
  if (other === state.selected.id) {
    state.userStateError = 'Cannot compare a session with itself';
    renderInspector();
    return;
  }
  await loadWorkspaceRoute({mode: 'compare', left: state.selected.id, right: other, align: 'prompt'}, true);
}

function renderInspectorNotes(el, c) {
  var marks = Object.keys(markSetFor(c.id));
  var querySummary = [];
  if (state.query) querySummary.push('query=' + state.query);
  if (state.origin) querySummary.push('origin=' + state.origin);
  var html = '<div class="inspector-section"><h4>Marks</h4>';
  if (marks.length) {
    html += '<div class="user-state-row"><span class="label">Active</span><span class="value">' + esc(marks.sort().join(', ')) + '</span></div>';
  } else {
    html += '<div class="inspector-empty">No marks on this session</div>';
  }
  html += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:8px">'
    + '<button class="user-action" onclick="toggleMark(\'star\')">Star</button>'
    + '<button class="user-action" onclick="toggleMark(\'pin\')">Pin</button>'
    + '<button class="user-action" onclick="toggleMark(\'archive\')">Archive</button>'
    + '</div></div>';
  html += '<div class="inspector-section"><h4>Saved Views</h4>'
    + '<div class="user-state-row"><span class="label">Current</span><span class="value">' + esc(querySummary.join(' / ') || 'all sessions') + '</span></div>'
    + '<button class="user-action" onclick="saveCurrentView()">Save current view</button>';
  if (state.savedViews.length) {
    html += '<div class="saved-view-list" style="margin-top:8px">';
    state.savedViews.forEach(function(v) {
      var q = v.query || {};
      var bits = [];
      if (q.query) bits.push('query=' + q.query);
      if (q.origin) bits.push('origin=' + q.origin);
      html += '<div class="saved-view-item" data-view-id="' + escAttr(v.view_id) + '"><div><div>' + esc(v.name || v.view_id) + '</div>'
        + '<div class="value">' + esc(bits.join(' / ') || 'all sessions') + '</div></div>'
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
    + renderAnnotationComposer(c);
  var annotations = annotationsFor(c.id);
  if (annotations.length) {
    html += '<div class="annotation-list">';
    annotations.forEach(function(a) {
      var target = a.target_type === 'message' ? ('message ' + (a.message_id || a.target_id)) : 'session';
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
    html += '<div class="inspector-empty">No annotations on this session</div>';
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
      await sendJSON('/api/user/marks?session_id=' + encodeURIComponent(id) + '&mark_type=' + encodeURIComponent(markType), 'DELETE');
      setMarkLocal(id, markType, false);
    } else {
      await sendJSON('/api/user/marks', 'POST', {session_id: id, mark_type: markType});
      setMarkLocal(id, markType, true);
    }
    state.userStateError = '';
  } catch(e) {
    state.userStateError = 'Failed to update mark';
  }
  renderSessions();
  renderMain();
  renderInspector();
}

function applySavedView(viewId) {
  var view = state.savedViews.find(function(v) { return v.view_id === viewId; });
  if (!view) return;
  var query = view.query || {};
  state.query = query.query || '';
  state.origin = query.origin || '';
  state.offset = 0;
  document.getElementById('search').value = state.query;
  loadSessions();
  loadFacets();
  renderInspector();
}

function findAnnotation(annotationId) {
  var cid = state.selected ? state.selected.id : '';
  var annotations = annotationsFor(cid);
  return annotations.find(function(a) { return a.annotation_id === annotationId; }) || null;
}

function messageTargetLabel(message) {
  var role = message.role || message.author_role || 'message';
  var text = (message.text || message.content || '').replace(/\s+/g, ' ').trim();
  if (text.length > 54) text = text.slice(0, 51) + '...';
  return role + (text ? ': ' + text : '');
}

function renderAnnotationComposer(c) {
  var html = '<div id="annotation-composer" class="annotation-composer" data-editing-id="">'
    + '<label for="annotation-target-select">Target</label>'
    + '<select id="annotation-target-select">'
    + '<option value="session::' + escAttr(c.id) + '">Session</option>';
  (c.messages || []).forEach(function(message, idx) {
    var messageId = message.id || message.message_id;
    if (!messageId) return;
    html += '<option value="message::' + escAttr(messageId) + '">Message ' + esc(String(idx + 1)) + ' - '
      + esc(messageTargetLabel(message)) + '</option>';
  });
  html += '</select>'
    + '<label for="annotation-note-input">Note</label>'
    + '<textarea id="annotation-note-input" placeholder="Add an operator note for this target"></textarea>'
    + '<div class="annotation-actions">'
    + '<button class="user-action" onclick="saveAnnotation()">Save note</button>'
    + '<button class="user-action" onclick="clearAnnotationComposer()">Clear</button>'
    + '</div></div>';
  return html;
}

function clearAnnotationComposer() {
  var composer = document.getElementById('annotation-composer');
  var target = document.getElementById('annotation-target-select');
  var note = document.getElementById('annotation-note-input');
  if (composer) composer.dataset.editingId = '';
  if (target) target.selectedIndex = 0;
  if (note) note.value = '';
}

function annotationTargetFromComposer() {
  var target = document.getElementById('annotation-target-select');
  var selected = target ? target.value : '';
  var parts = selected.split('::');
  var targetType = parts[0] || 'session';
  var targetId = parts.slice(1).join('::') || (state.selected ? state.selected.id : '');
  return {targetType: targetType, targetId: targetId};
}

async function saveAnnotation(annotationId) {
  if (!state.selected) return;
  var composer = document.getElementById('annotation-composer');
  var noteInput = document.getElementById('annotation-note-input');
  var note = noteInput ? noteInput.value.trim() : '';
  if (!note) {
    state.userStateError = 'Annotation note is required';
    renderInspector();
    return;
  }
  var target = annotationTargetFromComposer();
  var editingId = annotationId || (composer && composer.dataset.editingId) || '';
  var id = editingId || ('annotation-' + Date.now().toString(36));
  var payload = {
    annotation_id: id,
    session_id: state.selected.id,
    note_text: note,
    target_type: target.targetType
  };
  if (target.targetType === 'message') {
    payload.message_id = target.targetId;
  } else {
    payload.target_id = state.selected.id;
  }
  try {
    await sendJSON('/api/user/annotations', 'POST', payload);
    state.userStateError = '';
    await loadUserState();
  } catch(e) {
    state.userStateError = 'Failed to save annotation';
    renderInspector();
  }
}

function editAnnotation(annotationId) {
  var existing = findAnnotation(annotationId);
  if (!existing) return;
  var composer = document.getElementById('annotation-composer');
  var target = document.getElementById('annotation-target-select');
  var note = document.getElementById('annotation-note-input');
  if (composer) composer.dataset.editingId = annotationId;
  if (note) note.value = existing.note_text || '';
  if (target) {
    var targetValue = existing.target_type === 'message'
      ? ('message::' + (existing.message_id || existing.target_id || ''))
      : ('session::' + existing.session_id);
    target.value = targetValue;
  }
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
  if (!area) return;
  area.innerHTML = '<div style="color:var(--text-dim);font-size:var(--small);padding:8px 0">Loading artifact list...</div>';
  try {
    await loadSessionRaw(id);
    var raw = state.selectedRaw;
    if (!raw) { area.innerHTML = '<div class="inspector-empty">No raw artifact metadata available</div>'; return; }
    var artifacts = raw.raw_artifacts || ((raw.raw_id || raw.content_hash) ? [raw] : []);
    var total = raw.raw_artifacts_total != null ? raw.raw_artifacts_total : artifacts.length;
    var html = '<div class="inspector-section"><h4>Raw artifact metadata (' + esc(String(total)) + ')</h4>';
    if (!artifacts.length) {
      html += '<div class="inspector-empty">No raw artifacts are linked to this session.</div>';
    }
    artifacts.slice(0, 20).forEach(function(artifact) {
      var rows = [
        ['Raw id', artifact.raw_id],
        ['Source', artifact.source_name || artifact.source_kind],
        ['Source path', artifact.source_path_display],
        ['Content hash', artifact.content_hash],
        ['Blob size', artifact.blob_size_bytes != null ? Number(artifact.blob_size_bytes).toLocaleString() + ' B' : ''],
        ['Acquired', artifact.acquired_at],
        ['Parsed', artifact.parsed_at],
        ['Validation', artifact.validation_status],
        ['Quarantine', artifact.quarantined ? (artifact.quarantine_reason || 'yes') : 'no']
      ];
      html += '<div class="raw-block" style="margin-bottom:6px">';
      rows.forEach(function(row) {
        if (row[1] == null || row[1] === '') return;
        html += '<div class="inspector-field"><span class="label">' + esc(row[0]) + '</span><span class="value">' + esc(String(row[1])) + '</span></div>';
      });
      if (artifact.source_path_is_absolute || artifact.storage_path) {
        html += '<div class="inspector-field"><span class="label">Path policy</span><span class="value">absolute path retained in source tier; shell shows sanitized metadata only</span></div>';
      }
      html += '</div>';
    });
    if (artifacts.length > 20) {
      html += '<div class="inspector-empty">+' + esc(String(artifacts.length - 20)) + ' more artifacts omitted from the shell preview.</div>';
    }
    html += '<div class="inspector-empty" style="padding-top:8px">Raw bytes are behind the bounded provenance preview button above.</div></div>';
    area.innerHTML = html;
  } catch(e) { area.innerHTML = '<div class="inspector-empty">Failed to load artifact list</div>'; }
}

async function selectSession(id, updateURL, opts) {
  // ``opts.liveTail`` is set by the SSE handler when a message.appended
  // event lands for the currently-open session. We skip the loading
  // placeholder so the message list visibly stays put while the new
  // message animates in, instead of flickering through an empty state.
  var isLiveTail = !!(opts && opts.liveTail);
  var priorMessageIds = {};
  if (isLiveTail && state.selected && state.selected.messages) {
    state.selected.messages.forEach(function(m) { priorMessageIds[m.id] = true; });
  } else {
    document.getElementById('msg-list').innerHTML = '<div class="main-empty"><h3>Loading...</h3></div>';
    document.getElementById('inspector-content').innerHTML = '<div class="inspector-empty">Loading...</div>';
  }
  await loadSession(id, updateURL);
  if (isLiveTail) {
    // Animate any message rendered into the DOM whose id was not present
    // before the live-tail reload — the per-row data-msg-id lookup mirrors
    // the renderer in web_shell_reader.py.
    var rows = document.querySelectorAll('#msg-list [data-msg-id]');
    rows.forEach(function(row) {
      var mid = row.getAttribute('data-msg-id');
      if (mid && !priorMessageIds[mid]) animateAppendedMessage(row);
    });
  }
  restartRealtimeForView();
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
    if (state.query) { state.query = ''; document.getElementById('search').value = ''; state.offset = 0; loadSessions(); return; }
    if (state.origin) { state.origin = ''; state.offset = 0; loadSessions(); renderFacets(); return; }
    return;
  }
  if (e.key === 'j' || e.key === 'k') {
    // When a session is open the MK3 reader slice owns j/k for
    // message-card navigation (installed via installReaderShortcuts in
    // web_shell_reader.py). It runs as a capture-phase handler and will
    // have already preventDefault()'d in that case. Here we only drive
    // the sidebar list when no session messages are loaded.
    var convOpen = !!(state.selected && state.selected.messages && state.selected.messages.length);
    if (convOpen) return;
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
    if (state.offset + state.limit < state.total) { state.offset += state.limit; loadSessions(); }
  }
  if (e.key === 'p') {
    e.preventDefault();
    if (state.offset > 0) { state.offset = Math.max(0, state.offset - state.limit); loadSessions(); }
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
  searchTimer = setTimeout(function() { state.query = e.target.value; state.offset = 0; loadSessions(); loadFacets(); }, 280);
});

document.getElementById('facet-bar').addEventListener('click', function(e) {
  var chip = e.target.closest('.facet-chip');
  if (!chip) return;
  var facet = chip.dataset.facet;
  var value = chip.dataset.value;
  if (facet === 'origin') { state.origin = value || ''; state.offset = 0; loadSessions(); renderFacets(); }
});

attachBulkHandlers();

document.getElementById('inspector-tabs').addEventListener('click', function(e) {
  if (e.target.tagName !== 'BUTTON') return;
  state.inspectorTab = e.target.dataset.tab;
  document.querySelectorAll('#inspector-tabs button').forEach(function(b) { b.classList.remove('active'); });
  e.target.classList.add('active');
  renderInspector();
});

loadSessions().then(function() {
  var route = getWorkspaceRouteFromURL();
  if (route) { loadWorkspaceRoute(route, false); return; }
  var cid = getSessionIdFromURL();
  if (cid) selectSession(cid, false);
});
loadFacets();
loadReadViewProfiles();
loadUserState();
loadStatus();

__REALTIME_JS__
__READER_JS__
__PASTE_JS__
__ATTACHMENT_JS__
</script>
</body>
</html>""".replace("__WORKSPACE_CSS__", WORKSPACE_CSS)
    .replace("__WORKSPACE_HTML__", WORKSPACE_HTML)
    .replace("__WORKSPACE_JS__", WORKSPACE_JS)
    .replace("__BULK_CSS__", BULK_CSS)
    .replace("__BULK_TOOLBAR_HTML__", BULK_TOOLBAR_HTML)
    .replace("__BULK_PREVIEW_HTML__", BULK_PREVIEW_HTML)
    .replace("__BULK_JS__", BULK_JS)
    .replace("__PROVENANCE_JS__", PROVENANCE_JS)
    .replace("__LINEAGE_JS__", LINEAGE_JS)
    .replace("__SIMILAR_JS__", SIMILAR_JS)
    .replace("__READER_CSS__", READER_CSS)
    .replace("__READER_HELP_HTML__", READER_HELP_HTML)
    .replace("__READER_JS__", READER_JS)
    .replace("__REALTIME_JS__", REALTIME_JS)
    .replace("__PASTE_CSS__", PASTE_CSS)
    .replace("__PASTE_JS__", PASTE_JS)
    .replace("__ATTACHMENT_CSS__", ATTACHMENT_CSS)
    .replace("__ATTACHMENT_JS__", ATTACHMENT_JS)
)
