"""Mission-control projection for the daemon web shell.

The web panel is a projection over ``/api/agents/coordination``. It must not
grow a separate coordination ontology; CLI, MCP, and web all consume the same
``AgentCoordinationPayload``.
"""

COORDINATION_CSS = r"""
.mission-card { border:1px solid var(--border); background:var(--panel-subtle); border-radius:4px; padding:8px; margin-bottom:8px; }
.mission-card h4 { font-size:11px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.6px; margin-bottom:6px; }
.mission-row { display:flex; justify-content:space-between; gap:8px; padding:3px 0; border-bottom:1px solid var(--border); font-size:var(--small); }
.mission-row:last-child { border-bottom:0; }
.mission-row .label { color:var(--text-muted); flex-shrink:0; }
.mission-row .value { color:var(--text); font-family:var(--font-mono); font-size:11px; text-align:right; word-break:break-word; }
.mission-list { display:flex; flex-direction:column; gap:5px; }
.mission-item { border-left:2px solid var(--border-strong); padding-left:7px; color:var(--text-muted); font-size:var(--small); }
.mission-item.warning { border-left-color:var(--warn); color:var(--warn); }
.mission-item.critical { border-left-color:var(--err); color:var(--err); }
"""

COORDINATION_JS = r"""
// Mission-control projection over AgentCoordinationPayload (#bby.9).
async function loadCoordinationPanel() {
  try {
    var payload = await fetchJSON('/api/agents/coordination?view=status&limit=8', {timeoutMs: 5000});
    state.coordinationPayload = payload;
  } catch(e) {
    state.coordinationPayload = {error: String(e)};
  }
  if (state.inspectorTab === 'mission') renderInspector();
}

function missionValue(value) {
  if (value == null || value === '') return '&mdash;';
  return esc(String(value));
}

function renderMissionRow(label, value) {
  return '<div class="mission-row"><span class="label">' + esc(label) + '</span><span class="value">' + missionValue(value) + '</span></div>';
}

function renderMissionList(items, renderItem, emptyText) {
  if (!items || !items.length) return '<div class="inspector-field"><span class="value muted">' + esc(emptyText) + '</span></div>';
  return '<div class="mission-list">' + items.map(renderItem).join('') + '</div>';
}

function renderInspectorMission(el) {
  var payload = state.coordinationPayload;
  if (!payload) {
    el.innerHTML = '<div class="inspector-empty">Loading mission control...</div>';
    loadCoordinationPanel();
    return;
  }
  if (payload.error) {
    el.innerHTML = renderInlineRouteFailure('Mission control unavailable', payload, 'loadCoordinationPanel()');
    return;
  }
  var repo = payload.repo || {};
  var self = payload.self || {};
  var work = payload.work_item || {};
  var beads = payload.beads || {};
  var archive = payload.archive || {};
  var merge = beads.merge_slot || {};
  var html = '';
  html += '<div class="mission-card"><h4>Current Work</h4>'
    + renderMissionRow('Work item', (work.ref || 'none') + ' · ' + (work.source || 'unknown') + ' · ' + (work.confidence == null ? '?' : work.confidence))
    + renderMissionRow('Repo', repo.root || repo.cwd)
    + renderMissionRow('Branch', (repo.branch || 'n/a') + '@' + (repo.head || 'n/a'))
    + renderMissionRow('Agent', (self.agent_kind || 'agent') + ' pid=' + (self.pid || 'n/a'))
    + '</div>';
  html += '<div class="mission-card"><h4>Beads / Archive</h4>'
    + renderMissionRow('Hooks', beads.hooks_all_installed === true ? 'ok' : (beads.hooks_all_installed === false ? 'incomplete' : 'unknown'))
    + renderMissionRow('Open gates', beads.open_gate_count)
    + renderMissionRow('Merge slot', (merge.id || 'n/a') + ' · ' + (merge.status || merge.available || merge.error || 'unknown'))
    + renderMissionRow('Index schema', archive.index_user_version)
    + renderMissionRow('Archive root', archive.archive_root)
    + '</div>';
  html += '<div class="mission-card"><h4>Active Agents</h4>' + renderMissionList(payload.peers, function(peer) {
    return '<div class="mission-item">' + esc(peer.kind || 'agent') + ' pid=' + esc(peer.pid) + '<br><span class="muted">' + esc(peer.cwd || 'n/a') + '</span></div>';
  }, 'No peer agents detected.') + '</div>';
  html += '<div class="mission-card"><h4>Resources</h4>' + renderMissionList(payload.resource_episodes, function(res) {
    return '<div class="mission-item">' + esc(res.kind || 'resource') + ' pid=' + esc(res.pid) + ' · ' + esc(res.status || 'unknown') + '</div>';
  }, 'No heavy resources detected.') + '</div>';
  html += '<div class="mission-card"><h4>Overlap Awareness</h4>' + renderMissionList(payload.overlaps, function(overlap) {
    var cls = 'mission-item ' + esc(overlap.severity || 'info');
    var blocking = overlap.blocking ? 'blocking' : 'awareness';
    return '<div class="' + cls + '">' + esc(overlap.severity || 'info') + ' / ' + blocking + ': ' + esc(overlap.summary || '') + '</div>';
  }, 'No overlap warnings.') + '</div>';
  el.innerHTML = html;
}
"""

__all__ = ["COORDINATION_CSS", "COORDINATION_JS"]
