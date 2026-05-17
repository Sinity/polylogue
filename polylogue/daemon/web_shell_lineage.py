"""Lineage-tab assets for the daemon-served reader shell (#1121).

Extracted so the parent ``web_shell.py`` stays within its declared
file-size budget (``docs/plans/file-size-budgets.yaml``). Pure string
constants — composed into the served HTML via ``str.replace`` from
``web_shell.WEB_SHELL_HTML``.
"""

from __future__ import annotations

LINEAGE_CSS = r"""
/* Lineage tree (#1121): renders the SessionTopology rooted view as an
   indented list with edge-kind chips. Each node is clickable; the
   focused conversation is highlighted with the same accent border as
   the sidebar selected item. Quality chips reuse the MK3 vocabulary. */
.lineage-tree { font-size: var(--small); line-height: 1.5; }
.lineage-node { display: flex; align-items: center; gap: 6px; padding: 3px 4px;
  border-radius: 3px; cursor: pointer; color: var(--text-muted); }
.lineage-node:hover { background: var(--panel-elevated); color: var(--text); }
.lineage-node.focused { background: var(--accent-bg); color: var(--accent);
  border-left: 2px solid var(--accent); padding-left: 2px; }
.lineage-node .indent { display: inline-block; color: var(--text-dim);
  font-family: var(--font-mono); white-space: pre; flex-shrink: 0; }
.lineage-node .node-title { flex: 1; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.lineage-node .node-provider { font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); flex-shrink: 0; }
.lineage-edge-kind { font-size: 10px; padding: 0 4px; border-radius: 2px;
  background: var(--panel-subtle); border: 1px solid var(--border);
  color: var(--text-dim); flex-shrink: 0; }
.lineage-edge-kind.kind-continuation { color: var(--ok); border-color: var(--ok); }
.lineage-edge-kind.kind-sidechain { color: var(--accent); border-color: var(--accent-soft); }
.lineage-edge-kind.kind-fork { color: var(--warn); border-color: var(--warn); }
.lineage-edge-kind.kind-subagent { color: var(--role-tool); border-color: var(--accent-soft); }
.lineage-edge-kind.kind-unresolved_native { color: var(--err); border-color: var(--err); }
.lineage-summary { display: flex; gap: 6px; flex-wrap: wrap; padding: 4px 0;
  border-bottom: 1px solid var(--border); margin-bottom: 6px; }
.lineage-actions { display: flex; gap: 4px; margin-top: 8px; padding-top: 6px;
  border-top: 1px solid var(--border); }
"""

LINEAGE_JS = r"""
// Lineage tab (#1121): consumes /api/conversations/:id/topology.
// Renders an explicit readiness chip, the bounded rooted tree as an
// indented list, and an explicit empty state when no lineage exists.
async function loadLineage(id) {
  if (state.lineageLoading) return;
  state.lineageLoading = true;
  state.lineageError = '';
  try {
    state.lineage = await fetchJSON('/api/conversations/' + encodeURIComponent(id) + '/topology');
  } catch(e) {
    state.lineage = null;
    state.lineageError = 'Failed to load lineage';
  } finally {
    state.lineageLoading = false;
  }
  if (state.selected && state.selected.id === id && state.inspectorTab === 'lineage') {
    renderInspector();
  }
}

function renderInspectorLineage(el, c) {
  if (state.lineageLoading) {
    el.innerHTML = '<div class="inspector-empty">Loading lineage...</div>';
    return;
  }
  if (state.lineageError) {
    el.innerHTML = '<div class="inspector-empty">' + esc(state.lineageError) + '</div>';
    return;
  }
  if (!state.lineage || state.lineage.target_id !== c.id) {
    el.innerHTML = '<div class="inspector-empty">Loading lineage...</div>';
    loadLineage(c.id);
    return;
  }
  var topo = state.lineage;
  // Readiness chip uses the MK3 vocabulary so an operator can tell at a
  // glance whether the graph is complete, partial (truncated/unresolved/
  // cycle), or genuinely empty (single-node — no parent and no children).
  var readinessClass = 'q-canonical';
  var readinessLabel = 'complete';
  if (topo.readiness === 'empty') { readinessClass = 'q-unavailable'; readinessLabel = 'no lineage'; }
  else if (topo.readiness === 'partial') { readinessClass = 'q-partial'; readinessLabel = 'partial'; }
  var summaryHtml = '<div class="lineage-summary">'
    + '<span class="chip ' + readinessClass + '" title="Lineage readiness">' + esc(readinessLabel) + '</span>'
    + '<span class="chip">' + topo.node_count + ' nodes</span>'
    + '<span class="chip">' + topo.edge_count + ' edges</span>';
  if (topo.unresolved_edge_count > 0) {
    summaryHtml += '<span class="chip q-unresolved" title="Provider parent IDs that did not resolve to a stored conversation">'
      + topo.unresolved_edge_count + ' unresolved</span>';
  }
  if (topo.cycle_detected) {
    summaryHtml += '<span class="chip q-unresolved" title="Cycle in ancestry — quarantine the archive slice">cycle</span>';
  }
  if (topo.truncated_count > 0) {
    summaryHtml += '<span class="chip q-partial" title="Tree truncated to bounded node limit">+' + topo.truncated_count + ' hidden</span>';
  }
  summaryHtml += '</div>';

  if (topo.readiness === 'empty') {
    el.innerHTML = summaryHtml + '<div class="inspector-empty">This conversation has no parent or child sessions.</div>';
    return;
  }

  // Build a parent-of map from edges so we can render the rooted tree
  // as an indented BFS list. The substrate already orders nodes BFS
  // from root; we re-derive depth via parent links because edges are
  // the durable structural signal.
  var parentOf = {};
  var kindOf = {};
  (topo.edges || []).forEach(function(edge) {
    if (edge.resolved !== false && edge.parent_id) {
      parentOf[String(edge.child_id)] = String(edge.parent_id);
      kindOf[String(edge.child_id)] = edge.kind || 'unknown';
    }
  });
  var children = {};
  Object.keys(parentOf).forEach(function(cid) {
    var pid = parentOf[cid];
    if (!children[pid]) children[pid] = [];
    children[pid].push(cid);
  });
  var byId = {};
  (topo.nodes || []).forEach(function(n) { byId[String(n.conversation_id)] = n; });

  var rootId = String(topo.root_id);
  var lines = [];
  function emit(nodeId, depth) {
    var node = byId[nodeId];
    if (!node) return;
    var indent = '';
    for (var i = 0; i < depth; i++) indent += '  ';
    if (depth > 0) indent += '\u2514\u2500 ';  // └─
    var kind = kindOf[nodeId];
    var kindChip = kind && depth > 0
      ? '<span class="lineage-edge-kind kind-' + esc(kind) + '">' + esc(kind) + '</span>'
      : '';
    var focused = nodeId === String(topo.target_id) ? ' focused' : '';
    var title = esc((node.title || 'Untitled').substring(0, 80));
    var provider = node.provider_name ? '<span class="node-provider">' + esc(node.provider_name) + '</span>' : '';
    lines.push(
      '<div class="lineage-node' + focused + '" data-conv-id="' + escAttr(nodeId) + '" onclick="selectConversation(\'' + escAttr(nodeId) + '\')">'
      + '<span class="indent">' + esc(indent) + '</span>'
      + kindChip
      + '<span class="node-title">' + title + '</span>'
      + provider
      + '</div>'
    );
    var kids = children[nodeId] || [];
    kids.sort();
    kids.forEach(function(cid) { emit(cid, depth + 1); });
  }
  emit(rootId, 0);

  // Surface unresolved native parent edges as a separate section since
  // they cannot be navigated to a stored conversation.
  var unresolvedHtml = '';
  var unresolvedEdges = (topo.edges || []).filter(function(e) { return e.resolved === false; });
  if (unresolvedEdges.length) {
    unresolvedHtml = '<div class="inspector-section"><h4>Unresolved parent IDs</h4>';
    unresolvedEdges.forEach(function(edge) {
      unresolvedHtml += '<div class="inspector-field"><span class="label">'
        + esc(String(edge.child_id).substring(0, 12)) + '\u2026</span>'
        + '<span class="value">' + esc(String(edge.parent_native_id || '-')) + '</span></div>';
    });
    unresolvedHtml += '</div>';
  }

  var actionsHtml = '';
  var parentId = parentOf[c.id];
  if (parentId) {
    actionsHtml = '<div class="lineage-actions">'
      + '<button class="user-action" onclick="selectConversation(\'' + escAttr(parentId) + '\')">Open parent</button>'
      + '<button class="user-action" title="Compare with parent in workspace view" onclick="openCompareWith(\'' + escAttr(parentId) + '\',\'' + escAttr(c.id) + '\')">Compare with parent</button>'
      + '</div>';
  }

  el.innerHTML = summaryHtml
    + '<div class="lineage-tree">' + lines.join('') + '</div>'
    + unresolvedHtml
    + actionsHtml;
}

// Compare-with-parent shortcut (#1121). Delegates to the existing
// compare workspace route added in #1118; if the route helper is not
// wired (older builds), falls back to opening the parent directly.
function openCompareWith(leftId, rightId) {
  if (typeof loadWorkspaceRoute === 'function') {
    loadWorkspaceRoute({mode: 'compare', left: leftId, right: rightId, align: 'prompt'}, true);
    return;
  }
  selectConversation(leftId);
}
"""
