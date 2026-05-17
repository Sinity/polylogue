"""Lineage inspector JS fragment for the web reader (#1121).

Kept as a sibling of :mod:`polylogue.daemon.web_shell` so the main
single-page shell stays under its file-size budget. The fragment is
interpolated into ``WEB_SHELL_HTML`` at module import time. It owns:

- the inspector "Lineage" tab rendering (rooted-tree BFS list with
  edge-kind chips, depth indentation, focused-node highlight);
- the ``/api/conversations/{id}/topology`` fetch call;
- the "Open parent" / "Compare with parent" affordances that delegate
  to existing reader actions (``selectConversation`` and the
  ``openCompareWith`` workspace route from #1124).

The view never queries lineage tables directly: it consumes the public
``SessionTopology`` envelope from #866 via the topology endpoint.
"""

from __future__ import annotations

LINEAGE_JS = r"""
// --- Lineage inspector (#1121) ------------------------------------------
// Renders the rooted-tree projection of SessionTopology (#866) for the
// currently-selected conversation. The fetch is bounded server-side; the
// reader never plots a node outside the envelope it received.

async function loadLineage(id) {
  state.lineage = undefined;
  try {
    var data = await fetchJSON('/api/conversations/' + encodeURIComponent(id) + '/topology');
    state.lineage = data;
  } catch(e) {
    state.lineage = {error: String(e)};
  }
  if (state.selected && state.selected.id === id && state.inspectorTab === 'lineage') {
    renderInspector();
  }
}

function lineageReadinessChip(readiness) {
  if (readiness === 'empty') return '<span class="chip q-unavailable">no lineage</span>';
  if (readiness === 'partial') return '<span class="chip q-partial">partial</span>';
  return '<span class="chip q-canonical">ok</span>';
}

function lineageEdgeKindClass(kind) {
  if (kind === 'unresolved_native') return 'q-unresolved';
  if (kind === 'subagent' || kind === 'sidechain') return 'q-heuristic';
  if (kind === 'fork') return 'q-estimated';
  return 'q-canonical';
}

function lineageNodeLabel(node) {
  var label = node.title || node.conversation_id;
  if (label.length > 48) label = label.substring(0, 45) + '\u2026';
  return label;
}

function findLineageParent(data, conversationId) {
  if (!data || !data.edges) return null;
  for (var i = 0; i < data.edges.length; i++) {
    var edge = data.edges[i];
    if (edge.resolved && edge.child_id === conversationId && edge.parent_id) {
      return edge.parent_id;
    }
  }
  return null;
}

function renderLineageNodeRow(node, edgeKindByChild, focusedId) {
  var depth = Math.max(0, Number(node.depth || 0));
  var indent = 'padding-left:' + (depth * 14) + 'px';
  var focused = node.conversation_id === focusedId;
  var bg = focused ? 'background:var(--panel-elevated);' : '';
  var edgeKind = edgeKindByChild[node.conversation_id];
  var chip = '';
  if (node.is_root) {
    chip = '<span class="chip q-canonical" style="margin-left:4px">root</span>';
  } else if (edgeKind) {
    chip = '<span class="chip ' + esc(lineageEdgeKindClass(edgeKind)) + '" style="margin-left:4px">' + esc(edgeKind) + '</span>';
  }
  var btn = focused
    ? '<span class="value" style="color:var(--accent)">' + esc(lineageNodeLabel(node)) + '</span>'
    : '<button class="user-action" style="padding:0;border:none;background:none;color:var(--accent);cursor:pointer;text-align:left" onclick="selectConversation(\'' + escAttr(node.conversation_id) + '\', true)">' + esc(lineageNodeLabel(node)) + '</button>';
  return '<div class="inspector-field" style="' + indent + ';' + bg + '">'
    + '<span class="label">d' + depth + '</span>'
    + '<span class="value">' + btn + chip + '</span>'
    + '</div>';
}

function renderInspectorLineage(el, c) {
  var data = state.lineage;
  if (data === undefined) {
    el.innerHTML = '<div class="inspector-empty">Loading lineage...</div>';
    loadLineage(c.id);
    return;
  }
  if (data && data.error) {
    el.innerHTML = '<div class="inspector-empty">Lineage unavailable</div>';
    return;
  }
  var html = '';
  html += '<div class="inspector-section"><h4>Lineage</h4>';
  html += '<div style="margin-bottom:6px">' + lineageReadinessChip(data.readiness) + '</div>';
  html += '<div class="inspector-field"><span class="label">Nodes</span>'
    + '<span class="value">' + (data.node_count || 0) + (data.truncated_count ? ' (+' + data.truncated_count + ' truncated)' : '') + '</span></div>';
  if (data.cycle_detected) {
    html += '<div class="inspector-field"><span class="label">Cycle</span>'
      + '<span class="value"><span class="chip q-unresolved">cycle detected</span></span></div>';
  }
  if (data.unresolved_edge_count) {
    html += '<div class="inspector-field"><span class="label">Unresolved</span>'
      + '<span class="value">' + data.unresolved_edge_count + ' native pointer(s)</span></div>';
  }
  html += '</div>';

  if (data.readiness === 'empty' && (!data.nodes || data.nodes.length <= 1)) {
    html += '<div class="inspector-section"><h4>No related sessions</h4>'
      + '<div style="font-size:var(--small);color:var(--text-dim)">This conversation has no resolved parent or descendants.</div></div>';
    el.innerHTML = html;
    return;
  }

  // Build edge-kind map so each non-root row labels its incoming edge.
  var edgeKindByChild = {};
  (data.edges || []).forEach(function(edge) {
    if (edge.resolved && edge.parent_id) {
      edgeKindByChild[edge.child_id] = edge.kind;
    }
  });

  html += '<div class="inspector-section"><h4>Tree</h4>';
  (data.nodes || []).forEach(function(node) {
    html += renderLineageNodeRow(node, edgeKindByChild, c.id);
  });
  html += '</div>';

  // Unresolved native pointer block — provider-native parent IDs that did
  // not resolve to a stored conversation. Surfaced as a dedicated section
  // so late-arriving parents are visible to the operator.
  var unresolvedEdges = (data.edges || []).filter(function(edge) { return !edge.resolved; });
  if (unresolvedEdges.length) {
    html += '<div class="inspector-section"><h4>Unresolved parents</h4>';
    unresolvedEdges.forEach(function(edge) {
      html += '<div class="inspector-field"><span class="label">' + esc(edge.parent_native_id || '?') + '</span>'
        + '<span class="value"><span class="chip q-unresolved">' + esc(edge.kind) + '</span></span></div>';
    });
    html += '</div>';
  }

  // Open parent / Compare with parent actions — both delegate to existing
  // reader entry points; this view stays read-only (#1121 out-of-scope).
  var parentId = findLineageParent(data, c.id);
  html += '<div class="inspector-section"><h4>Actions</h4>';
  if (parentId) {
    html += '<button class="user-action" style="margin-right:6px" onclick="selectConversation(\'' + escAttr(parentId) + '\', true)">Open parent</button>';
    html += '<button class="user-action" onclick="openCompareWithParent()">Compare with parent</button>';
  } else {
    html += '<div style="font-size:var(--small);color:var(--text-dim)">No resolved parent to open or compare against.</div>';
  }
  html += '</div>';

  el.innerHTML = html;
}

async function openCompareWithParent() {
  if (!state.selected) return;
  var data = state.lineage;
  if (!data || data.error) return;
  var parentId = findLineageParent(data, state.selected.id);
  if (!parentId) return;
  await loadWorkspaceRoute({mode: 'compare', left: parentId, right: state.selected.id, align: 'prompt'}, true);
}
"""

__all__ = ["LINEAGE_JS"]
