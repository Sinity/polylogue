"""Inspector "Similar" tab JS fragment for the web reader (#1123).

Kept as a sibling of :mod:`polylogue.daemon.web_shell` so the main
single-page shell stays under its file-size budget. The fragment is
interpolated into ``WEB_SHELL_HTML`` at module import time. It owns:

- the inspector "Similar" tab rendering,
- the ``/api/sessions/{id}/similar`` fetch call,
- the four explicit visual states the endpoint contract defines
  (``ready`` / ``disabled`` / ``unavailable`` / ``not_embedded``).

The reader never calls the embedding provider directly — it consumes
the daemon-side similarity surface. When that surface reports the
embedding pipeline is dormant, this fragment surfaces the specific
``reason`` and the corresponding operator-facing guidance string,
rather than rendering an empty result list that would look like a
successful "no matches" lookup.
"""

from __future__ import annotations

SIMILAR_JS = r"""
function loadSimilarPanel(id) {
  state.similarPanels = state.similarPanels || {};
  fetchJSON('/api/sessions/' + encodeURIComponent(id) + '/similar?limit=10')
    .then(function(data) {
      state.similarPanels[id] = data;
      if (state.selected && state.selected.id === id && state.inspectorTab === 'similar') {
        renderInspector();
      }
    })
    .catch(function(e) {
      state.similarPanels[id] = {error: String(e)};
      if (state.selected && state.selected.id === id && state.inspectorTab === 'similar') {
        renderInspector();
      }
    });
}

function similarConfidenceChip(tag) {
  // Map the substrate's q-* vocabulary into a short visible label.
  // High/Med/Low keeps the row scannable; the chip class colors it.
  var label = tag === 'q-canonical' ? 'high' : (tag === 'q-estimated' ? 'med' : 'low');
  return '<span class="chip ' + esc(tag) + '" title="confidence: ' + esc(tag) + '">' + esc(label) + '</span>';
}

function renderSimilarReadyResults(data) {
  if (!data.results || !data.results.length) {
    return '<div class="inspector-empty" style="padding-top:8px">'
      + 'No similar sessions found yet. Embedding coverage may still be growing.'
      + '</div>';
  }
  var html = '<div class="inspector-section"><h4>Top matches</h4>';
  data.results.forEach(function(hit) {
    var title = hit.title || hit.session_id;
    var scoreStr = (hit.score != null ? hit.score.toFixed(3) : '0.000');
    var matched = hit.matched_message_count || 0;
    var providerStr = hit.source_name ? esc(hit.source_name) : '\u2014';
    html += '<div class="similar-hit" style="padding:6px 0;border-bottom:1px solid var(--border)">'
      + '<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">'
      +   '<a href="#" onclick="openSimilarHit(\'' + escAttr(hit.session_id) + '\');return false;" '
      +     'style="color:var(--accent);font-size:var(--small);text-decoration:none;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
      +     esc(String(title))
      +   '</a>'
      +   '<span style="font-family:var(--font-mono);font-size:10px;color:var(--text)">' + esc(scoreStr) + '</span>'
      +   similarConfidenceChip(hit.confidence || 'q-heuristic')
      + '</div>'
      + '<div style="font-size:10px;color:var(--text-dim);margin-top:2px">'
      +   providerStr + ' \u00b7 ' + matched + ' message hit' + (matched === 1 ? '' : 's')
      + '</div>'
      + '</div>';
  });
  html += '</div>';
  html += '<div style="font-size:10px;color:var(--text-dim);margin-top:8px">'
    + 'Source session has ' + (data.source_embedded_messages || 0)
    + ' embedded message' + ((data.source_embedded_messages || 0) === 1 ? '' : 's')
    + '.</div>';
  return html;
}

function renderSimilarDisabledState(data) {
  var reason = data.reason || 'unknown';
  var guidance = reason === 'no_voyage_api_key'
    ? 'Set VOYAGE_API_KEY and enable embedding_enabled in polylogue.toml to populate similarity vectors.'
    : (reason === 'embeddings_not_enabled'
        ? 'Set embedding_enabled = true (and supply voyage_api_key) in polylogue.toml to populate similarity vectors.'
        : 'Embedding similarity is currently disabled by configuration.');
  return '<div class="inspector-section"><h4>Embeddings disabled</h4>'
    + '<div style="font-size:var(--small);color:var(--text-muted);line-height:1.5">'
    +   esc(guidance)
    + '</div>'
    + '<div style="font-size:10px;color:var(--text-dim);margin-top:6px">'
    +   'reason: ' + esc(reason)
    + '</div>'
    + '</div>';
}

function renderSimilarUnavailableState(data) {
  var reason = data.reason || 'unknown';
  return '<div class="inspector-section"><h4>Similarity surface unavailable</h4>'
    + '<div style="font-size:var(--small);color:var(--text-muted);line-height:1.5">'
    +   'The embedding runtime is not ready on this archive.'
    + '</div>'
    + '<div style="font-size:10px;color:var(--text-dim);margin-top:6px">'
    +   'reason: ' + esc(reason)
    + '</div>'
    + '</div>';
}

function renderSimilarNotEmbeddedState() {
  return '<div class="inspector-section"><h4>Not yet embedded</h4>'
    + '<div style="font-size:var(--small);color:var(--text-muted);line-height:1.5">'
    +   'This session has no stored message vectors. The daemon will populate them when the embedding stage catches up.'
    + '</div>'
    + '</div>';
}

function renderInspectorSimilar(el, c) {
  state.similarPanels = state.similarPanels || {};
  var data = state.similarPanels[c.id];
  if (data === undefined) {
    el.innerHTML = '<div class="inspector-empty">Loading similar sessions...</div>';
    loadSimilarPanel(c.id);
    return;
  }
  if (data && data.error) {
    el.innerHTML = '<div class="inspector-empty">Similar sessions unavailable</div>';
    return;
  }
  var status = data.status || 'unknown';
  var html;
  if (status === 'disabled') {
    html = renderSimilarDisabledState(data);
  } else if (status === 'unavailable') {
    html = renderSimilarUnavailableState(data);
  } else if (status === 'not_embedded') {
    html = renderSimilarNotEmbeddedState();
  } else if (status === 'ready') {
    html = renderSimilarReadyResults(data);
  } else {
    html = '<div class="inspector-empty">Unknown similarity status: ' + esc(String(status)) + '</div>';
  }
  el.innerHTML = html;
}

function openSimilarHit(sessionId) {
  // Replace the selected session with the clicked similar hit by
  // navigating through the existing single-session route. The
  // inspector re-renders against the new selection and re-issues a
  // similarity lookup if the operator opens this tab again.
  if (!sessionId) return;
  if (typeof loadWorkspaceRoute === 'function') {
    loadWorkspaceRoute({mode: 'single', id: sessionId}, true).catch(function() {});
    return;
  }
  window.location.hash = '#/s/' + encodeURIComponent(sessionId);
}
"""

__all__ = ["SIMILAR_JS"]
