"""Realtime SSE channel JS for the daemon web reader (#1204).

Kept separate from :mod:`polylogue.daemon.web_shell` so the file-budget
lint (#1224) can govern web_shell.py without blocking incremental
realtime-channel growth. The constant is injected into the page via the
``__REALTIME_JS__`` placeholder substitution in ``web_shell.py``.

Topic vocabulary and selective-subscription policy:

* Legacy opaque kinds (``ingestion_batch`` / ``ingest`` / ``reset`` /
  ``operation``) stay on the wire so older consumers keep working.
* Granular kinds (``session.appended`` / ``session.updated`` /
  ``message.appended`` / ``insight.updated`` / ``progress.update`` /
  ``progress.complete``) split the channel by topic.
* The list view subscribes to legacy + ``session.*`` + ``progress.*``.
  The session view additionally subscribes to ``message.appended``
  and ``insight.updated`` so it can live-tail without polling.
* ``snapshot`` is the coalesced backpressure frame; clients react by
  refetching the materialised view and skipping row-level animations.
"""

from __future__ import annotations

REALTIME_JS = r"""
// --- Realtime channel (#1204) -------------------------------------------
// Subscribe to /api/events (SSE) when available, scoped by current view:
//   * list view subscribes to session.* and the legacy batch kinds
//   * session view also subscribes to message.appended + insight.updated
//   * progress.update / progress.complete are always streamed so the
//     status chip and #1218 watch-mode consumer can advance their UIs
// EventSource handles reconnects automatically; on persistent failure we
// fall back to polling. New-row animations decorate just-appended rows
// without rerendering the full list.
var realtime = {
  source: null,
  lastEventId: 0,
  pollTimer: null,
  refreshTimer: null,
  status: 'connecting',
  subscribedKinds: null,
  lastTickTs: null
};

// All event kinds the client knows how to dispatch. Order matters only
// for selective subscription URL construction.
var REALTIME_LEGACY_KINDS = ['ingestion_batch', 'ingest', 'reset', 'operation'];
var REALTIME_GRANULAR_KINDS = [
  'session.appended',
  'session.updated',
  'message.appended',
  'insight.updated',
  'progress.update',
  'progress.complete',
  'snapshot'
];

function realtimeKindsForView() {
  // Always include legacy kinds so existing consumers keep working;
  // granular kinds are scoped by current view to reduce wakeups on a
  // slow link. The session view subscribes to message.appended
  // and insight.updated for live tail.
  var kinds = REALTIME_LEGACY_KINDS.slice();
  kinds.push('session.appended');
  kinds.push('session.updated');
  kinds.push('progress.update');
  kinds.push('progress.complete');
  kinds.push('snapshot');
  if (state && state.selectedConvId) {
    kinds.push('message.appended');
    kinds.push('insight.updated');
  }
  return kinds;
}

function setLiveChip(status, lastSeen) {
  var el = document.getElementById('status-live');
  if (!el) return;
  el.className = 'chip' + (status === 'live' ? ' accent' : '');
  var label = 'live: ' + status;
  if (lastSeen) label += ' \u00b7 #' + lastSeen;
  if (realtime.lastTickTs) {
    var age = Math.max(0, Math.round((Date.now() - realtime.lastTickTs) / 1000));
    if (age > 0) label += ' \u00b7 ' + age + 's';
  }
  el.textContent = label;
  realtime.status = status;
}

function scheduleRefresh() {
  if (realtime.refreshTimer) return;
  realtime.refreshTimer = setTimeout(function() {
    realtime.refreshTimer = null;
    loadSessions({animateNewIds: realtime.pendingAnimateIds || null});
    realtime.pendingAnimateIds = null;
    loadFacets();
    loadStatus();
  }, 250);
}

function flagAppendedRow(convId) {
  // Mark a session row for the fade-in animation on next render.
  if (!convId) return;
  realtime.pendingAnimateIds = realtime.pendingAnimateIds || {};
  realtime.pendingAnimateIds[convId] = true;
}

function maybeAnimateExistingRow(convId) {
  // If the row is already in the DOM, attach the highlight directly.
  if (!convId) return;
  var row = document.querySelector('.conv-item[data-id="' + cssEscape(convId) + '"]');
  if (row) {
    row.classList.add('row-appended');
    setTimeout(function() { row.classList.remove('row-appended'); }, 1800);
  }
}

function cssEscape(s) {
  // Lightweight CSS attribute selector escape — sufficient for archive ids.
  return String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

function animateAppendedMessage(messageEl) {
  if (!messageEl) return;
  messageEl.classList.add('message-appended');
  setTimeout(function() { messageEl.classList.remove('message-appended'); }, 1800);
}

function liveTailCurrentSession(payload) {
  // Reload messages for the current session if the event targets
  // it, or reload unconditionally when the event is unscoped. Newly
  // rendered messages get the appended animation.
  if (!state || !state.selectedConvId) return;
  var convId = payload && payload.payload && payload.payload.session_id;
  if (convId && convId !== state.selectedConvId) return;
  // Reuse selectSession to refresh the message list; mark new ones.
  selectSession(state.selectedConvId, false, {liveTail: true});
}

function handleRealtimeEvent(payload) {
  if (!payload || typeof payload !== 'object') return;
  if (typeof payload.id === 'number') realtime.lastEventId = payload.id;
  realtime.lastTickTs = Date.now();
  setLiveChip('live', realtime.lastEventId);
  var kind = payload.kind || '';
  var data = payload.payload || {};
  switch (kind) {
    case 'message.appended':
      liveTailCurrentSession(payload);
      return;
    case 'session.appended':
    case 'session.updated':
      if (data && data.session_id) {
        flagAppendedRow(data.session_id);
        maybeAnimateExistingRow(data.session_id);
      }
      scheduleRefresh();
      return;
    case 'insight.updated':
      if (typeof renderInspector === 'function') renderInspector();
      return;
    case 'progress.update':
    case 'progress.complete':
      // Surface progress in the chip — operator can see live %.
      var opKind = (data && data.operation_kind) || 'op';
      var label = 'live';
      if (kind === 'progress.complete') {
        label = 'done: ' + opKind;
      } else if (typeof data.fraction === 'number') {
        label = opKind + ' ' + Math.round(data.fraction * 100) + '%';
      } else if (typeof data.completed === 'number') {
        label = opKind + ' ' + data.completed;
      }
      setLiveChip(label, realtime.lastEventId);
      return;
    case 'snapshot':
      // Coalesced burst — refetch the materialised view, skip animations.
      scheduleRefresh();
      return;
    case 'ingestion_batch':
    case 'ingest':
    case 'reset':
    case 'operation':
      scheduleRefresh();
      return;
    default:
      return;
  }
}

function buildEventsURL(opts) {
  var qs = ['since=' + realtime.lastEventId];
  if (opts && opts.poll) qs.push('poll=1');
  var kinds = realtimeKindsForView();
  realtime.subscribedKinds = kinds.slice();
  qs.push('kinds=' + encodeURIComponent(kinds.join(',')));
  return '/api/events?' + qs.join('&');
}

function startPollingFallback() {
  if (realtime.pollTimer) return;
  setLiveChip('polling', realtime.lastEventId);
  realtime.pollTimer = setInterval(async function() {
    try {
      var data = await fetchJSON(buildEventsURL({poll: true}));
      var events = data.events || [];
      events.forEach(handleRealtimeEvent);
      if (typeof data.last_event_id === 'number') realtime.lastEventId = data.last_event_id;
      loadStatus();
    } catch(e) {
      setLiveChip('disconnected', realtime.lastEventId);
    }
  }, 5000);
}

function startRealtimeChannel() {
  if (typeof EventSource === 'undefined') { startPollingFallback(); return; }
  try {
    var url = buildEventsURL({});
    realtime.source = new EventSource(url);
    setLiveChip('connecting', realtime.lastEventId);
    realtime.source.onopen = function() { setLiveChip('live', realtime.lastEventId); };
    var consumeMessage = function(e) {
      var data = null;
      try { data = JSON.parse(e.data); } catch(_) { return; }
      handleRealtimeEvent(data);
    };
    realtime.source.onmessage = consumeMessage;
    REALTIME_LEGACY_KINDS.concat(REALTIME_GRANULAR_KINDS).forEach(function(kind) {
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
          setLiveChip('disconnected', realtime.lastEventId);
          startPollingFallback();
        }
      }, 15000);
    };
  } catch(e) {
    startPollingFallback();
  }
}

function restartRealtimeForView() {
  // Reopen the SSE channel with an updated ?kinds= subscription when the
  // user switches between list and session views. The session
  // view adds message.appended + insight.updated; switching back removes
  // them so we don't fire live-tail handlers for a dormant view.
  if (!realtime.source) return;
  var currentKinds = (realtime.subscribedKinds || []).join(',');
  var nextKinds = realtimeKindsForView().join(',');
  if (currentKinds === nextKinds) return;
  try { realtime.source.close(); } catch(_) {}
  realtime.source = null;
  startRealtimeChannel();
}

startRealtimeChannel();
"""

__all__ = ["REALTIME_JS"]
