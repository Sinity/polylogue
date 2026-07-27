"""Realtime SSE channel JS for the daemon web reader (#1204).

Kept separate from :mod:`polylogue.daemon.web_shell` so the file-budget
lint (#1224) can govern web_shell.py without blocking incremental
realtime-channel growth. The constant is injected into the page via the
``__REALTIME_JS__`` placeholder substitution in ``web_shell.py``.

Topic vocabulary and selective-subscription policy:

* Legacy opaque kinds (``ingestion_batch`` / ``ingest`` / ``reset`` /
  ``operation``) stay on the wire so older consumers keep working.
* Granular kinds (``session.appended`` / ``session.updated`` /
  ``message.appended``) split the channel by topic. Each carries a real
  ``session_id`` (polylogue-20d.13) when the producer resolved one, so a
  reader with session A open ignores an event scoped to session B instead
  of refreshing unconditionally.
* The list view subscribes to legacy + ``session.*``. The session view
  additionally subscribes to ``message.appended`` so it can live-tail
  without polling.
* ``snapshot`` is the coalesced backpressure frame; clients react by
  refetching the materialised view and skipping row-level animations.
* ``insight.updated`` / ``progress.update`` / ``progress.complete`` were
  retired (polylogue-20d.13): no production code ever emitted them, so
  advertising subscriptions for them was itself an identity/completeness
  defect. Reintroduce only alongside a real producer.
"""

from __future__ import annotations

REALTIME_JS = r"""
// --- Realtime channel (#1204) -------------------------------------------
// Subscribe to /api/events (SSE) when available, scoped by current view:
//   * list view subscribes to session.* and the legacy batch kinds
//   * session view also subscribes to message.appended for live tail
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
  lastTickTs: null,
  reauthenticating: false
};

// All event kinds the client knows how to dispatch. Order matters only
// for selective subscription URL construction.
var REALTIME_LEGACY_KINDS = ['ingestion_batch', 'ingest', 'reset', 'operation'];
var REALTIME_GRANULAR_KINDS = [
  'session.appended',
  'session.updated',
  'message.appended',
  'snapshot'
];

function realtimeKindsForView() {
  // Always include legacy kinds so existing consumers keep working;
  // granular kinds are scoped by current view to reduce wakeups on a
  // slow link. The session view subscribes to message.appended for
  // live tail.
  var kinds = REALTIME_LEGACY_KINDS.slice();
  kinds.push('session.appended');
  kinds.push('session.updated');
  kinds.push('snapshot');
  if (currentSelectedSessionId()) {
    kinds.push('message.appended');
  }
  return kinds;
}

function currentSelectedSessionId() {
  if (!state) return '';
  if (state.selected && state.selected.id) return state.selected.id;
  return state.selectedConvId || '';
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
  var selectedId = currentSelectedSessionId();
  if (!selectedId) return;
  var convId = payload && payload.payload && payload.payload.session_id;
  if (convId && convId !== selectedId) return;
  // Reuse selectSession to refresh the message list; mark new ones.
  selectSession(selectedId, false, {liveTail: true});
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
    realtime.source = new EventSource(url, {withCredentials: true});
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
    realtime.source.onerror = async function() {
      setLiveChip('stale', realtime.lastEventId);
      if (!realtime.reauthenticating) {
        realtime.reauthenticating = true;
        try {
          await bootstrapWebCredential();
          try { realtime.source && realtime.source.close(); } catch(_) {}
          realtime.source = null;
          realtime.reauthenticating = false;
          startRealtimeChannel();
          return;
        } catch(_) {
          realtime.reauthenticating = false;
        }
      }
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
  // view adds message.appended; switching back removes it so we don't
  // fire live-tail handlers for a dormant view.
  if (!realtime.source) return;
  var currentKinds = (realtime.subscribedKinds || []).join(',');
  var nextKinds = realtimeKindsForView().join(',');
  if (currentKinds === nextKinds) return;
  try { realtime.source.close(); } catch(_) {}
  realtime.source = null;
  startRealtimeChannel();
}

ensureWebCredential().then(startRealtimeChannel).catch(startPollingFallback);
"""

__all__ = ["REALTIME_JS"]
