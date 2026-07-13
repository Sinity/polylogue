const OPERATOR_STATUS = Object.freeze({
  safe: { label: "Safe", tone: "ok", detail: "This conversation is saved and available in Polylogue." },
  catching_up: { label: "Catching up", tone: "warn", detail: "Polylogue has the capture and is making it queryable." },
  needs_attention: { label: "Needs attention", tone: "warn", detail: "Polylogue needs a concrete conversation or a receiver check before it can save this page." },
  failed: { label: "Failed", tone: "bad", detail: "The capture was rejected or could not be parsed." },
  not_saved: { label: "Not saved", tone: "warn", detail: "This conversation has not been saved yet." },
});

function operatorStatusForState(state = {}) {
  if (state.error === "unauthorized") {
    return { ...OPERATOR_STATUS.needs_attention, tone: "bad" };
  }
  if (state.online === false) {
    return { ...OPERATOR_STATUS.failed };
  }
  const archiveState = state.archive_state?.state;
  let status;
  if (archiveState === "failed" || state.error) status = OPERATOR_STATUS.failed;
  else if (["spooled_only", "ingest_pending", "stale"].includes(archiveState)) status = OPERATOR_STATUS.catching_up;
  else if (archiveState === "missing") status = OPERATOR_STATUS.not_saved;
  else if (archiveState === "archived" || state.captured) status = OPERATOR_STATUS.safe;
  else status = OPERATOR_STATUS.needs_attention;

  return {
    ...status,
    partialFidelity: state.capture_mode === "dom_degraded",
  };
}

function operatorStatusLabel(state = {}) {
  const status = operatorStatusForState(state);
  return status.partialFidelity ? `${status.label} · Partial fidelity` : status.label;
}

function operatorPresentationForState(state) {
  if (!state) return {
    badge: ["warn", "idle"], archive: "Not checked", headline: "No receiver state yet.",
    detail: "The popup refreshes automatically on open. If this stays idle, the service worker has not returned a status payload.",
  };
  if (state.active_page_state === "unsupported") return {
    badge: ["warn", "idle"], archive: "Unsupported", headline: "This page is not a supported conversation.",
    detail: "Open a ChatGPT, Claude.ai, or Grok/X conversation tab. The extension will update this state when the active tab changes.",
  };
  if (!state.online) {
    if (state.error === "unauthorized") return {
      badge: ["bad", "offline"], archive: "Unauthorized", headline: "Receiver requires a pairing token.",
      detail: 'Run `polylogued browser-capture token show` and paste the value into "Receiver token" below, then Save.',
    };
    return {
      badge: ["bad", "offline"], archive: "Offline", headline: "Receiver offline.",
      detail: `Start the local receiver, then refresh. ${state.error || ""}`.trim(),
    };
  }
  const archiveState = state.archive_state?.state || null;
  if (archiveState === "failed" || state.error) return {
    badge: ["bad", "failed"], archive: "Failed", headline: OPERATOR_STATUS.failed.detail,
    detail: state.archive_state?.latest_failure || state.error || "Open the debug log and match the request id in the receiver log.",
  };
  if (archiveState === "stale") return {
    badge: ["warn", "stale"], archive: "Stale", headline: "Receiver spool is newer than the indexed archive.",
    detail: "The daemon has not caught up to the latest browser capture yet. Keep the daemon running; this should converge without a manual repair step.",
  };
  if (archiveState === "ingest_pending") return {
    badge: ["warn", "pending"], archive: "Ingest pending", headline: "Capture reached source.db but is not queryable yet.",
    detail: "The daemon still needs to materialize the indexed session and messages.",
  };
  if (archiveState === "spooled_only") return {
    badge: ["warn", "spooled"], archive: "Spooled", headline: "Receiver wrote the capture artifact.",
    detail: "The daemon has not acquired the spool artifact into source.db yet.",
  };
  if (archiveState === "missing") return {
    badge: ["warn", "missing"], archive: "Not archived", headline: OPERATOR_STATUS.not_saved.detail,
    detail: "Use Capture page for the active conversation. The extension checks archive state automatically when tabs activate or finish loading.",
  };
  if (archiveState === "archived" || state.captured) return {
    badge: ["ok", "captured"], archive: "Archived",
    headline: state.last_capture
      ? `Last capture: ${state.last_capture.provider} / ${state.last_capture.provider_session_id}`
      : "The latest capture is visible in the archive.",
    detail: "Archive evidence includes receiver spool, source raw row, indexed session, and indexed messages.",
  };
  if (state.capture_mode === "dom_degraded") return {
    badge: ["warn", "dom"], archive: "DOM fallback", headline: "Captured from visible DOM, not provider-native app data.",
    detail: "DOM fallback is useful but is not provider-native app data; it may omit branches, provider ids, timestamps, or attachments. Reload the page, wait for the conversation API response, then capture again.",
  };
  if (state.active_page_state === "supported_no_session") return {
    badge: ["warn", "ready"], archive: "Ready", headline: "Supported site open, but no conversation id is visible.",
    detail: "Open or select a concrete conversation. The extension does not read page content until Capture page or Sync open tabs is used.",
  };
  return {
    badge: ["warn", "online"], archive: "Receiver online", headline: "Receiver online. Open a supported conversation to capture.",
    detail: "Supported pages are ChatGPT, Claude.ai, and Grok/X conversation routes.",
  };
}

globalThis.PolylogueOperatorStatus = Object.freeze({ operatorStatusForState, operatorStatusLabel, operatorPresentationForState });
