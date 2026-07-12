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
  else if (archiveState === "archived" || state.captured) status = OPERATOR_STATUS.safe;
  else if (archiveState === "missing") status = OPERATOR_STATUS.not_saved;
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

globalThis.PolylogueOperatorStatus = Object.freeze({ operatorStatusForState, operatorStatusLabel });
