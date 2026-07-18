(function (root) {
  const OPERATOR_STATUS = Object.freeze({
    not_conversation: Object.freeze({
      code: "not_conversation",
      label: "No conversation",
      tone: "neutral",
      detail: "The current page is not a conversation, so conversation capture status does not apply.",
    }),
    safe_current: Object.freeze({
      code: "safe_current",
      label: "Safe / current",
      tone: "ok",
      detail: "This conversation is saved and current in Polylogue.",
    }),
    catching_up: Object.freeze({
      code: "catching_up",
      label: "Catching up",
      tone: "warn",
      detail: "Polylogue has durable capture evidence and is making it queryable.",
    }),
    needs_attention: Object.freeze({
      code: "needs_attention",
      label: "Needs attention",
      tone: "warn",
      detail: "A concrete operator action is required before Polylogue can finish this work.",
    }),
    failed: Object.freeze({
      code: "failed",
      label: "Failed",
      tone: "bad",
      detail: "The operation was rejected or could not be completed.",
    }),
    receiver_offline: Object.freeze({
      code: "receiver_offline",
      label: "Receiver offline",
      tone: "warn",
      detail: "The paired local receiver is unavailable. Existing local queues and evidence are preserved.",
    }),
    provider_warning: Object.freeze({
      code: "provider_warning",
      label: "Provider warning",
      tone: "warn",
      detail: "The provider asked Polylogue to slow down or stopped this operation for safety.",
    }),
    partial_fidelity: Object.freeze({
      code: "partial_fidelity",
      label: "Partial fidelity",
      tone: "warn",
      detail: "This capture came from visible page content rather than complete provider-native data.",
    }),
  });

  const WORK_STATUS = Object.freeze({
    queued: Object.freeze({ code: "queued", label: "Queued", tone: "neutral" }),
    running: Object.freeze({ code: "running", label: "Running", tone: "ok" }),
    completed: Object.freeze({ code: "completed", label: "Completed", tone: "ok" }),
    stopped: Object.freeze({ code: "stopped", label: "Stopped", tone: "neutral" }),
  });

  const PROVIDER_WARNING_REASONS = new Set([
    "rate_limited",
    "safety_locked",
    "provider_rate_limited",
    "provider_soft_warning",
    "provider_hard_rate_limit",
    "provider_safety_lock",
  ]);

  const AUTOMATIC_WAIT_REASONS = new Set([
    "cadence",
    "transport_backoff",
    "network_error",
  ]);

  function withDetail(status, detail = null) {
    return detail ? { ...status, detail } : { ...status };
  }

  function latestEvent(job) {
    const events = Array.isArray(job?.events) ? job.events : [];
    return events[events.length - 1] || null;
  }

  function providerWarning(job) {
    const latest = latestEvent(job);
    return PROVIDER_WARNING_REASONS.has(job?.cooldown_reason)
      || ["soft_warning", "hard_rate_limit", "safety_lock"].includes(latest?.provider_state);
  }

  function operatorStatusForState(state = {}) {
    let status;
    if (["unsupported", "supported_no_session"].includes(state.active_page_state)) {
      status = withDetail(
        OPERATOR_STATUS.not_conversation,
        state.active_page_state === "supported_no_session"
          ? "No conversation is selected on this provider page. Polylogue remains available and observes conversations automatically."
          : OPERATOR_STATUS.not_conversation.detail,
      );
    } else if (state.receiver_pairing?.state === "mismatch" || state.error === "receiver_pairing_mismatch") {
      status = withDetail(
        OPERATOR_STATUS.needs_attention,
        "The configured endpoint answered as a different receiver. Reset pairing only after verifying the local receiver.",
      );
    } else if (state.error === "unauthorized") {
      status = withDetail(OPERATOR_STATUS.needs_attention, "The receiver requires its local pairing token.");
      status.tone = "bad";
    } else if (state.online === false) {
      status = withDetail(OPERATOR_STATUS.receiver_offline);
    } else if (state.provider_warning || providerWarning(state)) {
      status = withDetail(OPERATOR_STATUS.provider_warning, state.provider_warning || null);
    } else {
      const archiveState = state.archive_state?.state;
      if (archiveState === "failed" || state.error) {
        status = withDetail(OPERATOR_STATUS.failed, state.archive_state?.latest_failure || state.error || null);
      } else if (state.capture_freshness) {
        status = withDetail(
          OPERATOR_STATUS.catching_up,
          state.capture_freshness.last_error
            ? `Freshness check will retry: ${state.capture_freshness.last_error}`
            : "A provider change was observed and canonical capture is checking the latest turn.",
        );
      } else if (["spooled_only", "ingest_pending", "stale"].includes(archiveState)) {
        status = withDetail(OPERATOR_STATUS.catching_up);
      } else if (archiveState === "missing") {
        status = withDetail(
          OPERATOR_STATUS.catching_up,
          "This conversation is not saved yet. Automatic capture is acquiring it now or will retry when the provider is ready.",
        );
      } else if (archiveState === "archived" || state.captured) {
        status = withDetail(OPERATOR_STATUS.safe_current);
      } else {
        status = withDetail(OPERATOR_STATUS.catching_up, "Polylogue is checking the current conversation automatically.");
      }
    }

    return {
      ...status,
      partialFidelity: status.code !== "not_conversation" && state.capture_mode === "dom_degraded",
      fidelity: status.code !== "not_conversation" && state.capture_mode === "dom_degraded"
        ? { ...OPERATOR_STATUS.partial_fidelity }
        : null,
    };
  }

  function operatorStatusLabel(state = {}) {
    const status = operatorStatusForState(state);
    return status.partialFidelity ? `${status.label} · ${OPERATOR_STATUS.partial_fidelity.label}` : status.label;
  }

  function operatorPresentationForState(state) {
    if (!state) return {
      status: { ...OPERATOR_STATUS.needs_attention },
      badge: ["warn", "needs attention"],
      archive: "Not checked",
      headline: "No receiver state yet.",
      detail: "Polylogue is waiting for the service worker to return the current receiver and conversation state.",
    };

    if (state.active_page_state === "unsupported") return {
      status: { ...OPERATOR_STATUS.not_conversation },
      badge: ["neutral", "idle"],
      archive: "Not applicable",
      headline: "Ordinary webpage",
      detail: "This page is not a conversation. Polylogue has no conversation work to report here.",
    };

    if (state.active_page_state === "supported_no_session") return {
      status: operatorStatusForState(state),
      badge: ["neutral", "idle"],
      archive: "Not applicable",
      headline: "No conversation selected",
      detail: "Polylogue will observe and capture conversations automatically when one is opened.",
    };

    const status = operatorStatusForState(state);
    if (state.receiver_pairing?.state === "mismatch" || state.error === "receiver_pairing_mismatch") return {
      status,
      badge: ["bad", "needs attention"],
      archive: status.label,
      headline: "Receiver identity changed.",
      detail: status.detail,
    };
    if (state.error === "unauthorized") return {
      status,
      badge: ["bad", "needs attention"],
      archive: status.label,
      headline: "Receiver requires its pairing token.",
      detail: 'Run `polylogued browser-capture token show` and paste the value into “Receiver token”, then Save.',
    };
    if (state.online === false) {
      const lastSeen = state.receiver_pairing?.last_seen_at
        ? ` Last contact: ${new Date(state.receiver_pairing.last_seen_at).toLocaleString()}.`
        : "";
      return {
        status,
        badge: ["warn", "receiver offline"],
        archive: status.label,
        headline: "Receiver offline; local work remains intact.",
        detail: `${status.detail}${lastSeen}`,
      };
    }

    const archiveState = state.archive_state?.state || null;
    if (archiveState === "failed" || state.error) return {
      status,
      badge: ["bad", "failed"],
      archive: status.label,
      headline: status.detail,
      detail: state.archive_state?.latest_failure || state.error || "Use the request id in the debug log to inspect the receiver failure.",
    };
    if (archiveState === "stale") return {
      status,
      badge: ["warn", "catching up"],
      archive: status.label,
      headline: "The receiver spool is newer than the indexed archive.",
      detail: "The daemon is catching up to the latest durable browser capture; no duplicate capture is needed.",
    };
    if (archiveState === "ingest_pending") return {
      status,
      badge: ["warn", "catching up"],
      archive: status.label,
      headline: "Capture reached source.db and is awaiting index materialization.",
      detail: "The daemon still needs to materialize the indexed session and messages.",
    };
    if (archiveState === "spooled_only") return {
      status,
      badge: ["warn", "catching up"],
      archive: status.label,
      headline: "Receiver durably wrote the capture artifact.",
      detail: "The daemon has not acquired the spool artifact into source.db yet.",
    };
    if (archiveState === "missing") return {
      status,
      badge: ["warn", "catching up"],
      archive: status.label,
      headline: "This conversation is not saved yet.",
      detail: "Automatic capture is acquiring it now or will retry when the provider is ready.",
    };
    if (archiveState === "archived" || state.captured) return {
      status,
      badge: [status.partialFidelity ? "warn" : "ok", status.partialFidelity ? "partial fidelity" : "safe / current"],
      archive: status.label,
      headline: state.last_capture
        ? `Current capture: ${state.last_capture.provider} / ${state.last_capture.provider_session_id}`
        : "The latest capture is visible in the archive.",
      detail: status.partialFidelity
        ? OPERATOR_STATUS.partial_fidelity.detail
        : "Archive evidence includes the receiver spool, source raw row, indexed session, and indexed messages.",
    };
    if (state.capture_mode === "dom_degraded") return {
      status,
      badge: ["warn", "partial fidelity"],
      archive: OPERATOR_STATUS.partial_fidelity.label,
      headline: "Captured from visible page content.",
      detail: "Provider ids, timestamps, branches, or attachments can be incomplete. Reload, wait for provider-native data, then capture again.",
    };
    return {
      status,
      badge: ["warn", "needs attention"],
      archive: "Receiver online",
      headline: "Receiver online. Open a supported conversation to capture.",
      detail: "Supported pages are ChatGPT, Claude.ai, and Grok/X conversation routes.",
    };
  }

  function humanize(value, fallback = "—") {
    const text = String(value || "").trim();
    if (!text) return fallback;
    return text
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function providerLabel(provider) {
    const labels = { chatgpt: "ChatGPT", "claude-ai": "Claude.ai", claude: "Claude.ai", grok: "Grok" };
    return labels[provider] || humanize(provider, "Unknown provider");
  }

  function compactDate(value) {
    const parsed = Date.parse(value || "");
    if (!Number.isFinite(parsed)) return null;
    return new Date(parsed).toISOString().slice(0, 10);
  }

  function nextAttemptLabel(value) {
    const parsed = Date.parse(value || "");
    if (!Number.isFinite(parsed)) return "Due now";
    const seconds = Math.ceil((parsed - Date.now()) / 1000);
    if (seconds <= 0) return "Due now";
    if (seconds < 60) return `Due in ${seconds}s`;
    if (seconds < 3600) return `Due in ${Math.ceil(seconds / 60)}m`;
    return `Due in ${Math.ceil(seconds / 3600)}h`;
  }

  function backfillWorkStatus(job, receiverOnline) {
    if (receiverOnline === false && !["complete", "completed", "cancelled"].includes(job?.status)) {
      return { ...OPERATOR_STATUS.receiver_offline };
    }
    if (providerWarning(job)) return { ...OPERATOR_STATUS.provider_warning };
    if (["complete", "completed"].includes(job?.status)) return { ...WORK_STATUS.completed };
    if (job?.status === "cancelled") return { ...WORK_STATUS.stopped };
    if (job?.status === "failed") return { ...OPERATOR_STATUS.failed };
    if (job?.status === "paused" && !AUTOMATIC_WAIT_REASONS.has(job?.cooldown_reason)) {
      return { ...OPERATOR_STATUS.needs_attention };
    }
    if (job?.cooldown_reason && !AUTOMATIC_WAIT_REASONS.has(job.cooldown_reason)) {
      return { ...OPERATOR_STATUS.needs_attention };
    }
    if (job?.status === "running" && !job?.cooldown_reason) return { ...WORK_STATUS.running };
    return { ...WORK_STATUS.queued };
  }

  function captureQueueItems(captureQueue, receiverOnline) {
    const entries = Array.isArray(captureQueue?.entries) ? captureQueue.entries : [];
    return entries.map((entry, index) => {
      const session = entry.envelope?.session || {};
      const provider = session.provider || entry.provider || "unknown";
      const sessionId = session.provider_session_id || entry.provider_session_id || "unknown session";
      const title = session.title || `${providerLabel(provider)} conversation ${sessionId}`;
      return {
        id: entry.id || `capture-${index}`,
        kind: "capture_retry",
        title,
        status: receiverOnline === false ? { ...OPERATOR_STATUS.receiver_offline } : { ...WORK_STATUS.queued },
        phase: "Waiting to deliver capture",
        cadence: nextAttemptLabel(entry.next_attempt_at),
        owner: "This extension",
        cooldown: entry.last_error || null,
        receipt: "Receiver acknowledgement pending",
        updated_at: entry.enqueued_at || null,
        raw: entry,
      };
    });
  }

  function backfillItems(backfillJobs, receiverOnline) {
    return (Array.isArray(backfillJobs) ? backfillJobs : []).map((job, index) => {
      const since = compactDate(job.cutoff);
      const title = job.job_title || `${providerLabel(job.provider)} history backfill${since ? ` since ${since}` : ""}`;
      const cadenceSeconds = Math.max(0, Math.round((Number(job.learned_cadence_ms) || 0) / 1000));
      const phase = job.phase
        || (job.inventory_complete ? "Capturing discovered conversations" : "Discovering provider inventory");
      return {
        id: job.id || `backfill-${index}`,
        kind: "backfill",
        title,
        status: backfillWorkStatus(job, receiverOnline),
        phase,
        cadence: cadenceSeconds ? `${cadenceSeconds}s cadence` : "Adaptive cadence",
        owner: job.lease_owner || job.owner_instance_id || "This extension",
        cooldown: job.cooldown_reason || job.last_error || null,
        receipt: job.last_ack?.receiver_request_id
          ? `Last receiver ACK ${job.last_ack.receiver_request_id}`
          : "No receiver ACK yet",
        updated_at: job.updated_at || job.created_at || null,
        raw: job,
      };
    });
  }

  function freshnessItems(freshnessQueue, receiverOnline) {
    const entries = freshnessQueue?.entries && typeof freshnessQueue.entries === "object"
      ? Object.values(freshnessQueue.entries)
      : [];
    return entries.map((entry, index) => ({
      id: entry.key || `freshness-${index}`,
      kind: "freshness",
      title: `${providerLabel(entry.provider)} conversation ${entry.native_id}`,
      status: receiverOnline === false
        ? { ...OPERATOR_STATUS.receiver_offline }
        : { ...(entry.lease_owner ? WORK_STATUS.running : WORK_STATUS.queued) },
      phase: entry.lease_owner ? "Checking latest provider turn" : "Waiting for freshness check",
      cadence: nextAttemptLabel(new Date(entry.next_attempt_at_ms || 0).toISOString()),
      owner: entry.lease_owner || "This extension",
      cooldown: entry.last_error || null,
      receipt: (entry.reasons || []).join(", ") || "Provider change observed",
      updated_at: entry.hinted_at || null,
      raw: entry,
    }));
  }

  function normalizeWorkItems({
    captureQueue = null,
    freshnessQueue = null,
    backfillJobs = [],
    receiverOnline = true,
  } = {}) {
    const priority = {
      running: 0,
      provider_warning: 1,
      needs_attention: 2,
      receiver_offline: 3,
      queued: 4,
      failed: 5,
      completed: 6,
      stopped: 7,
    };
    return [
      ...captureQueueItems(captureQueue, receiverOnline),
      ...freshnessItems(freshnessQueue, receiverOnline),
      ...backfillItems(backfillJobs, receiverOnline),
    ].sort((left, right) => {
      const statusOrder = (priority[left.status.code] ?? 99) - (priority[right.status.code] ?? 99);
      if (statusOrder) return statusOrder;
      return String(right.updated_at || "").localeCompare(String(left.updated_at || ""));
    });
  }

  function eventPresentation(entry = {}) {
    const labels = {
      captured: "Captured and acknowledged",
      detected_new: "New conversation detected",
      held_with_reason: "Held; no unsafe action taken",
      first_seen: "Conversation observed",
      observed_no_action: "Observed; no action needed",
      queued_for_retry: "Capture queued for retry",
      receiver_recovered: "Receiver connection recovered",
    };
    const details = {
      already_safe: "Archive was already current, so Polylogue did not create a duplicate capture.",
      receiver_already_processing: "Durable evidence already existed and the receiver was still catching up.",
      archive_state_checked: "Polylogue checked the receiver and recorded the result.",
      provider_still_running: "The provider response is still running; Polylogue will check again automatically.",
      provider_head_current: "The latest provider turn was captured through a terminal assistant response.",
    };
    return {
      label: labels[entry.event] || humanize(entry.event, "Observed"),
      detail: details[entry.detail] || [entry.reason, entry.detail].filter(Boolean).join(" · "),
    };
  }

  function receiverPairingPresentation({ pairing = null, health = null, configuredUrl = "" } = {}) {
    if (pairing?.state === "mismatch" || health?.status === "pairing_mismatch") {
      return {
        status: { ...OPERATOR_STATUS.needs_attention, tone: "bad" },
        headline: "Receiver identity changed",
        detail: "The configured endpoint answered as a different receiver. Verify it before resetting pairing.",
      };
    }
    if (health?.status === "unauthorized") {
      return {
        status: { ...OPERATOR_STATUS.needs_attention, tone: "bad" },
        headline: "Pairing token required",
        detail: configuredUrl || pairing?.endpoint || "Local receiver",
      };
    }
    if (["unreachable", "offline"].includes(health?.status) || pairing?.state === "offline") {
      return {
        status: { ...OPERATOR_STATUS.receiver_offline },
        headline: pairing?.receiver_id ? `Paired receiver ${pairing.receiver_id}` : "Receiver offline",
        detail: pairing?.last_seen_at ? `Last seen ${new Date(pairing.last_seen_at).toLocaleString()}` : "No successful contact recorded",
      };
    }
    if (health?.status === "recovered") {
      return {
        status: { ...OPERATOR_STATUS.safe_current },
        headline: "Receiver recovered",
        detail: `Recovered at ${health.endpoint || configuredUrl}`,
      };
    }
    if (pairing?.receiver_id) {
      return {
        status: { ...OPERATOR_STATUS.safe_current },
        headline: `Paired receiver ${pairing.receiver_id}`,
        detail: `${pairing.api_schema || "legacy schema"} · ${pairing.endpoint || configuredUrl}`,
      };
    }
    return {
      status: { ...OPERATOR_STATUS.needs_attention },
      headline: "Receiver identity unavailable",
      detail: "This receiver is reachable but does not advertise stable pairing metadata.",
    };
  }

  // The exception-driven popup surfaces at most one attention item at a
  // time (polylogue-yyvg.7 AC3). This is a strict typed priority list, not a
  // free-form aggregation: auth/pairing mismatch (the receiver cannot be
  // trusted) outranks an unconfirmed action outcome (a stuck submit, never
  // auto-retried), which outranks a typed capability mismatch on queued
  // work, which outranks a hard archive failure on the current conversation.
  // Anything else is healthy/automatic and returns null — silence is the
  // correct answer far more often than not.
  function computeAttention({
    conversationState = null,
    pairing = null,
    health = null,
    workItems = [],
    browserActions = [],
  } = {}) {
    if (pairing?.state === "mismatch" || health?.status === "pairing_mismatch") {
      return {
        kind: "auth_pairing_mismatch",
        tone: "bad",
        headline: "Receiver identity changed",
        detail: "The configured endpoint answered as a different receiver. Verify it, then reset pairing.",
        actionId: "reset-pairing",
        actionLabel: "Reset pairing",
      };
    }
    if (health?.status === "unauthorized" || conversationState?.error === "unauthorized") {
      return {
        kind: "auth_pairing_mismatch",
        tone: "bad",
        headline: "Receiver requires its pairing token",
        detail: "Run `polylogued browser-capture token show` and paste the value into the receiver token field.",
        actionId: "receiver-token",
        actionLabel: "Open receiver settings",
      };
    }

    const unknownAction = (Array.isArray(browserActions) ? browserActions : [])
      .find((action) => action?.status === "outcome_unknown");
    if (unknownAction) {
      return {
        kind: "action_outcome_unknown",
        tone: "bad",
        headline: "A browser action's outcome could not be confirmed",
        detail: unknownAction.last_error
          || "The provider connection ended without a receipt. Check the conversation before retrying.",
        actionId: null,
        actionLabel: "Details",
      };
    }

    const capabilityItem = (Array.isArray(workItems) ? workItems : [])
      .find((item) => ["capability_mismatch", "receiver_contract_incompatible"].includes(item?.raw?.cooldown_reason));
    if (capabilityItem) {
      return {
        kind: "capability_mismatch",
        tone: "bad",
        headline: `${capabilityItem.title} needs a compatible provider selection`,
        detail: capabilityItem.cooldown || "The receiver rejected an unsupported provider capability.",
        actionId: null,
        actionLabel: "Details",
      };
    }

    if (conversationState && (conversationState.archive_state?.state === "failed" || conversationState.error)) {
      return {
        kind: "archive_failed",
        tone: "bad",
        headline: "This conversation failed to archive",
        detail: conversationState.archive_state?.latest_failure
          || conversationState.error
          || "Check the debug log request id.",
        actionId: null,
        actionLabel: "Details",
      };
    }

    return null;
  }

  function pendingBrowserActionCount(browserActions = []) {
    const pendingStatuses = new Set(["queued", "leased", "preparing", "submit_intent"]);
    return (Array.isArray(browserActions) ? browserActions : [])
      .filter((action) => pendingStatuses.has(action?.status)).length;
  }

  root.PolylogueOperatorStatus = Object.freeze({
    OPERATOR_STATUS,
    WORK_STATUS,
    computeAttention,
    eventPresentation,
    normalizeWorkItems,
    operatorPresentationForState,
    operatorStatusForState,
    operatorStatusLabel,
    pendingBrowserActionCount,
    receiverPairingPresentation,
  });
})(globalThis);
