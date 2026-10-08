(function () {
  if (window.__polylogueChatgptCaptureInstalled === 2) return;
  window.__polylogueChatgptCaptureInstalled = 2;

  // In-page Layer 1 (polylogue-ys30): capture-status dot + save action mounted
  // next to each detected message. Reused across every capture trigger below
  // (badge click, popup, background auto-capture) so the dots always reflect
  // the most recent capture outcome for the whole session.
  const MESSAGE_CONTAINER_SELECTOR = '[data-testid^="conversation-turn-"], article, [data-message-author-role]';
  let messageLayer = null;

  const nativeAdapterName = "chatgpt-native-v1";
  const nativeCaptureMessage = "polylogue.chatgpt.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.chatgpt.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.chatgpt.nativeFetchResponse";
  let nativeCapture = null;
  let nativeHeader = null;
  let nativeHeaderPromise = Promise.resolve();
  const nativeFetchResponses = new Map();
  const freshnessHintTimers = new Map();
  const pendingFreshnessObservations = new Map();
  const lifecycleObservationHistory = new Map();
  const lifecycleRuntime = new Map();
  let domFreshnessScanTimer = null;
  let freshnessObserver = null;
  let scheduleDomFreshnessScan = null;
  let freshnessSuspended = false;
  let lastDomFreshnessSignature = null;

  const nativeProgressStages = new Set(["throttle", "pending_header", "restore", "staging", "provider_auth", "provider_response", "body", "header", "canonical", "publication"]);
  function rememberNativeAttempt(diagnostic, diagnostics) {
    if (!diagnostics || diagnostics.cancelled) return;
    diagnostics.entries.push(Object.hasOwn(diagnostic, "state") ? diagnostic : { attempted_at: new Date().toISOString(), ...diagnostic });
    if (diagnostics.entries.length > 6) {
      const dropped = diagnostics.entries.length - 6;
      diagnostics.entries.splice(0, dropped);
      diagnostics.dropped += dropped;
    }
  }
  function nativeProgress(diagnostics, stage, state) {
    rememberNativeAttempt({ stage, state }, diagnostics);
  }
  const backgroundProgressStages = new Set(["normalize_admission", "native_prepare", "native_assets", "native_finalize"]);
  function nativeProgressSnapshot(diagnostics) {
    return diagnostics.entries.filter((entry) => ["BEGIN", "END"].includes(entry.state) &&
      ((Object.keys(entry).length === 2 && nativeProgressStages.has(entry.stage)) ||
       (Object.keys(entry).length === 3 && entry.source === "background_debug_log" && backgroundProgressStages.has(entry.stage)))).map((entry) => ({ ...entry }));
  }

  function observeNativePreparation(rawRef, nativeRequestId, signal, diagnostics) {
    // This is the already-created MAIN fetch request, not a staging UUID or
    // invocation authority. Missing correlation leaves diagnostics unknown.
    if (typeof nativeRequestId !== "string" || !/^polylogue-native-fetch-\d+-[a-z0-9]+$/.test(nativeRequestId)) return () => {};
    let admitted = false; let stopped = false;
    const equal = (a, b) => ["at", "stage", "phase", "state", "acquisition_ref", "native_request_id"].every(key => a?.[key] === b?.[key]);
    const changed = (changes, area) => {
      if (stopped || signal.aborted || area !== "local") return;
      const change = changes?.polylogueDebugLog;
      if (!Array.isArray(change?.newValue)) return;
      const prior = Array.isArray(change.oldValue) ? change.oldValue : [];
      for (const row of [...change.newValue].reverse()) {
        if (!row || Object.keys(row).length !== 6 || row.stage !== "native_preparation_progress" ||
            row.acquisition_ref !== rawRef?.id || row.native_request_id !== nativeRequestId ||
            !backgroundProgressStages.has(row.phase) || !["BEGIN", "END"].includes(row.state) || prior.some(old => equal(old, row))) continue;
        if (row.phase === "normalize_admission" && row.state === "BEGIN") {
          // Repeated admission for the same request is ambiguous. Retained
          // oldValue rows do not count as another observed admission.
          if (admitted) {
            diagnostics.entries = diagnostics.entries.filter(entry => entry.source !== "background_debug_log");
            stop(); return;
          }
          admitted = true;
        }
        if (!admitted) continue;
        rememberNativeAttempt({ stage: row.phase, state: row.state, source: "background_debug_log" }, diagnostics);
      }
    };
    const stop = () => {
      if (stopped) return; stopped = true;
      try { chrome.storage?.onChanged?.removeListener(changed); } catch { /* Unavailable diagnostics cannot mask capture. */ }
      signal.removeEventListener("abort", stop);
    };
    try {
      chrome.storage?.onChanged?.addListener(changed);
      signal.addEventListener("abort", stop, { once: true });
    } catch { stop(); }
    if (signal.aborted) stop();
    return stop;
  }

  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    const marker = parts.indexOf("c");
    return marker >= 0 && parts[marker + 1] ? parts[marker + 1] : null;
  }

  // A ChatGPT temporary chat never navigates to /c/<id> -- the visible URL
  // stays at /?temporary-chat=true for the whole session -- but the page
  // still fetches /backend-api/conversation/<ephemeral-id> to render it, and
  // that fetch is intercepted the same as any other (chatgpt_bridge.js's
  // window.fetch override matches on path shape only). Zero temporary chats
  // had ever landed in the archive before this because every id-matching
  // step downstream (parseNativeCapture, fetchNativePayloadOnDemand) treated
  // "no id in the URL" as "no conversation on this page", discarding a
  // capture that had already been intercepted correctly.
  function isTemporaryChatUrl(url = window.location.href) {
    try {
      return new URL(url).searchParams.get("temporary-chat") === "true";
    } catch {
      return false;
    }
  }

  function queueFreshnessHint(
    reason,
    nativeId = conversationIdFromUrl(),
    delayMs = 5000,
    providerUpdatedAt = null,
    generationObservation = null,
  ) {
    if (freshnessSuspended) return;
    if (!nativeId || !/^[A-Za-z0-9_-]{1,256}$/.test(nativeId)) return;
    if (generationObservation) {
      const pending = pendingFreshnessObservations.get(nativeId) || [];
      const byId = new Map(pending.map((observation) => [observation.observation_id, observation]));
      byId.set(generationObservation.observation_id, generationObservation);
      pendingFreshnessObservations.set(nativeId, [...byId.values()]);

      const history = lifecycleObservationHistory.get(nativeId) || [];
      const historyById = new Map(history.map((observation) => [observation.observation_id, observation]));
      historyById.set(generationObservation.observation_id, generationObservation);
      lifecycleObservationHistory.set(nativeId, [...historyById.values()]);
    }
    const existingTimer = freshnessHintTimers.get(nativeId);
    if (existingTimer) clearTimeout(existingTimer);
    const timer = setTimeout(() => {
      freshnessHintTimers.delete(nativeId);
      const observations = pendingFreshnessObservations.get(nativeId) || [];
      pendingFreshnessObservations.delete(nativeId);
      chrome.runtime.sendMessage({
        type: "polylogue.captureFreshnessHint",
        provider: "chatgpt",
        provider_session_id: nativeId,
        provider_updated_at: providerUpdatedAt,
        reason,
        delay_ms: delayMs,
        generation_observations: observations,
      }).catch(() => undefined);
    }, 750);
    freshnessHintTimers.set(nativeId, timer);
  }

  function displayedElapsedMs(label) {
    const text = String(label || "").replace(/\s+/g, " ").trim();
    if (!/^Worked for\b/i.test(text)) return null;
    const hours = Number(text.match(/\b(\d+)\s*h\b/i)?.[1] || 0);
    const minutes = Number(text.match(/\b(\d+)\s*m\b/i)?.[1] || 0);
    const seconds = Number(text.match(/\b(\d+)\s*s\b/i)?.[1] || 0);
    if (![hours, minutes, seconds].every(Number.isFinite) || hours + minutes + seconds === 0) return null;
    return ((hours * 60 + minutes) * 60 + seconds) * 1000;
  }

  function lifecycleTurnIdentity(node) {
    return node?.getAttribute?.("data-turn-id")
      || node?.getAttribute?.("data-message-id")
      || node?.getAttribute?.("data-testid")
      || null;
  }

  function completedDurationControl() {
    const turns = [...document.querySelectorAll(MESSAGE_CONTAINER_SELECTOR)].reverse();
    for (const turn of turns) {
      const role = turn.getAttribute("data-turn") || turn.getAttribute("data-message-author-role") || "";
      if (role && role !== "assistant") continue;
      for (const button of turn.querySelectorAll("button")) {
        const label = String(button.innerText || button.textContent || "").replace(/\s+/g, " ").trim();
        if (/^Worked for\b/i.test(label)) return { turn, label };
      }
    }
    return null;
  }

  function observeGenerationLifecycle(trigger) {
    const nativeId = conversationIdFromUrl();
    if (!nativeId) return null;
    const nowMs = Date.now();
    const previous = lifecycleRuntime.get(nativeId) || {};
    const stopButton = document.querySelector(
      '[data-testid="stop-button"], button[aria-label="Stop generating"], button[aria-label="Stop streaming"]',
    );
    const completed = completedDurationControl();
    let observation = null;
    let next = previous;

    if (stopButton) {
      const stopTurnId = lifecycleTurnIdentity(stopButton.closest(MESSAGE_CONTAINER_SELECTOR));
      const completedKey = completed
        ? `${lifecycleTurnIdentity(completed.turn) || "active"}:${completed.label}`
        : null;
      const startedAtMs = previous.running ? (previous.started_at_ms || nowMs) : nowMs;
      const state = previous.running ? "in_progress" : "started";
      if (state === "started" || nowMs - (previous.last_progress_at_ms || 0) >= 30_000) {
        observation = {
          observation_id: `${nativeId}:${stopTurnId || "active"}:${state}:${state === "started" ? startedAtMs : Math.floor(nowMs / 30_000)}`,
          state,
          observed_at: new Date(nowMs).toISOString(),
          evidence_source: "dom_control",
          fidelity: "observed",
          duration_semantics: "dom_observed_wall",
          turn_provider_id: stopTurnId,
          wall_elapsed_ms: Math.max(0, nowMs - startedAtMs),
          trigger,
        };
      }
      next = {
        ...previous,
        running: true,
        started_at_ms: startedAtMs,
        last_progress_at_ms: observation ? nowMs : previous.last_progress_at_ms,
        turn_provider_id: stopTurnId || previous.turn_provider_id,
        baseline_completed_key: previous.running ? previous.baseline_completed_key : completedKey,
      };
    } else if (completed || previous.running) {
      const completedKey = completed
        ? `${lifecycleTurnIdentity(completed.turn) || "active"}:${completed.label}`
        : null;
      const effectiveCompleted = previous.running
        && completedKey
        && completedKey === previous.baseline_completed_key
        ? null
        : completed;
      const turn = effectiveCompleted?.turn || null;
      const label = effectiveCompleted?.label || null;
      const turnId = lifecycleTurnIdentity(turn) || previous.turn_provider_id || "active";
      const elapsedMs = displayedElapsedMs(label);
      const terminalKey = `${turnId}:${label || "stop_disappeared"}`;
      if (previous.terminal_key !== terminalKey) {
        observation = {
          observation_id: `${nativeId}:${turnId}:completed:${window.polylogueCapture.fnv1a(terminalKey)}`,
          state: "completed",
          observed_at: new Date(nowMs).toISOString(),
          evidence_source: effectiveCompleted ? "dom_duration_control" : "dom_control_transition",
          fidelity: effectiveCompleted ? "observed" : "inferred",
          duration_semantics: effectiveCompleted ? "provider_ui_elapsed" : "dom_observed_wall",
          turn_provider_id: turnId === "active" ? null : turnId,
          displayed_elapsed_ms: elapsedMs,
          wall_elapsed_ms: previous.started_at_ms ? Math.max(0, nowMs - previous.started_at_ms) : null,
          raw_label: label,
          trigger,
        };
      }
      next = { ...previous, running: false, terminal_key: terminalKey, turn_provider_id: turnId };
    }

    lifecycleRuntime.set(nativeId, next);
    if (observation) {
      queueFreshnessHint(
        `generation_${observation.state}`,
        nativeId,
        observation.state === "completed" ? 0 : 1000,
        null,
        observation,
      );
    }
    return observation;
  }

  async function rememberNativeCapture(capture, requestedConversationId = null) {
    if (!capture?.ok || !capture.bodyRef) return null;
    if (requestedConversationId) {
      const source = new URL(capture.url, window.location.origin);
      const expectedPath = `/backend-api/conversation/${encodeURIComponent(requestedConversationId)}`;
      const visibleId = conversationIdFromUrl();
      if (source.origin !== window.location.origin || source.pathname !== expectedPath ||
          (visibleId && visibleId !== requestedConversationId)) return null;
    }
    const result = await window.polylogueAssetStream.nativeHeaders("chatgpt", capture);
    capture = { ...result.capture, invocationRef: capture.invocationRef || null, nativeRequestId: capture.nativeRequestId };
    const headers = result.headers;
    const nativeId = String(headers.conversation_id || headers.id || "");
    const bound = requestedConversationId || conversationIdFromUrl();
    if ((bound && nativeId !== bound) || (!bound && (!isTemporaryChatUrl() || headers.is_temporary !== true))) return null;
    const selected = result.cache;
    if (window.polylogueCapture.cacheCaptureIsNewer(nativeCapture, selected.capture)) { nativeCapture = { ...selected.capture, invocationRef: capture.invocationRef || null }; nativeHeader = selected.headers; }
    return { capture, headers };
  }
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== nativeCaptureMessage || !data.capture) return;
    nativeHeaderPromise = rememberNativeCapture(data.capture).then((revision) => {
      if (revision && data.capture.source !== "polylogue_native_fetch") {
        queueFreshnessHint("provider_native_observed", String(revision.headers.conversation_id || revision.headers.id), 3000, revision.headers.update_time);
      }
    }).catch(() => undefined);
  });

  navigator.serviceWorker?.addEventListener?.("message", (event) => {
    const data = event.data || {};
    if (data.type !== "new-message") return;
    const nativeId = String(data.conversation_id || data.data?.conversation_id || "");
    queueFreshnessHint("provider_push_new_message", nativeId, 2000);
  });

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== nativeFetchResponseMessage || !data.requestId) return;
    const pending = nativeFetchResponses.get(data.requestId);
    if (!pending) return;
    if (Object.hasOwn(data, "progress")) {
      const progress = data.progress;
      if (Object.keys(event.data).length === 3 && progress && typeof progress === "object" &&
          Object.keys(progress).length === 2 && ["staging", "provider_auth", "provider_response", "body"].includes(progress.stage) &&
          ["BEGIN", "END"].includes(progress.state)) {
        nativeProgress(pending.diagnostics, progress.stage, progress.state);
      }
      return;
    }
    nativeFetchResponses.delete(data.requestId);
    pending.resolve({ capture: data.capture || null, error: data.error || null, requestId: data.requestId });
  });

  function latestNativePayload(expectedConversationId = conversationIdFromUrl()) {
    const nativeId = nativeHeader?.conversation_id || nativeHeader?.id;
    if (!nativeCapture || (expectedConversationId && nativeId !== expectedConversationId)) return null;
    if (!expectedConversationId && (!isTemporaryChatUrl() || nativeHeader?.is_temporary !== true)) return null;
    return nativeHeader;
  }

  function nativePayloadUpdatedAt(payload) {
    const value = payload?.update_time;
    if (typeof value === "number" && Number.isFinite(value)) {
      return value < 10_000_000_000 ? value * 1000 : value;
    }
    if (typeof value === "string" && value.trim()) {
      const parsed = Date.parse(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }

  async function cachedFreshTerminalPayload(expectedConversationId, minimumUpdatedAt, signal, attribution) {
    const payload = latestNativePayload(expectedConversationId);
    if (!payload) return null;
    // Without a provider revision hint, the cache cannot distinguish an
    // unchanged terminal response from a stale page-load response (or prove
    // that the next provider read would not return a 429). Keep the ordinary
    // provider path in that case so a throttle remains typed and no cached
    // payload can trigger asset downloads after it.
    if (!minimumUpdatedAt) return null;
    const minimumMs = nativePayloadUpdatedAt({ update_time: minimumUpdatedAt });
    const cachedMs = nativePayloadUpdatedAt(payload);
    if (!Number.isFinite(minimumMs) || !Number.isFinite(cachedMs) || cachedMs < minimumMs) return null;
    const candidate = nativeCapture;
    const summary = await window.polylogueAssetStream.nativeEnvelope({ provider: "chatgpt", capture: candidate, nativeId: expectedConversationId, signal, attribution, summaryOnly: true });
    if (typeof summary?.needs_follow_up !== "boolean") throw new Error("native_capture_summary_invalid");
    return summary.needs_follow_up === false ? { capture: candidate, headers: payload } : null;
  }

  async function requestNativeCaptureFromPage(conversationId, signal, invocationRef = null, diagnostics = null) {
    const requestId = `polylogue-native-fetch-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      nativeFetchResponses.set(requestId, { resolve, diagnostics });
    });
    window.polylogueAssetStream.pageMessage(
      {
        type: nativeFetchRequestMessage,
        requestId,
        conversationId,
        invocationRef
      },
      chrome.runtime.id
    );
    const onAbort = () => {
      window.polylogueAssetStream.pageMessage({ type: "polylogue.chatgpt.cancelRequest", requestId }, chrome.runtime.id);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
    return responsePromise.finally(() => signal.removeEventListener("abort", onAbort));
  }

  function retryAfterMilliseconds(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    const seconds = Number(value);
    const deadline = Date.parse(value);
    const delay = Number.isFinite(seconds) && seconds >= 0 ? Math.ceil(seconds * 1000)
      : Number.isFinite(deadline) ? Math.max(0, deadline - Date.now()) : null;
    if (delay !== null && (!Number.isFinite(delay) || !Number.isFinite(new Date(Date.now() + delay).getTime()))) {
      const error = new Error("provider_retry_after_unrepresentable"); error.code = "provider_retry_after_unrepresentable"; throw error;
    }
    return delay;
  }

  function rateLimitedNativeFetch(capture, requestId) {
    return { payload: null, rateLimited: true, retryAfterMs: retryAfterMilliseconds(capture.retryAfter), providerResponse: { status: capture.status, url: capture.url }, requestId };
  }

  async function providerThrottle() {
    try {
      return await chrome.runtime.sendMessage({ type: "polylogue.providerThrottle", provider: "chatgpt" });
    } catch {
      return { outcome: "provider_throttle_authority_unavailable" };
    }
  }

  async function recordProviderRateLimit(retryAfterMs, providerResponse, requestId) {
    try {
      return await chrome.runtime.sendMessage({
        type: "polylogue.providerRateLimited",
        provider: "chatgpt",
        provider_response: providerResponse, request_id: requestId,
        retry_after_seconds: retryAfterMs === null ? null : Math.ceil(retryAfterMs / 1000),
      });
    } catch {
      return { ok: false, outcome: "provider_throttle_authority_unavailable" };
    }
  }

  const freshConversationWaitIntervalMs = 300;

  // ChatGPT does not update the URL to /c/<id> until the backend accepts the
  // first turn of a brand-new conversation. A capture triggered in that
  // narrow window (in-page save button, DOM-mutation freshness observer) has
  // no id to fetch by yet and no intercepted response either -- this was the
  // only case where DOM scraping ever produced real captures (2026-07-17:
  // every chatgpt-dom-v1 capture landed inside one such window, and every
  // one was a degraded 3-turn shadow of a conversation the native path
  // picked up moments later with the full turn history). Wait for the SPA
  // router to publish the id instead of falling back to a DOM scrape.
  async function waitForConversationId(signal) {
    let id = conversationIdFromUrl();
    while (!id) {
      signal.throwIfAborted();
      await new Promise((resolve, reject) => {
        const onAbort = () => { window.clearTimeout(timer); reject(signal.reason); };
        const timer = window.setTimeout(() => { signal.removeEventListener("abort", onAbort); resolve(); }, freshConversationWaitIntervalMs);
        signal.addEventListener("abort", onAbort, { once: true });
      });
      id = conversationIdFromUrl();
    }
    return id;
  }

  async function fetchNativePayloadOnDemand(requestedConversationId = null, { preferCachedTerminal = false, minimumUpdatedAt = null, signal, invocationRef = null, attribution = {}, diagnostics = null } = {}) {
    nativeProgress(diagnostics, "pending_header", "BEGIN");
    await window.polylogueAssetStream.settleNativeHeaders(nativeHeaderPromise, signal);
    nativeProgress(diagnostics, "pending_header", "END");
    let nativeId = requestedConversationId || conversationIdFromUrl();
    if (!nativeId && !isTemporaryChatUrl()) nativeId = await waitForConversationId(signal);
    if (!nativeId) return { capture: nativeCapture, headers: nativeHeader, retryAfterMs: null };
    nativeProgress(diagnostics, "restore", "BEGIN");
    const restored = await window.polylogueAssetStream.restoreNative("chatgpt", nativeId, signal, invocationRef);
    nativeProgress(diagnostics, "restore", "END");
    if (restored) { nativeCapture = { ...restored, invocationRef }; nativeHeader = restored.headers; }
    if (preferCachedTerminal) {
      const cached = await cachedFreshTerminalPayload(nativeId, minimumUpdatedAt, signal, attribution);
      if (cached) return { capture: cached.capture, headers: cached.headers, retryAfterMs: null, cached: true };
    }
    const result = await requestNativeCaptureFromPage(nativeId, signal, invocationRef, diagnostics);
    const acquired = result?.capture ? { ...result.capture, invocationRef, nativeRequestId: result.requestId } : null;
    if (acquired?.status === 429) return rateLimitedNativeFetch(acquired, result.requestId);
    signal.throwIfAborted();
    nativeProgress(diagnostics, "header", "BEGIN");
    const revision = await rememberNativeCapture(acquired, nativeId);
    nativeProgress(diagnostics, "header", "END");
    rememberNativeAttempt({ stage: "page_bridge_fetch", ok: acquired?.ok ?? null, status: acquired?.status ?? null,
      accepted: Boolean(revision), error: result?.error || acquired?.error || null }, diagnostics);
    return revision ? { ...revision, retryAfterMs: null } : { capture: null, retryAfterMs: null };
  }

  // --- Assistant-produced asset acquisition (sandbox + file-service) ------
  //
  // Deliverable files surface only as expiring links; the conversation JSON
  // never carries bytes. Fetch them through the page bridge (authenticated
  // session) at capture time and ship them as envelope attachments. Failures
  // are normal (links expire with the sandbox container) and must never fail
  // the capture itself — per-asset outcomes are disclosed in provider_meta.
  const assetFetchRequestMessage = "polylogue.chatgpt.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.chatgpt.assetFetchResponse";
  const assetResponses = new Map();

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = window.polylogueAssetStream.readPageMessage(event);
    if (!data) return;
    if (data.type !== assetFetchResponseMessage || !data.requestId) return;
    const pending = assetResponses.get(data.requestId);
    if (!pending) return;
    assetResponses.delete(data.requestId);
    if (data.outcome && typeof data.outcome === "object") {
      pending.resolve(data.outcome);
    } else {
      pending.resolve({ status: "request_failed", detail: "bridge_response_missing" });
    }
  });

  function requestAssetFromPage(request, signal, acquisition) {
    const requestId = `polylogue-asset-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    return window.polylogueAssetStream.request({ provider: "chatgpt", requestId, signal, purpose: { acquisition }, start: () => {
      const responsePromise = new Promise((resolve) => assetResponses.set(requestId, { resolve }));
      const onAbort = () => window.polylogueAssetStream.pageMessage({ type: "polylogue.chatgpt.cancelRequest", requestId }, chrome.runtime.id);
      signal.addEventListener("abort", onAbort, { once: true });
      window.polylogueAssetStream.pageMessage({ type: assetFetchRequestMessage, requestId, request }, chrome.runtime.id);
      if (signal.aborted) onAbort();
      return responsePromise.finally(() => signal.removeEventListener("abort", onAbort));
    } });
  }

  async function acquireAssets(descriptors, conversationId, signal, rawRef, recordKey, attachmentOrdinal) {
    const outcome = {
      attempted: descriptors.length,
      acquired: 0,
      failed: [],
      acquired_assets: [],
      status_counts: {},
    };
    const attachments = [];
    for (const descriptor of descriptors) {
      signal.throwIfAborted();
      const request = {
        kind: descriptor.url?.startsWith("sandbox:") ? "sandbox" : "file",
        conversationId,
        messageId: descriptor.message_provider_id,
        sandboxPath: descriptor.url?.startsWith("sandbox:") ? descriptor.url.slice("sandbox:".length) : null,
        fileId: descriptor.provider_meta?.provider_file_id || null,
      };
      let result;
      try {
        result = request.kind === "file" && !request.fileId
          ? { status: "no_resolvable_source" }
          : await requestAssetFromPage(request, signal, { raw_id: rawRef, native_id: conversationId, record_key: recordKey, attachment_id: descriptor.provider_attachment_id, attachment_ordinal: attachmentOrdinal });
      }
      catch (error) {
        signal.throwIfAborted();
        result = { status: error.outcome === "rate_limited" ? "rate_limited" : "request_failed", detail: typeof error.code === "string" ? error.code : (error.message || "request_failed") };
      }
      signal.throwIfAborted();
      const status = typeof result.status === "string" ? result.status : "request_failed";
      const contentSha256 = result.asset && result.asset.sha256;
      const acquiredIsValid =
        status === "acquired" &&
        result.asset &&
        result.asset.staged_asset &&
        typeof contentSha256 === "string" &&
        /^[0-9a-f]{64}$/.test(contentSha256);
      const recordedStatus = acquiredIsValid ? "acquired" : status === "acquired" ? "invalid_response" : status;
      outcome.status_counts[recordedStatus] = (outcome.status_counts[recordedStatus] || 0) + 1;
      if (acquiredIsValid) {
        outcome.acquired += 1;
        outcome.acquired_assets.push({
          provider_attachment_id: descriptor.provider_attachment_id,
          attachment_kind: descriptor.attachment_kind,
          sha256: contentSha256,
          size_bytes: result.asset.size_bytes || 0
        });
        attachments.push({
          provider_attachment_id: descriptor.provider_attachment_id,
          message_provider_id: descriptor.message_provider_id,
          attachment_kind: descriptor.attachment_kind,
          name: result.asset.name || descriptor.name,
          mime_type: result.asset.mime_type || descriptor.mime_type,
          size_bytes: result.asset.size_bytes || null,
          staged_asset: result.asset.staged_asset,
          provider_meta: {
            capture_source: "chatgpt_page_asset_fetch",
            asset_kind: descriptor.attachment_kind,
            sandbox_path: request.sandboxPath,
            content_sha256: contentSha256
          }
        });
      } else {
        outcome.failed.push({
          provider_attachment_id: descriptor.provider_attachment_id,
          attachment_kind: descriptor.attachment_kind,
          status: recordedStatus,
          error: recordedStatus,
          phase: result.phase || null,
          http_status: typeof result.http_status === "number" ? result.http_status : null,
          detail: status === "acquired" ? "acquired_asset_missing_sha256" : result.detail || null
        });
      }
    }
    return { attachments, outcome };
  }

  async function performCapture(reason = null, requestedConversationId = null, deferReceiver = false,
    generationObservationsOverride = [], providerUpdatedAt = null, invocationRef = null, signal, diagnostics) {
    signal.throwIfAborted();
    nativeProgress(diagnostics, "throttle", "BEGIN");
    const throttle = await providerThrottle();
    nativeProgress(diagnostics, "throttle", "END");
    if (throttle?.ok !== true) return { ok: false, error: throttle?.outcome || "provider_throttle_authority_unavailable",
      outcome: throttle?.outcome, retry_after_seconds: throttle?.retry_after_seconds ?? null };
    const selectedId = requestedConversationId || conversationIdFromUrl();
    const observationsFor = (id) => {
      const observations = new Map();
      for (const observation of [...(lifecycleObservationHistory.get(id) || []), ...(Array.isArray(generationObservationsOverride) ? generationObservationsOverride : [])]) {
        if (observation?.observation_id) observations.set(observation.observation_id, observation);
      }
      return { generation_observations: [...observations.values()] };
    };
    const attribution = observationsFor(selectedId);
    const acquired = await fetchNativePayloadOnDemand(requestedConversationId, {
      preferCachedTerminal: reason === "freshness_convergence", minimumUpdatedAt: providerUpdatedAt, signal, invocationRef, attribution, diagnostics });
    if (acquired.rateLimited) {
      const recorded = await recordProviderRateLimit(acquired.retryAfterMs, acquired.providerResponse, acquired.requestId);
      if (!recorded?.ok) return { ok: false, error: "provider_throttle_authority_unavailable" };
      return { ok: false, error: "rate_limited", outcome: "rate_limited", retry_after_seconds: acquired.retryAfterMs === null ? null : Math.ceil(acquired.retryAfterMs / 1000), native_attempts: diagnostics.entries.slice(), native_attempts_dropped: diagnostics.dropped };
    }
    const source = acquired.capture;
    if (!source) return { ok: false, error: "native_capture_unavailable", native_attempts: diagnostics.entries.slice(), native_attempts_dropped: diagnostics.dropped };
    const nativeId = String(acquired.headers?.conversation_id || acquired.headers?.id || requestedConversationId || conversationIdFromUrl() || "");
    nativeProgress(diagnostics, "canonical", "BEGIN");
    const stopPreparationProgress = observeNativePreparation(source.bodyRef, source.nativeRequestId, signal, diagnostics);
    let finalEnvelope;
    try {
      finalEnvelope = await window.polylogueAssetStream.nativeEnvelope({ provider: "chatgpt", capture: source, nativeId, signal,
        attribution: nativeId === selectedId ? attribution : observationsFor(nativeId) });
    } finally { stopPreparationProgress(); }
    nativeProgress(diagnostics, "canonical", "END");
    signal.throwIfAborted();
    if (deferReceiver) return { ok: true, envelope: finalEnvelope, deferred: true };
    nativeProgress(diagnostics, "publication", "BEGIN");
    const captureResult = await window.polylogueCapture.sendCapture(finalEnvelope, reason, signal);
    nativeProgress(diagnostics, "publication", "END");
    if (!captureResult?.ok) {
      messageLayer?.reportOutcome({ ok: false });
      return { ok: false, envelope: finalEnvelope, captureResult, error: captureResult?.error || "capture_rejected", timelineRecorded: true };
    }
    const archiveState = await window.polylogueCapture.refreshArchiveState("chatgpt", finalEnvelope.session.provider_session_id);
    messageLayer?.reportOutcome({ ok: true, acceptedIdentities: captureResult.accepted_identities });
    return { ok: true, envelope: finalEnvelope, captureResult, archiveState };
  }

  const recordOperations = new Set();
  function acquireRecord(message) {
    const controller = new AbortController();
    const operation = { controller, captureRef: message.capture_ref, promise: null };
    operation.promise = acquireAssets(message.attachments, message.nativeId, controller.signal, message.capture_ref, message.recordKey, message.attachmentOrdinal)
      .then((acquisition) => ({ ok: true, acquisition }))
      .catch((error) => ({ ok: false, error: controller.signal.aborted ? "capture_cancelled" : String(error.message || error) }))
      .finally(() => recordOperations.delete(operation));
    recordOperations.add(operation);
    return operation.promise;
  }
  async function cancelRecord(captureRef) {
    const owned = [...recordOperations].filter((operation) => operation.captureRef === captureRef);
    for (const operation of owned) operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    await Promise.allSettled(owned.map((operation) => operation.promise));
    return { ok: true, outcome: "cancelled" };
  }

  const captureOperations = new Set();
  function capture(...args) {
    const controller = new AbortController();
    const diagnostics = { entries: [], dropped: 0, cancelled: false };
    const stopProgress = () => { diagnostics.cancelled = true; };
    controller.signal.addEventListener("abort", stopProgress, { once: true });
    const operation = { controller, invocationRef: args[5] || null, promise: null };
    operation.promise = performCapture(...Array.from({ length: 6 }, (_, index) => args[index]), controller.signal, diagnostics)
      .then((result) => ({ ...result, native_progress: nativeProgressSnapshot(diagnostics) }))
      .catch((error) => {
        if (controller.signal.aborted) return { ok: false, error: "capture_cancelled", outcome: "cancelled", native_progress: nativeProgressSnapshot(diagnostics) };
        throw error;
      }).finally(() => { controller.signal.removeEventListener("abort", stopProgress); captureOperations.delete(operation); });
    operation.promise.nativeProgress = () => nativeProgressSnapshot(diagnostics);
    captureOperations.add(operation);
    return operation.promise;
  }
  async function cancelCapture(invocationRef = null) {
    const owned = [...captureOperations].filter((operation) => !invocationRef ||
      (operation.invocationRef?.id === invocationRef.id && operation.invocationRef?.token === invocationRef.token));
    for (const operation of owned) operation.controller.abort(new globalThis.DOMException("capture_cancelled", "AbortError"));
    const closing = owned.filter((operation) => operation.invocationRef).map((operation) =>
      chrome.runtime.sendMessage({ type: "polylogue.closeNativeInvocation", invocation_ref: operation.invocationRef }));
    await Promise.allSettled(closing);
    await Promise.allSettled(owned.map((operation) => operation.promise));
    return { ok: true, outcome: "cancelled", drained: owned.length };
  }
  window.addEventListener("pagehide", () => {
    freshnessSuspended = true;
    freshnessObserver?.disconnect();
    if (domFreshnessScanTimer !== null) clearTimeout(domFreshnessScanTimer);
    domFreshnessScanTimer = null;
    for (const timer of freshnessHintTimers.values()) clearTimeout(timer);
    freshnessHintTimers.clear();
    void cancelCapture();
  });
  window.addEventListener("pageshow", () => {
    freshnessSuspended = false;
    for (const nativeId of pendingFreshnessObservations.keys()) {
      queueFreshnessHint("provider_document_restored", nativeId);
    }
    freshnessObserver?.observe(document.documentElement, { childList: true, characterData: true, subtree: true });
    scheduleDomFreshnessScan?.();
  });
  window.polylogueCapture.cancelCapture = cancelCapture;
  window.polylogueCapture.capturePage = capture;
  if (window.polylogueMessageLayer) {
    messageLayer = window.polylogueMessageLayer.mount({
      containerSelector: MESSAGE_CONTAINER_SELECTOR,
      identityForNode: (node) => window.polylogueCapture.identityObservation({
        provider: "chatgpt", conversationId: conversationIdFromUrl() || latestNativePayload()?.conversation_id || latestNativePayload()?.id || null,
        // Only an explicit provider-native message id is authoritative. A
        // turn test id, DOM ordinal, and visible text are diagnostic hints.
        messageId: node.getAttribute("data-message-id"),
        text: node.innerText || node.textContent || "", adapterName: nativeAdapterName,
        adapterVersion: chrome.runtime.getManifest().version,
        fidelity: node.getAttribute("data-message-id") ? "native" : "unknown",
      }),
      onSave: () => {
        capture("message_layer_save").catch(() => undefined);
      },
    });
  }
  if (typeof MutationObserver !== "undefined") {
    const domFreshnessSignature = () => [...document.querySelectorAll(MESSAGE_CONTAINER_SELECTOR)]
      .map((node) => {
        const text = String(node.innerText || node.textContent || "").replace(/\s+/g, " ").trim();
        return [
          node.getAttribute("data-message-id") || node.getAttribute("data-testid") || "",
          node.getAttribute("data-message-author-role") || "",
          text.length,
          text.slice(-80),
        ].join(":");
      })
      .join("|");
    lastDomFreshnessSignature = domFreshnessSignature();
    observeGenerationLifecycle("initial_scan");
    scheduleDomFreshnessScan = () => {
      if (freshnessSuspended) return;
      observeGenerationLifecycle("dom_mutation");
      if (domFreshnessScanTimer) clearTimeout(domFreshnessScanTimer);
      domFreshnessScanTimer = setTimeout(() => {
        domFreshnessScanTimer = null;
        const lifecycleObservation = observeGenerationLifecycle("dom_mutation");
        const signature = domFreshnessSignature();
        if (!signature || signature === lastDomFreshnessSignature) return;
        lastDomFreshnessSignature = signature;
        if (!lifecycleObservation) {
          queueFreshnessHint("provider_dom_changed", conversationIdFromUrl(), 5000);
        }
      }, 750);
    };
    freshnessObserver = new MutationObserver(scheduleDomFreshnessScan);
    const observeFreshness = () => freshnessObserver.observe(document.documentElement, {
      childList: true,
      characterData: true,
      subtree: true,
    });
    observeFreshness();
    window.addEventListener("pageshow", () => {
      observeGenerationLifecycle("page_restore");
      observeFreshness();
    });
    window.addEventListener("pagehide", () => {
      freshnessObserver.disconnect();
      if (domFreshnessScanTimer) clearTimeout(domFreshnessScanTimer);
      domFreshnessScanTimer = null;
    });
  }
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "polylogue.acquireRecordAssets") { acquireRecord(message).then(sendResponse); return true; }
    if (message.type === "polylogue.cancelRecordAssets") { cancelRecord(message.capture_ref).then(sendResponse); return true; }
    if (message.type === "polylogue.cancelCapture") {
      cancelCapture(message.invocationRef || null).then(sendResponse);
      return true;
    }
    if (message.type === "polylogue.captureIdentity") {
      const payload = message.expectedUrl === window.location.href ? latestNativePayload() : null;
      const id = payload?.conversation_id || payload?.id;
      sendResponse({ provider_session_id: typeof id === "string" && /^[A-Za-z0-9_-]{1,256}$/.test(id) ? id : null });
      return true;
    }
    if (message.type !== "polylogue.capturePage") return false;
    const operation = capture(
      message.reason || null,
      message.providerSessionId || null,
      message.deferReceiver === true,
      message.generationObservations ?? [],
      message.providerUpdatedAt || null,
      message.invocationRef || null,
    );
    operation.then(sendResponse)
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error), native_progress: operation.nativeProgress() }));
    return true;
  });
})();
