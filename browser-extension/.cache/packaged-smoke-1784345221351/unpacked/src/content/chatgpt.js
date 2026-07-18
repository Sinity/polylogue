(function () {
  if (window.__polylogueChatgptCaptureInstalled) return;
  window.__polylogueChatgptCaptureInstalled = true;

  // In-page Layer 1 (polylogue-ys30): capture-status dot + save action mounted
  // next to each detected message. Reused across every capture trigger below
  // (badge click, popup, background auto-capture) so the dots always reflect
  // the most recent capture outcome for the whole session.
  const MESSAGE_CONTAINER_SELECTOR = '[data-testid^="conversation-turn-"], article, [data-message-author-role]';
  let messageLayer = null;

  const domAdapterName = "chatgpt-dom-v1";
  const nativeAdapterName = "chatgpt-native-v1";
  const nativeCaptureMessage = "polylogue.chatgpt.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.chatgpt.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.chatgpt.nativeFetchResponse";
  const nativeFetchTimeoutMs = 8000;
  const nativeCaptures = [];
  const nativeFetchResponses = new Map();
  const nativeAttemptDiagnostics = [];
  const freshnessHintTimers = new Map();
  const pendingFreshnessObservations = new Map();
  const lifecycleObservationHistory = new Map();
  const lifecycleRuntime = new Map();
  let domFreshnessScanTimer = null;
  let lastDomFreshnessSignature = null;

  function rememberNativeAttempt(diagnostic) {
    nativeAttemptDiagnostics.push({
      attempted_at: new Date().toISOString(),
      ...diagnostic
    });
    if (nativeAttemptDiagnostics.length > 6) {
      nativeAttemptDiagnostics.splice(0, nativeAttemptDiagnostics.length - 6);
    }
  }

  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    const marker = parts.indexOf("c");
    return marker >= 0 && parts[marker + 1] ? parts[marker + 1] : null;
  }

  function queueFreshnessHint(
    reason,
    nativeId = conversationIdFromUrl(),
    delayMs = 5000,
    providerUpdatedAt = null,
    generationObservation = null,
  ) {
    if (!nativeId || !/^[A-Za-z0-9_-]{1,256}$/.test(nativeId)) return;
    if (generationObservation) {
      const pending = pendingFreshnessObservations.get(nativeId) || [];
      const byId = new Map(pending.map((observation) => [observation.observation_id, observation]));
      byId.set(generationObservation.observation_id, generationObservation);
      pendingFreshnessObservations.set(nativeId, [...byId.values()].slice(-24));

      const history = lifecycleObservationHistory.get(nativeId) || [];
      const historyById = new Map(history.map((observation) => [observation.observation_id, observation]));
      historyById.set(generationObservation.observation_id, generationObservation);
      lifecycleObservationHistory.set(nativeId, [...historyById.values()].slice(-64));
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
      const startedAtMs = previous.started_at_ms || nowMs;
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

  function nativeCaptureIdentity(capture) {
    if (!capture?.ok || typeof capture.body !== "string") return null;
    try {
      const payload = JSON.parse(capture.body);
      const nativeId = String(payload.conversation_id || payload.id || "");
      if (!/^[A-Za-z0-9_-]{1,256}$/.test(nativeId)) return null;
      const updatedAt = typeof payload.update_time === "number"
        ? new Date(payload.update_time < 10_000_000_000 ? payload.update_time * 1000 : payload.update_time).toISOString()
        : typeof payload.update_time === "string" ? payload.update_time : null;
      return { nativeId, updatedAt };
    } catch {
      return null;
    }
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== nativeCaptureMessage || !data.capture) return;
    nativeCaptures.push(data.capture);
    if (nativeCaptures.length > 8) nativeCaptures.splice(0, nativeCaptures.length - 8);
    const identity = nativeCaptureIdentity(data.capture);
    if (identity) queueFreshnessHint("provider_native_observed", identity.nativeId, 3000, identity.updatedAt);
  });

  navigator.serviceWorker?.addEventListener?.("message", (event) => {
    const data = event.data || {};
    if (data.type !== "new-message") return;
    const nativeId = String(data.conversation_id || data.data?.conversation_id || "");
    queueFreshnessHint("provider_push_new_message", nativeId, 2000);
  });

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== nativeFetchResponseMessage || !data.requestId) return;
    const pending = nativeFetchResponses.get(data.requestId);
    if (!pending) return;
    nativeFetchResponses.delete(data.requestId);
    pending.resolve({ capture: data.capture || null, error: data.error || null });
  });

  function roleFromNode(node, index) {
    const testId = node.getAttribute("data-testid") || "";
    if (testId.includes("user")) return "user";
    if (testId.includes("assistant")) return "assistant";
    const labelled = node.getAttribute("aria-label") || "";
    if (/you|user/i.test(labelled)) return "user";
    if (/chatgpt|assistant/i.test(labelled)) return "assistant";
    return index % 2 === 0 ? "user" : "assistant";
  }

  function attachmentNameFromNode(node) {
    const label = node.getAttribute("aria-label") || "";
    const download = node.getAttribute("download") || "";
    const alt = node.getAttribute("alt") || "";
    const text = window.polylogueCapture.visibleText(node);
    const href = node.getAttribute("href") || node.getAttribute("src") || "";
    const basename = href.split(/[/?#]/).filter(Boolean).at(-1) || "";
    const candidates = [label, download, alt, text, basename]
      .map((value) => String(value || "").trim())
      .filter(Boolean);
    const extensionPattern =
      "zip|tar|tgz|gz|bz2|xz|7z|rar|md|txt|pdf|doc|docx|json|jsonl|csv|tsv|py|js|ts|tsx|jsx|rs|go|java|c|cc|cpp|h|hpp|png|jpe?g|gif|webp|svg|mp3|mp4|wav|webm";
    const filePattern = new RegExp(`(?:^|\\s)([^\\s@/]+\\.(?:${extensionPattern}))(?:\\s|$)`, "i");
    for (const candidate of candidates) {
      const fileNameMatch = candidate.match(filePattern);
      if (fileNameMatch) return fileNameMatch[1].trim();
    }
    return null;
  }

  function collectAttachments(node, turnIndex) {
    const selector = [
      '[role="group"][aria-label]',
      "a[download]",
      "a[href][aria-label]",
      "img[alt]",
      "img[src]"
    ].join(",");
    const seen = new Set();
    const attachments = [];
    for (const candidate of node.querySelectorAll(selector)) {
      const name = attachmentNameFromNode(candidate);
      if (!name) continue;
      const rawHref = candidate.getAttribute("href") || candidate.getAttribute("src") || null;
      const url = rawHref && /^https?:\/\//i.test(rawHref) ? rawHref : null;
      const key = `${name}\n${url || ""}`;
      if (seen.has(key)) continue;
      seen.add(key);
      attachments.push({
        provider_attachment_id: `dom:${window.polylogueCapture.fnv1a(`${turnIndex}:${key}`)}`,
        name,
        url,
        provider_meta: {
          dom_selector_index: turnIndex,
          dom_label: candidate.getAttribute("aria-label") || null,
          dom_text: window.polylogueCapture.visibleText(candidate) || null,
          capture_source: "chatgpt_dom_attachment"
        }
      });
    }
    return attachments;
  }

  async function collectTurns() {
    const nodes = [
      ...document.querySelectorAll('[data-testid^="conversation-turn-"], article, [data-message-author-role]')
    ];
    const turns = [];
    for (const [index, node] of nodes.entries()) {
      const explicitRole = node.getAttribute("data-message-author-role");
      const role = explicitRole || roleFromNode(node, index);
      const text = window.polylogueCapture.visibleText(node);
      const domAttachments = collectAttachments(node, index);
      // chatgpt-dom-v1 has no backend-api mapping to resolve file/sandbox ids
      // from (polylogue-83u.3) — the DOM itself is the only evidence of an
      // attachment. When a chip's own href/src is already a concrete https
      // URL (rendered by the page, e.g. an inline image), fetch it through
      // the SAME authenticated page-bridge mechanism the native adapter uses
      // for sandbox/file assets (see acquireAssets/requestAssetFromPage), not
      // a new one. No resolvable URL stays honestly byte_count=0.
      const attachments = domAttachments.length ? await acquireDomAttachmentBytes(domAttachments) : domAttachments;
      if (text || attachments.length) {
        turns.push({ role, text, attachments, provider_meta: { selector_index: index } });
      }
    }
    return turns;
  }

  function extractContentText(content) {
    const parts = content && content.parts;
    if (Array.isArray(parts)) {
      const textParts = [];
      for (const part of parts) {
        if (typeof part === "string" && part) textParts.push(part);
        if (part && typeof part === "object" && typeof part.text === "string" && part.text) {
          textParts.push(part.text);
        }
      }
      if (textParts.length) return textParts.join("\n");
    }
    if (typeof content?.text === "string" && content.text) return content.text;
    if (typeof content?.result === "string" && content.result) return content.result;
    return "";
  }

  function timestampFromSeconds(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return new Date(value * 1000).toISOString();
  }

  function timeoutError(label) {
    const error = new Error(`${label}_timeout_after_${nativeFetchTimeoutMs}ms`);
    error.name = "PolylogueTimeoutError";
    return error;
  }

  function withTimeout(promise, label) {
    let timer = 0;
    const timeout = new Promise((_resolve, reject) => {
      timer = window.setTimeout(() => reject(timeoutError(label)), nativeFetchTimeoutMs);
    });
    return Promise.race([promise, timeout]).finally(() => {
      if (timer) window.clearTimeout(timer);
    });
  }

  function roleFromRaw(raw) {
    if (raw === "user" || raw === "assistant" || raw === "system" || raw === "tool") return raw;
    if (raw === "function" || raw === "tool_use" || raw === "tool_result") return "tool";
    return "unknown";
  }

  function collectNativeTurns(payload) {
    const mapping = payload && payload.mapping;
    if (!mapping || typeof mapping !== "object") return [];
    const turns = [];
    for (const [nodeId, node] of Object.entries(mapping)) {
      const message = node && node.message;
      const content = message && message.content;
      if (!message || !content) continue;
      const text = extractContentText(content);
      const role = roleFromRaw(message.author && message.author.role);
      const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
      const hasAttachmentEvidence =
        (Array.isArray(metadata.attachments) && metadata.attachments.length > 0)
        || (Array.isArray(content.parts) && content.parts.some((part) =>
          part && typeof part === "object" && (
            typeof part.asset_pointer === "string" || typeof part.file_id === "string"
          )
        ));
      if (!text && !hasAttachmentEvidence) continue;
      turns.push({
        provider_turn_id: String(message.id || node.id || nodeId),
        role,
        text,
        timestamp: timestampFromSeconds(message.create_time),
        parent_turn_id: node.parent ? String(node.parent) : null,
        provider_meta: {
          node_id: nodeId,
          content_type: content.content_type || "text",
          status: message.status || null,
          model_slug: metadata.model_slug || null,
          capture_source: "chatgpt_backend_api"
        }
      });
    }
    return turns;
  }

  function parseNativeCapture(capture, expectedConversationId = conversationIdFromUrl()) {
    if (!capture || !capture.ok || typeof capture.body !== "string") return null;
    if (!expectedConversationId || !String(capture.url || "").includes(`/conversation/${expectedConversationId}`)) {
      return null;
    }
    try {
      const payload = JSON.parse(capture.body);
      if (!payload || typeof payload !== "object" || !payload.mapping) return null;
      const payloadConversationId = payload.conversation_id || payload.id;
      if (payloadConversationId && String(payloadConversationId) !== expectedConversationId) return null;
      return payload;
    } catch {
      return null;
    }
  }

  function latestNativePayload(expectedConversationId = conversationIdFromUrl()) {
    for (let index = nativeCaptures.length - 1; index >= 0; index -= 1) {
      const payload = parseNativeCapture(nativeCaptures[index], expectedConversationId);
      if (payload) return payload;
    }
    return null;
  }

  async function requestNativeCaptureFromPage(conversationId) {
    const requestId = `polylogue-native-fetch-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        nativeFetchResponses.delete(requestId);
        resolve({ capture: null, error: "timeout" });
      }, nativeFetchTimeoutMs);
      nativeFetchResponses.set(requestId, {
        resolve(value) {
          window.clearTimeout(timeout);
          resolve(value);
        }
      });
    });
    window.postMessage(
      {
        type: nativeFetchRequestMessage,
        requestId,
        conversationId
      },
      window.location.origin
    );
    return responsePromise;
  }

  async function fetchNativePayloadFromContentScript(conversationId) {
    try {
      const controller = new globalThis.AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(timeoutError("content_script_fetch")), nativeFetchTimeoutMs);
      let response;
      try {
        response = await fetch(`/backend-api/conversation/${encodeURIComponent(conversationId)}`, {
          credentials: "include",
          cache: "no-store",
          signal: controller.signal
        });
      } finally {
        window.clearTimeout(timeoutId);
      }
      const contentType = response.headers.get("content-type") || "";
      if (!response.ok || !contentType.includes("application/json")) {
        rememberNativeAttempt({
          stage: "content_script_fetch",
          ok: response.ok,
          status: response.status,
          content_type: contentType,
          accepted: false
        });
        return null;
      }
      const payload = await withTimeout(response.clone().json(), "content_script_json");
      if (!payload || typeof payload !== "object" || !payload.mapping) {
        rememberNativeAttempt({
          stage: "content_script_fetch",
          ok: response.ok,
          status: response.status,
          content_type: contentType,
          accepted: false,
          reason: "missing_mapping"
        });
        return null;
      }
      const payloadConversationId = payload.conversation_id || payload.id;
      if (payloadConversationId && String(payloadConversationId) !== conversationId) {
        rememberNativeAttempt({
          stage: "content_script_fetch",
          ok: response.ok,
          status: response.status,
          content_type: contentType,
          accepted: false,
          reason: "conversation_id_mismatch"
        });
        return null;
      }
      rememberNativeAttempt({
        stage: "content_script_fetch",
        ok: response.ok,
        status: response.status,
        content_type: contentType,
        accepted: true
      });
      return payload;
    } catch (error) {
      rememberNativeAttempt({
        stage: "content_script_fetch",
        accepted: false,
        error: String(error && error.message ? error.message : error)
      });
      return null;
    }
  }

  async function fetchNativePayloadOnDemand(requestedConversationId = null) {
    const conversationId = requestedConversationId || conversationIdFromUrl();
    if (!conversationId) return null;
    if (!/^[A-Za-z0-9_-]{1,256}$/.test(conversationId)) return null;
    const pageResult = await requestNativeCaptureFromPage(conversationId);
    const pageCapture = pageResult && pageResult.capture;
    const pagePayload = parseNativeCapture(pageCapture, conversationId);
    rememberNativeAttempt({
      stage: "page_bridge_fetch",
      ok: pageCapture?.ok ?? null,
      status: pageCapture?.status ?? null,
      content_type: pageCapture?.contentType || null,
      body_bytes: typeof pageCapture?.body === "string" ? pageCapture.body.length : 0,
      accepted: Boolean(pagePayload),
      error: pageResult?.error || null
    });
    if (pagePayload) return pagePayload;
    return fetchNativePayloadFromContentScript(conversationId);
  }

  function modelFromNativePayload(payload) {
    const mapping = payload && payload.mapping;
    if (!mapping || typeof mapping !== "object") return null;
    for (const node of Object.values(mapping)) {
      const metadata = node?.message?.metadata;
      if (metadata && typeof metadata.model_slug === "string" && metadata.model_slug) {
        return metadata.model_slug;
      }
    }
    return null;
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
  const assetFetchTimeoutMs = 9000;
  const assetMaxBytesPerFile = 25 * 1024 * 1024;
  const assetMaxBytesTotal = 75 * 1024 * 1024;
  // Wall-clock budget for the WHOLE acquisition pass and a per-kind circuit
  // breaker: a conversation can reference dozens of sandbox files, and once
  // the sandbox container is gone every one of them fails the same way --
  // without these bounds a capture could stall for minutes on dead links.
  // Must fit inside the 35s capturePage message timeout raced by popup and
  // background (CAPTURE_MESSAGE_TIMEOUT_MS) with headroom for the
  // conversation fetch itself. Dead links answer fast; anything slower is
  // skipped and disclosed -- a later re-capture backfills idempotently.
  const assetTotalTimeBudgetMs = 10000;
  const assetConsecutiveFailureLimit = 3;
  const assetResponses = new Map();
  const sandboxLinkPattern = /sandbox:(\/mnt\/data\/[^\s)\]"'>]+)/g;

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== assetFetchResponseMessage || !data.requestId) return;
    const pending = assetResponses.get(data.requestId);
    if (!pending) return;
    assetResponses.delete(data.requestId);
    if (data.outcome && typeof data.outcome === "object") {
      pending.resolve(data.outcome);
    } else if (data.asset) {
      pending.resolve({ status: "acquired", phase: "legacy_bridge", asset: data.asset });
    } else {
      pending.resolve({ status: "request_failed", phase: "legacy_bridge", detail: "legacy_bridge_error" });
    }
  });

  function requestAssetFromPage(request) {
    const requestId = `polylogue-asset-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        assetResponses.delete(requestId);
        resolve({ status: "request_failed", phase: "content_bridge", detail: "response_timeout" });
      }, assetFetchTimeoutMs);
      assetResponses.set(requestId, {
        resolve(value) {
          window.clearTimeout(timeout);
          resolve(value);
        }
      });
    });
    window.postMessage({ type: assetFetchRequestMessage, requestId, request }, window.location.origin);
    return responsePromise;
  }

  // chatgpt-dom-v1 attachment byte acquisition (polylogue-83u.3): the DOM
  // scrape only ever recorded the chip name (byte_count=0) because there is
  // no backend-api mapping in this degraded capture mode to resolve a
  // file/sandbox id from. Any chip whose own href/src the page already
  // rendered as a concrete https URL is fetched through the bridge's "url"
  // asset kind — the same authenticated page-bridge fetch+hash mechanism
  // acquireAssets uses for the native adapter's sandbox/file assets, just
  // skipping the metadata round trip since the URL is already in hand.
  // Bounded by the same per-file/total-byte/time/failure budgets so a
  // degraded capture with many broken image chips cannot stall or balloon.
  async function acquireDomAttachmentBytes(attachments) {
    const startedAt = Date.now();
    let totalBytes = 0;
    let consecutiveFailures = 0;
    const acquired = [];
    for (const attachment of attachments) {
      if (
        !attachment.url
        || totalBytes >= assetMaxBytesTotal
        || Date.now() - startedAt >= assetTotalTimeBudgetMs
        || consecutiveFailures >= assetConsecutiveFailureLimit
      ) {
        acquired.push(attachment);
        continue;
      }
      const request = {
        kind: "url",
        url: attachment.url,
        name: attachment.name,
        maxBytes: Math.min(assetMaxBytesPerFile, assetMaxBytesTotal - totalBytes)
      };
      const result = await requestAssetFromPage(request);
      const status = typeof result.status === "string" ? result.status : "request_failed";
      const contentSha256 = result.asset && result.asset.sha256;
      const acquiredIsValid =
        status === "acquired" &&
        result.asset &&
        result.asset.base64 &&
        typeof contentSha256 === "string" &&
        /^[0-9a-f]{64}$/.test(contentSha256);
      if (acquiredIsValid) {
        totalBytes += result.asset.size_bytes || 0;
        consecutiveFailures = 0;
        acquired.push({
          ...attachment,
          mime_type: result.asset.mime_type || attachment.mime_type || null,
          size_bytes: result.asset.size_bytes || null,
          inline_base64: result.asset.base64,
          provider_meta: {
            ...attachment.provider_meta,
            asset_kind: "url",
            content_sha256: contentSha256
          }
        });
      } else {
        consecutiveFailures += 1;
        acquired.push({
          ...attachment,
          provider_meta: {
            ...attachment.provider_meta,
            asset_kind: "url",
            asset_fetch_status: status,
            asset_fetch_detail: result.detail || null
          }
        });
      }
    }
    return acquired;
  }

  function sandboxPathsFromText(text) {
    const paths = [];
    for (const match of String(text).matchAll(sandboxLinkPattern)) {
      const path = match[1].replace(/[.,;:!?*`]+$/, "");
      if (path !== "/mnt/data/" && !paths.includes(path)) paths.push(path);
    }
    return paths;
  }

  function collectAssetDescriptors(payload) {
    const mapping = payload && payload.mapping;
    if (!mapping || typeof mapping !== "object") return [];
    const descriptors = [];
    const seen = new Set();
    const add = (descriptor) => {
      if (descriptor.provider_attachment_id && !seen.has(descriptor.provider_attachment_id)) {
        seen.add(descriptor.provider_attachment_id);
        descriptors.push(descriptor);
      }
    };
    // ChatGPT's mapping insertion order is not conversation order. In a
    // branched conversation it can put old, expired interpreter assets before
    // the selected branch's newest deliverable. Because acquisition is
    // deliberately bounded by a failure breaker and a wall-clock budget, that
    // incidental order can prevent the current node's live asset from ever
    // being attempted. Walk the selected lineage newest-first, then cover the
    // remaining branches newest-first as best-effort backfill.
    const orderedNodeIds = [];
    const orderedNodeIdSet = new Set();
    let lineageNodeId = typeof payload.current_node === "string" ? payload.current_node : null;
    while (lineageNodeId && !orderedNodeIdSet.has(lineageNodeId)) {
      const node = mapping[lineageNodeId];
      if (!node) break;
      orderedNodeIds.push(lineageNodeId);
      orderedNodeIdSet.add(lineageNodeId);
      lineageNodeId = typeof node.parent === "string" ? node.parent : null;
    }
    const remainingNodeIds = Object.keys(mapping)
      .filter((nodeId) => !orderedNodeIdSet.has(nodeId))
      .sort((left, right) => {
        const leftTime = Number(mapping[left]?.message?.create_time) || 0;
        const rightTime = Number(mapping[right]?.message?.create_time) || 0;
        return rightTime - leftTime;
      });

    for (const nodeId of [...orderedNodeIds, ...remainingNodeIds]) {
      const node = mapping[nodeId];
      const message = node && node.message;
      if (!message) continue;
      const messageId = String(message.id || node.id || nodeId);
      const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
      for (const attachment of Array.isArray(metadata.attachments) ? metadata.attachments : []) {
        if (attachment && attachment.id) {
          add({
            kind: "file",
            fileId: String(attachment.id),
            provider_attachment_id: String(attachment.id),
            message_provider_id: messageId,
            name: attachment.name ? String(attachment.name) : null,
            mime_type: attachment.mime_type ? String(attachment.mime_type) : null
          });
        }
      }
      const content = message.content;
      const parts = content && Array.isArray(content.parts) ? content.parts : [];
      const role = message.author && message.author.role;
      for (const part of parts) {
        if (part && typeof part === "object" && typeof part.asset_pointer === "string" && part.asset_pointer) {
          const pointer = part.asset_pointer;
          const pointerPath = pointer.includes("://") ? pointer.split("://").at(-1) : pointer;
          const fileIdMatch = pointerPath.match(/file[-_][A-Za-z0-9]+/);
          if (fileIdMatch) {
            add({
              kind: "file",
              fileId: fileIdMatch[0],
              provider_attachment_id: pointer,
              message_provider_id: messageId,
              name: null,
              mime_type: null
            });
          }
        }
        if (typeof part === "string" && role === "assistant") {
          for (const path of sandboxPathsFromText(part)) {
            add({
              kind: "sandbox",
              sandboxPath: path,
              provider_attachment_id: `sandbox:${messageId}:${path}`,
              message_provider_id: messageId,
              name: path.replace(/\/+$/, "").split("/").at(-1) || null,
              mime_type: null
            });
          }
        }
      }
    }
    return descriptors;
  }

  async function acquireAssets(payload, conversationId) {
    const descriptors = collectAssetDescriptors(payload);
    const outcome = {
      attempted: descriptors.length,
      acquired: 0,
      failed: [],
      acquired_assets: [],
      status_counts: {},
      skipped_over_budget: 0,
      skipped_time_budget: 0,
      skipped_circuit_breaker: 0
    };
    const attachments = [];
    let totalBytes = 0;
    const startedAt = Date.now();
    const consecutiveFailuresByKind = { sandbox: 0, file: 0 };
    for (const descriptor of descriptors) {
      if (totalBytes >= assetMaxBytesTotal) {
        outcome.skipped_over_budget += 1;
        continue;
      }
      if (Date.now() - startedAt >= assetTotalTimeBudgetMs) {
        outcome.skipped_time_budget += 1;
        continue;
      }
      if (consecutiveFailuresByKind[descriptor.kind] >= assetConsecutiveFailureLimit) {
        // e.g. the sandbox container is dead: every sandbox link fails
        // identically, so stop burning the time budget on the rest.
        outcome.skipped_circuit_breaker += 1;
        continue;
      }
      const request = {
        kind: descriptor.kind,
        conversationId,
        messageId: descriptor.message_provider_id,
        sandboxPath: descriptor.sandboxPath || null,
        fileId: descriptor.fileId || null,
        maxBytes: Math.min(assetMaxBytesPerFile, assetMaxBytesTotal - totalBytes)
      };
      const result = await requestAssetFromPage(request);
      const status = typeof result.status === "string" ? result.status : "request_failed";
      const contentSha256 = result.asset && result.asset.sha256;
      const acquiredIsValid =
        status === "acquired" &&
        result.asset &&
        result.asset.base64 &&
        typeof contentSha256 === "string" &&
        /^[0-9a-f]{64}$/.test(contentSha256);
      const recordedStatus = acquiredIsValid ? "acquired" : status === "acquired" ? "invalid_response" : status;
      outcome.status_counts[recordedStatus] = (outcome.status_counts[recordedStatus] || 0) + 1;
      if (acquiredIsValid) {
        totalBytes += result.asset.size_bytes || 0;
        consecutiveFailuresByKind[descriptor.kind] = 0;
        outcome.acquired += 1;
        outcome.acquired_assets.push({
          provider_attachment_id: descriptor.provider_attachment_id,
          sha256: contentSha256,
          size_bytes: result.asset.size_bytes || 0
        });
        attachments.push({
          provider_attachment_id: descriptor.provider_attachment_id,
          message_provider_id: descriptor.message_provider_id,
          name: result.asset.name || descriptor.name,
          mime_type: result.asset.mime_type || descriptor.mime_type,
          size_bytes: result.asset.size_bytes || null,
          inline_base64: result.asset.base64,
          provider_meta: {
            capture_source: "chatgpt_page_asset_fetch",
            asset_kind: descriptor.kind,
            sandbox_path: descriptor.sandboxPath || null,
            content_sha256: contentSha256
          }
        });
      } else {
        consecutiveFailuresByKind[descriptor.kind] += 1;
        outcome.failed.push({
          provider_attachment_id: descriptor.provider_attachment_id,
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

  function buildNativeEnvelope(
    payload,
    assetAcquisition = null,
    requestedConversationId = null,
    generationObservations = [],
  ) {
    const turns = collectNativeTurns(payload);
    if (!turns.length) return null;
    return window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: nativeAdapterName,
      turns,
      providerSessionId: String(payload.conversation_id || payload.id || requestedConversationId || conversationIdFromUrl()),
      sessionKind: payload.is_temporary === true ? "temporary" : null,
      title: typeof payload.title === "string" && payload.title ? payload.title : null,
      createdAt: timestampFromSeconds(payload.create_time),
      updatedAt: timestampFromSeconds(payload.update_time),
      model: modelFromNativePayload(payload),
      providerMeta: {
        capture_source: "chatgpt_backend_api",
        current_node: payload.current_node || null,
        mapping_node_count: payload.mapping ? Object.keys(payload.mapping).length : 0,
        is_temporary: payload.is_temporary === true,
        session_kind: payload.is_temporary === true ? "temporary" : null,
        asset_acquisition: assetAcquisition ? assetAcquisition.outcome : null,
        generation_observations: generationObservations,
      },
      rawProviderPayload: payload,
      attachments: assetAcquisition ? assetAcquisition.attachments : []
    });
  }

  async function capture(
    reason = null,
    requestedConversationId = null,
    deferReceiver = false,
    nativePayloadOverride = null,
    generationObservationsOverride = [],
  ) {
    // Intercepted responses are only a bootstrap/fallback cache. A long-running
    // conversation can grow substantially after the response observed at page
    // load, so every explicit capture first asks ChatGPT for current native
    // detail with cache: "no-store". Falling back preserves degraded/offline
    // capture without allowing an old response to outrank fresh provider state.
    let nativePayload = nativePayloadOverride;
    if (nativePayload !== null) {
      if (!nativePayload || typeof nativePayload !== "object" || !nativePayload.mapping) {
        throw new Error("provided_native_payload_invalid");
      }
      const suppliedId = nativePayload.conversation_id || nativePayload.id;
      if (requestedConversationId && suppliedId && String(suppliedId) !== requestedConversationId) {
        throw new Error("provided_native_payload_identity_mismatch");
      }
    } else {
      nativePayload =
        (await fetchNativePayloadOnDemand(requestedConversationId))
        || latestNativePayload(requestedConversationId || conversationIdFromUrl());
    }
    let assetAcquisition = null;
    if (nativePayload) {
      try {
        assetAcquisition = await acquireAssets(
          nativePayload,
          String(nativePayload.conversation_id || nativePayload.id || requestedConversationId || conversationIdFromUrl())
        );
      } catch {
        assetAcquisition = {
          attachments: [],
          outcome: {
            attempted: 0,
            acquired: 0,
            acquired_assets: [],
            status_counts: { request_failed: 1 },
            failed: [
              {
                provider_attachment_id: null,
                status: "request_failed",
                error: "request_failed",
                detail: "asset_acquisition_failed"
              }
            ],
            skipped_over_budget: 0,
            skipped_time_budget: 0,
            skipped_circuit_breaker: 0
          }
        };
      }
    }
    const nativeId = String(
      nativePayload?.conversation_id
      || nativePayload?.id
      || requestedConversationId
      || conversationIdFromUrl()
      || "",
    );
    const generationObservations = [
      ...(lifecycleObservationHistory.get(nativeId) || []),
      ...(Array.isArray(generationObservationsOverride) ? generationObservationsOverride : []),
    ].reduce((byId, observation) => {
      if (observation?.observation_id) byId.set(observation.observation_id, observation);
      return byId;
    }, new Map());
    const normalizedGenerationObservations = [...generationObservations.values()].slice(-64);
    const envelope = nativePayload
      ? buildNativeEnvelope(
        nativePayload,
        assetAcquisition,
        requestedConversationId,
        normalizedGenerationObservations,
      )
      : null;
    const fallbackEnvelope = async () => {
      const turns = await collectTurns();
      if (!turns.length) return null;
      return window.polylogueCapture.buildEnvelope({
        provider: "chatgpt",
        adapterName: domAdapterName,
        turns,
        providerMeta: {
          capture_fidelity: "dom_degraded",
          native_attempts: nativeAttemptDiagnostics.slice(-6),
          generation_observations: normalizedGenerationObservations,
        }
      });
    };
    const finalEnvelope = envelope || (requestedConversationId ? null : await fallbackEnvelope());
    if (!finalEnvelope) return { ok: false, error: "no_turns" };
    if (deferReceiver) return { ok: true, envelope: finalEnvelope, deferred: true };
    const captureResult = await window.polylogueCapture.sendCapture(finalEnvelope, reason);
    if (!captureResult?.ok) {
      messageLayer?.reportOutcome({ ok: false, turnCount: finalEnvelope.session.turns.length });
      return {
        ok: false,
        envelope: finalEnvelope,
        captureResult,
        error: captureResult?.error || "capture_rejected",
        timelineRecorded: true,
      };
    }
    const archiveState = await window.polylogueCapture.refreshArchiveState(
      "chatgpt",
      finalEnvelope.session.provider_session_id
    );
    messageLayer?.reportOutcome({ ok: true, turnCount: finalEnvelope.session.turns.length });
    return { ok: true, envelope: finalEnvelope, captureResult, archiveState };
  }

  window.polylogueCapture.capturePage = capture;
  if (window.polylogueMessageLayer) {
    messageLayer = window.polylogueMessageLayer.mount({
      containerSelector: MESSAGE_CONTAINER_SELECTOR,
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
    const freshnessObserver = new MutationObserver(() => {
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
    });
    freshnessObserver.observe(document.documentElement, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.capturePage") return false;
    capture(
      message.reason || null,
      message.providerSessionId || null,
      message.deferReceiver === true,
      message.nativePayload ?? null,
      message.generationObservations ?? [],
    )
      .then(sendResponse)
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
})();
