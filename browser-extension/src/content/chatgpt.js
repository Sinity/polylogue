(function () {
  const domAdapterName = "chatgpt-dom-v1";
  const nativeAdapterName = "chatgpt-native-v1";
  const nativeCaptureMessage = "polylogue.chatgpt.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.chatgpt.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.chatgpt.nativeFetchResponse";
  const nativeCaptures = [];
  const nativeFetchResponses = new Map();
  const nativeAttemptDiagnostics = [];

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

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== nativeCaptureMessage || !data.capture) return;
    nativeCaptures.push(data.capture);
    if (nativeCaptures.length > 8) nativeCaptures.splice(0, nativeCaptures.length - 8);
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

  function collectTurns() {
    const nodes = [
      ...document.querySelectorAll('[data-testid^="conversation-turn-"], article, [data-message-author-role]')
    ];
    return nodes
      .map((node, index) => {
        const explicitRole = node.getAttribute("data-message-author-role");
        const role = explicitRole || roleFromNode(node, index);
        const text = window.polylogueCapture.visibleText(node);
        return text ? { role, text, provider_meta: { selector_index: index } } : null;
      })
      .filter(Boolean);
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
      if (!text) continue;
      const role = roleFromRaw(message.author && message.author.role);
      const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
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

  function parseNativeCapture(capture) {
    if (!capture || !capture.ok || typeof capture.body !== "string") return null;
    const currentConversationId = conversationIdFromUrl();
    if (!currentConversationId || !String(capture.url || "").includes(`/conversation/${currentConversationId}`)) {
      return null;
    }
    try {
      const payload = JSON.parse(capture.body);
      if (!payload || typeof payload !== "object" || !payload.mapping) return null;
      return payload;
    } catch {
      return null;
    }
  }

  function latestNativePayload() {
    for (let index = nativeCaptures.length - 1; index >= 0; index -= 1) {
      const payload = parseNativeCapture(nativeCaptures[index]);
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
      }, 10000);
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
      const response = await fetch(`/backend-api/conversation/${encodeURIComponent(conversationId)}`, {
        credentials: "include",
        cache: "no-store"
      });
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
      const payload = await response.clone().json();
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

  async function fetchNativePayloadOnDemand() {
    const conversationId = conversationIdFromUrl();
    if (!conversationId) return null;
    const pageResult = await requestNativeCaptureFromPage(conversationId);
    const pageCapture = pageResult && pageResult.capture;
    const pagePayload = parseNativeCapture(pageCapture);
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

  function buildNativeEnvelope(payload) {
    const turns = collectNativeTurns(payload);
    if (!turns.length) return null;
    return window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: nativeAdapterName,
      turns,
      providerSessionId: String(payload.conversation_id || payload.id || conversationIdFromUrl()),
      title: typeof payload.title === "string" && payload.title ? payload.title : null,
      createdAt: timestampFromSeconds(payload.create_time),
      updatedAt: timestampFromSeconds(payload.update_time),
      model: modelFromNativePayload(payload),
      providerMeta: {
        capture_source: "chatgpt_backend_api",
        current_node: payload.current_node || null,
        mapping_node_count: payload.mapping ? Object.keys(payload.mapping).length : 0
      },
      rawProviderPayload: payload
    });
  }

  async function capture() {
    const nativePayload = latestNativePayload() || (await fetchNativePayloadOnDemand());
    const envelope = nativePayload ? buildNativeEnvelope(nativePayload) : null;
    const fallbackEnvelope = () => {
      const turns = collectTurns();
      if (!turns.length) return null;
      return window.polylogueCapture.buildEnvelope({
        provider: "chatgpt",
        adapterName: domAdapterName,
        turns,
        providerMeta: {
          native_attempts: nativeAttemptDiagnostics.slice(-6)
        }
      });
    };
    const finalEnvelope = envelope || fallbackEnvelope();
    if (!finalEnvelope) return { ok: false, error: "no_turns" };
    const captureResult = await window.polylogueCapture.sendCapture(finalEnvelope);
    const archiveState = await window.polylogueCapture.refreshArchiveState(
      "chatgpt",
      finalEnvelope.session.provider_session_id
    );
    return { ok: true, envelope: finalEnvelope, captureResult, archiveState };
  }

  let timer = 0;
  function scheduleCapture() {
    window.clearTimeout(timer);
    timer = window.setTimeout(capture, 1200);
  }

  scheduleCapture();
  window.polylogueCapture.capturePage = capture;
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.capturePage") return false;
    capture().then(sendResponse).catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
  new MutationObserver(scheduleCapture).observe(document.documentElement, {
    childList: true,
    subtree: true,
    characterData: true
  });
})();
