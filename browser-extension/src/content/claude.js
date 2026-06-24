(function () {
  if (window.__polylogueClaudeCaptureInstalled) return;
  window.__polylogueClaudeCaptureInstalled = true;

  const domAdapterName = "claude-ai-dom-v1";
  const nativeAdapterName = "claude-ai-native-v1";
  const nativeCaptureMessage = "polylogue.claude.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.claude.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.claude.nativeFetchResponse";
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
    return parts[0] === "chat" && parts[1] ? parts[1] : null;
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
    const role = node.getAttribute("data-message-author-role") || node.getAttribute("data-testid") || "";
    if (/human|user/i.test(role)) return "user";
    if (/assistant|claude/i.test(role)) return "assistant";
    return index % 2 === 0 ? "user" : "assistant";
  }

  function collectTurns() {
    const nodes = [
      ...document.querySelectorAll('[data-testid*="message"], [data-message-author-role], article')
    ];
    return nodes
      .map((node, index) => {
        const text = window.polylogueCapture.visibleText(node);
        return text ? { role: roleFromNode(node, index), text, provider_meta: { selector_index: index } } : null;
      })
      .filter(Boolean);
  }

  function textFromMessage(message) {
    if (!message || typeof message !== "object") return "";
    if (typeof message.text === "string" && message.text) return message.text;
    if (typeof message.content === "string" && message.content) return message.content;
    if (Array.isArray(message.content)) {
      return message.content
        .map((part) => {
          if (typeof part === "string") return part;
          if (part && typeof part === "object" && typeof part.text === "string") return part.text;
          return "";
        })
        .filter(Boolean)
        .join("\n");
    }
    return "";
  }

  function roleFromNativeMessage(message) {
    const raw = message && (message.sender || message.role || message.author);
    if (raw === "human" || raw === "user") return "user";
    if (raw === "assistant" || raw === "claude") return "assistant";
    if (raw === "system" || raw === "tool") return raw;
    return "unknown";
  }

  function collectNativeTurns(payload) {
    const messages = payload && payload.chat_messages;
    if (!Array.isArray(messages)) return [];
    return messages
      .map((message, index) => {
        const text = textFromMessage(message);
        if (!text) return null;
        return {
          provider_turn_id: String(message.uuid || message.id || `claude-message-${index}`),
          role: roleFromNativeMessage(message),
          text,
          timestamp: message.created_at || message.updated_at || null,
          parent_turn_id: message.parent_message_uuid || message.parent_uuid || null,
          provider_meta: {
            model: message.model || null,
            sender: message.sender || message.role || null,
            capture_source: "claude_chat_conversations_api"
          }
        };
      })
      .filter(Boolean);
  }

  function parseNativeCapture(capture) {
    if (!capture || !capture.ok || typeof capture.body !== "string") return null;
    const currentConversationId = conversationIdFromUrl();
    if (!currentConversationId || !String(capture.url || "").includes(`/chat_conversations/${currentConversationId}`)) {
      return null;
    }
    try {
      const payload = JSON.parse(capture.body);
      if (!payload || typeof payload !== "object" || !Array.isArray(payload.chat_messages)) return null;
      if (payload.uuid && String(payload.uuid) !== currentConversationId) return null;
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
    const requestId = `polylogue-claude-native-fetch-${Date.now()}-${Math.random().toString(36).slice(2)}`;
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
      error: pageResult?.error || pageCapture?.error || null
    });
    return pagePayload;
  }

  function modelFromNativePayload(payload) {
    const messages = payload && payload.chat_messages;
    if (!Array.isArray(messages)) return null;
    for (const message of messages) {
      if (typeof message.model === "string" && message.model) return message.model;
    }
    return null;
  }

  function buildNativeEnvelope(payload) {
    const turns = collectNativeTurns(payload);
    if (!turns.length) return null;
    return window.polylogueCapture.buildEnvelope({
      provider: "claude-ai",
      adapterName: nativeAdapterName,
      turns,
      providerSessionId: String(payload.uuid || conversationIdFromUrl()),
      title: typeof payload.name === "string" && payload.name ? payload.name : null,
      createdAt: payload.created_at || null,
      updatedAt: payload.updated_at || null,
      model: modelFromNativePayload(payload),
      providerMeta: {
        capture_source: "claude_chat_conversations_api",
        message_count: Array.isArray(payload.chat_messages) ? payload.chat_messages.length : 0
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
        provider: "claude-ai",
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
      "claude-ai",
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
