(function () {
  const domAdapterName = "chatgpt-dom-v1";
  const nativeAdapterName = "chatgpt-native-v1";
  const nativeCaptureMessage = "polylogue.chatgpt.nativeCapture";
  const nativeCaptures = [];

  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    return parts[0] === "c" && parts[1] ? parts[1] : null;
  }

  function injectNativeCaptureBridge() {
    const root = document.documentElement || document.head || document.body;
    if (!root) {
      window.setTimeout(injectNativeCaptureBridge, 0);
      return;
    }
    const script = document.createElement("script");
    script.textContent = `(() => {
      const messageType = ${JSON.stringify(nativeCaptureMessage)};
      const currentOrigin = window.location.origin;
      window.__polylogueCapturedFetches = Array.isArray(window.__polylogueCapturedFetches)
        ? window.__polylogueCapturedFetches
        : [];
      function post(capture) {
        window.postMessage({ type: messageType, capture }, currentOrigin);
      }
      function remember(capture) {
        window.__polylogueCapturedFetches.push(capture);
        if (window.__polylogueCapturedFetches.length > 8) {
          window.__polylogueCapturedFetches.splice(0, window.__polylogueCapturedFetches.length - 8);
        }
        post(capture);
      }
      const existingCaptures = window.__polylogueCapturedFetches.slice(-8);
      window.__polylogueCapturedFetches = existingCaptures;
      for (const capture of existingCaptures) post(capture);
      if (window.__polylogueFetchHookInstalled) return;
      window.__polylogueFetchHookInstalled = true;
      const originalFetch = window.fetch;
      window.fetch = async function polylogueFetch(input) {
        const response = await originalFetch.apply(this, arguments);
        try {
          const url = typeof input === "string" ? input : input && input.url;
          const absolute = new URL(url, window.location.href);
          const isConversation = absolute.origin === currentOrigin
            && /\\/backend-api\\/conversation\\/[^/?#]+/.test(absolute.pathname)
            && !absolute.pathname.endsWith("/init");
          const contentType = response.headers.get("content-type") || "";
          if (isConversation && contentType.includes("application/json")) {
            const body = await response.clone().text();
            remember({
              url: absolute.href,
              status: response.status,
              ok: response.ok,
              contentType,
              body,
              capturedAt: new Date().toISOString()
            });
          }
        } catch (_error) {
          // Capture must never perturb the ChatGPT page's own request path.
        }
        return response;
      };
    })();`;
    root.appendChild(script);
    script.remove();
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== nativeCaptureMessage || !data.capture) return;
    nativeCaptures.push(data.capture);
    if (nativeCaptures.length > 8) nativeCaptures.splice(0, nativeCaptures.length - 8);
  });

  injectNativeCaptureBridge();

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
    const nativePayload = latestNativePayload();
    const envelope = nativePayload ? buildNativeEnvelope(nativePayload) : null;
    const fallbackEnvelope = () => {
      const turns = collectTurns();
      if (!turns.length) return null;
      return window.polylogueCapture.buildEnvelope({
        provider: "chatgpt",
        adapterName: domAdapterName,
        turns
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
