(function () {
  if (window.__polylogueGeminiCaptureInstalled) return;
  window.__polylogueGeminiCaptureInstalled = true;

  // Gemini deliberately has no private API dependency here. These custom
  // elements are the provider's stable conversation boundary; keeping the
  // fixture contract at that boundary means a DOM redesign fails in CI
  // instead of silently producing an empty capture.
  const TURN_SELECTORS = ["user-query", "model-response"];

  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    return parsed.searchParams.get("conversation") || parsed.searchParams.get("id") ||
      parsed.pathname.match(/\/app\/([A-Za-z0-9_-]+)/)?.[1] || null;
  }

  function roleForElement(element) {
    return element.localName === "user-query" ? "user" : "assistant";
  }

  function textForElement(element) {
    const content = element.querySelector("message-content, .message-content, [data-message-content]") || element;
    return (content.innerText || content.textContent || "").replace(/\s+/g, " ").trim();
  }

  function collectTurns() {
    const elements = collectTurnElements();
    return elements.map((element, ordinal) => {
      const text = textForElement(element);
      return {
        provider_turn_id: element.getAttribute("data-message-id") || `gemini-dom-${ordinal}`,
        role: roleForElement(element),
        text,
        timestamp: null,
        provider_meta: { capture_source: "gemini_dom", tag_name: element.localName },
      };
    }).filter((turn) => turn.text);
  }

  function collectTurnElements() {
    return TURN_SELECTORS.flatMap((selector) => [...document.querySelectorAll(selector)])
      .sort((left, right) => (left.compareDocumentPosition(right) & 4) ? -1 : 1);
  }

  async function capture(reason = null) {
    const providerSessionId = conversationIdFromUrl();
    if (!providerSessionId) return { ok: false, error: "cannot_capture_gemini_without_conversation_id" };
    const visibleTurnCount = collectTurnElements().length;
    const turns = collectTurns();
    if (!turns.length) return { ok: false, error: "no_turns" };
    if (visibleTurnCount > turns.length) {
      await chrome.runtime.sendMessage({
        type: "polylogue.captureHealth",
        event: "capture_gap",
        provider: "gemini",
        provider_session_id: providerSessionId,
        visible_count: visibleTurnCount,
        captured_count: turns.length,
        reason: "visible_turn_not_captureable",
      });
    }
    const envelope = window.polylogueCapture.buildEnvelope({
      provider: "gemini",
      adapterName: "gemini-dom-v1",
      providerSessionId,
      title: document.title || providerSessionId,
      turns,
      providerMeta: { visible_turn_count: visibleTurnCount },
    });
    const result = await window.polylogueCapture.sendCapture(envelope, reason);
    if (!result?.ok) {
      await chrome.runtime.sendMessage({
        type: "polylogue.captureHealth",
        event: "capture_error",
        provider: "gemini",
        provider_session_id: providerSessionId,
        reason: result?.error || "capture_rejected",
      });
      return { ok: false, envelope, captureResult: result, error: result?.error || "capture_rejected" };
    }
    const archiveState = await window.polylogueCapture.refreshArchiveState("gemini", providerSessionId);
    return { ok: true, envelope, captureResult: result, archiveState, visibleTurnCount };
  }

  window.polylogueCapture.capturePage = capture;
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.capturePage") return false;
    capture(message.reason || null).then(sendResponse).catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
})();
