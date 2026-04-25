(function () {
  const adapterName = "chatgpt-dom-v1";

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

  async function capture() {
    const turns = collectTurns();
    if (!turns.length) return { ok: false, error: "no_turns" };
    const envelope = window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName,
      turns
    });
    const captureResult = await window.polylogueCapture.sendCapture(envelope);
    const archiveState = await window.polylogueCapture.refreshArchiveState(
      "chatgpt",
      envelope.session.provider_session_id
    );
    return { ok: true, envelope, captureResult, archiveState };
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
