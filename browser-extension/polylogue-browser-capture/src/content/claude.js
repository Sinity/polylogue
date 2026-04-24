(function () {
  const adapterName = "claude-ai-dom-v1";

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

  async function capture() {
    const turns = collectTurns();
    if (!turns.length) return { ok: false, error: "no_turns" };
    const envelope = window.polylogueCapture.buildEnvelope({
      provider: "claude-ai",
      adapterName,
      turns
    });
    const captureResult = await window.polylogueCapture.sendCapture(envelope);
    const archiveState = await window.polylogueCapture.refreshArchiveState(
      "claude-ai",
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
