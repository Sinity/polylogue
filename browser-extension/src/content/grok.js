(function () {
  if (window.__polylogueGrokCaptureInstalled) return;
  window.__polylogueGrokCaptureInstalled = true;

  const domAdapterName = "grok-dom-v1";

  function roleFromNode(node, index) {
    const attrs = [
      node.getAttribute("data-testid"),
      node.getAttribute("data-message-author-role"),
      node.getAttribute("aria-label"),
      node.className,
    ]
      .filter(Boolean)
      .join(" ");
    if (/user|you|human/i.test(attrs)) return "user";
    if (/assistant|grok|ai/i.test(attrs)) return "assistant";
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
    const filePattern =
      /(?:^|\s)([^\s@/]+\.(?:zip|tar|tgz|gz|bz2|xz|7z|rar|md|txt|pdf|doc|docx|json|jsonl|csv|tsv|py|js|ts|tsx|jsx|rs|go|java|c|cc|cpp|h|hpp|png|jpe?g|gif|webp|svg|mp3|mp4|wav|webm))(?:\s|$)/i;
    for (const candidate of candidates) {
      const match = candidate.match(filePattern);
      if (match) return match[1].trim();
    }
    return null;
  }

  function collectAttachments(node, turnIndex) {
    const selector = [
      '[role="group"][aria-label]',
      "a[download]",
      "a[href][aria-label]",
      "img[alt]",
      "img[src]",
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
          capture_source: "grok_dom_attachment",
        },
      });
    }
    return attachments;
  }

  function collectTurns() {
    const nodes = [
      ...document.querySelectorAll(
        '[data-testid*="conversation"], [data-testid*="message"], [data-message-author-role], article, main [role="article"]',
      ),
    ];
    return nodes
      .map((node, index) => {
        const text = window.polylogueCapture.visibleText(node);
        const attachments = collectAttachments(node, index);
        return text || attachments.length
          ? {
              role: roleFromNode(node, index),
              text,
              attachments,
              provider_meta: {
                selector_index: index,
                capture_source: "grok_dom",
              },
            }
          : null;
      })
      .filter(Boolean);
  }

  async function capture(reason = null) {
    const turns = collectTurns();
    if (!turns.length) return { ok: false, error: "no_turns" };
    const envelope = window.polylogueCapture.buildEnvelope({
      provider: "grok",
      adapterName: domAdapterName,
      turns,
      providerMeta: {
        capture_source: "grok_dom",
        capture_fidelity: "dom_degraded",
        native_attempts: [],
      },
    });
    const captureResult = await window.polylogueCapture.sendCapture(envelope, reason);
    const archiveState = await window.polylogueCapture.refreshArchiveState(
      "grok",
      envelope.session.provider_session_id,
    );
    return { ok: true, envelope, captureResult, archiveState };
  }

  window.polylogueCapture.capturePage = capture;
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.capturePage") return false;
    capture(message.reason || null).then(sendResponse).catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
})();
