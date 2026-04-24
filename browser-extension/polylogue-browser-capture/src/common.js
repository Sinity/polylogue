(function () {
  const SCHEMA_VERSION = 1;
  const CAPTURE_KIND = "browser_llm_session";

  function fnv1a(text) {
    let hash = 0x811c9dc5;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16).padStart(8, "0");
  }

  function conversationIdFromUrl(provider, url) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    const conversationToken = parts.at(-1) || parsed.pathname || parsed.hostname;
    return `${provider}:${conversationToken}:${fnv1a(parsed.origin + parsed.pathname)}`;
  }

  function visibleText(node) {
    return (node?.innerText || node?.textContent || "").replace(/\s+\n/g, "\n").trim();
  }

  function buildEnvelope({ provider, adapterName, turns, model = null }) {
    const sourceUrl = window.location.href;
    const providerSessionId = conversationIdFromUrl(provider, sourceUrl);
    const now = new Date().toISOString();
    return {
      polylogue_capture_kind: CAPTURE_KIND,
      schema_version: SCHEMA_VERSION,
      capture_id: `${provider}:${providerSessionId}`,
      source: "browser-extension",
      provenance: {
        source_url: sourceUrl,
        page_title: document.title || null,
        captured_at: now,
        extension_id: chrome.runtime.id,
        adapter_name: adapterName,
        adapter_version: chrome.runtime.getManifest().version,
        capture_mode: "snapshot"
      },
      session: {
        provider,
        provider_session_id: providerSessionId,
        title: document.title || providerSessionId,
        updated_at: now,
        model,
        turns: turns.map((turn, ordinal) => ({
          provider_turn_id: turn.provider_turn_id || `${providerSessionId}:turn:${ordinal}:${fnv1a(turn.role + ":" + turn.text)}`,
          role: turn.role,
          text: turn.text,
          ordinal,
          provider_meta: turn.provider_meta || {}
        }))
      }
    };
  }

  async function sendCapture(envelope) {
    const response = await chrome.runtime.sendMessage({ type: "polylogue.capture", envelope });
    return response;
  }

  async function refreshArchiveState(provider, providerSessionId) {
    return chrome.runtime.sendMessage({
      type: "polylogue.archiveState",
      provider,
      provider_session_id: providerSessionId
    });
  }

  window.polylogueCapture = {
    buildEnvelope,
    conversationIdFromUrl,
    capturePage: null,
    fnv1a,
    refreshArchiveState,
    sendCapture,
    visibleText
  };
})();
