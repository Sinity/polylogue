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

  function sessionIdFromUrl(provider, url) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (provider === "chatgpt") {
      const marker = parts.indexOf("c");
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
    }
    if (provider === "claude-ai" && parts[0] === "chat" && parts[1]) {
      return parts[1];
    }
    const sessionToken = parts.at(-1) || parsed.pathname || parsed.hostname;
    return `${provider}:${sessionToken}:${fnv1a(parsed.origin + parsed.pathname)}`;
  }

  function visibleText(node) {
    return (node?.innerText || node?.textContent || "").replace(/\s+\n/g, "\n").trim();
  }

  function buildEnvelope({
    provider,
    adapterName,
    turns,
    model = null,
    providerSessionId = null,
    title = null,
    createdAt = null,
    updatedAt = null,
    providerMeta = {},
    rawProviderPayload = null
  }) {
    const sourceUrl = window.location.href;
    const stableProviderSessionId = providerSessionId || sessionIdFromUrl(provider, sourceUrl);
    const stableCaptureId = stableProviderSessionId.startsWith(`${provider}:`)
      ? stableProviderSessionId
      : `${provider}:${stableProviderSessionId}`;
    const now = new Date().toISOString();
    const envelope = {
      polylogue_capture_kind: CAPTURE_KIND,
      schema_version: SCHEMA_VERSION,
      capture_id: stableCaptureId,
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
        provider_session_id: stableProviderSessionId,
        title: title || document.title || stableProviderSessionId,
        created_at: createdAt,
        updated_at: updatedAt || now,
        model,
        provider_meta: providerMeta,
        turns: turns.map((turn, ordinal) => ({
          provider_turn_id: turn.provider_turn_id || `${stableProviderSessionId}:turn:${ordinal}:${fnv1a(turn.role + ":" + turn.text)}`,
          role: turn.role,
          text: turn.text,
          timestamp: turn.timestamp || null,
          ordinal,
          parent_turn_id: turn.parent_turn_id || null,
          provider_meta: turn.provider_meta || {}
        }))
      }
    };
    if (rawProviderPayload && typeof rawProviderPayload === "object") {
      envelope.raw_provider_payload = rawProviderPayload;
    }
    return envelope;
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
    sessionIdFromUrl,
    capturePage: null,
    fnv1a,
    refreshArchiveState,
    sendCapture,
    visibleText
  };
})();
