(function () {
  const SCHEMA_VERSION = 1;
  const CAPTURE_KIND = "browser_llm_session";
  const TEMPORARY_CHAT_ID_KEY = "polylogue:chatgpt-temporary-session-id";

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
      if (parsed.searchParams.get("temporary-chat") === "true") return "__polylogue_temporary_chat__";
      return null;
    }
    if (provider === "claude-ai" && parts[0] === "chat" && parts[1]) {
      return parts[1];
    }
    if (provider === "claude-ai") return null;
    const sessionToken = parts.at(-1) || parsed.pathname || parsed.hostname;
    return `${provider}:${sessionToken}:${fnv1a(parsed.origin + parsed.pathname)}`;
  }

  function visibleText(node) {
    return (node?.innerText || node?.textContent || "").replace(/\s+\n/g, "\n").trim();
  }

  function randomHex(length) {
    const bytes = new Uint8Array(Math.ceil(length / 2));
    if (globalThis.crypto?.getRandomValues) {
      globalThis.crypto.getRandomValues(bytes);
      return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("").slice(0, length);
    }
    return `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`.slice(0, length).padEnd(length, "0");
  }

  function temporarySessionId() {
    try {
      const existing = window.sessionStorage.getItem(TEMPORARY_CHAT_ID_KEY);
      if (existing && /^temporary:[0-9a-f]{24}$/.test(existing)) return existing;
      const created = `temporary:${randomHex(24)}`;
      window.sessionStorage.setItem(TEMPORARY_CHAT_ID_KEY, created);
      return created;
    } catch {
      return `temporary:${randomHex(24)}`;
    }
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
    const urlSessionId = providerSessionId || sessionIdFromUrl(provider, sourceUrl);
    const stableProviderSessionId =
      urlSessionId === "__polylogue_temporary_chat__"
      ? temporarySessionId()
        : urlSessionId;
    if (!stableProviderSessionId) {
      throw new Error(`cannot capture ${provider} page without a provider-native conversation id`);
    }
    const stableCaptureId = stableProviderSessionId.startsWith(`${provider}:`)
      ? stableProviderSessionId
      : `${provider}:${stableProviderSessionId}`;
    const sessionProviderMeta = { ...providerMeta };
    if (urlSessionId === "__polylogue_temporary_chat__" || stableProviderSessionId.startsWith("temporary:")) {
      sessionProviderMeta.session_kind = "temporary";
    }
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
        provider_meta: sessionProviderMeta,
        turns: turns.map((turn, ordinal) => ({
          provider_turn_id: turn.provider_turn_id || `${stableProviderSessionId}:turn:${ordinal}:${fnv1a(turn.role + ":" + (turn.text || ""))}`,
          role: turn.role,
          text: turn.text || null,
          timestamp: turn.timestamp || null,
          ordinal,
          parent_turn_id: turn.parent_turn_id || null,
          attachments: Array.isArray(turn.attachments) ? turn.attachments : [],
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

  const existingCapture = window.polylogueCapture || {};
  window.polylogueCapture = {
    ...existingCapture,
    buildEnvelope,
    sessionIdFromUrl,
    capturePage: existingCapture.capturePage || null,
    fnv1a,
    refreshArchiveState,
    sendCapture,
    temporarySessionId,
    visibleText
  };
})();
