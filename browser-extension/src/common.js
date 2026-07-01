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
    if (provider === "grok") {
      const grokPathId = parts.find((part, index) => parts[index - 1] === "chat" || parts[index - 1] === "grok");
      if (grokPathId) return grokPathId;
      const queryId = parsed.searchParams.get("conversation") || parsed.searchParams.get("conversationId");
      if (queryId) return queryId;
      return `dom:${fnv1a(parsed.origin + parsed.pathname + parsed.search)}`;
    }
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
    sessionKind = null,
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
    const sessionProviderMeta = {
      capture_fidelity: rawProviderPayload ? "native_full" : "dom_degraded",
      ...providerMeta,
    };
    if (urlSessionId === "__polylogue_temporary_chat__" || stableProviderSessionId.startsWith("temporary:")) {
      sessionProviderMeta.session_kind = "temporary";
    }
    const stableSessionKind =
      sessionKind === "temporary" ||
      sessionProviderMeta.session_kind === "temporary" ||
      stableProviderSessionId.startsWith("temporary:")
        ? "temporary"
        : "standard";
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
        session_kind: stableSessionKind,
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

  function firstMatch(selectors) {
    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (node) return node;
    }
    return null;
  }

  function setComposerText(node, text) {
    node.focus();
    // ProseMirror / Lexical / contenteditable composers register input through
    // beforeinput+input. execCommand("insertText") drives that path most
    // reliably; selecting all first replaces any existing draft.
    try {
      const selection = window.getSelection();
      if (selection) {
        const range = document.createRange();
        range.selectNodeContents(node);
        selection.removeAllRanges();
        selection.addRange(range);
      }
      const inserted = document.execCommand && document.execCommand("insertText", false, text);
      if (inserted) return true;
    } catch {
      /* fall through to manual path */
    }
    // Fallback: set content directly and dispatch an input event.
    if (typeof node.value === "string") {
      node.value = text;
    } else {
      node.textContent = text;
    }
    node.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: true, data: text, inputType: "insertText" }));
    return true;
  }

  // Deliver a post command into a provider composer. SAFETY: this fills the
  // composer and only clicks send when command.submit === true. The default is a
  // dry-run that leaves the drafted text in place without submitting, so an
  // accidental dispatch never posts to a live thread.
  async function postReplyToComposer({ command, composerSelectors, sendSelectors }) {
    const observed_url = window.location.href;
    const text = command && typeof command.text === "string" ? command.text : "";
    if (!text) return { status: "failed", detail: "empty_text", observed_url };
    const composer = firstMatch(composerSelectors);
    if (!composer) return { status: "failed", detail: "composer_not_found", observed_url };
    setComposerText(composer, text);
    if (command.submit !== true) {
      return { status: "submitted", detail: "dry_run_filled_not_sent", observed_url };
    }
    const sendButton = firstMatch(sendSelectors);
    if (!sendButton) return { status: "failed", detail: "send_button_not_found", observed_url };
    if (sendButton.disabled) return { status: "failed", detail: "send_button_disabled", observed_url };
    sendButton.click();
    return { status: "submitted", detail: "sent", observed_url };
  }

  const existingCapture = window.polylogueCapture || {};
  window.polylogueCapture = {
    ...existingCapture,
    buildEnvelope,
    sessionIdFromUrl,
    capturePage: existingCapture.capturePage || null,
    fnv1a,
    postReplyToComposer,
    refreshArchiveState,
    sendCapture,
    temporarySessionId,
    visibleText
  };
})();
