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

  // A turn without a provider-native id is named by a SHA-256 digest of every
  // semantic field it carries plus an occurrence counter, never its position:
  // a DOM that inserts an earlier turn must not move an existing durable
  // message id onto other content, and tool-only turns with the same role and
  // empty text still differ by timestamp, blocks or attachments.
  function semanticTurnIdFactory(sessionId) {
    const occurrences = new Map();
    return (turn) => {
      const digest = sha256Hex(JSON.stringify([
        turn.role ?? null,
        turn.text ?? null,
        turn.timestamp ?? null,
        Array.isArray(turn.blocks) ? turn.blocks : [],
        Array.isArray(turn.attachments) ? turn.attachments : [],
      ]));
      const occurrence = occurrences.get(digest) || 0;
      occurrences.set(digest, occurrence + 1);
      return `${sessionId}:turn:${digest}:${occurrence}`;
    };
  }

  let sha256Parameters = null;

  function sha256Hex(text) {
    const bytes = [];
    for (const character of String(text)) {
      const point = character.codePointAt(0);
      if (point < 0x80) bytes.push(point);
      else if (point < 0x800) bytes.push(0xc0 | (point >>> 6), 0x80 | (point & 0x3f));
      else if (point < 0x10000) bytes.push(0xe0 | (point >>> 12), 0x80 | ((point >>> 6) & 0x3f), 0x80 | (point & 0x3f));
      else bytes.push(0xf0 | (point >>> 18), 0x80 | ((point >>> 12) & 0x3f), 0x80 | ((point >>> 6) & 0x3f), 0x80 | (point & 0x3f));
    }
    const bitLength = bytes.length * 8;
    bytes.push(0x80);
    while (bytes.length % 64 !== 56) bytes.push(0);
    const high = Math.floor(bitLength / 0x100000000);
    const low = bitLength >>> 0;
    for (const value of [high, low]) bytes.push((value >>> 24) & 255, (value >>> 16) & 255, (value >>> 8) & 255, value & 255);
    if (sha256Parameters === null) {
      const primes = [];
      for (let candidate = 2; primes.length < 64; candidate += 1) {
        if (primes.every((prime) => candidate % prime !== 0)) primes.push(candidate);
      }
      sha256Parameters = {
        constants: primes.map((prime) => Math.floor((Math.cbrt(prime) % 1) * 0x100000000) >>> 0),
        initialState: primes.slice(0, 8).map((prime) => Math.floor((Math.sqrt(prime) % 1) * 0x100000000) >>> 0),
      };
    }
    const constants = sha256Parameters.constants;
    const state = sha256Parameters.initialState.slice();
    const words = new Uint32Array(64);
    const rotate = (value, count) => (value >>> count) | (value << (32 - count));
    for (let offset = 0; offset < bytes.length; offset += 64) {
      for (let i = 0; i < 16; i += 1) {
        const at = offset + i * 4;
        words[i] = ((bytes[at] << 24) | (bytes[at + 1] << 16) | (bytes[at + 2] << 8) | bytes[at + 3]) >>> 0;
      }
      for (let i = 16; i < 64; i += 1) {
        const x = words[i - 15], y = words[i - 2];
        const s0 = rotate(x, 7) ^ rotate(x, 18) ^ (x >>> 3);
        const s1 = rotate(y, 17) ^ rotate(y, 19) ^ (y >>> 10);
        words[i] = (words[i - 16] + s0 + words[i - 7] + s1) >>> 0;
      }
      let [a, b, c, d, e, f, g, h] = state;
      for (let i = 0; i < 64; i += 1) {
        const sum1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
        const choice = (e & f) ^ (~e & g);
        const t1 = (h + sum1 + choice + constants[i] + words[i]) >>> 0;
        const sum0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
        const majority = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (sum0 + majority) >>> 0;
        [h, g, f, e, d, c, b, a] = [g, f, e, (d + t1) >>> 0, c, b, a, (t1 + t2) >>> 0];
      }
      [a, b, c, d, e, f, g, h].forEach((value, i) => { state[i] = (state[i] + value) >>> 0; });
    }
    return state.map((value) => value.toString(16).padStart(8, "0")).join("");
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
    if (provider === "gemini") {
      const marker = parts.indexOf("app");
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
      const conversation = parsed.searchParams.get("conversation") || parsed.searchParams.get("id");
      return conversation || null;
    }
    if (provider === "claude-ai" && parts[0] === "chat" && parts[1]) {
      return parts[1];
    }
    if (provider === "claude-ai") return null;
    if (provider === "grok") {
      // grok.com's own conversation URLs are /c/<uuid> (verified live,
      // 2026-07-31), matching the same convention ChatGPT/Claude use. The
      // /chat/ and /grok/ segment guesses below predate that verification
      // and are kept only as a defensive fallback for URL shapes this has
      // not been checked against.
      const marker = parts.indexOf("c");
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
      const grokPathId = parts.find((part, index) => parts[index - 1] === "chat" || parts[index - 1] === "grok");
      if (grokPathId) return grokPathId;
      const queryId = parsed.searchParams.get("conversation") || parsed.searchParams.get("conversationId");
      if (queryId) return queryId;
      return null;
    }
    const sessionToken = parts.at(-1) || parsed.pathname || parsed.hostname;
    return `${provider}:${sessionToken}:${fnv1a(parsed.origin + parsed.pathname)}`;
  }

  function visibleText(node) {
    return (node?.innerText || node?.textContent || "").replace(/\s+\n/g, "\n").trim();
  }

  const ARCHIVE_ORIGINS = {
    chatgpt: "chatgpt-export",
    "claude-ai": "claude-ai-export",
    claude: "claude-ai-export",
  };

  function identityObservation({ provider, conversationId, messageId, parentMessageId = null, text = null,
    ordinal = null, adapterName, adapterVersion = null, fidelity = "unknown", degradedReason = null }) {
    const origin = ARCHIVE_ORIGINS[provider] || "unknown-export";
    const observation = {
      origin,
      provider_conversation_id: conversationId || null,
      provider_message_id: messageId || null,
      parent_provider_message_id: parentMessageId,
      content_fingerprint: text ? `sha256:${sha256Hex(String(text).replace(/\s+/g, " ").trim())}` : null,
      dom_ordinal: Number.isInteger(ordinal) ? ordinal : null,
      adapter_name: adapterName || "",
      adapter_version: adapterVersion,
      // Provider adapters may supply a provider timestamp when available.
      // Do not stamp wall-clock time here: identical native snapshots must
      // produce identical envelopes and deduplication hashes.
      observed_at: null,
      fidelity,
      degraded_reason: degradedReason,
    };
    return observation;
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
    attachments = [],
    observationRef = null
  }) {
    const sourceUrl = window.location.href;
    const urlSessionId = providerSessionId || sessionIdFromUrl(provider, sourceUrl);
    const stableProviderSessionId =
      urlSessionId === "__polylogue_temporary_chat__"
      ? temporarySessionId()
        : urlSessionId;
    const fallbackTurnId = semanticTurnIdFactory(stableProviderSessionId);
    if (!stableProviderSessionId) {
      throw new Error(`cannot capture ${provider} page without a provider-native conversation id`);
    }
    const stableCaptureId = stableProviderSessionId.startsWith(`${provider}:`)
      ? stableProviderSessionId
      : `${provider}:${stableProviderSessionId}`;
    const sessionProviderMeta = {
      capture_fidelity: "dom_degraded",
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
    const sessionTitle = title || document.title || stableProviderSessionId;
    const sessionTitleSource = title ? "provider" : document.title ? "page" : "session-id";
    const envelope = {
      ...(observationRef ? { capture_observation_ref: observationRef } : {}),
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
        title: sessionTitle,
        title_source: sessionTitleSource,
        created_at: createdAt,
        updated_at: updatedAt || now,
        model,
        provider_meta: sessionProviderMeta,
        turns: turns.map((turn, ordinal) => ({
          provider_turn_id: turn.provider_turn_id || fallbackTurnId(turn),
          role: turn.role,
          text: turn.text || null,
          timestamp: turn.timestamp || null,
          ordinal,
          parent_turn_id: turn.parent_turn_id || null,
          attachments: Array.isArray(turn.attachments) ? turn.attachments : [],
          // Structured content (tool_use/tool_result/thinking/...) an
          // adapter observed for this turn. Previously dropped here even
          // when a caller (e.g. chatgpt.js's collectNativeTurns) populated
          // it, silently flattening every native capture's tool call/result
          // structure back down to prose (polylogue-ah21 regressed).
          blocks: Array.isArray(turn.blocks) ? turn.blocks : [],
          provider_meta: turn.provider_meta || {},
          ...(turn.identity_observation ? { identity_observation: turn.identity_observation } : {}),
        })),
        attachments: Array.isArray(attachments) ? attachments : []
      }
    };
    return envelope;
  }

  async function sendCapture(envelope, reason = null, signal = null) {
    signal?.throwIfAborted();
    const requestId = crypto.randomUUID();
    const message = { type: "polylogue.capture", request_id: requestId, envelope };
    if (reason) message.reason = reason;
    let cancellation = null;
    let response;
    const abort = () => { cancellation = chrome.runtime.sendMessage({ type: "polylogue.cancelCaptureDelivery", request_id: requestId }).then(
      (result) => ({ result }), (error) => ({ error }),
    ); };
    signal?.addEventListener("abort", abort, { once: true });
    let failure;
    try {
      response = await chrome.runtime.sendMessage(message);
    } catch (error) {
      failure = error;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
    if (cancellation) {
      const settled = await cancellation;
      if (settled.error && !response?.ok) throw settled.error;
    }
    if (failure) throw failure;
    return response;
  }

  async function refreshArchiveState(provider, providerSessionId) {
    try {
      return await chrome.runtime.sendMessage({
        type: "polylogue.archiveState",
        provider,
        provider_session_id: providerSessionId
      });
    } catch (error) {
      // Every caller refreshes after an accepted capture. The read failure is
      // explicit but cannot turn that delivery's ACK into a capture failure.
      return { ok: false, error: "archive_state_refresh_failed", detail: String(error.message || error) };
    }
  }

  function cacheCaptureIsNewer(current, incoming) {
    if (!current || current.nativeId !== incoming.nativeId) return true;
    const stamp = (value) => typeof value === "number" ? value * (value < 10_000_000_000 ? 1000 : 1) : Date.parse(value || "");
    const prior = stamp(current.providerUpdatedAt); const next = stamp(incoming.providerUpdatedAt);
    if (Number.isFinite(prior) && Number.isFinite(next) && prior !== next) return next > prior;
    if (Number.isSafeInteger(current.acquisitionSequence) && Number.isSafeInteger(incoming.acquisitionSequence)) {
      return incoming.acquisitionSequence >= current.acquisitionSequence;
    }
    const priorObserved = Date.parse(current.capturedAt || ""); const nextObserved = Date.parse(incoming.capturedAt || "");
    if (Number.isFinite(priorObserved) && Number.isFinite(nextObserved) && priorObserved !== nextObserved) return nextObserved > priorObserved;
    return current.bodyRef?.id === incoming.bodyRef?.id;
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
    visibleText,
    identityObservation,
    cacheCaptureIsNewer,
  };
})();
