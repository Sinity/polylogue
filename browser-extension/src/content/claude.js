(function () {
  if (window.__polylogueClaudeCaptureInstalled) return;
  window.__polylogueClaudeCaptureInstalled = true;

  // In-page Layer 1 (polylogue-ys30): capture-status dot + save action mounted
  // next to each detected message. Reused across every capture trigger below
  // (badge click, popup, background auto-capture) so the dots always reflect
  // the most recent capture outcome for the whole session.
  const MESSAGE_CONTAINER_SELECTOR = '[data-testid*="message"], [data-message-author-role], article';
  let messageLayer = null;

  const nativeAdapterName = "claude-ai-native-v1";
  const nativeCaptureMessage = "polylogue.claude.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.claude.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.claude.nativeFetchResponse";
  const nativeFetchTimeoutMs = 8000;
  const nativeCaptures = [];
  const nativeFetchResponses = new Map();
  const nativeAttemptDiagnostics = [];

  function rememberNativeAttempt(diagnostic) {
    nativeAttemptDiagnostics.push({
      attempted_at: new Date().toISOString(),
      ...diagnostic
    });
    if (nativeAttemptDiagnostics.length > 6) {
      nativeAttemptDiagnostics.splice(0, nativeAttemptDiagnostics.length - 6);
    }
  }

  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    return parts[0] === "chat" && parts[1] ? parts[1] : null;
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== nativeCaptureMessage || !data.capture) return;
    nativeCaptures.push(data.capture);
    if (nativeCaptures.length > 8) nativeCaptures.splice(0, nativeCaptures.length - 8);
  });

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== nativeFetchResponseMessage || !data.requestId) return;
    const pending = nativeFetchResponses.get(data.requestId);
    if (!pending) return;
    nativeFetchResponses.delete(data.requestId);
    pending.resolve({ capture: data.capture || null, error: data.error || null });
  });

  function textFromMessage(message) {
    if (!message || typeof message !== "object") return "";
    if (typeof message.text === "string" && message.text) return message.text;
    if (typeof message.content === "string" && message.content) return message.content;
    if (Array.isArray(message.content)) {
      return message.content
        .map((part) => {
          if (typeof part === "string") return part;
          if (part && typeof part === "object" && typeof part.text === "string") return part.text;
          return "";
        })
        .filter(Boolean)
        .join("\n");
    }
    return "";
  }

  function roleFromNativeMessage(message) {
    const raw = message && (message.sender || message.role || message.author);
    if (raw === "human" || raw === "user") return "user";
    if (raw === "assistant" || raw === "claude") return "assistant";
    if (raw === "system" || raw === "tool") return raw;
    return "unknown";
  }

  // Claude's chat_conversations API returns the same segment shape as the
  // GDPR export, so the structure the archive-side parser understands is
  // already in hand here -- flattening it to prose would discard tool_use /
  // tool_result / thinking that the export path parses in full.
  // BlockType names and the is_error contract mirror BrowserCaptureBlock
  // (polylogue/browser_capture/models.py): outcome fields are read from the
  // provider's own segment, never inferred from rendered text.
  // tool_result.content per Anthropic's API is either a plain string or an
  // array of content blocks (type: "text" | "image" | ...). Join the text of
  // "text"-typed blocks, newline-separated; return null only when no text
  // content is present at all (e.g. an image-only result).
  function textFromToolResultContent(content) {
    if (typeof content === "string") return content || null;
    if (Array.isArray(content)) {
      const texts = content
        .filter((part) => part && typeof part === "object" && part.type === "text" && typeof part.text === "string")
        .map((part) => part.text);
      return texts.length ? texts.join("\n") : null;
    }
    return null;
  }

  function nativeTurnBlocks(message) {
    const segments = Array.isArray(message && message.content) ? message.content : [];
    const blocks = [];
    for (const segment of segments) {
      if (!segment || typeof segment !== "object") continue;
      const type = segment.type;
      if (type === "text" && typeof segment.text === "string" && segment.text) {
        blocks.push({ type: "text", text: segment.text });
      } else if (type === "thinking") {
        const thinking = typeof segment.thinking === "string" ? segment.thinking : segment.text;
        if (thinking) blocks.push({ type: "thinking", text: thinking, metadata: { content_type: type } });
      } else if (type === "tool_use") {
        blocks.push({
          type: "tool_use",
          tool_name: segment.name || null,
          tool_id: segment.id || null,
          tool_input: segment.input && typeof segment.input === "object" && !Array.isArray(segment.input)
            ? segment.input
            : null
        });
      } else if (type === "tool_result") {
        blocks.push({
          type: "tool_result",
          tool_id: segment.tool_use_id || null,
          // Anthropic's tool_result content can be a plain string OR an array
          // of content blocks (text/image/...) -- join the text blocks'
          // `text` fields when it's an array. Never invent text for
          // non-text blocks (e.g. images); text stays null only when no
          // text content exists at all.
          text: textFromToolResultContent(segment.content),
          // Provider-reported outcome. Absent stays null (unknown), never false.
          is_error: typeof segment.is_error === "boolean" ? segment.is_error : null,
          metadata: { tool_name: segment.name || null }
        });
      }
    }
    return blocks;
  }

  // Two distinct attachment channels, both present in the conversations API:
  //   attachments[] carries extracted_content inline (text already extracted
  //     by the provider) but NO id -- synthesise a stable one.
  //   files[]       carries a real file_uuid but no bytes; recorded as a
  //     reference so a later byte acquisition can join on the uuid.
  function nativeTurnAttachments(message, index) {
    const out = [];
    const messageId = String(message.uuid || message.id || `claude-message-${index}`);
    const messageAttachments = Array.isArray(message.attachments) ? message.attachments : [];
    for (const [attachmentPosition, attachment] of messageAttachments.entries()) {
      if (!attachment || typeof attachment !== "object") continue;
      const name = attachment.file_name || attachment.name || null;
      if (!name) continue;
      const size = Number.parseInt(attachment.file_size, 10);
      // Include the loop index in the hash input: two attachments with the
      // same name and size in one message would otherwise collide on the
      // same synthesised id (name/size alone are not unique within a message).
      out.push({
        provider_attachment_id: `claude-attachment:${window.polylogueCapture.fnv1a(`${messageId}:${attachmentPosition}:${name}:${attachment.file_size || ""}`)}`,
        message_provider_id: messageId,
        name,
        mime_type: attachment.file_type || null,
        size_bytes: Number.isFinite(size) ? size : null,
        extracted_content: typeof attachment.extracted_content === "string" ? attachment.extracted_content : null,
        provider_meta: { capture_source: "claude_chat_conversations_api", channel: "attachments" }
      });
    }
    for (const file of Array.isArray(message.files) ? message.files : []) {
      if (!file || typeof file !== "object") continue;
      const name = file.file_name || file.name || null;
      const uuid = file.file_uuid || file.uuid || null;
      if (!name && !uuid) continue;
      out.push({
        provider_attachment_id: uuid ? `claude-file:${uuid}` : `claude-file:${window.polylogueCapture.fnv1a(`${messageId}:${name}`)}`,
        message_provider_id: messageId,
        name,
        provider_meta: {
          capture_source: "claude_chat_conversations_api",
          channel: "files",
          file_uuid: uuid || null
        }
      });
    }
    return out;
  }

  function collectNativeTurns(payload) {
    const messages = payload && payload.chat_messages;
    if (!Array.isArray(messages)) return [];
    return messages
      .map((message, index) => {
        const text = textFromMessage(message);
        const blocks = nativeTurnBlocks(message);
        const attachments = nativeTurnAttachments(message, index);
        // A turn with attachments or structured blocks but no prose is still a
        // real turn -- requiring text would drop image-only and tool-only turns.
        if (!text && !blocks.length && !attachments.length) return null;
        return {
          provider_turn_id: String(message.uuid || message.id || `claude-message-${index}`),
          role: roleFromNativeMessage(message),
          text,
          timestamp: message.created_at || message.updated_at || null,
          parent_turn_id: message.parent_message_uuid || message.parent_uuid || null,
          blocks,
          attachments,
          provider_meta: {
            model: message.model || null,
            sender: message.sender || message.role || null,
            capture_source: "claude_chat_conversations_api"
          }
        };
      })
      .filter(Boolean);
  }

  function parseNativeCapture(capture) {
    if (!capture || !capture.ok || typeof capture.body !== "string") return null;
    const currentConversationId = conversationIdFromUrl();
    if (!currentConversationId || !String(capture.url || "").includes(`/chat_conversations/${currentConversationId}`)) {
      return null;
    }
    try {
      const payload = JSON.parse(capture.body);
      if (!payload || typeof payload !== "object" || !Array.isArray(payload.chat_messages)) return null;
      if (payload.uuid && String(payload.uuid) !== currentConversationId) return null;
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

  async function requestNativeCaptureFromPage(conversationId) {
    const requestId = `polylogue-claude-native-fetch-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        nativeFetchResponses.delete(requestId);
        resolve({ capture: null, error: "timeout" });
      }, nativeFetchTimeoutMs);
      nativeFetchResponses.set(requestId, {
        resolve(value) {
          window.clearTimeout(timeout);
          resolve(value);
        }
      });
    });
    window.postMessage(
      {
        type: nativeFetchRequestMessage,
        requestId,
        conversationId
      },
      window.location.origin
    );
    return responsePromise;
  }

  async function fetchNativePayloadOnDemand() {
    const conversationId = conversationIdFromUrl();
    if (!conversationId) return null;
    const pageResult = await requestNativeCaptureFromPage(conversationId);
      const pageCapture = pageResult && pageResult.capture;
      const pagePayload = parseNativeCapture(pageCapture);
    rememberNativeAttempt({
      stage: "page_bridge_fetch",
      ok: pageCapture?.ok ?? null,
      status: pageCapture?.status ?? null,
      content_type: pageCapture?.contentType || null,
      body_bytes: typeof pageCapture?.body === "string" ? pageCapture.body.length : 0,
      accepted: Boolean(pagePayload),
      error: pageResult?.error || pageCapture?.error || null
    });
    return pagePayload;
  }

  function modelFromNativePayload(payload) {
    const messages = payload && payload.chat_messages;
    if (!Array.isArray(messages)) return null;
    for (const message of messages) {
      if (typeof message.model === "string" && message.model) return message.model;
    }
    return null;
  }

  function buildNativeEnvelope(payload) {
    const turns = collectNativeTurns(payload);
    if (!turns.length) return null;
    return window.polylogueCapture.buildEnvelope({
      provider: "claude-ai",
      adapterName: nativeAdapterName,
      turns,
      providerSessionId: String(payload.uuid || conversationIdFromUrl()),
      sessionKind: payload.is_temporary === true ? "temporary" : null,
      title: typeof payload.name === "string" && payload.name ? payload.name : null,
      createdAt: payload.created_at || null,
      updatedAt: payload.updated_at || null,
      model: modelFromNativePayload(payload),
      providerMeta: {
        capture_source: "claude_chat_conversations_api",
        message_count: Array.isArray(payload.chat_messages) ? payload.chat_messages.length : 0,
        is_temporary: payload.is_temporary === true,
        session_kind: payload.is_temporary === true ? "temporary" : null
      },
      rawProviderPayload: payload
    });
  }

  async function capture(reason = null) {
    const nativePayload = latestNativePayload() || (await fetchNativePayloadOnDemand());
    const finalEnvelope = nativePayload ? buildNativeEnvelope(nativePayload) : null;
    if (!finalEnvelope) {
      return { ok: false, error: "native_capture_unavailable", native_attempts: nativeAttemptDiagnostics.slice(-6) };
    }
    const captureResult = await window.polylogueCapture.sendCapture(finalEnvelope, reason);
    if (!captureResult?.ok) {
      messageLayer?.reportOutcome({ ok: false, turnCount: finalEnvelope.session.turns.length });
      return {
        ok: false,
        envelope: finalEnvelope,
        captureResult,
        error: captureResult?.error || "capture_rejected",
        timelineRecorded: true,
      };
    }
    const archiveState = await window.polylogueCapture.refreshArchiveState(
      "claude-ai",
      finalEnvelope.session.provider_session_id
    );
    messageLayer?.reportOutcome({ ok: true, turnCount: finalEnvelope.session.turns.length });
    return { ok: true, envelope: finalEnvelope, captureResult, archiveState };
  }

  window.polylogueCapture.capturePage = capture;
  // Test-only exposure of the pure structured-block/attachment extractors so
  // tests exercise the real implementation instead of a hand-copied one that
  // can silently drift (polylogue-ah21's dropped-`blocks` regression was
  // caused by exactly that drift). Not used by any runtime capture path.
  window.polylogueCapture.__claudeNativeInternals = { nativeTurnBlocks, nativeTurnAttachments };
  if (window.polylogueMessageLayer) {
    messageLayer = window.polylogueMessageLayer.mount({
      containerSelector: MESSAGE_CONTAINER_SELECTOR,
      onSave: () => {
        capture("message_layer_save").catch(() => undefined);
      },
    });
  }
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.capturePage") return false;
    capture(message.reason || null).then(sendResponse).catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
})();
