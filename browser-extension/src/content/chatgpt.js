(function () {
  if (window.__polylogueChatgptCaptureInstalled) return;
  window.__polylogueChatgptCaptureInstalled = true;

  const domAdapterName = "chatgpt-dom-v1";
  const nativeAdapterName = "chatgpt-native-v1";
  const nativeCaptureMessage = "polylogue.chatgpt.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.chatgpt.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.chatgpt.nativeFetchResponse";
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
    const marker = parts.indexOf("c");
    return marker >= 0 && parts[marker + 1] ? parts[marker + 1] : null;
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

  function roleFromNode(node, index) {
    const testId = node.getAttribute("data-testid") || "";
    if (testId.includes("user")) return "user";
    if (testId.includes("assistant")) return "assistant";
    const labelled = node.getAttribute("aria-label") || "";
    if (/you|user/i.test(labelled)) return "user";
    if (/chatgpt|assistant/i.test(labelled)) return "assistant";
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
    const extensionPattern =
      "zip|tar|tgz|gz|bz2|xz|7z|rar|md|txt|pdf|doc|docx|json|jsonl|csv|tsv|py|js|ts|tsx|jsx|rs|go|java|c|cc|cpp|h|hpp|png|jpe?g|gif|webp|svg|mp3|mp4|wav|webm";
    const filePattern = new RegExp(`(?:^|\\s)([^\\s@/]+\\.(?:${extensionPattern}))(?:\\s|$)`, "i");
    for (const candidate of candidates) {
      const fileNameMatch = candidate.match(filePattern);
      if (fileNameMatch) return fileNameMatch[1].trim();
    }
    return null;
  }

  function collectAttachments(node, turnIndex) {
    const selector = [
      '[role="group"][aria-label]',
      "a[download]",
      "a[href][aria-label]",
      "img[alt]",
      "img[src]"
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
          capture_source: "chatgpt_dom_attachment"
        }
      });
    }
    return attachments;
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
        const attachments = collectAttachments(node, index);
        return text || attachments.length
          ? { role, text, attachments, provider_meta: { selector_index: index } }
          : null;
      })
      .filter(Boolean);
  }

  function extractContentText(content) {
    const parts = content && content.parts;
    if (Array.isArray(parts)) {
      const textParts = [];
      for (const part of parts) {
        if (typeof part === "string" && part) textParts.push(part);
        if (part && typeof part === "object" && typeof part.text === "string" && part.text) {
          textParts.push(part.text);
        }
      }
      if (textParts.length) return textParts.join("\n");
    }
    if (typeof content?.text === "string" && content.text) return content.text;
    if (typeof content?.result === "string" && content.result) return content.result;
    return "";
  }

  function timestampFromSeconds(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return new Date(value * 1000).toISOString();
  }

  function timeoutError(label) {
    const error = new Error(`${label}_timeout_after_${nativeFetchTimeoutMs}ms`);
    error.name = "PolylogueTimeoutError";
    return error;
  }

  function withTimeout(promise, label) {
    let timer = 0;
    const timeout = new Promise((_resolve, reject) => {
      timer = window.setTimeout(() => reject(timeoutError(label)), nativeFetchTimeoutMs);
    });
    return Promise.race([promise, timeout]).finally(() => {
      if (timer) window.clearTimeout(timer);
    });
  }

  function roleFromRaw(raw) {
    if (raw === "user" || raw === "assistant" || raw === "system" || raw === "tool") return raw;
    if (raw === "function" || raw === "tool_use" || raw === "tool_result") return "tool";
    return "unknown";
  }

  function collectNativeTurns(payload) {
    const mapping = payload && payload.mapping;
    if (!mapping || typeof mapping !== "object") return [];
    const turns = [];
    for (const [nodeId, node] of Object.entries(mapping)) {
      const message = node && node.message;
      const content = message && message.content;
      if (!message || !content) continue;
      const text = extractContentText(content);
      if (!text) continue;
      const role = roleFromRaw(message.author && message.author.role);
      const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
      turns.push({
        provider_turn_id: String(message.id || node.id || nodeId),
        role,
        text,
        timestamp: timestampFromSeconds(message.create_time),
        parent_turn_id: node.parent ? String(node.parent) : null,
        provider_meta: {
          node_id: nodeId,
          content_type: content.content_type || "text",
          status: message.status || null,
          model_slug: metadata.model_slug || null,
          capture_source: "chatgpt_backend_api"
        }
      });
    }
    return turns;
  }

  function parseNativeCapture(capture) {
    if (!capture || !capture.ok || typeof capture.body !== "string") return null;
    const currentConversationId = conversationIdFromUrl();
    if (!currentConversationId || !String(capture.url || "").includes(`/conversation/${currentConversationId}`)) {
      return null;
    }
    try {
      const payload = JSON.parse(capture.body);
      if (!payload || typeof payload !== "object" || !payload.mapping) return null;
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
    const requestId = `polylogue-native-fetch-${Date.now()}-${Math.random().toString(36).slice(2)}`;
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

  async function fetchNativePayloadFromContentScript(conversationId) {
    try {
      const controller = new globalThis.AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(timeoutError("content_script_fetch")), nativeFetchTimeoutMs);
      let response;
      try {
        response = await fetch(`/backend-api/conversation/${encodeURIComponent(conversationId)}`, {
          credentials: "include",
          cache: "no-store",
          signal: controller.signal
        });
      } finally {
        window.clearTimeout(timeoutId);
      }
      const contentType = response.headers.get("content-type") || "";
      if (!response.ok || !contentType.includes("application/json")) {
        rememberNativeAttempt({
          stage: "content_script_fetch",
          ok: response.ok,
          status: response.status,
          content_type: contentType,
          accepted: false
        });
        return null;
      }
      const payload = await withTimeout(response.clone().json(), "content_script_json");
      if (!payload || typeof payload !== "object" || !payload.mapping) {
        rememberNativeAttempt({
          stage: "content_script_fetch",
          ok: response.ok,
          status: response.status,
          content_type: contentType,
          accepted: false,
          reason: "missing_mapping"
        });
        return null;
      }
      const payloadConversationId = payload.conversation_id || payload.id;
      if (payloadConversationId && String(payloadConversationId) !== conversationId) {
        rememberNativeAttempt({
          stage: "content_script_fetch",
          ok: response.ok,
          status: response.status,
          content_type: contentType,
          accepted: false,
          reason: "conversation_id_mismatch"
        });
        return null;
      }
      rememberNativeAttempt({
        stage: "content_script_fetch",
        ok: response.ok,
        status: response.status,
        content_type: contentType,
        accepted: true
      });
      return payload;
    } catch (error) {
      rememberNativeAttempt({
        stage: "content_script_fetch",
        accepted: false,
        error: String(error && error.message ? error.message : error)
      });
      return null;
    }
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
      error: pageResult?.error || null
    });
    if (pagePayload) return pagePayload;
    return fetchNativePayloadFromContentScript(conversationId);
  }

  function modelFromNativePayload(payload) {
    const mapping = payload && payload.mapping;
    if (!mapping || typeof mapping !== "object") return null;
    for (const node of Object.values(mapping)) {
      const metadata = node?.message?.metadata;
      if (metadata && typeof metadata.model_slug === "string" && metadata.model_slug) {
        return metadata.model_slug;
      }
    }
    return null;
  }

  // --- Assistant-produced asset acquisition (sandbox + file-service) ------
  //
  // Deliverable files surface only as expiring links; the conversation JSON
  // never carries bytes. Fetch them through the page bridge (authenticated
  // session) at capture time and ship them as envelope attachments. Failures
  // are normal (links expire with the sandbox container) and must never fail
  // the capture itself — per-asset outcomes are disclosed in provider_meta.
  const assetFetchRequestMessage = "polylogue.chatgpt.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.chatgpt.assetFetchResponse";
  const assetFetchTimeoutMs = 35000;
  const assetMaxBytesPerFile = 25 * 1024 * 1024;
  const assetMaxBytesTotal = 75 * 1024 * 1024;
  // Wall-clock budget for the WHOLE acquisition pass and a per-kind circuit
  // breaker: a conversation can reference dozens of sandbox files, and once
  // the sandbox container is gone every one of them fails the same way --
  // without these bounds a capture could stall for minutes on dead links.
  const assetTotalTimeBudgetMs = 45000;
  const assetConsecutiveFailureLimit = 3;
  const assetResponses = new Map();
  const sandboxLinkPattern = /sandbox:(\/mnt\/data\/[^\s)\]"'>]+)/g;

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== assetFetchResponseMessage || !data.requestId) return;
    const pending = assetResponses.get(data.requestId);
    if (!pending) return;
    assetResponses.delete(data.requestId);
    pending.resolve({ asset: data.asset || null, error: data.error || null });
  });

  function requestAssetFromPage(request) {
    const requestId = `polylogue-asset-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        assetResponses.delete(requestId);
        resolve({ asset: null, error: "timeout" });
      }, assetFetchTimeoutMs);
      assetResponses.set(requestId, {
        resolve(value) {
          window.clearTimeout(timeout);
          resolve(value);
        }
      });
    });
    window.postMessage({ type: assetFetchRequestMessage, requestId, request }, window.location.origin);
    return responsePromise;
  }

  function sandboxPathsFromText(text) {
    const paths = [];
    for (const match of String(text).matchAll(sandboxLinkPattern)) {
      const path = match[1].replace(/[.,;:!?*`]+$/, "");
      if (path !== "/mnt/data/" && !paths.includes(path)) paths.push(path);
    }
    return paths;
  }

  function collectAssetDescriptors(payload) {
    const mapping = payload && payload.mapping;
    if (!mapping || typeof mapping !== "object") return [];
    const descriptors = [];
    const seen = new Set();
    const add = (descriptor) => {
      if (descriptor.provider_attachment_id && !seen.has(descriptor.provider_attachment_id)) {
        seen.add(descriptor.provider_attachment_id);
        descriptors.push(descriptor);
      }
    };
    for (const [nodeId, node] of Object.entries(mapping)) {
      const message = node && node.message;
      if (!message) continue;
      const messageId = String(message.id || node.id || nodeId);
      const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
      for (const attachment of Array.isArray(metadata.attachments) ? metadata.attachments : []) {
        if (attachment && attachment.id) {
          add({
            kind: "file",
            fileId: String(attachment.id),
            provider_attachment_id: String(attachment.id),
            message_provider_id: messageId,
            name: attachment.name ? String(attachment.name) : null,
            mime_type: attachment.mime_type ? String(attachment.mime_type) : null
          });
        }
      }
      const content = message.content;
      const parts = content && Array.isArray(content.parts) ? content.parts : [];
      const role = message.author && message.author.role;
      for (const part of parts) {
        if (part && typeof part === "object" && typeof part.asset_pointer === "string" && part.asset_pointer) {
          const pointer = part.asset_pointer;
          const pointerPath = pointer.includes("://") ? pointer.split("://").at(-1) : pointer;
          const fileIdMatch = pointerPath.match(/file[-_][A-Za-z0-9]+/);
          if (fileIdMatch) {
            add({
              kind: "file",
              fileId: fileIdMatch[0],
              provider_attachment_id: pointer,
              message_provider_id: messageId,
              name: null,
              mime_type: null
            });
          }
        }
        if (typeof part === "string" && role === "assistant") {
          for (const path of sandboxPathsFromText(part)) {
            add({
              kind: "sandbox",
              sandboxPath: path,
              provider_attachment_id: `sandbox:${messageId}:${path}`,
              message_provider_id: messageId,
              name: path.replace(/\/+$/, "").split("/").at(-1) || null,
              mime_type: null
            });
          }
        }
      }
    }
    return descriptors;
  }

  async function acquireAssets(payload, conversationId) {
    const descriptors = collectAssetDescriptors(payload);
    const outcome = {
      attempted: descriptors.length,
      acquired: 0,
      failed: [],
      skipped_over_budget: 0,
      skipped_time_budget: 0,
      skipped_circuit_breaker: 0
    };
    const attachments = [];
    let totalBytes = 0;
    const startedAt = Date.now();
    const consecutiveFailuresByKind = { sandbox: 0, file: 0 };
    for (const descriptor of descriptors) {
      if (totalBytes >= assetMaxBytesTotal) {
        outcome.skipped_over_budget += 1;
        continue;
      }
      if (Date.now() - startedAt >= assetTotalTimeBudgetMs) {
        outcome.skipped_time_budget += 1;
        continue;
      }
      if (consecutiveFailuresByKind[descriptor.kind] >= assetConsecutiveFailureLimit) {
        // e.g. the sandbox container is dead: every sandbox link fails
        // identically, so stop burning the time budget on the rest.
        outcome.skipped_circuit_breaker += 1;
        continue;
      }
      const request = {
        kind: descriptor.kind,
        conversationId,
        messageId: descriptor.message_provider_id,
        sandboxPath: descriptor.sandboxPath || null,
        fileId: descriptor.fileId || null,
        maxBytes: Math.min(assetMaxBytesPerFile, assetMaxBytesTotal - totalBytes)
      };
      const result = await requestAssetFromPage(request);
      if (result.asset && result.asset.base64) {
        totalBytes += result.asset.size_bytes || 0;
        consecutiveFailuresByKind[descriptor.kind] = 0;
        outcome.acquired += 1;
        attachments.push({
          provider_attachment_id: descriptor.provider_attachment_id,
          message_provider_id: descriptor.message_provider_id,
          name: result.asset.name || descriptor.name,
          mime_type: result.asset.mime_type || descriptor.mime_type,
          size_bytes: result.asset.size_bytes || null,
          inline_base64: result.asset.base64,
          provider_meta: {
            capture_source: "chatgpt_page_asset_fetch",
            asset_kind: descriptor.kind,
            sandbox_path: descriptor.sandboxPath || null
          }
        });
      } else {
        consecutiveFailuresByKind[descriptor.kind] += 1;
        outcome.failed.push({
          provider_attachment_id: descriptor.provider_attachment_id,
          error: result.error || "unknown"
        });
      }
    }
    return { attachments, outcome };
  }

  function buildNativeEnvelope(payload, assetAcquisition = null) {
    const turns = collectNativeTurns(payload);
    if (!turns.length) return null;
    return window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: nativeAdapterName,
      turns,
      providerSessionId: String(payload.conversation_id || payload.id || conversationIdFromUrl()),
      sessionKind: payload.is_temporary === true ? "temporary" : null,
      title: typeof payload.title === "string" && payload.title ? payload.title : null,
      createdAt: timestampFromSeconds(payload.create_time),
      updatedAt: timestampFromSeconds(payload.update_time),
      model: modelFromNativePayload(payload),
      providerMeta: {
        capture_source: "chatgpt_backend_api",
        current_node: payload.current_node || null,
        mapping_node_count: payload.mapping ? Object.keys(payload.mapping).length : 0,
        is_temporary: payload.is_temporary === true,
        session_kind: payload.is_temporary === true ? "temporary" : null,
        asset_acquisition: assetAcquisition ? assetAcquisition.outcome : null
      },
      rawProviderPayload: payload,
      attachments: assetAcquisition ? assetAcquisition.attachments : []
    });
  }

  async function capture() {
    const nativePayload = latestNativePayload() || (await fetchNativePayloadOnDemand());
    let assetAcquisition = null;
    if (nativePayload) {
      try {
        assetAcquisition = await acquireAssets(
          nativePayload,
          String(nativePayload.conversation_id || nativePayload.id || conversationIdFromUrl())
        );
      } catch (error) {
        assetAcquisition = {
          attachments: [],
          outcome: { attempted: 0, acquired: 0, failed: [{ provider_attachment_id: null, error: String(error && error.message ? error.message : error) }], skipped_over_budget: 0 }
        };
      }
    }
    const envelope = nativePayload ? buildNativeEnvelope(nativePayload, assetAcquisition) : null;
    const fallbackEnvelope = () => {
      const turns = collectTurns();
      if (!turns.length) return null;
      return window.polylogueCapture.buildEnvelope({
        provider: "chatgpt",
        adapterName: domAdapterName,
        turns,
        providerMeta: {
          capture_fidelity: "dom_degraded",
          native_attempts: nativeAttemptDiagnostics.slice(-6)
        }
      });
    };
    const finalEnvelope = envelope || fallbackEnvelope();
    if (!finalEnvelope) return { ok: false, error: "no_turns" };
    const captureResult = await window.polylogueCapture.sendCapture(finalEnvelope);
    const archiveState = await window.polylogueCapture.refreshArchiveState(
      "chatgpt",
      finalEnvelope.session.provider_session_id
    );
    return { ok: true, envelope: finalEnvelope, captureResult, archiveState };
  }

  window.polylogueCapture.capturePage = capture;
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.capturePage") return false;
    capture().then(sendResponse).catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
  // Outbound posting (reverse channel). Selectors target the ChatGPT composer
  // as of 2026-06 and need live re-verification when the UI changes: composer is
  // the ProseMirror contenteditable #prompt-textarea (verified live). The submit
  // button has NO data-testid="send-button" in the current UI — it is the
  // composer-submit slot (class composer-submit-button-color); its aria-label is
  // "Start Voice"/dictation when empty and becomes a send label only with text,
  // so match by the submit-slot class first, aria-label last.
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.postReply") return false;
    window.polylogueCapture
      .postReplyToComposer({
        command: message.command,
        composerSelectors: [
          "#prompt-textarea",
          'div[contenteditable="true"]#prompt-textarea',
          'form div[contenteditable="true"]'
        ],
        sendSelectors: [
          "button.composer-submit-button-color",
          'button[data-testid="composer-send-button"]',
          'button[data-testid="send-button"]',
          'button[aria-label*="Send" i]'
        ]
      })
      .then(sendResponse)
      .catch((error) =>
        sendResponse({ status: "failed", detail: String(error.message || error), observed_url: window.location.href })
      );
    return true;
  });
})();
