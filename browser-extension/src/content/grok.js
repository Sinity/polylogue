(function () {
  if (window.__polylogueGrokCaptureInstalled) return;
  window.__polylogueGrokCaptureInstalled = true;

  // Native-only adapter (polylogue Grok native-capture upgrade, 2026-07-31).
  //
  // grok.js used to be the only provider adapter with no MAIN-world bridge:
  // it scraped whatever turn nodes happened to be mounted in the DOM and
  // hardcoded `native_attempts: []`. Every other provider's DOM path has
  // proven lossy by roughly two orders of magnitude once a conversation
  // scrolls past the virtualized viewport (measured live: one ChatGPT
  // conversation held 5 DOM nodes against 996 API messages). grok.com's own
  // `/rest/app-chat/conversations/<id>` + `.../responses` REST surface
  // (verified live 2026-07-31 via CDP against an authenticated session) is
  // strictly superior for every field DOM scraping could ever observe, so
  // there is no DOM fallback here at all -- a native-capture failure fails
  // loud (diagnostics attached) rather than silently degrading to an
  // under-observed capture that looks the same as a complete one.
  const nativeAdapterName = "grok-native-v1";
  const nativeCaptureMessage = "polylogue.grok.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.grok.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.grok.nativeFetchResponse";
  const nativeFetchTimeoutMs = 8000;
  const assetFetchRequestMessage = "polylogue.grok.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.grok.assetFetchResponse";
  const assetFetchTimeoutMs = 10000;
  const assetMaxBytesPerFile = 25 * 1024 * 1024;
  const assetMaxBytesTotal = 75 * 1024 * 1024;
  const assetTotalTimeBudgetMs = 10000;
  const assetConsecutiveFailureLimit = 3;

  const nativeCaptures = [];
  const nativeFetchResponses = new Map();
  const assetResponses = new Map();
  const nativeAttemptDiagnostics = [];

  function rememberNativeAttempt(diagnostic) {
    nativeAttemptDiagnostics.push({ attempted_at: new Date().toISOString(), ...diagnostic });
    if (nativeAttemptDiagnostics.length > 8) {
      nativeAttemptDiagnostics.splice(0, nativeAttemptDiagnostics.length - 8);
    }
  }

  // grok.com conversation URLs are /c/<uuid> (verified live 2026-07-31).
  function conversationIdFromUrl(url = window.location.href) {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    const marker = parts.indexOf("c");
    if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
    return null;
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type === nativeCaptureMessage && data.capture) {
      nativeCaptures.push(data.capture);
      if (nativeCaptures.length > 8) nativeCaptures.splice(0, nativeCaptures.length - 8);
      return;
    }
    if (data.type === nativeFetchResponseMessage && data.requestId) {
      const pending = nativeFetchResponses.get(data.requestId);
      if (!pending) return;
      nativeFetchResponses.delete(data.requestId);
      pending.resolve({ capture: data.capture || null, error: data.error || null });
      return;
    }
    if (data.type === assetFetchResponseMessage && data.requestId) {
      const pending = assetResponses.get(data.requestId);
      if (!pending) return;
      assetResponses.delete(data.requestId);
      pending.resolve(data.outcome && typeof data.outcome === "object" ? data.outcome : { status: "request_failed", detail: "bridge_response_missing" });
    }
  });

  async function requestNativeCaptureFromPage(conversationId) {
    const requestId = `polylogue-grok-native-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        nativeFetchResponses.delete(requestId);
        resolve({ capture: null, error: "timeout" });
      }, nativeFetchTimeoutMs);
      nativeFetchResponses.set(requestId, {
        resolve(value) {
          window.clearTimeout(timeout);
          resolve(value);
        },
      });
    });
    window.postMessage({ type: nativeFetchRequestMessage, requestId, conversationId }, window.location.origin);
    return responsePromise;
  }

  function parseNativeCapture(capture, expectedConversationId) {
    if (!capture || !capture.ok || typeof capture.body !== "string") return null;
    let payload;
    try {
      payload = JSON.parse(capture.body);
    } catch {
      return null;
    }
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.responses)) return null;
    const payloadConversationId = payload.conversationId;
    if (expectedConversationId && payloadConversationId && String(payloadConversationId) !== expectedConversationId) {
      return null;
    }
    return payload;
  }

  function latestNativePayload(expectedConversationId) {
    for (let index = nativeCaptures.length - 1; index >= 0; index -= 1) {
      const payload = parseNativeCapture(nativeCaptures[index], expectedConversationId);
      if (payload) return payload;
    }
    return null;
  }

  async function fetchNativePayloadOnDemand(requestedConversationId) {
    const conversationId = requestedConversationId || conversationIdFromUrl();
    if (!conversationId || !/^[A-Za-z0-9_-]{1,256}$/.test(conversationId)) {
      rememberNativeAttempt({ stage: "conversation_id_resolution", accepted: false, reason: "no_conversation_id_in_url" });
      return null;
    }
    const pageResult = await requestNativeCaptureFromPage(conversationId);
    const payload = parseNativeCapture(pageResult && pageResult.capture, conversationId);
    rememberNativeAttempt({
      stage: "page_bridge_fetch",
      ok: pageResult?.capture?.ok ?? null,
      status: pageResult?.capture?.status ?? null,
      response_count: Array.isArray(payload?.responses) ? payload.responses.length : null,
      accepted: Boolean(payload),
      error: pageResult?.error || pageResult?.capture?.error || null,
    });
    if (payload) return payload;
    return latestNativePayload(conversationId);
  }

  function roleFromSender(sender) {
    const normalized = String(sender || "").toLowerCase();
    if (normalized === "human") return "user";
    if (normalized === "assistant") return "assistant";
    return "unknown";
  }

  function timestampFromValue(value) {
    if (typeof value === "string" && value) return value;
    if (typeof value === "number" && Number.isFinite(value)) {
      return new Date(value < 10_000_000_000 ? value * 1000 : value).toISOString();
    }
    return null;
  }

  // Structured evidence a Grok response node carries beyond its `message`
  // prose: reasoning steps, web/X/RAG/connector search evidence, and raw
  // tool responses. Each recognized shape becomes a typed
  // BrowserCaptureBlock; anything with content this function does not
  // recognize is still emitted (never dropped) as a tool_result block
  // flagged `metadata.unrecognized_shape = true` so a payload shape change
  // shows up as a visible diagnostic instead of a silently-shrunken capture.
  function stepBlocks(response) {
    const steps = Array.isArray(response.steps) ? response.steps : [];
    return steps.flatMap((step, index) => {
      const text = Array.isArray(step?.text) ? step.text.filter((line) => typeof line === "string" && line).join("\n") : "";
      const blocks = [];
      if (text) {
        blocks.push({ type: "thinking", text, metadata: { step_index: index, tags: Array.isArray(step?.tags) ? step.tags : [] } });
      }
      for (const usage of Array.isArray(step?.toolUsageResults) ? step.toolUsageResults : []) {
        blocks.push(toolUsageBlock(usage, response.responseId, index));
      }
      for (const card of Array.isArray(step?.toolUsageCards) ? step.toolUsageCards : []) {
        blocks.push(toolUsageBlock(card, response.responseId, index, "tool_usage_card"));
      }
      return blocks;
    });
  }

  function toolUsageBlock(usage, ownId, stepIndex, sourceLabel = "tool_usage_result") {
    const toolName =
      (usage && typeof usage === "object" && (usage.toolName || usage.tool_name || usage.name || usage.type)) || null;
    if (toolName && typeof toolName === "string") {
      return {
        type: "tool_result",
        tool_id: ownId,
        tool_name: toolName,
        text: typeof usage.text === "string" ? usage.text : null,
        metadata: { step_index: stepIndex, source: sourceLabel, raw: usage },
      };
    }
    return {
      type: "tool_result",
      tool_id: ownId,
      text: JSON.stringify(usage),
      metadata: { step_index: stepIndex, source: sourceLabel, unrecognized_shape: true },
    };
  }

  const RESULT_LIST_FIELDS = [
    ["webSearchResults", "web_search"],
    ["citedWebSearchResults", "web_search"],
    ["xposts", "x_search"],
    ["citedXposts", "x_search"],
    ["ragResults", "rag_search"],
    ["citedRagResults", "rag_search"],
    ["searchProductResults", "product_search"],
    ["connectorSearchResults", "connector_search"],
    ["citedConnectorSearchResults", "connector_search"],
    ["collectionSearchResults", "collection_search"],
    ["citedCollectionSearchResults", "collection_search"],
  ];

  function resultListBlocks(response) {
    const blocks = [];
    for (const [field, toolName] of RESULT_LIST_FIELDS) {
      const value = response[field];
      if (!Array.isArray(value) || !value.length) continue;
      blocks.push({
        type: "tool_result",
        tool_id: response.responseId,
        tool_name: toolName,
        metadata: { field, count: value.length, results: value },
      });
    }
    if (response.query) {
      blocks.unshift({
        type: "tool_use",
        tool_id: response.responseId,
        tool_name: "web_search",
        tool_input: { query: response.query, query_type: response.queryType || null },
      });
    }
    return blocks;
  }

  function toolResponseBlocks(response) {
    const toolResponses = Array.isArray(response.toolResponses) ? response.toolResponses : [];
    return toolResponses.map((entry, index) => {
      const name = entry && typeof entry === "object" && (entry.toolName || entry.tool_name || entry.name);
      if (typeof name === "string" && name) {
        return {
          type: "tool_use",
          tool_id: `${response.responseId}:tool_response:${index}`,
          tool_name: name,
          tool_input: entry.input || entry.tool_input || null,
          metadata: { source: "toolResponses", index },
        };
      }
      return {
        type: "tool_result",
        tool_id: `${response.responseId}:tool_response:${index}`,
        text: JSON.stringify(entry),
        metadata: { source: "toolResponses", index, unrecognized_shape: true },
      };
    });
  }

  function nativeTurnBlocks(response) {
    return [...stepBlocks(response), ...resultListBlocks(response), ...toolResponseBlocks(response)];
  }

  // Attachments carry no bytes in `/responses` -- only identifying metadata
  // plus a `fileUri`/`key` resolvable through assets.grok.com (see
  // grok_bridge.js). fileAttachmentsMetadata and fileAttachmentAssetMetadata
  // both key by the same id (verified live 2026-07-31); merge them so a
  // single attachment descriptor carries both the display metadata and the
  // acquirable content key.
  function attachmentDescriptorsForResponse(response) {
    const descriptors = [];
    const seen = new Set();
    const assetById = new Map(
      (Array.isArray(response.fileAttachmentAssetMetadata) ? response.fileAttachmentAssetMetadata : [])
        .filter((asset) => asset && typeof asset.assetId === "string")
        .map((asset) => [asset.assetId, asset]),
    );
    for (const meta of Array.isArray(response.fileAttachmentsMetadata) ? response.fileAttachmentsMetadata : []) {
      if (!meta || typeof meta.fileMetadataId !== "string" || !meta.fileMetadataId) continue;
      if (seen.has(meta.fileMetadataId)) continue;
      seen.add(meta.fileMetadataId);
      const asset = assetById.get(meta.fileMetadataId) || null;
      descriptors.push({
        provider_attachment_id: meta.fileMetadataId,
        message_provider_id: response.responseId,
        name: meta.fileName || asset?.name || null,
        mime_type: meta.fileMimeType || asset?.mimeType || null,
        size_bytes: typeof asset?.sizeBytes === "number" ? asset.sizeBytes : null,
        asset_key: meta.fileUri || asset?.key || null,
        provider_meta: { capture_source: "grok_app_chat_api", file_source: meta.fileSource || asset?.fileSource || null },
      });
    }
    for (const url of Array.isArray(response.generatedImageUrls) ? response.generatedImageUrls : []) {
      if (typeof url !== "string" || !url) continue;
      const id = `generated:${window.polylogueCapture.fnv1a(url)}`;
      if (seen.has(id)) continue;
      seen.add(id);
      descriptors.push({
        provider_attachment_id: id,
        message_provider_id: response.responseId,
        name: null,
        mime_type: null,
        size_bytes: null,
        url: /^https?:\/\//i.test(url) ? url : null,
        provider_meta: { capture_source: "grok_generated_image" },
      });
    }
    for (const url of Array.isArray(response.imageEditUris) ? response.imageEditUris : []) {
      if (typeof url !== "string" || !url) continue;
      const id = `image_edit:${window.polylogueCapture.fnv1a(url)}`;
      if (seen.has(id)) continue;
      seen.add(id);
      descriptors.push({
        provider_attachment_id: id,
        message_provider_id: response.responseId,
        name: null,
        mime_type: null,
        size_bytes: null,
        url: /^https?:\/\//i.test(url) ? url : null,
        provider_meta: { capture_source: "grok_image_edit" },
      });
    }
    // imageAttachments' element shape has not been observed live with
    // content (empty in every fixture captured so far). Record it rather
    // than silently guessing a shape for it.
    if (Array.isArray(response.imageAttachments) && response.imageAttachments.length) {
      rememberNativeAttempt({
        stage: "attachment_shape",
        accepted: false,
        reason: "imageAttachments_shape_unverified",
        sample: response.imageAttachments.slice(0, 2),
      });
    }
    return descriptors;
  }

  function requestAssetFromPage(request) {
    const requestId = `polylogue-grok-asset-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const responsePromise = new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        assetResponses.delete(requestId);
        resolve({ status: "request_failed", detail: "response_timeout" });
      }, assetFetchTimeoutMs);
      assetResponses.set(requestId, {
        resolve(value) {
          window.clearTimeout(timeout);
          resolve(value);
        },
      });
    });
    window.postMessage({ type: assetFetchRequestMessage, requestId, request }, window.location.origin);
    return responsePromise;
  }

  async function acquireAssets(descriptors) {
    const outcome = {
      attempted: descriptors.length,
      acquired: 0,
      failed: [],
      status_counts: {},
      skipped_over_budget: 0,
      skipped_time_budget: 0,
      skipped_circuit_breaker: 0,
    };
    const attachments = [];
    let totalBytes = 0;
    let consecutiveFailures = 0;
    const startedAt = Date.now();
    for (const descriptor of descriptors) {
      const resolvable = descriptor.asset_key || descriptor.url;
      if (!resolvable) {
        attachments.push({ ...descriptor, provider_meta: { ...descriptor.provider_meta, byte_acquisition: "no_resolvable_source" } });
        continue;
      }
      if (totalBytes >= assetMaxBytesTotal) {
        outcome.skipped_over_budget += 1;
        attachments.push(descriptor);
        continue;
      }
      if (Date.now() - startedAt >= assetTotalTimeBudgetMs) {
        outcome.skipped_time_budget += 1;
        attachments.push(descriptor);
        continue;
      }
      if (consecutiveFailures >= assetConsecutiveFailureLimit) {
        outcome.skipped_circuit_breaker += 1;
        attachments.push(descriptor);
        continue;
      }
      const request = descriptor.asset_key
        ? { key: descriptor.asset_key, name: descriptor.name, maxBytes: Math.min(assetMaxBytesPerFile, assetMaxBytesTotal - totalBytes) }
        : { key: descriptor.url, name: descriptor.name, maxBytes: Math.min(assetMaxBytesPerFile, assetMaxBytesTotal - totalBytes) };
      const result = await requestAssetFromPage(request);
      const status = typeof result.status === "string" ? result.status : "request_failed";
      const contentSha256 = result.asset && result.asset.sha256;
      const acquiredIsValid =
        status === "acquired" && result.asset && result.asset.base64 && typeof contentSha256 === "string" && /^[0-9a-f]{64}$/.test(contentSha256);
      outcome.status_counts[status] = (outcome.status_counts[status] || 0) + 1;
      if (acquiredIsValid) {
        totalBytes += result.asset.size_bytes || 0;
        consecutiveFailures = 0;
        outcome.acquired += 1;
        attachments.push({
          ...descriptor,
          mime_type: result.asset.mime_type || descriptor.mime_type,
          size_bytes: result.asset.size_bytes || descriptor.size_bytes,
          inline_base64: result.asset.base64,
          provider_meta: { ...descriptor.provider_meta, content_sha256: contentSha256, byte_acquisition: "acquired" },
        });
      } else {
        consecutiveFailures += 1;
        outcome.failed.push({ provider_attachment_id: descriptor.provider_attachment_id, status, detail: result.detail || null });
        attachments.push({ ...descriptor, provider_meta: { ...descriptor.provider_meta, byte_acquisition: status, byte_acquisition_detail: result.detail || null } });
      }
    }
    return { attachments, outcome };
  }

  function collectNativeTurns(payload) {
    const responses = Array.isArray(payload.responses) ? payload.responses : [];
    const turns = [];
    for (const response of responses) {
      if (!response || typeof response !== "object" || typeof response.responseId !== "string") {
        rememberNativeAttempt({ stage: "turn_shape", accepted: false, reason: "response_missing_id" });
        continue;
      }
      const text = typeof response.message === "string" ? response.message : "";
      const blocks = nativeTurnBlocks(response);
      if (!text && !blocks.length) continue;
      turns.push({
        provider_turn_id: response.responseId,
        role: roleFromSender(response.sender),
        text: text || null,
        timestamp: timestampFromValue(response.createTime),
        parent_turn_id: response.parentResponseId || null,
        blocks,
        provider_meta: {
          capture_source: "grok_app_chat_api",
          model: response.model || null,
          partial: response.partial === true,
          manual: response.manual === true,
          shared: response.shared === true,
          stream_error_count: Array.isArray(response.streamErrors) ? response.streamErrors.length : 0,
          stream_errors: Array.isArray(response.streamErrors) && response.streamErrors.length ? response.streamErrors : null,
        },
      });
    }
    turns.sort((left, right) => {
      const leftTime = left.timestamp ? Date.parse(left.timestamp) : 0;
      const rightTime = right.timestamp ? Date.parse(right.timestamp) : 0;
      return leftTime - rightTime;
    });
    return turns;
  }

  function modelFromNativePayload(payload) {
    for (const response of Array.isArray(payload.responses) ? payload.responses : []) {
      if (typeof response?.model === "string" && response.model) return response.model;
    }
    return null;
  }

  async function buildNativeEnvelope(payload, generationDiagnostics) {
    const turns = collectNativeTurns(payload);
    if (!turns.length) return null;
    const descriptors = (Array.isArray(payload.responses) ? payload.responses : []).flatMap((response) =>
      attachmentDescriptorsForResponse(response),
    );
    const assetAcquisition = descriptors.length ? await acquireAssets(descriptors) : { attachments: [], outcome: null };
    const inflightCount = Array.isArray(payload.inflightResponses) ? payload.inflightResponses.length : 0;
    return window.polylogueCapture.buildEnvelope({
      provider: "grok",
      adapterName: nativeAdapterName,
      turns,
      providerSessionId: String(payload.conversationId),
      sessionKind: payload.temporary === true ? "temporary" : null,
      title: typeof payload.title === "string" && payload.title ? payload.title : null,
      createdAt: timestampFromValue(payload.createTime),
      updatedAt: timestampFromValue(payload.modifyTime),
      model: modelFromNativePayload(payload),
      providerMeta: {
        capture_source: "grok_app_chat_api",
        response_count: turns.length,
        inflight_response_count: inflightCount,
        conversation_temporary: payload.temporary === true,
        session_kind: payload.temporary === true ? "temporary" : null,
        asset_acquisition: assetAcquisition.outcome,
        native_attempts: generationDiagnostics.slice(-8),
      },
      rawProviderPayload: payload,
      attachments: assetAcquisition.attachments,
    });
  }

  async function capture(reason = null, requestedConversationId = null, deferReceiver = false) {
    const nativePayload = await fetchNativePayloadOnDemand(requestedConversationId);
    if (!nativePayload) {
      return {
        ok: false,
        error: "native_capture_unavailable",
        native_attempts: nativeAttemptDiagnostics.slice(-8),
      };
    }
    const envelope = await buildNativeEnvelope(nativePayload, nativeAttemptDiagnostics);
    if (!envelope) {
      return {
        ok: false,
        error: "no_turns",
        native_attempts: nativeAttemptDiagnostics.slice(-8),
      };
    }
    if (deferReceiver) return { ok: true, envelope, deferred: true };
    const captureResult = await window.polylogueCapture.sendCapture(envelope, reason);
    if (!captureResult?.ok) {
      return {
        ok: false,
        envelope,
        captureResult,
        error: captureResult?.error || "capture_rejected",
        timelineRecorded: true,
      };
    }
    const archiveState = await window.polylogueCapture.refreshArchiveState("grok", envelope.session.provider_session_id);
    return { ok: true, envelope, captureResult, archiveState };
  }

  window.polylogueCapture.capturePage = capture;
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "polylogue.capturePage") return false;
    capture(message.reason || null, message.providerSessionId || null, message.deferReceiver === true)
      .then(sendResponse)
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  });
})();
