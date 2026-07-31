(function () {
  // MAIN-world companion to src/content/grok.js. Grok is the only provider
  // that previously had no bridge at all -- grok.js hardcoded
  // `native_attempts: []` and scraped the DOM, which this extension's other
  // adapters have shown is lossy by roughly two orders of magnitude once a
  // conversation scrolls past the virtualized viewport. grok.com exposes a
  // clean, cookie-authenticated REST surface
  // (/rest/app-chat/conversations/<id>, .../responses, .../response-node)
  // that this bridge fetches with the page's own credentials, exactly the
  // way chatgpt_bridge.js/claude_bridge.js do for their providers.
  const nativeCaptureMessage = "polylogue.grok.nativeCapture";
  const nativeFetchRequestMessage = "polylogue.grok.nativeFetchRequest";
  const nativeFetchResponseMessage = "polylogue.grok.nativeFetchResponse";
  const assetFetchRequestMessage = "polylogue.grok.assetFetchRequest";
  const assetFetchResponseMessage = "polylogue.grok.assetFetchResponse";
  const currentOrigin = window.location.origin;
  const nativeFetchTimeoutMs = 8000;
  const assetFetchTimeoutMs = 9000;
  const assetAbsoluteMaxBytes = 25 * 1024 * 1024;

  window.__polylogueGrokCapturedFetches = Array.isArray(window.__polylogueGrokCapturedFetches)
    ? window.__polylogueGrokCapturedFetches
    : [];

  function post(capture) {
    window.postMessage({ type: nativeCaptureMessage, capture }, currentOrigin);
  }

  function remember(capture) {
    window.__polylogueGrokCapturedFetches.push(capture);
    if (window.__polylogueGrokCapturedFetches.length > 8) {
      window.__polylogueGrokCapturedFetches.splice(0, window.__polylogueGrokCapturedFetches.length - 8);
    }
    post(capture);
  }

  const existingCaptures = window.__polylogueGrokCapturedFetches.slice(-8);
  window.__polylogueGrokCapturedFetches = existingCaptures;
  for (const capture of existingCaptures) post(capture);

  if (window.__polylogueGrokFetchHookInstalled) return;
  window.__polylogueGrokFetchHookInstalled = true;

  const originalFetch = window.fetch;

  function timeoutError(label, timeoutMs) {
    const error = new Error(`${label}_timeout_after_${timeoutMs}ms`);
    error.name = "PolylogueTimeoutError";
    return error;
  }

  function conversationUrl(conversationId, suffix = "") {
    return new URL(
      `/rest/app-chat/conversations/${encodeURIComponent(String(conversationId))}${suffix}`,
      currentOrigin,
    );
  }

  async function fetchJson(url, label) {
    const controller = new globalThis.AbortController();
    const timeoutId = window.setTimeout(
      () => controller.abort(timeoutError(label, nativeFetchTimeoutMs)),
      nativeFetchTimeoutMs,
    );
    let response;
    try {
      response = await originalFetch.call(window, url.href, {
        credentials: "include",
        cache: "no-store",
        signal: controller.signal,
      });
    } finally {
      window.clearTimeout(timeoutId);
    }
    const contentType = response.headers.get("content-type") || "";
    const bodyText = contentType.includes("application/json") ? await response.clone().text() : "";
    let parsed = null;
    if (bodyText) {
      try {
        parsed = JSON.parse(bodyText);
      } catch {
        parsed = null;
      }
    }
    return { ok: response.ok, status: response.status, contentType, bodyText, parsed };
  }

  // A Grok conversation's content lives across three independent endpoints:
  // metadata+identity (title/temporary/timestamps), the actual turn content
  // (`/responses` -- confirmed live 2026-07-31 to carry `message` text,
  // `steps`, attachments, tool/search evidence; `/response-node` alone is
  // just an id/parent skeleton with no message content), and any
  // still-generating tail (`inflightResponses`, best-effort). Bundle all
  // three into one combined JSON body so grok.js has a single capture
  // artifact to parse, exactly like the other bridges hand back one
  // conversation payload.
  async function fetchConversation(conversationId) {
    const conversationResult = await fetchJson(conversationUrl(conversationId), "conversation_metadata");
    if (!conversationResult.ok || !conversationResult.parsed) {
      return {
        url: conversationUrl(conversationId).href,
        status: conversationResult.status,
        ok: false,
        contentType: conversationResult.contentType,
        body: conversationResult.bodyText,
        capturedAt: new Date().toISOString(),
        error: "conversation_metadata_fetch_failed",
      };
    }
    const responsesResult = await fetchJson(conversationUrl(conversationId, "/responses"), "conversation_responses");
    if (!responsesResult.ok || !Array.isArray(responsesResult.parsed?.responses)) {
      return {
        url: conversationUrl(conversationId, "/responses").href,
        status: responsesResult.status,
        ok: false,
        contentType: responsesResult.contentType,
        body: responsesResult.bodyText,
        capturedAt: new Date().toISOString(),
        error: "conversation_responses_fetch_failed",
      };
    }
    // response-node is best-effort: it only ever adds inflight-generation
    // skeleton entries, never turn content, so its failure must not fail
    // the whole capture.
    let inflightResponses = [];
    try {
      const responseNodeResult = await fetchJson(conversationUrl(conversationId, "/response-node"), "response_node");
      if (responseNodeResult.ok && Array.isArray(responseNodeResult.parsed?.inflightResponses)) {
        inflightResponses = responseNodeResult.parsed.inflightResponses;
      }
    } catch {
      inflightResponses = [];
    }
    const combined = {
      ...conversationResult.parsed,
      responses: responsesResult.parsed.responses,
      inflightResponses,
    };
    return {
      url: conversationUrl(conversationId).href,
      status: 200,
      ok: true,
      contentType: "application/json",
      body: JSON.stringify(combined),
      capturedAt: new Date().toISOString(),
    };
  }

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = event.data || {};
    if (data.type !== nativeFetchRequestMessage || !data.requestId || !data.conversationId) return;
    try {
      const capture = await fetchConversation(data.conversationId);
      if (capture.ok && capture.body) remember(capture);
      window.postMessage({ type: nativeFetchResponseMessage, requestId: data.requestId, capture }, currentOrigin);
    } catch (error) {
      window.postMessage(
        {
          type: nativeFetchResponseMessage,
          requestId: data.requestId,
          error: String(error && error.message ? error.message : error),
        },
        currentOrigin,
      );
    }
  });

  // --- Attachment byte acquisition -----------------------------------
  //
  // File/image attachments never carry bytes in `/responses` -- only a
  // `fileUri`/`key` such as `users/<uid>/<assetId>/content`, served from the
  // separate `assets.grok.com` host. Verified live 2026-07-31: that host
  // requires the same grok.com session credentials (403 with
  // credentials:"omit", 200 with credentials:"include" from a grok.com
  // page), unlike ChatGPT's signed object-store URLs which must NOT receive
  // page credentials cross-origin. So, deliberately unlike
  // chatgpt_bridge.js's asset fetch, this always sends credentials.
  function assetOutcome(status, { httpStatus = null, detail = null, sizeBytes = null, asset = null } = {}) {
    const outcome = { status };
    if (httpStatus !== null) outcome.http_status = httpStatus;
    if (detail !== null) outcome.detail = detail;
    if (sizeBytes !== null) outcome.size_bytes = sizeBytes;
    if (asset !== null) outcome.asset = asset;
    return outcome;
  }

  function boundedMaxBytes(request) {
    const requested = Number(request.maxBytes);
    if (!Number.isFinite(requested) || requested <= 0) return assetAbsoluteMaxBytes;
    return Math.min(requested, assetAbsoluteMaxBytes);
  }

  function declaredContentLength(response) {
    const raw = response.headers.get("content-length");
    if (!raw || !/^\d+$/.test(raw)) return null;
    const parsed = Number(raw);
    return Number.isSafeInteger(parsed) ? parsed : null;
  }

  async function readBoundedBody(response, maxBytes) {
    if (!response.body || typeof response.body.getReader !== "function") {
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength > maxBytes) return { tooLarge: true, sizeBytes: buffer.byteLength };
      return { tooLarge: false, buffer };
    }
    const reader = response.body.getReader();
    const chunks = [];
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel().catch(() => undefined);
        return { tooLarge: true, sizeBytes: total };
      }
      chunks.push(value);
    }
    const merged = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return { tooLarge: false, buffer: merged.buffer };
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + chunkSize));
    }
    return window.btoa(binary);
  }

  function bytesToHex(buffer) {
    return [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  async function sha256Hex(buffer) {
    if (!globalThis.crypto?.subtle) throw new Error("asset_sha256_unavailable");
    return bytesToHex(await globalThis.crypto.subtle.digest("SHA-256", buffer));
  }

  async function fetchAssetBytes(request) {
    let assetUrl;
    try {
      assetUrl = new URL(`/${String(request.key).replace(/^\/+/, "")}`, "https://assets.grok.com");
    } catch {
      return assetOutcome("invalid_request", { detail: "asset_key_invalid" });
    }
    if (typeof request.key !== "string" || !request.key || assetUrl.protocol !== "https:") {
      return assetOutcome("invalid_request", { detail: "asset_key_invalid" });
    }
    const controller = new globalThis.AbortController();
    const timeoutId = window.setTimeout(
      () => controller.abort(timeoutError("asset_bytes_fetch", assetFetchTimeoutMs)),
      assetFetchTimeoutMs,
    );
    let response;
    try {
      response = await originalFetch.call(window, assetUrl.href, {
        credentials: "include",
        cache: "no-store",
        signal: controller.signal,
      });
    } catch (error) {
      const timedOut = error?.name === "AbortError" || error?.name === "PolylogueTimeoutError";
      return assetOutcome("request_failed", { detail: timedOut ? "request_timeout" : "request_failed" });
    } finally {
      window.clearTimeout(timeoutId);
    }
    if ([401, 403, 404, 410].includes(response.status)) {
      return assetOutcome("signed_url_expired", { httpStatus: response.status, detail: `asset_http_${response.status}` });
    }
    if (!response.ok) {
      return assetOutcome("request_failed", { httpStatus: response.status, detail: `asset_http_${response.status}` });
    }
    const maxBytes = boundedMaxBytes(request);
    const contentLength = declaredContentLength(response);
    if (contentLength !== null && contentLength > maxBytes) {
      return assetOutcome("too_large", { httpStatus: response.status, detail: "content_length_over_limit", sizeBytes: contentLength });
    }
    const bodyResult = await readBoundedBody(response, maxBytes);
    if (bodyResult.tooLarge) {
      return assetOutcome("too_large", { httpStatus: response.status, detail: "downloaded_bytes_over_limit", sizeBytes: bodyResult.sizeBytes });
    }
    const buffer = bodyResult.buffer;
    let contentSha256;
    try {
      contentSha256 = await sha256Hex(buffer);
    } catch {
      return assetOutcome("integrity_error", { detail: "sha256_unavailable" });
    }
    return assetOutcome("acquired", {
      httpStatus: response.status,
      asset: {
        base64: arrayBufferToBase64(buffer),
        size_bytes: buffer.byteLength,
        sha256: contentSha256,
        mime_type: response.headers.get("content-type") || null,
        name: request.name || null,
      },
    });
  }

  window.addEventListener("message", async (event) => {
    if (event.source !== window || event.origin !== currentOrigin) return;
    const data = event.data || {};
    if (data.type !== assetFetchRequestMessage || !data.requestId || !data.request) return;
    try {
      const outcome = await fetchAssetBytes(data.request);
      window.postMessage({ type: assetFetchResponseMessage, requestId: data.requestId, outcome }, currentOrigin);
    } catch (error) {
      const timedOut = error?.name === "AbortError" || error?.name === "PolylogueTimeoutError";
      window.postMessage(
        {
          type: assetFetchResponseMessage,
          requestId: data.requestId,
          outcome: assetOutcome("request_failed", { detail: timedOut ? "request_timeout" : "request_failed" }),
        },
        currentOrigin,
      );
    }
  });
})();
