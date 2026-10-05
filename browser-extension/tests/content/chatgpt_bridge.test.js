import { stagingRuntime, attachmentBytes } from "../infra/capture-staging.js";
import { Buffer } from "node:buffer";
import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { TextEncoder } from "node:util";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const testDirectory = dirname(fileURLToPath(import.meta.url));
const assetStreamSource = readFileSync(resolve(testDirectory, "../../src/content/asset_stream.js"), "utf8");
const bridgeSource = readFileSync(resolve(testDirectory, "../../src/content/chatgpt_bridge.js"), "utf8");
const commonSource = readFileSync(resolve(testDirectory, "../../src/common.js"), "utf8");
const contentSource = readFileSync(resolve(testDirectory, "../../src/content/chatgpt.js"), "utf8");
const bearerToken = "synthetic-bearer-must-not-cross-the-page-bridge";
const chatGptAccountId = "synthetic-account-must-not-cross-the-page-bridge";
const signedUrl = "https://files.example.test/download/kit.zip?signature=synthetic-signed-secret";
const assetBytes = new TextEncoder().encode("polylogue authenticated interpreter asset\n");
const expectedSha256 = createHash("sha256").update(assetBytes).digest("hex");
const openDoms = [];

function jsonResponse(body, status = 200) {
  return new globalThis.Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function byteResponse(bytes, status = 200, declaredSize = null) {
  const headers = { "content-type": "application/zip" };
  if (declaredSize !== null) headers["content-length"] = String(declaredSize);
  return new globalThis.Response(bytes, { status, headers });
}

function authorizationHeader(options) {
  return new globalThis.Headers(options?.headers || {}).get("authorization");
}

function sandboxPlan(path = "/mnt/data/kit.zip", messageId = "assistant-message-1", recordKey = "assistant-node", ordinal = 0) {
  return { ordinal, descriptor: { provider_attachment_id: `sandbox:${messageId}:${path}`,
    message_provider_id: messageId, attachment_kind: "sandbox_file", name: path.split("/").at(-1), url: `sandbox:${path}`,
    original_record_key: recordKey, original_record_ordinal: ordinal, provider_meta: {} } };
}

function conversationPayload() {
  return {
    id: "conversation-1",
    conversation_id: "conversation-1",
    title: "Authenticated capture fixture",
    create_time: 1781366400,
    update_time: 1781366460,
    current_node: "assistant-node",
    mapping: {
      "assistant-node": {
        id: "assistant-node",
        parent: null,
        children: [],
        message: {
          id: "assistant-message-1",
          author: { role: "assistant" },
          create_time: 1781366460,
          content: {
            content_type: "text",
            parts: ["Kit ready: [download](sandbox:/mnt/data/kit.zip)"],
          },
          metadata: { model_slug: "gpt-test" },
        },
      },
    },
  };
}

function syntheticEndpointAdapter({
  authStatus = 200,
  authBody = { accessToken: bearerToken, account: { id: chatGptAccountId } },
  metadataStatus = 200,
  signedDownloadUrl = signedUrl,
  metadataBody = { download_url: signedDownloadUrl, file_name: "kit.zip" },
  signedStatus = 200,
  signedBytes = assetBytes,
  declaredSize = null,
} = {}) {
  const calls = [];
  const fetch = vi.fn(async (input, options = {}) => {
    const url = new URL(String(input), "https://chatgpt.com");
    calls.push({ url, options });
    if (url.origin === "https://chatgpt.com" && url.pathname === "/api/auth/session") {
      return jsonResponse(authBody, authStatus);
    }
    if (url.origin === "https://chatgpt.com" && url.pathname === "/backend-api/conversation/conversation-1") {
      if (authorizationHeader(options) !== `Bearer ${bearerToken}`) {
        return jsonResponse({ detail: "Unauthorized" }, 401);
      }
      return jsonResponse(conversationPayload());
    }
    if (
      url.origin === "https://chatgpt.com" &&
      url.pathname === "/backend-api/conversation/conversation-1/interpreter/download"
    ) {
      if (authorizationHeader(options) !== `Bearer ${bearerToken}`) {
        return jsonResponse({ detail: "Unauthorized" }, 401);
      }
      return jsonResponse(metadataBody, metadataStatus);
    }
    if (url.href === signedDownloadUrl) {
      return byteResponse(signedBytes, signedStatus, declaredSize);
    }
    throw new Error(`unexpected synthetic request: ${url.origin}${url.pathname}`);
  });
  return { calls, fetch };
}

function makeDom(adapter, url = "https://chatgpt.com/c/conversation-1") {
  const dom = new JSDOM("<!doctype html><title>ChatGPT fixture</title>", {
    url,
    runScripts: "outside-only",
  });
  openDoms.push(dom);
  // Node 20 rejects ArrayBuffers created by a separate jsdom VM realm. Keep
  // the production dependency on Web Crypto, but adapt test bytes into the
  // host realm before invoking the real digest implementation.
  const cryptoAdapter = {
    randomUUID: () => webcrypto.randomUUID(),
    subtle: {
      digest(algorithm, data) {
        return webcrypto.subtle.digest(algorithm, Buffer.from(new dom.window.Uint8Array(data)));
      },
    },
  };
  Object.defineProperty(dom.window, "crypto", { configurable: true, value: cryptoAdapter });
  Object.defineProperty(dom.window, "fetch", { configurable: true, value: adapter.fetch });
  return dom;
}

function installBridge(adapter, source = bridgeSource, { bootstrapToken = null } = {}) {
  const dom = makeDom(adapter);
  if (bootstrapToken) {
    const bootstrap = dom.window.document.createElement("script");
    bootstrap.id = "client-bootstrap";
    bootstrap.textContent = JSON.stringify({ session: { accessToken: bootstrapToken } });
    dom.window.document.body.appendChild(bootstrap);
  }
  const pending = new Map();
  const posted = [];
  dom.__captureRuntime = stagingRuntime();
  Object.defineProperty(dom.window, "chrome", { configurable: true, value: { storage: { local: { get: async () => ({ polylogueAmbientSettings: {}, polylogueReceiverPairing: { receiver_id: "neutral", state: "online" } }) } }, runtime: { id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", sendMessage: dom.__captureRuntime.sendMessage } } });
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(data) {
      posted.push(data);
      dom.window.queueMicrotask(() => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin, data })));
      const resolve = pending.get(data?.requestId);
      if (data?.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.assetFetchResponse" && resolve) {
        pending.delete(data.requestId);
        resolve(data.outcome);
      }
    },
  });
  new Script(assetStreamSource).runInContext(dom.getInternalVMContext());
  new Script(source).runInContext(dom.getInternalVMContext());

  function requestAsset(overrides = {}) {
    const requestId = `asset-request-${pending.size + 1}-${posted.length}`;
    const response = new Promise((resolve) => pending.set(requestId, resolve));
    return dom.window.polylogueAssetStream.request({ provider: "chatgpt", requestId, signal: new dom.window.AbortController().signal, start: () => {
      dom.window.postMessage({
        type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.assetFetchRequest", requestId,
        request: { kind: "sandbox", conversationId: "conversation-1",
          messageId: "assistant-message-1", sandboxPath: "/mnt/data/kit.zip", ...overrides },
      }, dom.window.location.origin);
      return response;
    } });
  }

  return { dom, posted, requestAsset };
}

function installFullCapture(adapter, { url, beforeInstall, plan = [], summary = {} } = {}) {
  const dom = makeDom(adapter, url);
  beforeInstall?.(dom.window.document);
  const posted = [];
  const runtimeMessages = [];
  const runtimeListeners = [];
  const chrome = {
    storage: { local: { get: async () => ({ polylogueAmbientSettings: {}, polylogueReceiverPairing: { receiver_id: "neutral", state: "online" } }) } },
    runtime: {
      id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
        const assetResult = await dom.__captureRuntime.sendMessage(message);
        if (assetResult !== undefined) return assetResult;
        runtimeMessages.push(message);
        if (message.type === "polylogue.capture") {
          return {
            ok: true,
            provider: "chatgpt",
            provider_session_id: "conversation-1",
            receiver_request_id: "synthetic-request",
          };
        }
        if (message.type === "polylogue.archiveState") return { captured: true, state: "archived" };
        return { ok: true };
      },
    },
  };
  dom.__captureRuntime = stagingRuntime();
  dom.__captureRuntime.nativeContract.plan = plan;
  dom.__captureRuntime.nativeContract.summary = summary;
  Object.defineProperty(dom.window, "chrome", { configurable: true, value: chrome });
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(data) {
      posted.push(data);
      dom.window.queueMicrotask(() => {
        dom.window.dispatchEvent(
          new dom.window.MessageEvent("message", {
            source: dom.window,
            origin: dom.window.location.origin,
            data,
          }),
        );
      });
    },
  });
  const context = dom.getInternalVMContext();
  new Script(assetStreamSource).runInContext(context);
  new Script(bridgeSource).runInContext(context);
  new Script(commonSource).runInContext(context);
  new Script(contentSource).runInContext(context);
  function sendRuntimeMessage(message) {
    return new Promise((resolve, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolve) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  dom.__captureRuntime.setDispatch(sendRuntimeMessage);
  async function materializeResult(result) {
    if (result?.envelope) result.envelope = await dom.__captureRuntime.materialize(result.envelope);
    return result;
  }
  return { dom, posted, runtimeListeners, runtimeMessages,
    sendRuntimeMessage: async (message) => materializeResult(await sendRuntimeMessage(message)),
    capturePage: async () => materializeResult(await dom.window.polylogueCapture.capturePage()),
  };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) {
    dom.window.dispatchEvent(new dom.window.Event("pagehide"));
    dom.window.close();
  }
});

describe("ChatGPT authenticated interpreter bridge response contract", () => {
  it("cancels a pending capture clone without waiting for or cancelling the app response", async () => {
    const { dom } = installBridge({ fetch: vi.fn() });
    let producer;
    const original = new globalThis.Response(new globalThis.ReadableStream({ start(controller) { producer = controller; } }));
    const controller = new dom.window.AbortController();
    const capture = dom.window.polylogueAssetStream.stream(original.clone(), "pending-clone", controller.signal, { borrowedResponse: true });
    const refused = expect(capture).rejects.toMatchObject({ name: "AbortError" });
    await Promise.resolve();
    controller.abort(new dom.window.DOMException("capture_cancelled", "AbortError"));
    // This must settle while the original branch is still waiting for bytes.
    await refused;
    producer.enqueue(new TextEncoder().encode("original app bytes"));
    producer.close();
    expect(await original.text()).toBe("original app bytes");
  });

  it("does not use a bootstrap token or request assets after the auth endpoint returns429", async () => {
    const calls = [];
    const adapter = { fetch: vi.fn(async (input) => {
      const url = new URL(String(input)); calls.push(url.pathname);
      return new globalThis.Response("rate limited", { status: 429, headers: { "Retry-After": "172800" } });
    }) };
    const { requestAsset } = installBridge(adapter, bridgeSource, { bootstrapToken: bearerToken });
    expect(await requestAsset()).toMatchObject({ status: "rate_limited", http_status: 429, retry_after: "172800",
      response_url: "https://chatgpt.com/api/auth/session" });
    expect(calls).toEqual(["/api/auth/session"]);
  });

  it("refuses a clone chunk without waiting for the unfinished app branch", async () => {
    const { dom } = installBridge({ fetch: vi.fn() });
    let producer;
    const original = new globalThis.Response(new globalThis.ReadableStream({ start(controller) { producer = controller; } }),
      { headers: { "content-type": "application/json" } });
    const runtimeRequest = dom.window.chrome.runtime.sendMessage;
    dom.window.chrome.runtime.sendMessage = async (message) => message.type === "polylogue.asset.chunk"
      ? { ok: false, error: "capture_staging_write_failed" } : runtimeRequest(message);
    const capture = dom.window.polylogueAssetStream.stageResponse(original.clone(), "chatgpt", new dom.window.AbortController().signal);
    const refused = expect(capture).rejects.toThrow("capture_staging_write_failed");
    producer.enqueue(new TextEncoder().encode("original prefix "));
    await refused;
    producer.enqueue(new TextEncoder().encode("and suffix")); producer.close();
    expect(await original.text()).toBe("original prefix and suffix");
  });

  it("cancels one pending auth read without failing another capture's asset acquisition", async () => {
    const ordinary = syntheticEndpointAdapter();
    let count = 0; let ready; let releaseSecond;
    const started = new Promise((resolve) => { ready = resolve; });
    const adapter = { fetch: vi.fn(async (input, options) => {
      if (new URL(String(input)).pathname !== "/api/auth/session") return ordinary.fetch(input, options);
      count += 1;
      if (count === 1) return new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true }));
      ready();
      return new Promise((resolve) => { releaseSecond = () => resolve(jsonResponse({ accessToken: bearerToken, account: { id: chatGptAccountId } })); });
    }) };
    const { dom, posted, requestAsset } = installBridge(adapter);
    const first = requestAsset(); const second = requestAsset();
    await started;
    const firstRequest = posted.find((message) => message.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.assetFetchRequest");
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin,
      data: { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.cancelRequest", requestId: firstRequest.requestId } }));
    expect(await first).toMatchObject({ status: "cancelled" });
    releaseSecond();
    expect(await second).toMatchObject({ status: "acquired", asset: { staged_asset: { id: expect.any(String) } } });
    expect(count).toBe(2);
  });
  it.each([
    {
      name: "missing access token",
      adapter: { authStatus: 401, authBody: { detail: "Unauthorized" } },
      expected: { status: "unauthorized", phase: "access_token", detail: "access_token_unavailable" },
    },
    {
      name: "provider-reported expired pod",
      adapter: { metadataBody: { detail: "ace_pod_expired" } },
      expected: { status: "pod_expired", phase: "metadata", detail: "ace_pod_expired", http_status: 200 },
    },
    {
      name: "provider-reported expired pod on a generic 403",
      adapter: { metadataStatus: 403, metadataBody: { detail: "ace_pod_expired" } },
      expected: { status: "pod_expired", phase: "metadata", detail: "ace_pod_expired", http_status: 403 },
    },
    {
      name: "interpreter file missing",
      adapter: { metadataStatus: 404, metadataBody: { detail: "Interpreter file not found" } },
      expected: {
        status: "missing",
        phase: "metadata",
        detail: "interpreter_file_not_found",
        http_status: 404,
      },
    },
    {
      name: "expired signed URL",
      adapter: { signedStatus: 403 },
      expected: {
        status: "signed_url_expired",
        phase: "signed_bytes",
        detail: "signed_url_http_403",
        http_status: 403,
      },
    },
  ])("classifies $name without collapsing it into a generic HTTP error", async ({ adapter, expected }) => {
    const harness = installBridge(syntheticEndpointAdapter(adapter));

    await expect(harness.requestAsset()).resolves.toMatchObject(expected);
  });

  it("acquires signed bytes with a deterministic SHA-256 and no credential disclosure", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installBridge(adapter);

    const first = await harness.requestAsset();
    const second = await harness.requestAsset();

    expect(first).toMatchObject({
      status: "acquired",
      phase: "complete",
      asset: {
        size_bytes: assetBytes.byteLength,
        sha256: expectedSha256,
        mime_type: "application/zip",
        name: "kit.zip",
      },
    });
    expect(second.asset.sha256).toBe(first.asset.sha256);
    expect(Buffer.from(await attachmentBytes(harness.dom.__captureRuntime.staging, first.asset))).toEqual(Buffer.from(assetBytes));
    const metadataCalls = adapter.calls.filter((call) => call.url.pathname.endsWith("/interpreter/download"));
    const signedCalls = adapter.calls.filter((call) => call.url.origin === "https://files.example.test");
    const authCalls = adapter.calls.filter((call) => call.url.pathname === "/api/auth/session");
    expect(metadataCalls).toHaveLength(2);
    expect(authorizationHeader(metadataCalls[0].options)).toBe(`Bearer ${bearerToken}`);
    expect(metadataCalls[0].options.credentials).toBe("include");
    expect(new globalThis.Headers(metadataCalls[0].options.headers).get("ChatGPT-Account-Id")).toBe(chatGptAccountId);
    expect(metadataCalls[0].url.searchParams.get("message_id")).toBe("assistant-message-1");
    expect(metadataCalls[0].url.searchParams.get("sandbox_path")).toBe("/mnt/data/kit.zip");
    expect(signedCalls).toHaveLength(2);
    expect(authorizationHeader(signedCalls[0].options)).toBe(null);
    expect(signedCalls[0].options.credentials).toBe("omit");
    expect(authCalls).toHaveLength(1);
    expect(authCalls[0].options.credentials).toBe("include");
    expect(authorizationHeader(authCalls[0].options)).toBe(null);
    const disclosed = JSON.stringify(harness.posted);
    expect(disclosed).not.toContain(bearerToken);
    expect(disclosed).not.toContain(chatGptAccountId);
    expect(disclosed).not.toContain("synthetic-signed-secret");
  });

  it("keeps cookies for same-origin estuary bytes without forwarding the bearer", async () => {
    const estuaryUrl = "https://chatgpt.com/backend-api/estuary/content?download=synthetic-secret";
    const adapter = syntheticEndpointAdapter({ signedDownloadUrl: estuaryUrl });
    const harness = installBridge(adapter);

    await expect(harness.requestAsset()).resolves.toMatchObject({
      status: "acquired",
      asset: { sha256: expectedSha256 },
    });
    const byteCall = adapter.calls.find((call) => call.url.href === estuaryUrl);
    expect(byteCall.options.credentials).toBe("include");
    expect(authorizationHeader(byteCall.options)).toBe(null);
    const disclosed = JSON.stringify(harness.posted);
    expect(disclosed).not.toContain(bearerToken);
    expect(disclosed).not.toContain("synthetic-secret");
  });

  it("prefers the current session token over a stale legacy bootstrap token", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installBridge(adapter, bridgeSource, { bootstrapToken: "stale-bootstrap-token" });

    await expect(harness.requestAsset()).resolves.toMatchObject({ status: "acquired" });
    const metadataCall = adapter.calls.find((call) => call.url.pathname.endsWith("/interpreter/download"));
    expect(authorizationHeader(metadataCall.options)).toBe(`Bearer ${bearerToken}`);
    expect(adapter.calls.filter((call) => call.url.pathname === "/api/auth/session")).toHaveLength(1);
  });

  it("falls back to the trusted bootstrap token when the session endpoint has none", async () => {
    const adapter = syntheticEndpointAdapter({ authStatus: 401, authBody: { detail: "Unauthorized" } });
    const harness = installBridge(adapter, bridgeSource, { bootstrapToken: bearerToken });

    await expect(harness.requestAsset()).resolves.toMatchObject({ status: "acquired" });
    const metadataCall = adapter.calls.find((call) => call.url.pathname.endsWith("/interpreter/download"));
    expect(authorizationHeader(metadataCall.options)).toBe(`Bearer ${bearerToken}`);
  });

  it("acquires the actual complete asset despite a larger declared Content-Length", async () => {
    const harness = installBridge(syntheticEndpointAdapter({ declaredSize: 2048 }));
    const result = await harness.requestAsset();
    expect(result.status).toBe("acquired");
    expect(Buffer.from(await attachmentBytes(harness.dom.__captureRuntime.staging, result.asset))).toEqual(Buffer.from(assetBytes));
  });


  it("fetches bytes directly from a DOM-rendered https URL with no metadata round trip", async () => {
    const domUrl = "https://files.example.test/dom/photo.png";
    const domBytes = new TextEncoder().encode("dom rendered photo bytes\n");
    const domSha256 = createHash("sha256").update(domBytes).digest("hex");
    const calls = [];
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      calls.push({ url });
      if (url.href === domUrl) return byteResponse(domBytes);
      throw new Error(`unexpected synthetic request: ${url.href}`);
    });
    const harness = installBridge({ calls, fetch });

    const outcome = await harness.requestAsset({ kind: "url", url: domUrl, name: "photo.png" });

    expect(outcome).toMatchObject({
      status: "acquired",
      phase: "complete",
      asset: {
        size_bytes: domBytes.byteLength,
        sha256: domSha256,
        mime_type: "application/zip",
        name: "photo.png",
      },
    });
    expect(Buffer.from(await attachmentBytes(harness.dom.__captureRuntime.staging, outcome.asset))).toEqual(Buffer.from(domBytes));
    // No auth/metadata/backend-api round trip at all -- straight to the URL.
    expect(calls).toHaveLength(1);
    expect(calls[0].url.href).toBe(domUrl);
  });

  it("never forwards page cookies to a cross-origin direct URL", async () => {
    const domUrl = "https://files.example.test/dom/photo.png";
    const domBytes = new TextEncoder().encode("cross-origin bytes\n");
    let capturedOptions = null;
    const fetch = vi.fn(async (input, options = {}) => {
      capturedOptions = options;
      return byteResponse(domBytes);
    });
    const harness = installBridge({ fetch });

    const outcome = await harness.requestAsset({ kind: "url", url: domUrl });

    expect(outcome.status).toBe("acquired");
    expect(capturedOptions.credentials).toBe("omit");
  });

  it("keeps page cookies for a same-origin direct URL", async () => {
    const sameOriginUrl = "https://chatgpt.com/dom-asset/photo.png";
    const domBytes = new TextEncoder().encode("same-origin dom bytes\n");
    let capturedOptions = null;
    const fetch = vi.fn(async (input, options = {}) => {
      capturedOptions = options;
      return byteResponse(domBytes);
    });
    const harness = installBridge({ fetch });

    const outcome = await harness.requestAsset({ kind: "url", url: sameOriginUrl });

    expect(outcome.status).toBe("acquired");
    expect(capturedOptions.credentials).toBe("include");
  });

  it("rejects a non-https direct URL without attempting a fetch", async () => {
    const fetch = vi.fn();
    const harness = installBridge({ fetch });

    await expect(
      harness.requestAsset({ kind: "url", url: "http://insecure.example.test/x.png" }),
    ).resolves.toMatchObject({ status: "invalid_request", phase: "request", detail: "url_not_https" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([true, false])("streams a complete direct URL asset with declared Content-Length=%s", async (declared) => {
    const domUrl = "https://files.example.test/dom/large.png";
    const bytes = new Uint8Array(2 * 1024 * 1024); bytes.fill(17);
    const harness = installBridge({ fetch: vi.fn(async () => byteResponse(bytes, 200, declared ? bytes.length : null)) });
    const result = await harness.requestAsset({ kind: "url", url: domUrl });
    expect(result.status).toBe("acquired");
    expect(result.asset.size_bytes).toBe(bytes.length);
    expect(result.asset.sha256).toBe(createHash("sha256").update(bytes).digest("hex"));
    expect(Buffer.from(await attachmentBytes(harness.dom.__captureRuntime.staging, result.asset))).toEqual(Buffer.from(bytes));
  }, 30000);

  it("acquires a direct URL body streamed without a declared Content-Length", async () => {
    const domUrl = "https://files.example.test/dom/chunked-ok.png";
    const bodyBytes = new TextEncoder().encode("streamed without content-length\n");
    const expectedSha256 = createHash("sha256").update(bodyBytes).digest("hex");
    const fetch = vi.fn(async () => byteResponse(bodyBytes));
    const harness = installBridge({ fetch });

    const outcome = await harness.requestAsset({ kind: "url", url: domUrl });

    expect(outcome).toMatchObject({ status: "acquired", asset: { size_bytes: bodyBytes.byteLength, sha256: expectedSha256 } });
    expect(Buffer.from(await attachmentBytes(harness.dom.__captureRuntime.staging, outcome.asset))).toEqual(Buffer.from(bodyBytes));
  });
});

describe("ChatGPT native coverage that replaced chatgpt-dom-v1", () => {
  it("captures a temporary chat from its intercepted native fetch despite no /c/<id> URL", async () => {
    // A temporary chat never navigates to /c/<id> -- the visible URL stays
    // at /?temporary-chat=true for the whole session -- but the page still
    // fetches /backend-api/conversation/<ephemeral-id> to render it, and
    // window.fetch is intercepted the same way for any conversation path.
    // Before this fix every id-matching step (parseNativeCapture,
    // fetchNativePayloadOnDemand, and background.js's own
    // conversationIdForUrl gate) treated "no id in the URL" as "no
    // conversation on this page", so zero temporary chats ever landed.
    const ephemeralId = "temp-conv-ephemeral-1";
    const harness = installFullCapture(syntheticEndpointAdapter(), { url: "https://chatgpt.com/?temporary-chat=true", summary: { session_kind: "temporary" } });

    const sourceUrl = `https://chatgpt.com/backend-api/conversation/${ephemeralId}`;
    const bodyRef = await harness.dom.window.polylogueAssetStream.stageResponse(new globalThis.Response(JSON.stringify({
              id: ephemeralId,
              conversation_id: ephemeralId,
              is_temporary: true,
              title: "Temporary chat",
              mapping: {
                node: {
                  id: "node",
                  parent: null,
                  message: { id: "message", author: { role: "assistant" }, content: { content_type: "text", parts: ["hello"] } },
                },
              },
            }), { headers: { "content-type": "application/json" } }), "chatgpt", new globalThis.AbortController().signal, sourceUrl);

    // Simulate what chatgpt_bridge.js's window.fetch override posts when it
    // intercepts the page's own render fetch -- that interception matches
    // on path shape alone and does not care what the visible URL is.
    harness.dom.window.dispatchEvent(
      new harness.dom.window.MessageEvent("message", {
        source: harness.dom.window,
        origin: harness.dom.window.location.origin,
        data: {
          type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture",
          capture: {
            ok: true,
            status: 200,
            contentType: "application/json",
            url: `https://chatgpt.com/backend-api/conversation/${ephemeralId}`,
            bodyRef,
          },
        },
      }),
    );

    const result = await harness.sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({
      ok: true,
      envelope: {
        session: {
          provider_session_id: ephemeralId,
          session_kind: "temporary",
          turns: [],
        },
      },
    });
  });

  it("waits for a brand-new conversation's URL instead of giving up immediately", async () => {
    // 2026-07-17: every chatgpt-dom-v1 capture that ever fired for real
    // landed inside one narrow window where a capture was triggered before
    // ChatGPT's SPA router had published /c/<id> for a just-created
    // conversation -- there was no id to fetch by yet. Native must now wait
    // for the id instead of falling back to a DOM scrape (removed).
    const freshId = "fresh-conversation-1";
    const fetch = vi.fn(async (input, options = {}) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/api/auth/session") return jsonResponse({ accessToken: bearerToken, account: { id: chatGptAccountId } });
      if (url.pathname === `/backend-api/conversation/${freshId}`) {
        if (authorizationHeader(options) !== `Bearer ${bearerToken}`) return jsonResponse({ detail: "Unauthorized" }, 401);
        return jsonResponse({
          id: freshId,
          conversation_id: freshId,
          mapping: {
            node: {
              id: "node",
              parent: null,
              message: { id: "message", author: { role: "user" }, content: { content_type: "text", parts: ["first turn"] } },
            },
          },
        });
      }
      throw new Error(`unexpected synthetic request: ${url.pathname}`);
    });
    const harness = installFullCapture({ fetch }, { url: "https://chatgpt.com/" });

    const resultPromise = harness.sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });
    // The SPA router publishes the id a little after the first turn is
    // sent -- well inside the wait budget, but after at least one poll tick.
    await new Promise((resolve) => globalThis.setTimeout(resolve, 400));
    harness.dom.reconfigure({ url: `https://chatgpt.com/c/${freshId}` });

    const result = await resultPromise;

    expect(result).toMatchObject({
      ok: true,
      envelope: { session: { provider_session_id: freshId, turns: [] } },
    });
  });
});

describe("ChatGPT authenticated asset capture envelope", () => {
  it("captures an exact conversation and its output bytes from a reusable transport page", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter, { url: "https://chatgpt.com/", plan: [sandboxPlan()] });

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "completion_monitor",
      providerSessionId: "conversation-1",
    });

    expect(result).toMatchObject({ ok: true, envelope: { session: { provider_session_id: "conversation-1", turns: [] } } });
    const receipt = harness.dom.__captureRuntime.nativeContract.receipts[0].result;
    expect(receipt.outcome).toMatchObject({ acquired: 1 });
    expect(receipt.attachments[0]).toMatchObject({ name: "kit.zip", provider_meta: { content_sha256: expectedSha256 } });
    expect(Buffer.from(await attachmentBytes(harness.dom.__captureRuntime.staging, receipt.attachments[0]))).toEqual(Buffer.from(assetBytes));
    expect(harness.dom.window.location.pathname).toBe("/");
  });

  it.each(["wrong-native-id", "navigation"])("refuses displaced reusable transport evidence: %s", async (cause) => {
    let release;
    let entered;
    const waiting = new Promise((resolve) => { entered = resolve; });
    const gate = new Promise((resolve) => { release = resolve; });
    const adapter = syntheticEndpointAdapter();
    const fetch = async (input, options) => {
      if (new URL(String(input)).pathname === "/backend-api/conversation/conversation-1") {
        entered();
        await gate;
        const payload = conversationPayload();
        if (cause === "wrong-native-id") payload.id = payload.conversation_id = "conversation-foreign";
        return jsonResponse(payload);
      }
      return adapter.fetch(input, options);
    };
    const harness = installFullCapture({ fetch }, { url: "https://chatgpt.com/" });
    const pending = harness.sendRuntimeMessage({ type: "polylogue.capturePage", providerSessionId: "conversation-1" });
    await waiting;
    if (cause === "navigation") harness.dom.reconfigure({ url: "https://chatgpt.com/c/conversation-foreign" });
    release();
    expect(await pending).toMatchObject({ ok: false, error: "native_capture_unavailable" });
    expect(harness.runtimeMessages.filter((message) => message.type === "polylogue.capture")).toEqual([]);
    expect(adapter.calls.some((call) => call.url.pathname.endsWith("/interpreter/download"))).toBe(false);
  });

  it("debounces full transcript freshness scans across streamed DOM mutations", async () => {
    let textReads = 0;
    const harness = installFullCapture(syntheticEndpointAdapter(), {
      beforeInstall(document) {
        for (let index = 0; index < 12; index += 1) {
          const turn = document.createElement("article");
          turn.setAttribute("data-message-id", `message-${index}`);
          turn.textContent = `turn ${index}`;
          Object.defineProperty(turn, "innerText", {
            configurable: true,
            get() {
              textReads += 1;
              return turn.textContent;
            },
          });
          document.body.appendChild(turn);
        }
      },
    });
    const baselineReads = textReads;
    const turns = [...harness.dom.window.document.querySelectorAll("article")];

    for (let index = 0; index < 6; index += 1) {
      turns[index].firstChild.data += ` streamed-${index}`;
      await Promise.resolve();
    }
    expect(textReads).toBe(baselineReads);

    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 800));
    expect(textReads).toBe(baselineReads + turns.length);
  });

  // Anti-vacuity: let a MAIN-world capture supply its own identity again and
  // the forged "conversation-foreign" wake reaches the background.
  it("wakes capture only for the conversation the tab URL names, never a page-supplied one", async () => {
    const harness = installFullCapture(syntheticEndpointAdapter());
    for (const conversationId of ["conversation-1", "conversation-foreign"]) {
      const sourceUrl = `https://chatgpt.com/backend-api/conversation/${conversationId}`;
      const bodyRef = await harness.dom.window.polylogueAssetStream.stageResponse(new globalThis.Response(JSON.stringify({ conversation_id: conversationId, mapping: {}, update_time: 1781366460 }), { headers: { "content-type": "application/json" } }), "chatgpt", new globalThis.AbortController().signal, sourceUrl);
      harness.dom.window.postMessage({
        type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture",
        capture: {
          ok: true,
          bodyRef,
          url: sourceUrl,
        },
      });
    }

    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 800));
    const hints = harness.runtimeMessages.filter((message) => message.type === "polylogue.captureFreshnessHint");
    expect(hints.map((message) => message.provider_session_id)).toEqual(["conversation-1"]);
  });

  // Anti-vacuity: require a URL-named id unconditionally and a temporary
  // chat's later turns never wake a recapture.
  it("wakes capture on a temporary-chat page only for a payload that declares itself temporary", async () => {
    const harness = installFullCapture(syntheticEndpointAdapter(), { url: "https://chatgpt.com/?temporary-chat=true", summary: { session_kind: "temporary" } });
    for (const [conversationId, isTemporary] of [["ephemeral-1", true], ["conversation-foreign", false]]) {
      const sourceUrl = `https://chatgpt.com/backend-api/conversation/${conversationId}`;
      const bodyRef = await harness.dom.window.polylogueAssetStream.stageResponse(new globalThis.Response(JSON.stringify({ conversation_id: conversationId, mapping: {}, is_temporary: isTemporary, update_time: 1781366460 }), { headers: { "content-type": "application/json" } }), "chatgpt", new globalThis.AbortController().signal, sourceUrl);
      harness.dom.window.postMessage({
        type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture",
        capture: {
          ok: true,
          bodyRef,
          url: sourceUrl,
        },
      });
    }

    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 800));
    const hints = harness.runtimeMessages.filter((message) => message.type === "polylogue.captureFreshnessHint");
    expect(hints.map((message) => message.provider_session_id)).toEqual(["ephemeral-1"]);
  });

  it("captures typed live generation start and terminal UI timing before native reconciliation", async () => {
    const harness = installFullCapture(syntheticEndpointAdapter());
    const turn = harness.dom.window.document.createElement("section");
    turn.setAttribute("data-testid", "conversation-turn-2");
    turn.setAttribute("data-turn", "assistant");
    turn.setAttribute("data-turn-id", "assistant-turn-2");
    const stop = harness.dom.window.document.createElement("button");
    stop.setAttribute("data-testid", "stop-button");
    stop.textContent = "Stop";
    turn.appendChild(stop);
    harness.dom.window.document.body.appendChild(turn);

    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 1600));
    const started = harness.runtimeMessages.find((message) =>
      message.type === "polylogue.captureFreshnessHint"
      && message.generation_observations?.some((observation) => observation.state === "started")
    );
    expect(started).toMatchObject({
      reason: "generation_started",
      provider_session_id: "conversation-1",
      delay_ms: 1000,
      generation_observations: [{
        state: "started",
        evidence_source: "dom_control",
        fidelity: "observed",
        duration_semantics: "dom_observed_wall",
        turn_provider_id: "assistant-turn-2",
      }],
    });

    stop.remove();
    const workedFor = harness.dom.window.document.createElement("button");
    workedFor.textContent = "Worked for 86m 30s";
    turn.appendChild(workedFor);

    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 1600));
    const completed = harness.runtimeMessages.findLast((message) =>
      message.type === "polylogue.captureFreshnessHint"
      && message.generation_observations?.some((observation) => observation.state === "completed")
    );
    expect(completed).toMatchObject({
      reason: "generation_completed",
      delay_ms: 0,
      generation_observations: [{
        state: "completed",
        evidence_source: "dom_duration_control",
        fidelity: "observed",
        duration_semantics: "provider_ui_elapsed",
        displayed_elapsed_ms: 5_190_000,
        raw_label: "Worked for 86m 30s",
        turn_provider_id: "assistant-turn-2",
      }],
    });
  });

  it("does not attribute an older Worked-for control to a newly completed turn", async () => {
    const harness = installFullCapture(syntheticEndpointAdapter());
    const priorTurn = harness.dom.window.document.createElement("section");
    priorTurn.setAttribute("data-testid", "conversation-turn-2");
    priorTurn.setAttribute("data-turn", "assistant");
    priorTurn.setAttribute("data-turn-id", "prior-assistant-turn");
    const priorWorkedFor = harness.dom.window.document.createElement("button");
    priorWorkedFor.textContent = "Worked for 4m 10s";
    priorTurn.appendChild(priorWorkedFor);
    harness.dom.window.document.body.appendChild(priorTurn);

    const activeTurn = harness.dom.window.document.createElement("section");
    activeTurn.setAttribute("data-testid", "conversation-turn-4");
    activeTurn.setAttribute("data-turn", "assistant");
    activeTurn.setAttribute("data-turn-id", "active-assistant-turn");
    const stop = harness.dom.window.document.createElement("button");
    stop.setAttribute("data-testid", "stop-button");
    activeTurn.appendChild(stop);
    harness.dom.window.document.body.appendChild(activeTurn);

    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 1600));
    stop.remove();
    activeTurn.appendChild(harness.dom.window.document.createTextNode("Finished"));
    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 1600));

    const completed = harness.runtimeMessages.findLast((message) =>
      message.type === "polylogue.captureFreshnessHint"
      && message.generation_observations?.some((observation) =>
        observation.state === "completed"
        && observation.turn_provider_id === "active-assistant-turn"
      )
    );
    expect(completed).toMatchObject({
      reason: "generation_completed",
      generation_observations: [{
        state: "completed",
        evidence_source: "dom_control_transition",
        fidelity: "inferred",
        duration_semantics: "dom_observed_wall",
        turn_provider_id: "active-assistant-turn",
        displayed_elapsed_ms: null,
        raw_label: null,
      }],
    });
  });

  it("reuses supplied native detail without a second conversation read", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter, { url: "https://chatgpt.com/c/conversation-1", plan: [sandboxPlan()] });

    const sourceUrl = "https://chatgpt.com/backend-api/conversation/conversation-1";
    const payload = conversationPayload();
    payload.mapping["assistant-node"].message.status = "finished_successfully";
    const bodyRef = await harness.dom.window.polylogueAssetStream.stageResponse(
      new globalThis.Response(JSON.stringify(payload), { headers: { "content-type": "application/json" } }),
      "chatgpt", new globalThis.AbortController().signal, sourceUrl,
    );
    harness.dom.window.postMessage({ type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture", capture: { ok: true, bodyRef, url: sourceUrl } });
    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "freshness_convergence",
      providerUpdatedAt: "2026-06-13T00:01:00.000Z",
      providerSessionId: "conversation-1",
    });

    expect(result).toMatchObject({
      ok: true,
      envelope: { session: { provider_session_id: "conversation-1" } },
    });
    expect(
      adapter.calls.filter((call) => call.url.pathname === "/backend-api/conversation/conversation-1"),
    ).toHaveLength(0);
    expect(harness.dom.__captureRuntime.nativeContract.receipts[0].result.attachments).toHaveLength(1);
  });

  it.each([false, true])("uses canonical follow-up summary %s to decide cached revision reuse", async (needsFollowUp) => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter, { url: "https://chatgpt.com/c/conversation-1", summary: { needs_follow_up: needsFollowUp } });
    const cachedPayload = {
      ...conversationPayload(),
      update_time: 1781366460,
      mapping: {
        "assistant-node": {
          ...conversationPayload().mapping["assistant-node"],
          message: {
            ...conversationPayload().mapping["assistant-node"].message,
            status: needsFollowUp ? "in_progress" : "finished_successfully",
          },
        },
      },
    };

    harness.dom.window.postMessage({
      type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture",
      capture: {
        ok: true,
        status: 200,
        contentType: "application/json",
        url: "https://chatgpt.com/backend-api/conversation/conversation-1",
        bodyRef: await harness.dom.window.polylogueAssetStream.stageResponse(
          new globalThis.Response(JSON.stringify(cachedPayload), { headers: { "content-type": "application/json" } }),
          "chatgpt", new globalThis.AbortController().signal, "https://chatgpt.com/backend-api/conversation/conversation-1",
        ),
      },
    });
    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 0));

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "freshness_convergence",
      providerSessionId: "conversation-1",
      providerUpdatedAt: "2026-06-13T00:01:00.000Z",
    });

    expect(result).toMatchObject({ ok: true, envelope: { session: { provider_session_id: "conversation-1" } } });
    expect(adapter.calls.filter((call) => call.url.pathname === "/backend-api/conversation/conversation-1")).toHaveLength(needsFollowUp ? 1 : 0);
  });

  it("fetches when freshness claims a revision newer than the intercepted payload", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter, { url: "https://chatgpt.com/c/conversation-1" });
    const cachedPayload = {
      ...conversationPayload(),
      update_time: 1781366460,
      current_node: "assistant-node",
      mapping: {
        "assistant-node": {
          ...conversationPayload().mapping["assistant-node"],
          message: {
            ...conversationPayload().mapping["assistant-node"].message,
            status: "finished_successfully",
          },
        },
      },
    };
    harness.dom.window.postMessage({
      type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture",
      capture: {
        ok: true,
        status: 200,
        contentType: "application/json",
        url: "https://chatgpt.com/backend-api/conversation/conversation-1",
        bodyRef: await harness.dom.window.polylogueAssetStream.stageResponse(
          new globalThis.Response(JSON.stringify(cachedPayload), { headers: { "content-type": "application/json" } }),
          "chatgpt", new globalThis.AbortController().signal, "https://chatgpt.com/backend-api/conversation/conversation-1",
        ),
      },
    });
    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 0));

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "freshness_convergence",
      providerSessionId: "conversation-1",
      providerUpdatedAt: "2026-06-13T17:02:00.000Z",
    });

    expect(result.ok).toBe(true);
    expect(adapter.calls.filter((call) => call.url.pathname === "/backend-api/conversation/conversation-1")).toHaveLength(1);
  });

  it("carries background-observed lifecycle evidence into the exact native envelope", async () => {
    const harness = installFullCapture(syntheticEndpointAdapter(), { url: "https://chatgpt.com/" });
    const observation = {
      observation_id: "conversation-1:assistant-turn-2:completed:worked-for",
      state: "completed",
      observed_at: "2026-07-16T01:26:30Z",
      evidence_source: "dom_duration_control",
      fidelity: "observed",
      displayed_elapsed_ms: 5_190_000,
    };

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "freshness_convergence",
      providerSessionId: "conversation-1",
      generationObservations: [observation],
    });

    expect(result.envelope.session.provider_meta.generation_observations).toEqual([observation]);
  });

  it("prefers fresh native detail over an intercepted page-load payload", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter);
    const stalePayload = {
      ...conversationPayload(),
      title: "Stale page-load title",
      update_time: 1781366401,
      current_node: "stale-node",
      mapping: {
        "stale-node": {
          id: "stale-node",
          parent: null,
          children: [],
          message: {
            id: "stale-message",
            author: { role: "user" },
            create_time: 1781366401,
            content: { content_type: "text", parts: ["opening prompt only"] },
            metadata: { model_slug: "gpt-test" },
          },
        },
      },
    };
    harness.dom.window.dispatchEvent(
      new harness.dom.window.MessageEvent("message", {
        source: harness.dom.window,
        origin: harness.dom.window.location.origin,
        data: {
          type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture",
          capture: {
            ok: true,
            status: 200,
            contentType: "application/json",
            url: "https://chatgpt.com/backend-api/conversation/conversation-1",
            bodyRef: await harness.dom.window.polylogueAssetStream.stageResponse(
              new globalThis.Response(JSON.stringify(stalePayload), { headers: { "content-type": "application/json" } }),
              "chatgpt", new globalThis.AbortController().signal, "https://chatgpt.com/backend-api/conversation/conversation-1",
            ),
          },
        },
      }),
    );

    const result = await harness.capturePage();

    expect((await harness.dom.__captureRuntime.retainedNativeReplies(result.envelope)).title).toBe("Authenticated capture fixture");
    expect((await harness.dom.__captureRuntime.retainedNativeReplies(result.envelope)).mapping["assistant-node"].message.id).toBe("assistant-message-1");
    expect(
      adapter.calls.filter((call) => call.url.pathname === "/backend-api/conversation/conversation-1"),
    ).toHaveLength(1);
  });

  it("records stable attachment identity, bytes, size, and SHA receipt across repeat capture", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter, { plan: [sandboxPlan()] });

    const result = await harness.capturePage();
    const repeated = await harness.capturePage();

    expect(result.ok).toBe(true);
    expect(repeated.ok).toBe(true);
    expect(repeated.envelope.session).toEqual(result.envelope.session);
    const [attachment] = harness.dom.__captureRuntime.nativeContract.receipts[0].result.attachments;
    expect(attachment).toMatchObject({
      provider_attachment_id: "sandbox:assistant-message-1:/mnt/data/kit.zip",
      message_provider_id: "assistant-message-1",
      name: "kit.zip",
      size_bytes: assetBytes.byteLength,
      provider_meta: {
        capture_source: "chatgpt_page_asset_fetch",
        asset_kind: "sandbox_file",
        sandbox_path: "/mnt/data/kit.zip",
        content_sha256: expectedSha256,
      },
    });
    expect(harness.dom.__captureRuntime.nativeContract.receipts[0].result.outcome).toMatchObject({
      attempted: 1,
      acquired: 1,
      status_counts: { acquired: 1 },
      failed: [],
    });
    const captureMessages = harness.runtimeMessages.filter((message) => message.type === "polylogue.capture");
    expect(captureMessages).toHaveLength(2);
    expect(captureMessages[0].envelope).toEqual(result.envelope);
    expect(captureMessages[1].envelope).toEqual(repeated.envelope);
    const durablePayload = JSON.stringify({ envelope: result.envelope, posted: harness.posted });
    expect(durablePayload).not.toContain(bearerToken);
    expect(durablePayload).not.toContain("synthetic-signed-secret");
  });

  it("attempts every asset and retains the selected branch bytes after independent stale-asset refusals", async () => {
    const targetPath = "/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit.zip";
    const stalePaths = ["/mnt/data/stale-1.zip", "/mnt/data/stale-2.zip", "/mnt/data/stale-3.zip"];
    const mapping = {};
    for (const [index, sandboxPath] of stalePaths.entries()) {
      const nodeId = `stale-node-${index + 1}`;
      mapping[nodeId] = {
        id: nodeId,
        parent: null,
        children: [],
        message: {
          id: `stale-message-${index + 1}`,
          author: { role: "assistant" },
          create_time: 1781366400 + index,
          content: { content_type: "text", parts: [`Old: sandbox:${sandboxPath}`] },
          metadata: { model_slug: "gpt-test" },
        },
      };
    }
    mapping["current-node"] = {
      id: "current-node",
      parent: null,
      children: [],
      message: {
        id: "current-message",
        author: { role: "assistant" },
        create_time: 1781366460,
        content: { content_type: "text", parts: [`Current: sandbox:${targetPath}`] },
        metadata: { model_slug: "gpt-test" },
      },
    };
    const payload = {
      ...conversationPayload(),
      current_node: "current-node",
      mapping,
    };
    const calls = [];
    const adapter = {
      calls,
      fetch: vi.fn(async (input, options = {}) => {
        const url = new URL(String(input), "https://chatgpt.com");
        calls.push({ url, options });
        if (url.pathname === "/api/auth/session") return jsonResponse({ accessToken: bearerToken });
        if (url.pathname === "/backend-api/conversation/conversation-1") return jsonResponse(payload);
        if (url.pathname === "/backend-api/conversation/conversation-1/interpreter/download") {
          const sandboxPath = url.searchParams.get("sandbox_path");
          const downloadUrl =
            sandboxPath === targetPath
              ? signedUrl
              : `https://files.example.test/expired/${encodeURIComponent(sandboxPath)}`;
          return jsonResponse({ download_url: downloadUrl, file_name: sandboxPath.split("/").at(-1) });
        }
        if (url.href === signedUrl) return byteResponse(assetBytes);
        if (url.origin === "https://files.example.test" && url.pathname.startsWith("/expired/")) {
          return byteResponse(new Uint8Array(), 403);
        }
        throw new Error(`unexpected synthetic request: ${url.href}`);
      }),
    };
    const plan = [...stalePaths.map((path, ordinal) => sandboxPlan(path, `old-message-${ordinal}`, `old-node-${ordinal}`, ordinal)), sandboxPlan(targetPath, "current-message", "current-node", stalePaths.length)];
    const harness = installFullCapture(adapter, { plan });

    const result = await harness.capturePage();

    expect(result.ok).toBe(true);
    const receipts = harness.dom.__captureRuntime.nativeContract.receipts;
    expect(receipts.map(({ result }) => result.outcome.acquired)).toEqual([0, 0, 0, 1]);
    expect(receipts.map(({ result }) => result.outcome.failed[0]?.status || "acquired")).toEqual(["signed_url_expired", "signed_url_expired", "signed_url_expired", "acquired"]);
    expect(receipts.at(-1).result.attachments[0]).toMatchObject({ provider_attachment_id: `sandbox:current-message:${targetPath}`, provider_meta: { content_sha256: expectedSha256 } });
    expect(await harness.dom.__captureRuntime.retainedNativeReplies(result.envelope)).toEqual(payload);
    const metadataPaths = calls
      .filter((call) => call.url.pathname.endsWith("/interpreter/download"))
      .map((call) => call.url.searchParams.get("sandbox_path"));
    expect(metadataPaths).toEqual([...stalePaths, targetPath]);
  });
});
