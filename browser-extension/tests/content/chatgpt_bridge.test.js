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
const bridgeSource = readFileSync(resolve(testDirectory, "../../src/content/chatgpt_bridge.js"), "utf8");
const commonSource = readFileSync(resolve(testDirectory, "../../src/common.js"), "utf8");
const contentSource = readFileSync(resolve(testDirectory, "../../src/content/chatgpt.js"), "utf8");
const bearerToken = "synthetic-bearer-must-not-cross-the-page-bridge";
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
  authBody = { accessToken: bearerToken },
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

function makeDom(adapter) {
  const dom = new JSDOM("<!doctype html><title>ChatGPT fixture</title>", {
    url: "https://chatgpt.com/c/conversation-1",
    runScripts: "outside-only",
  });
  openDoms.push(dom);
  // Node 20 rejects ArrayBuffers created by a separate jsdom VM realm. Keep
  // the production dependency on Web Crypto, but adapt test bytes into the
  // host realm before invoking the real digest implementation.
  const cryptoAdapter = {
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
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(data) {
      posted.push(data);
      const resolve = pending.get(data?.requestId);
      if (data?.type === "polylogue.chatgpt.assetFetchResponse" && resolve) {
        pending.delete(data.requestId);
        resolve(data.outcome);
      }
    },
  });
  new Script(source).runInContext(dom.getInternalVMContext());

  function requestAsset(overrides = {}) {
    const requestId = `asset-request-${pending.size + 1}-${posted.length}`;
    const response = new Promise((resolve) => pending.set(requestId, resolve));
    dom.window.dispatchEvent(
      new dom.window.MessageEvent("message", {
        source: dom.window,
        origin: dom.window.location.origin,
        data: {
          type: "polylogue.chatgpt.assetFetchRequest",
          requestId,
          request: {
            kind: "sandbox",
            conversationId: "conversation-1",
            messageId: "assistant-message-1",
            sandboxPath: "/mnt/data/kit.zip",
            maxBytes: 1024,
            ...overrides,
          },
        },
      }),
    );
    return response;
  }

  return { dom, posted, requestAsset };
}

function installFullCapture(adapter) {
  const dom = makeDom(adapter);
  const posted = [];
  const runtimeMessages = [];
  const runtimeListeners = [];
  const chrome = {
    runtime: {
      id: "synthetic-extension-id",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
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
  new Script(bridgeSource).runInContext(context);
  new Script(commonSource).runInContext(context);
  new Script(contentSource).runInContext(context);
  return { dom, posted, runtimeListeners, runtimeMessages };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("ChatGPT authenticated interpreter bridge response contract", () => {
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
    expect(first.asset.base64).toBe(Buffer.from(assetBytes).toString("base64"));
    const metadataCalls = adapter.calls.filter((call) => call.url.pathname.endsWith("/interpreter/download"));
    const signedCalls = adapter.calls.filter((call) => call.url.origin === "https://files.example.test");
    const authCalls = adapter.calls.filter((call) => call.url.pathname === "/api/auth/session");
    expect(metadataCalls).toHaveLength(2);
    expect(authorizationHeader(metadataCalls[0].options)).toBe(`Bearer ${bearerToken}`);
    expect(metadataCalls[0].options.credentials).toBe("include");
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

  it("rejects a declared body over the per-request cap before publishing bytes", async () => {
    const harness = installBridge(syntheticEndpointAdapter({ declaredSize: 2048 }));

    await expect(harness.requestAsset({ maxBytes: 1024 })).resolves.toMatchObject({
      status: "too_large",
      phase: "signed_bytes",
      detail: "content_length_over_limit",
      size_bytes: 2048,
    });
  });

  it("reproduces the old unauthorized behavior when the production bearer dependency is removed", async () => {
    const bearerRequest = '{ credentials: "include", cache: "no-store", headers: bearerHeaders(accessToken) }';
    const cookieOnlyRequest = '{ credentials: "include", cache: "no-store", headers: {} }';
    const unauthenticatedSource = bridgeSource.replace(bearerRequest, cookieOnlyRequest);
    expect(unauthenticatedSource).not.toBe(bridgeSource);
    const authenticated = installBridge(syntheticEndpointAdapter());
    const unauthenticated = installBridge(syntheticEndpointAdapter(), unauthenticatedSource);

    await expect(authenticated.requestAsset()).resolves.toMatchObject({ status: "acquired" });
    await expect(unauthenticated.requestAsset()).resolves.toMatchObject({
      status: "unauthorized",
      phase: "metadata",
      http_status: 401,
    });
  });
});

describe("ChatGPT authenticated asset capture envelope", () => {
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
          type: "polylogue.chatgpt.nativeCapture",
          capture: {
            ok: true,
            status: 200,
            contentType: "application/json",
            url: "https://chatgpt.com/backend-api/conversation/conversation-1",
            body: JSON.stringify(stalePayload),
          },
        },
      }),
    );

    const result = await harness.dom.window.polylogueCapture.capturePage();

    expect(result.envelope.session.title).toBe("Authenticated capture fixture");
    expect(result.envelope.session.turns[0].provider_turn_id).toBe("assistant-message-1");
    expect(
      adapter.calls.filter((call) => call.url.pathname === "/backend-api/conversation/conversation-1"),
    ).toHaveLength(1);
  });

  it("records stable attachment identity, bytes, size, and SHA receipt across repeat capture", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter);

    const result = await harness.dom.window.polylogueCapture.capturePage();
    const repeated = await harness.dom.window.polylogueCapture.capturePage();

    expect(result.ok).toBe(true);
    expect(repeated.ok).toBe(true);
    expect(repeated.envelope.session).toEqual(result.envelope.session);
    const [attachment] = result.envelope.session.attachments;
    expect(attachment).toMatchObject({
      provider_attachment_id: "sandbox:assistant-message-1:/mnt/data/kit.zip",
      message_provider_id: "assistant-message-1",
      name: "kit.zip",
      size_bytes: assetBytes.byteLength,
      inline_base64: Buffer.from(assetBytes).toString("base64"),
      provider_meta: {
        capture_source: "chatgpt_page_asset_fetch",
        asset_kind: "sandbox",
        sandbox_path: "/mnt/data/kit.zip",
        content_sha256: expectedSha256,
      },
    });
    expect(result.envelope.session.provider_meta.asset_acquisition).toMatchObject({
      attempted: 1,
      acquired: 1,
      status_counts: { acquired: 1 },
      acquired_assets: [
        {
          provider_attachment_id: "sandbox:assistant-message-1:/mnt/data/kit.zip",
          sha256: expectedSha256,
          size_bytes: assetBytes.byteLength,
        },
      ],
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

  it("attempts the selected branch's newest asset before stale off-branch assets trip the breaker", async () => {
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
    const harness = installFullCapture(adapter);

    const result = await harness.dom.window.polylogueCapture.capturePage();

    expect(result.envelope.session.provider_meta.asset_acquisition).toMatchObject({
      attempted: 4,
      acquired: 1,
      skipped_circuit_breaker: 0,
      status_counts: { acquired: 1, signed_url_expired: 3 },
      acquired_assets: [
        {
          provider_attachment_id: `sandbox:current-message:${targetPath}`,
          sha256: expectedSha256,
          size_bytes: assetBytes.byteLength,
        },
      ],
    });
    expect(result.envelope.session.attachments[0]).toMatchObject({
      provider_attachment_id: `sandbox:current-message:${targetPath}`,
      provider_meta: { content_sha256: expectedSha256 },
    });
    const metadataPaths = calls
      .filter((call) => call.url.pathname.endsWith("/interpreter/download"))
      .map((call) => call.url.searchParams.get("sandbox_path"));
    expect(metadataPaths).toEqual([targetPath, ...stalePaths.slice().reverse()]);
  });
});
