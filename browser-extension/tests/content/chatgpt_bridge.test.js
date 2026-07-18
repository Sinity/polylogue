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

function installFullCapture(adapter, { url, beforeInstall } = {}) {
  const dom = makeDom(adapter, url);
  beforeInstall?.(dom.window.document);
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
  function sendRuntimeMessage(message) {
    return new Promise((resolve, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolve) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  return { dom, posted, runtimeListeners, runtimeMessages, sendRuntimeMessage };
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

describe("ChatGPT bridge direct-URL asset kind (polylogue-83u.3, chatgpt-dom-v1 gap)", () => {
  // chatgpt-dom-v1 has no backend-api mapping to resolve a file/sandbox id
  // from -- the DOM chip's own href/src is the only evidence available. The
  // "url" request kind skips the metadata round trip entirely and reuses the
  // exact same fetch+budget+hash tail as sandbox/file, proving the DOM
  // adapter's byte fetch is the SAME mechanism, not a second one.
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

    const outcome = await harness.requestAsset({ kind: "url", url: domUrl, name: "photo.png", maxBytes: 1024 });

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
    expect(outcome.asset.base64).toBe(Buffer.from(domBytes).toString("base64"));
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

    const outcome = await harness.requestAsset({ kind: "url", url: domUrl, maxBytes: 1024 });

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

    const outcome = await harness.requestAsset({ kind: "url", url: sameOriginUrl, maxBytes: 1024 });

    expect(outcome.status).toBe("acquired");
    expect(capturedOptions.credentials).toBe("include");
  });

  it("rejects a non-https direct URL without attempting a fetch", async () => {
    const fetch = vi.fn();
    const harness = installBridge({ fetch });

    await expect(
      harness.requestAsset({ kind: "url", url: "http://insecure.example.test/x.png", maxBytes: 1024 }),
    ).resolves.toMatchObject({ status: "invalid_request", phase: "request", detail: "url_not_https" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("enforces the byte cap on a direct URL the same way as sandbox/file kinds", async () => {
    const domUrl = "https://files.example.test/dom/huge.png";
    const fetch = vi.fn(async () => byteResponse(new Uint8Array(2048), 200, 2048));
    const harness = installBridge({ fetch });

    await expect(
      harness.requestAsset({ kind: "url", url: domUrl, maxBytes: 1024 }),
    ).resolves.toMatchObject({ status: "too_large", phase: "signed_bytes", detail: "content_length_over_limit" });
  });
});

describe("chatgpt-dom-v1 fallback capture attachment byte acquisition (polylogue-83u.3)", () => {
  // Deterministic capture smoke: forces the native backend-api read to fail
  // (both the page-bridge attempt and chatgpt.js's own content-script fetch
  // hit the same failing endpoint), so `capture()` falls through to the
  // chatgpt-dom-v1 DOM adapter. Before this change, a DOM-scraped attachment
  // only ever recorded its chip name with byte_count=0 -- there was no fetch
  // attempt at all. Proves a captured attachment now carries real bytes.
  function domFallbackAdapter({ domUrl, domBytes }) {
    const calls = [];
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      calls.push(url.href);
      if (url.pathname === "/api/auth/session") return jsonResponse({});
      if (url.pathname === "/backend-api/conversation/conversation-1") {
        return jsonResponse({ detail: "not_found" }, 404);
      }
      if (url.href === domUrl) return byteResponse(domBytes);
      throw new Error(`unexpected synthetic request in DOM-fallback fixture: ${url.href}`);
    });
    return { calls, fetch };
  }

  it("acquires a DOM-scraped attachment's bytes with a real blob-addressable SHA-256", async () => {
    const domUrl = "https://files.example.test/dom/deliverable.png";
    const domBytes = new TextEncoder().encode("chatgpt-dom-v1 fallback attachment bytes\n");
    const expectedDomSha256 = createHash("sha256").update(domBytes).digest("hex");
    const adapter = domFallbackAdapter({ domUrl, domBytes });
    const harness = installFullCapture(adapter, {
      beforeInstall(document) {
        const turn = document.createElement("article");
        turn.setAttribute("data-message-author-role", "assistant");
        turn.textContent = "Here is the file you asked for.";
        const image = document.createElement("img");
        image.setAttribute("src", domUrl);
        image.setAttribute("alt", "deliverable.png");
        turn.appendChild(image);
        document.body.appendChild(turn);
      },
    });

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "message_layer_save",
    });

    expect(result.ok).toBe(true);
    expect(result.envelope.provenance.adapter_name).toBe("chatgpt-dom-v1");
    expect(result.envelope.provenance.capture_mode).toBe("snapshot");
    const [turn] = result.envelope.session.turns;
    const [attachment] = turn.attachments;
    expect(attachment).toMatchObject({
      name: "deliverable.png",
      size_bytes: domBytes.byteLength,
      inline_base64: Buffer.from(domBytes).toString("base64"),
      provider_meta: { content_sha256: expectedDomSha256, asset_kind: "url" },
    });
    // The declared SHA-256 is the true hash of the delivered bytes -- the
    // exact invariant the archive-side blob store re-derives and persists as
    // acquisition_status='acquired' (polylogue/storage/sqlite/archive_tiers/write.py).
    expect(createHash("sha256").update(Buffer.from(attachment.inline_base64, "base64")).digest("hex")).toBe(
      expectedDomSha256,
    );
    expect(adapter.calls).toContain(domUrl);
  });

  it("leaves a DOM-scraped attachment honestly byte_count=0 when its chip has no fetchable URL", async () => {
    // A sandbox-output chip: the DOM only ever exposes the file name (via
    // aria-label), never a real href/src -- this is the genuinely-unfetchable
    // shape (byte_count=0 stays honest) distinct from the DOM-rendered <img>
    // case above, which the fetch above proves is now reachable.
    const adapter = domFallbackAdapter({ domUrl: "https://unused.example.test/none.png", domBytes: new Uint8Array() });
    const harness = installFullCapture(adapter, {
      beforeInstall(document) {
        const turn = document.createElement("article");
        turn.setAttribute("data-message-author-role", "assistant");
        turn.textContent = "Here is the file you asked for.";
        const chip = document.createElement("div");
        chip.setAttribute("role", "group");
        chip.setAttribute("aria-label", "report.pdf");
        turn.appendChild(chip);
        document.body.appendChild(turn);
      },
    });

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "message_layer_save",
    });

    expect(result.ok).toBe(true);
    const [turn] = result.envelope.session.turns;
    const [attachment] = turn.attachments;
    expect(attachment.name).toBe("report.pdf");
    expect(attachment.url).toBeNull();
    expect(attachment.inline_base64).toBeUndefined();
    expect(attachment.size_bytes).toBeUndefined();
    expect(adapter.calls).not.toContain("https://unused.example.test/none.png");
  });
});

describe("ChatGPT authenticated asset capture envelope", () => {
  it("captures an exact conversation and its output bytes from a reusable transport page", async () => {
    const adapter = syntheticEndpointAdapter();
    const harness = installFullCapture(adapter, { url: "https://chatgpt.com/" });

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "completion_monitor",
      providerSessionId: "conversation-1",
    });

    expect(result).toMatchObject({
      ok: true,
      envelope: {
        session: {
          provider_session_id: "conversation-1",
          attachments: [{
            name: "kit.zip",
            provider_meta: { content_sha256: expectedSha256 },
          }],
        },
      },
    });
    expect(result.envelope.session.provider_meta.asset_acquisition).toMatchObject({ acquired: 1 });
    expect(harness.dom.window.location.pathname).toBe("/");
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

  it("debounces native freshness hints independently per conversation", async () => {
    const harness = installFullCapture(syntheticEndpointAdapter());
    for (const conversationId of ["conversation-a", "conversation-b"]) {
      harness.dom.window.postMessage({
        type: "polylogue.chatgpt.nativeCapture",
        capture: {
          ok: true,
          body: JSON.stringify({ conversation_id: conversationId, update_time: 1781366460 }),
        },
      });
    }

    await new Promise((resolve) => harness.dom.window.setTimeout(resolve, 800));
    const hints = harness.runtimeMessages.filter((message) => message.type === "polylogue.captureFreshnessHint");
    expect(hints.map((message) => message.provider_session_id).sort()).toEqual([
      "conversation-a",
      "conversation-b",
    ]);
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
    const harness = installFullCapture(adapter, { url: "https://chatgpt.com/" });

    const result = await harness.sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "completion_monitor",
      providerSessionId: "conversation-1",
      nativePayload: conversationPayload(),
    });

    expect(result).toMatchObject({
      ok: true,
      envelope: { session: { provider_session_id: "conversation-1" } },
    });
    expect(
      adapter.calls.filter((call) => call.url.pathname === "/backend-api/conversation/conversation-1"),
    ).toHaveLength(0);
    expect(result.envelope.session.attachments).toHaveLength(1);
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
      nativePayload: conversationPayload(),
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
