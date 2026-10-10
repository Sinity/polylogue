import { captureProvider, proofFailureReport } from "../../scripts/live_provider_proof.mjs";
import { stagingRuntime } from "../infra/capture-staging.js";
import { webcrypto } from "node:crypto";
/**
 * Tests for claude.js's native-capture contract (URL/role/turn extraction)
 * and claude_bridge.js's conversation-API URL resolution, driven through
 * the REAL source (common.js + claude_bridge.js + claude.js loaded via
 * vm.Script into a JSDOM window, the same technique
 * tests/content/chatgpt_bridge.test.js and tests/content/grok.test.js
 * already use) rather than hand-copied function bodies.
 *
 * This file used to keep local copies of roleFromNode (a DOM-scrape
 * function deleted from src/content/claude.js entirely -- native capture
 * now covers everything the DOM fallback used to), conversationIdFromUrl,
 * textFromMessage, roleFromNativeMessage, collectNativeTurns,
 * parseNativeCapture, and a `conversationApiUrlFromResources` +
 * `organizationIdFromStorageKeys` pair that had ALREADY silently drifted
 * from the real functions (which live in claude_bridge.js, are named
 * `conversationApiUrlFromResources`/`organizationIdFromLocalStorage`, and
 * take different parameters) with zero test failures, because the copies
 * only ever tested themselves. That is precisely the failure mode that let
 * src/common.js's buildEnvelope silently drop every turn's `blocks` field
 * (polylogue-ah21 regressed) go unnoticed. All coverage here now exercises
 * window.polylogueCapture.capturePage and the real message-bridge protocol
 * against the real IIFE bodies.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const testDirectory = dirname(fileURLToPath(import.meta.url));
const assetStreamSource = readFileSync(resolve(testDirectory, "../../src/content/asset_stream.js"), "utf8");
const bridgeSource = readFileSync(resolve(testDirectory, "../../src/content/claude_bridge.js"), "utf8");
const commonSource = readFileSync(resolve(testDirectory, "../../src/common.js"), "utf8");
const contentSource = readFileSync(resolve(testDirectory, "../../src/content/claude.js"), "utf8");
const openDoms = [];

function jsonResponse(body, status = 200) {
  return new globalThis.Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

// claude_bridge.js resolves the chat_conversations API URL from either
// window.performance resource-timing entries (page-observed requests) or a
// localStorage key carrying the organization id -- JSDOM has no real
// resource-timing buffer, so this fixture stubs both inputs directly rather
// than faking network-level resource entries.
function installClaude({ url = "https://claude.ai/chat/conversation-1", resourceUrls = [], localStorageEntries = {}, fetch } = {}) {
  const dom = new JSDOM("<!doctype html><title>Claude fixture</title>", { url, runScripts: "outside-only" });
  openDoms.push(dom);
  Object.defineProperty(dom.window, "crypto", { configurable: true, value: webcrypto });
  dom.__captureRuntime = stagingRuntime(undefined, { tab_id: 42, document_id: "synthetic-document", provider: "claude-ai" });
  Object.defineProperty(dom.window, "fetch", { configurable: true, value: fetch || (async () => jsonResponse({ detail: "not_found" }, 404)) });
  Object.defineProperty(dom.window.performance, "getEntriesByType", {
    configurable: true,
    value: (type) => (type === "resource" ? resourceUrls.map((name) => ({ name })) : []),
  });
  for (const [key, value] of Object.entries(localStorageEntries)) dom.window.localStorage.setItem(key, value);
  const runtimeListeners = [];
  const chrome = {
    runtime: {
      id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
        const staged = await dom.__captureRuntime.sendMessage(message);
        if (staged !== undefined) return staged;
        if (message.type === "polylogue.capture") {
          return { ok: true, provider: "claude-ai", provider_session_id: "conversation-1", receiver_request_id: "synthetic-request" };
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
      dom.window.queueMicrotask(() => {
        dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin, data }));
      });
    },
  });
  const context = dom.getInternalVMContext();
  new Script(assetStreamSource).runInContext(context);
  new Script(bridgeSource).runInContext(context);
  new Script(commonSource).runInContext(context);
  new Script(contentSource).runInContext(context);
  function sendRuntimeMessage(message) {
    return new Promise((resolvePromise, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolvePromise) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  dom.__captureRuntime.setDispatch(sendRuntimeMessage);
  return { dom, sendRuntimeMessage: async (message) => {
    const result = await sendRuntimeMessage(message);
    if (result?.envelope) result.envelope = await dom.__captureRuntime.materialize(result.envelope);
    return result;
  } };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("claude.js native capture (real source)", () => {
  it.each(["admission", "provider_fetch", "staging", "post_staging"])("retains the original caught %s boundary through real bridge/content and strict proof filtering", async (stage) => {
    const secret = "https://private.invalid/conversation?token=neutral-private-error";
    const fetch = vi.fn(async () => {
      if (stage === "provider_fetch") throw new Error(secret);
      const response = jsonResponse({ uuid: "conversation-1", chat_messages: [{ uuid: "u1", sender: "human", text: "Neutral message" }] });
      if (stage === "post_staging") {
        const originalGet = response.headers.get.bind(response.headers);
        let contentTypeReads = 0;
        response.headers.get = name => {
          if (name === "content-type" && ++contentTypeReads === 2) throw new Error(secret);
          return originalGet(name);
        };
      }
      return response;
    });
    const { dom, sendRuntimeMessage } = installClaude({
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: "11111111-1111-4111-8111-111111111111" }) }, fetch,
    });
    const originalSend = dom.__captureRuntime.sendMessage.bind(dom.__captureRuntime);
    dom.__captureRuntime.sendMessage = async message => {
      if (message.type === (stage === "admission" ? "polylogue.asset.begin" : stage === "staging" ? "polylogue.asset.seal" : "never")) return { ok: false, error: secret };
      return originalSend(message);
    };
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    const expectedStage = stage === "post_staging" ? "unknown" : stage;
    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable", native_attempts: [{ stage: "page_bridge_fetch", failure_stage: expectedStage, error: secret, accepted: false }] });
    if (stage === "admission") expect(fetch).not.toHaveBeenCalled();
    const provider = { provider: "claude-ai", nativeId: "conversation-1", url: dom.window.location.href };
    const __polylogueOwnedProviderProof = { capture: async () => result };
    const popup = { call: async (_method, params) => ({ result: { value: await new Script(params.expression).runInNewContext({ __polylogueOwnedProviderProof }) } }) };
    await captureProvider(popup, provider, 1);
    const report = proofFailureReport("summary", new Error("proof_capture_incomplete"));
    expect(report.capture_evidence[0].bridge).toMatchObject({ observed: true, accepted: false, category: "unknown", failure_stage: expectedStage, status: null });
    expect(JSON.stringify(report)).not.toContain(secret);
  });

  it("keeps a cancelled suspended provider read distinct from a caught acquisition failure and drains original work", async () => {
    let entered;
    const started = new Promise(resolve => { entered = resolve; });
    let rejected = false;
    const fetch = vi.fn((_url, { signal }) => new Promise((_resolve, reject) => {
      entered();
      signal.addEventListener("abort", () => { rejected = true; reject(signal.reason); }, { once: true });
    }));
    const { sendRuntimeMessage } = installClaude({
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: "11111111-1111-4111-8111-111111111111" }) }, fetch,
    });
    const capture = sendRuntimeMessage({ type: "polylogue.capturePage" });
    await started;
    const cancelled = await sendRuntimeMessage({ type: "polylogue.cancelCapture" });
    expect(cancelled).toMatchObject({ ok: true, outcome: "cancelled", drained: 1 });
    expect(rejected).toBe(true);
    expect(await capture).toMatchObject({ ok: false, error: "capture_cancelled", outcome: "cancelled" });
    expect((await capture).native_attempts).toBeUndefined();
  });

  it.each([undefined, "foreign", { stage: "admission", private: "neutral-secret" }, null])("refuses malformed or missing bridge stage while preserving original error", async (stage) => {
    const { dom, sendRuntimeMessage } = installClaude();
    const originalPost = dom.window.postMessage;
    dom.window.postMessage = data => {
      if (data.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.claude.nativeFetchRequest") originalPost({ type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.claude.nativeFetchResponse", requestId: data.requestId, error: "capture_staging_unavailable", failure_stage: stage });
      else originalPost(data);
    };
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    expect(result.native_attempts[0]).toMatchObject({ error: "capture_staging_unavailable", failure_stage: "unknown", accepted: false });
  });

  it.each([true, false])("acquires the current Claude revision despite an older intercepted capture: available=%s", async (available) => {
    const orgId = "11111111-1111-4111-8111-111111111111";
    const url = `https://claude.ai/api/organizations/${orgId}/chat_conversations/conversation-1`;
    const old = { uuid: "conversation-1", chat_messages: [
      { uuid: "u1", sender: "human", text: "Original neutral prompt" },
      { uuid: "a1", sender: "assistant", text: "Original neutral reply" },
    ] };
    const fresh = { ...old, chat_messages: [...old.chat_messages,
      { uuid: "u2", sender: "human", text: "New neutral follow-up" },
    ] };
    const fetch = vi.fn(async () => available ? jsonResponse(fresh) : jsonResponse({ detail: "unavailable" }, 503));
    const { dom, sendRuntimeMessage } = installClaude({
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: orgId }) }, fetch,
    });
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      source: dom.window, origin: dom.window.location.origin,
      data: { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.claude.nativeCapture", capture: {
        ok: true, status: 200, contentType: "application/json", url, body: JSON.stringify(old),
      } },
    }));
    const send = vi.spyOn(dom.window.chrome.runtime, "sendMessage");
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "auto_capture_unconverged_provider" });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(String(fetch.mock.calls[0][0])).toBe(`${url}?tree=True&rendering_mode=messages&render_all_tools=true&consistency=strong`);
    if (available) {
      expect(result.ok).toBe(true);
      expect((await dom.__captureRuntime.retainedNativeReplies(result.envelope)).chat_messages.map(message => message.uuid)).toEqual(["u1", "a1", "u2"]);
      expect(await dom.__captureRuntime.retainedNativeReplies(result.envelope)).toEqual(fresh);
    } else {
      expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
      expect(send.mock.calls.filter(([message]) => message.type === "polylogue.capture")).toEqual([]);
    }
  });

  it("leaves the provider page response available without waiting for its conversation body", async () => {
    let release;
    const body = new Promise(resolve => { release = resolve; });
    const response = { headers: new globalThis.Headers({ "content-type": "application/json" }),
      clone: () => ({ text: () => body }) };
    const { dom } = installClaude({ fetch: async () => response });
    let returned = false;
    const request = dom.window.fetch("https://claude.ai/api/organizations/org/chat_conversations/conversation-1")
      .then(value => { returned = true; return value; });
    try {
      await vi.waitFor(() => expect(returned).toBe(true));
      expect(await request).toBe(response);
    } finally { release('{"chat_messages":[]}'); await request; }
  });

  it("fetches the current revision when lifecycle recapture follows a cached response", async () => {
    const orgId = "11111111-1111-4111-8111-111111111111";
    let messages = [{ uuid: "u1", sender: "human", text: "first" }];
    const fetch = vi.fn(async () => jsonResponse({ uuid: "conversation-1", chat_messages: messages }));
    const { dom, sendRuntimeMessage } = installClaude({
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: orgId }) }, fetch,
    });
    const first = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    expect((await dom.__captureRuntime.retainedNativeReplies(first.envelope)).chat_messages).toEqual(messages);
    messages = [...messages, { uuid: "a1", sender: "assistant", text: "new turn" }];
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "auto_capture_unconverged_provider" });
    expect((await dom.__captureRuntime.retainedNativeReplies(result.envelope)).chat_messages).toEqual(messages);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("retains native Claude message roles and content verbatim for canonical receiver preparation", async () => {
    const orgId = "11111111-1111-4111-8111-111111111111";
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input));
      if (url.pathname === `/api/organizations/${orgId}/chat_conversations/conversation-1`) {
        return jsonResponse({
          uuid: "conversation-1",
          name: "Native Claude title",
          chat_messages: [
            { uuid: "u1", sender: "human", text: "Native user", created_at: "2026-06-24T00:00:00Z" },
            { uuid: "a1", sender: "assistant", content: [{ text: "Native answer" }], model: "claude-native", parent_message_uuid: "u1" },
            { uuid: "empty", sender: "assistant", text: "" },
            { uuid: "s1", sender: "system", text: "unrecognized-shaped system note" },
          ],
        });
      }
      return jsonResponse({ detail: "not_found" }, 404);
    });
    const { dom, sendRuntimeMessage } = installClaude({
      url: "https://claude.ai/chat/conversation-1",
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: orgId }) },
      fetch,
    });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result.ok).toBe(true);
    const retained = await dom.__captureRuntime.retainedNativeReplies(result.envelope);
    expect(retained.chat_messages.map((message) => message.sender)).toEqual(["human", "assistant", "assistant", "system"]);
    expect(retained.chat_messages[1]).toMatchObject({ model: "claude-native", parent_message_uuid: "u1", content: [{ text: "Native answer" }] });
    expect(retained.chat_messages[2].text).toBe("");
    expect(retained.name).toBe("Native Claude title");
    expect(result.envelope.session.provider_session_id).toBe("conversation-1");
    expect(result.envelope.receiver_native).toBeDefined();
    expect(result.envelope.session.turns).toEqual([]);
  });

  it("uses the selected organization cache before the page observes a conversation request", async () => {
    const orgId = "d83be663-5e28-4dfc-8a54-1c34bdbb8c44";
    let requestedUrl = null;
    const fetch = vi.fn(async (input) => {
      requestedUrl = String(input);
      return jsonResponse({ uuid: "conversation-1", name: "Resolved via localStorage", chat_messages: [{ uuid: "u1", sender: "human", text: "hi" }] });
    });
    const { sendRuntimeMessage } = installClaude({
      url: "https://claude.ai/chat/conversation-1",
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: orgId }) },
      fetch,
    });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result.ok).toBe(true);
    expect(requestedUrl).toBe(
      `https://claude.ai/api/organizations/${orgId}/chat_conversations/conversation-1?tree=True&rendering_mode=messages&render_all_tools=true&consistency=strong`,
    );
  });

  it.each([null, "{invalid", JSON.stringify({ orgUuid: "not-an-organization" })])("refuses an absent or malformed selected organization instead of inferring stale storage keys: %s", async (selector) => {
    const staleOrg = "00000000-0000-4000-8000-000000000043";
    const fetch = vi.fn(async () => jsonResponse({ uuid: "conversation-1", chat_messages: [] }));
    const { sendRuntimeMessage } = installClaude({
      localStorageEntries: { [`claude-mcp-has-connectors:${staleOrg}`]: "true", ...(selector === null ? {} : { "omelette-org-settings-cache": selector }) }, fetch,
    });
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("prefers an already-observed resource-timing chat_conversations URL over deriving one", async () => {
    const observedUrl = "https://claude.ai/api/organizations/org-observed/chat_conversations/conversation-1?tree=True&rendering_mode=messages";
    let requestedUrl = null;
    const fetch = vi.fn(async (input) => {
      requestedUrl = String(input);
      return jsonResponse({ uuid: "conversation-1", name: "Via resource entry", chat_messages: [{ uuid: "u1", sender: "human", text: "hi" }] });
    });
    const { sendRuntimeMessage } = installClaude({
      url: "https://claude.ai/chat/conversation-1",
      resourceUrls: ["https://claude.ai/api/bootstrap/org-fallback/current_user_access", observedUrl],
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: "00000000-0000-4000-8000-000000000042" }) },
      fetch,
    });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result.ok).toBe(true);
    expect(requestedUrl).toBe(observedUrl);
  });

  it("returns native_capture_unavailable when no conversation API URL can be resolved at all", async () => {
    const { sendRuntimeMessage } = installClaude({ url: "https://claude.ai/chat/conversation-1" });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
    let repeated = result;
    for (let attempt = 0; attempt < 8; attempt++) repeated = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    expect(repeated).toMatchObject({ ok: false, error: "native_capture_unavailable" });
    expect(repeated.native_attempts).toHaveLength(6);
    expect(repeated.native_attempts_dropped).toBe(3);
  });

  it("rejects a captured payload for a different conversation than the current URL", async () => {
    const { dom, sendRuntimeMessage } = installClaude({
      url: "https://claude.ai/chat/conversation-1",
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: "00000000-0000-4000-8000-000000000042" }) },
    });
    const bodyRef = await dom.window.polylogueAssetStream.stageResponse(
      jsonResponse({ uuid: "other-conversation", chat_messages: [{ uuid: "u1", sender: "human", text: "wrong conversation" }] }),
      "claude-ai", new globalThis.AbortController().signal,
    );
    // A staged response from another conversation cannot become this page's cache.
    dom.window.dispatchEvent(
      new dom.window.MessageEvent("message", {
        source: dom.window,
        origin: dom.window.location.origin,
        data: {
          type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.claude.nativeCapture",
          capture: {
            ok: true,
            status: 200,
            contentType: "application/json",
            url: "https://claude.ai/api/organizations/org-1/chat_conversations/other-conversation",
            bodyRef,
          },
        },
      }),
    );

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
  });

  it("acquires the current revision after an unrelated conversation response arrives", async () => {
    const fetch = vi.fn(async () => jsonResponse({ uuid: "conversation-1", chat_messages: [{ uuid: "current-message", sender: "human", text: "fresh-current-text" }] }));
    const { dom, sendRuntimeMessage } = installClaude({ localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: "00000000-0000-4000-8000-000000000042" }) }, fetch });
    for (const id of ["conversation-1", "other-conversation"]) {
      const bodyRef = await dom.window.polylogueAssetStream.stageResponse(jsonResponse({ uuid: id, chat_messages: [{ uuid: `message-${id}`, sender: "human", text: `text-${id}` }] }), "claude-ai", new globalThis.AbortController().signal, `https://claude.ai/api/organizations/00000000-0000-4000-8000-000000000042/chat_conversations/${id}`);
      dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin,
        data: { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.claude.nativeCapture", capture: { ok: true, bodyRef, capturedAt: new Date().toISOString(), url: `https://claude.ai/api/organizations/00000000-0000-4000-8000-000000000042/chat_conversations/${id}` } } }));
    }
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });
    expect(result.ok).toBe(true);
    expect(result.envelope.session.provider_session_id).toBe("conversation-1");
    expect((await dom.__captureRuntime.retainedNativeReplies(result.envelope)).chat_messages[0].text).toBe("fresh-current-text");
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("returns null outside a /chat/<id> conversation route", async () => {
    const { sendRuntimeMessage } = installClaude({ url: "https://claude.ai/new" });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
  });
});

// Structured native semantics are qualified through the actual receiver in
// test_native_receiver_complete_envelope_matches_canonical_provider_parsing,
// using native-rich-blocks-v1.json (tool use/results, arrays, thinking and
// signatures, token budgets) and native-attachment-order.json. The ordinary
// Claude catalog remains the source of canonical normalization semantics.

describe("Claude original native attachment custody", () => {
  it("retains both attachment channels and missing-ID out-of-order messages for the ordinary parser", async () => {
    const payload = JSON.parse(readFileSync(resolve(testDirectory, "../../../tests/fixtures/claude-ai/native-attachment-order.json"), "utf8"));
    const orgId = "00000000-0000-4000-8000-000000000042";
    const fetch = vi.fn(async () => jsonResponse(payload));
    const { dom, sendRuntimeMessage } = installClaude({ url: `https://claude.ai/chat/${payload.uuid}`,
      localStorageEntries: { "omelette-org-settings-cache": JSON.stringify({ orgUuid: orgId }) }, fetch });
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "synthetic-attachment-custody" });
    expect(result.ok).toBe(true);
    const retained = await dom.__captureRuntime.retainedNativeReplies(result.envelope);
    expect(retained).toEqual(payload);
    expect(result.envelope.receiver_native).toBeDefined();
    expect(result.envelope.session.turns).toEqual([]);
    expect(retained.chat_messages[0].attachments.map((item) => item.extracted_content)).toEqual(["first", "second"]);
    expect(retained.chat_messages[1].files[0].file_uuid).toBe("synthetic-file");
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
