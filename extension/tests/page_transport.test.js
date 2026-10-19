import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { executeProviderPageRequest as executeOwnedProviderPageRequest } from "../src/backfill/page_transport.js";
import { stagingRuntime } from "./infra/capture-staging.js";

const neutralOwner = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const executeProviderPageRequest = request => executeOwnedProviderPageRequest({ ...request, ownerId: neutralOwner });
const originalWindow = globalThis.window;
const selected = "22222222-2222-4222-8222-222222222222";
function installWindow(url, fetchImpl) {
  const events = new globalThis.EventTarget();
  const { staging } = stagingRuntime();
  const provider = new globalThis.URL(url).hostname === "chatgpt.com" ? "chatgpt" : new globalThis.URL(url).hostname === "claude.ai" ? "claude-ai" : "grok";
  const owner = { tab_id: 42, document_id: "synthetic", provider };
  globalThis.window = {
    location: new globalThis.URL(url), fetch: fetchImpl, preparedResponses: [],
    localStorage: { getItem: () => JSON.stringify({ orgUuid: selected }) },
    addEventListener: events.addEventListener.bind(events), removeEventListener: events.removeEventListener.bind(events),
    dispatchEvent: events.dispatchEvent.bind(events),
    polylogueAssetStream: { async prepareResponse(provider, signal, sourceUrl, bundle, kind, observationOnly, queueContext, accountHandle) {
      globalThis.window.preparedResponses.push({ provider, sourceUrl, bundle, kind, observationOnly, queueContext, accountHandle });
      return { consume: async (response) => {
        if (!response.ok) { await response.body?.cancel(); return null; }
        return globalThis.window.polylogueAssetStream.stageResponse(response, provider, signal);
      }, fail: async () => undefined };
    }, async stageResponse(response, _provider, signal) {
      const ref = await staging.begin(owner); const reader = response.body.getReader(); let sequence = 0;
      try {
        for (;;) {
          signal.throwIfAborted(); const { value, done } = await reader.read(); if (done) break;
          for (let offset = 0; offset < value.length; offset += 48 * 1024) await staging.append(ref, owner, sequence++, globalThis.Buffer.from(value.subarray(offset, offset + 48 * 1024)).toString("base64"));
        }
        await staging.seal(ref, owner); return ref;
      } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
    } },
  };
  return staging;
}
afterEach(() => { globalThis.window = originalWindow; vi.restoreAllMocks(); });
const auth = () => new globalThis.Response(JSON.stringify({ accessToken: "synthetic-token", account: { id: "synthetic-account" } }));

describe("first-party provider page transport", () => {
  it.each([
    ["chatgpt", "https://chatgpt.com/", "/api/auth/session"],
    ["claude-ai", "https://claude.ai/new", "/api/organizations"],
  ])("preserves %s identity429 and its48h Retry-After without subsequent provider reads", async (provider, page, path) => {
    const fetchImpl = vi.fn(async () => new globalThis.Response("rate limited", { status: 429, headers: { "Retry-After": "172800" } }));
    const staging = installWindow(page, fetchImpl);
    expect(await executeProviderPageRequest({ provider, operation: "identity", params: {} }))
      .toMatchObject({ ok: false, error: "provider_rate_limited", outcome: "rate_limited", status: 429, retryAfter: "172800", responseUrl: `${new globalThis.URL(page).origin}${path}` });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(globalThis.window.preparedResponses).toEqual([]);
    expect(staging.storage.files.size).toBe(0);
  });
  it("keeps ChatGPT credentials in MAIN and returns its stable account handle", async () => {
    const fetchImpl = vi.fn(async () => auth()); installWindow("https://chatgpt.com/", fetchImpl);
    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "identity", params: {} });
    expect(result).toEqual({ ok: true, response: { accountHandle: "synthetic-account" } });
    expect(JSON.stringify(result)).not.toContain("synthetic-token"); expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("validates the selected Claude organization for identity without picking the first organization", async () => {
    const fetchImpl = vi.fn(async () => new globalThis.Response(JSON.stringify([{ uuid: "33333333-3333-4333-8333-333333333333" }, { uuid: selected }])));
    installWindow("https://claude.ai/new", fetchImpl);
    expect(await executeProviderPageRequest({ provider: "claude-ai", operation: "identity", params: {} })).toEqual({ ok: true, response: { accountHandle: selected } });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("refuses a stale selected Claude identity without substituting another available organization", async () => {
    const fetchImpl = vi.fn(async () => new globalThis.Response(JSON.stringify([{ uuid: "33333333-3333-4333-8333-333333333333" }])));
    installWindow("https://claude.ai/new", fetchImpl);
    expect(await executeProviderPageRequest({ provider: "claude-ai", operation: "identity", params: {} }))
      .toEqual({ ok: false, error: "backfill_bridge_selected_organization_stale" });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(globalThis.window.preparedResponses).toEqual([]);
  });

  it("refuses a switched Claude conversation scope before staging or provider traffic", async () => {
    const fetchImpl = vi.fn(); installWindow("https://claude.ai/new", fetchImpl);
    expect(await executeProviderPageRequest({ provider: "claude-ai", operation: "conversation",
      params: { nativeId: "session", organizationId: "33333333-3333-4333-8333-333333333333" } }))
      .toEqual({ ok: false, error: "backfill_bridge_selected_organization_stale" });
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(globalThis.window.preparedResponses).toEqual([]);
  });

  it.each([
    ["chatgpt", "https://chatgpt.com/", "/backend-api/conversation/session", 2],
    ["claude-ai", "https://claude.ai/new", `/api/organizations/${selected}/chat_conversations/session`, 1],
    ["grok", "https://grok.com/", "/rest/app-chat/conversations/session", 1],
  ])("stages exact %s reply bytes while returning only a ref and response metadata", async (provider, page, expectedPath, requests) => {
    const literal = ' {"id":"session", "metadata":{"kept":"\\\\ \\u0061"}, "mapping":{}}\n';
    const calls = [];
    const fetchImpl = vi.fn(async (input, options) => {
      const url = new globalThis.URL(input); calls.push({ url, options });
      if (url.pathname === "/api/auth/session") return auth();
      if (provider === "chatgpt") {
        expect(new globalThis.Headers(options.headers).get("Authorization")).toBe("Bearer synthetic-token");
        expect(new globalThis.Headers(options.headers).get("ChatGPT-Account-Id")).toBe("synthetic-account");
      }
      return new globalThis.Response(literal, { headers: { "Content-Type": "application/json", "Content-Length": String(40 * 1024 * 1024) } });
    });
    const staging = installWindow(page, fetchImpl);
    const result = await executeProviderPageRequest({ provider, requestId: "request", operation: "conversation", params: { nativeId: "session", organizationId: selected } });
    expect(result.ok).toBe(true); expect(result.response.bodyRef).toMatchObject({ id: expect.any(String), token: expect.any(String) });
    expect(result.response).not.toHaveProperty("body");
    expect(await (await staging.file(result.response.bodyRef.id)).text()).toBe(literal);
    expect((await staging.metadata(result.response.bodyRef.id)).sha256).toBe(createHash("sha256").update(literal).digest("hex"));
    expect(calls).toHaveLength(requests); expect(calls.at(-1).url.pathname).toBe(expectedPath);
    if (provider === "claude-ai") expect(calls.at(-1).url.searchParams.get("render_all_tools")).toBe("true");
    expect(globalThis.window.preparedResponses).toHaveLength(1);
    expect(globalThis.window.preparedResponses[0].accountHandle).toBe(provider === "chatgpt" ? "synthetic-account" : provider === "claude-ai" ? selected : null);
  });

  it("streams a valid reply beyond the former bridge cap without projecting or truncating it", async () => {
    const chunk = globalThis.Buffer.from("x".repeat(48 * 1024)); const count = 700;
    let emitted = 0; const hash = createHash("sha256");
    const fetchImpl = vi.fn(async (input) => new globalThis.URL(input).pathname === "/api/auth/session" ? auth() : new globalThis.Response(new globalThis.ReadableStream({ pull(controller) {
      if (emitted++ < count) { hash.update(chunk); controller.enqueue(chunk); } else controller.close();
    } })));
    const staging = installWindow("https://chatgpt.com/", fetchImpl);
    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "session" } });
    expect(result.ok).toBe(true); const meta = await staging.metadata(result.response.bodyRef.id);
    expect(meta.bytes).toBe(count * chunk.length); expect(meta.sha256).toBe(hash.digest("hex"));
    expect(JSON.stringify(result).length).toBeLessThan(1024);
  }, 30_000);

  it("stages the original organization list once and carries the selected identity separately", async () => {
    const literal = JSON.stringify([{ uuid: "33333333-3333-4333-8333-333333333333" }, { uuid: selected }]);
    const fetchImpl = vi.fn(async () => new globalThis.Response(literal)); const staging = installWindow("https://claude.ai/new", fetchImpl);
    const result = await executeProviderPageRequest({ provider: "claude-ai", operation: "organizations", params: {} });
    expect(result.response.selectedOrganizationId).toBe(selected); expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(await (await staging.file(result.response.bodyRef.id)).text()).toBe(literal);
  });

  it("returns provider 429 and Retry-After without acquiring its body", async () => {
    const fetchImpl = vi.fn(async () => new globalThis.Response("rate limited", { status: 429, headers: { "Retry-After": "120" } }));
    const staging = installWindow("https://grok.com/", fetchImpl);
    const result = await executeProviderPageRequest({ provider: "grok", operation: "responses", params: { nativeId: "session" } });
    expect(result).toMatchObject({ ok: true, response: { ok: false, status: 429, retryAfter: "120", bodyRef: null } });
    expect(staging.storage.files.size).toBe(0); expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it.each(["pagehide", "polylogue.providerCancel"])("drains a pending provider request on %s", async (eventType) => {
    let started; let drained = false; const began = new Promise((resolve) => { started = resolve; });
    const fetchImpl = vi.fn(async (_input, options) => {
      started(); try { return await new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(new globalThis.DOMException("cancelled", "AbortError")), { once: true })); }
      finally { drained = true; }
    });
    installWindow("https://grok.com/", fetchImpl);
    const pending = executeProviderPageRequest({ provider: "grok", requestId: "owned", operation: "conversation", params: { nativeId: "session" } });
    await began; const event = new globalThis.Event(eventType); Object.defineProperty(event, "detail", { value: { requestId: "owned", ownerId: neutralOwner } }); globalThis.window.dispatchEvent(event);
    expect((await pending).ok).toBe(false); expect(drained).toBe(true);
  });

  it("refuses a mismatched provider before any request", async () => {
    const fetchImpl = vi.fn(); installWindow("https://grok.com/", fetchImpl);
    expect(await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "session" } })).toMatchObject({ ok: false, error: "backfill_bridge_provider_mismatch" });
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
