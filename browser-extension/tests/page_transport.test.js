import { afterEach, describe, expect, it, vi } from "vitest";

import { executeProviderPageRequest } from "../src/backfill/page_transport.js";

const { Headers, Response } = globalThis;
const originalWindow = globalThis.window;

function installWindow(url, fetchImpl, storage = {}) {
  const values = new Map(Object.entries(storage));
  globalThis.window = {
    location: new URL(url),
    fetch: fetchImpl,
    localStorage: { getItem: (key) => values.get(key) || null },
    setTimeout: globalThis.setTimeout.bind(globalThis),
    clearTimeout: globalThis.clearTimeout.bind(globalThis),
  };
}

afterEach(() => {
  globalThis.window = originalWindow;
  vi.restoreAllMocks();
});

describe("first-party provider page transport", () => {
  it("returns only the stable ChatGPT account handle for an identity request", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "account-stable-id";
    const fetchImpl = vi.fn(async (input) => {
      expect(new URL(input).pathname).toBe("/api/auth/session");
      return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({
      provider: "chatgpt",
      operation: "identity",
      params: {},
    });

    expect(result).toEqual({ ok: true, response: { accountHandle: accountId } });
    expect(JSON.stringify(result)).not.toContain(token);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("uses the exact selected Claude organization as its stable identity", async () => {
    const selected = "22222222-2222-4222-8222-222222222222";
    const fetchImpl = vi.fn(async (input) => {
      expect(new URL(input).pathname).toBe("/api/organizations");
      return new Response(JSON.stringify([{ uuid: selected }]));
    });
    installWindow("https://claude.ai/new", fetchImpl, {
      "omelette-org-settings-cache": JSON.stringify({ orgUuid: selected, settings: {} }),
    });

    const result = await executeProviderPageRequest({
      provider: "claude-ai",
      operation: "identity",
      params: {},
    });

    expect(result).toEqual({ ok: true, response: { accountHandle: selected } });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("rejects a stale cached Claude organization as an identity", async () => {
    const selected = "22222222-2222-4222-8222-222222222222";
    const current = "33333333-3333-4333-8333-333333333333";
    installWindow("https://claude.ai/new", vi.fn(async () => new Response(JSON.stringify([{ uuid: current }]))), {
      "omelette-org-settings-cache": JSON.stringify({ orgUuid: selected, settings: {} }),
    });

    const result = await executeProviderPageRequest({
      provider: "claude-ai",
      operation: "identity",
      params: {},
    });

    expect(result).toEqual({ ok: false, error: "backfill_bridge_selected_organization_stale" });
  });

  it("keeps ChatGPT bearer and selected account inside MAIN-world execution", async () => {
    const calls = [];
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input, options = {}) => {
      const url = new URL(input);
      calls.push({ url, options });
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }), { headers: { "Content-Type": "application/json" } });
      const headers = new Headers(options.headers);
      if (headers.get("Authorization") !== `Bearer ${token}` || headers.get("ChatGPT-Account-Id") !== accountId) {
        return new Response(JSON.stringify({ items: [], total: 0 }), { headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({ items: [{ id: "conversation-1" }], total: 2 }), { headers: { "Content-Type": "application/json" } });
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({
      provider: "chatgpt",
      operation: "inventory",
      params: { offset: 0, limit: 1, archived: false, starred: false },
      maxResponseBytes: 4096,
    });

    expect(result).toMatchObject({ ok: true, response: { ok: true, status: 200 } });
    expect(JSON.parse(result.response.body)).toMatchObject({ total: 2, items: [{ id: "conversation-1" }] });
    expect(JSON.stringify(result)).not.toContain(token);
    expect(JSON.stringify(result)).not.toContain(accountId);
    expect(calls.find((call) => call.url.pathname === "/backend-api/conversations").url.searchParams.get("is_archived")).toBe("false");
  });

  it("cancels a chunked response as soon as the byte cap is crossed", async () => {
    const cancel = vi.fn(async () => undefined);
    let reads = 0;
    const response = {
      ok: true,
      status: 200,
      headers: { get: () => null },
      body: { getReader: () => ({
        read: vi.fn(async () => {
          reads += 1;
          if (reads === 1) return { done: false, value: new Uint8Array(6) };
          if (reads === 2) return { done: false, value: new Uint8Array(6) };
          return { done: true };
        }),
        cancel,
      }) },
    };
    const selected = "22222222-2222-4222-8222-222222222222";
    installWindow("https://claude.ai/new", vi.fn(async () => response), {
      "omelette-org-settings-cache": JSON.stringify({ orgUuid: selected, settings: {} }),
    });

    const result = await executeProviderPageRequest({ provider: "claude-ai", operation: "organizations", params: {}, maxResponseBytes: 8 });

    expect(result.error).toMatch(/^backfill_bridge_response_too_large:observed_bytes=12;limit_bytes=8$/);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(reads).toBe(2);
  });

  it("projects a declared-over-32-MiB ChatGPT conversation before crossing the bridge, dropping only envelope-level noise", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        title: "Fixture",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "assistant" }, content: { parts: ["kept"] }, metadata: { model_slug: "fixture" }, create_time: 1 } },
        },
        ignored_large_provider_metadata: "x".repeat(4096),
      }), { headers: { "Content-Length": String(32 * 1024 * 1024 + 1) } });
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    expect(result).toMatchObject({ ok: true, response: { ok: true } });
    const body = JSON.parse(result.response.body);
    expect(body).toMatchObject({
      polylogue_bridge_projection: "chatgpt-native-compact-v1",
      mapping: { node: { message: { content: { parts: ["kept"] }, metadata: { model_slug: "fixture" } } } },
    });
    // Envelope-level fields not in the declared header (id/title/create_time/
    // update_time/current_node) are conversation-object noise, distinct from
    // the per-node content/metadata fidelity bug this fix addresses.
    expect(JSON.stringify(body)).not.toContain("ignored_large_provider_metadata");
  });

  it("forwards a ChatGPT `thoughts` reasoning node's full payload instead of collapsing it to {}", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        mapping: {
          "reasoning-node": {
            id: "reasoning-node",
            parent: null,
            message: {
              id: "reasoning-message",
              author: { role: "assistant" },
              content: {
                content_type: "thoughts",
                thoughts: [
                  { summary: "Considering the request", content: "The user wants X, so I should check Y first." },
                  { summary: "Checking constraints", content: "Y depends on Z, verified via tool call." },
                ],
              },
              metadata: { model_slug: "gpt-5-6-thinking" },
            },
          },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    const body = JSON.parse(result.response.body);
    expect(body.mapping["reasoning-node"].message.content).toEqual({
      content_type: "thoughts",
      thoughts: [
        { summary: "Considering the request", content: "The user wants X, so I should check Y first." },
        { summary: "Checking constraints", content: "Y depends on Z, verified via tool call." },
      ],
    });
  });

  it("forwards content_references, citations, and other metadata keys the old allowlist dropped", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const richMetadata = {
      model_slug: "gpt-5-6",
      content_references: [{ matched_text: "a source", url: "https://example.com/a", type: "webpage" }],
      citations: [{ start_ix: 0, end_ix: 5, metadata: { url: "https://example.com/b" } }],
      finish_details: { type: "stop", stop_tokens: [200002] },
      request_id: "req-abc123",
      reasoning_status: "reasoning_ended",
      search_result_groups: [{ domain: "example.com", entries: [{ url: "https://example.com/c" }] }],
      aggregate_result: { status: "success", final_expression_output: '{"result": 42}' },
      attachments: [{ id: "file-1", name: "data.csv", mime_type: "text/csv", size: 1234, extracted_content: "col_a,col_b\n1,2\n" }],
    };
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "assistant" }, content: { content_type: "text", parts: ["with citation"] }, metadata: richMetadata } },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    const body = JSON.parse(result.response.body);
    expect(body.mapping.node.message.metadata).toEqual(richMetadata);
  });

  it("preserves each node's `children` ordering across the projection (branch_index depends on it)", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        mapping: {
          root: { id: "root", parent: null, children: ["reply-b", "reply-a"], message: { id: "root-msg", author: { role: "user" }, content: { parts: ["question"] } } },
          "reply-a": { id: "reply-a", parent: "root", children: [], message: { id: "reply-a-msg", author: { role: "assistant" }, content: { parts: ["first regeneration"] } } },
          "reply-b": { id: "reply-b", parent: "root", children: [], message: { id: "reply-b-msg", author: { role: "assistant" }, content: { parts: ["second regeneration"] } } },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    const body = JSON.parse(result.response.body);
    // The order here (reply-b before reply-a) is deliberately not sorted --
    // polylogue.sources.parsers.chatgpt.extract_messages_from_mapping reads
    // this exact array to compute branch_index (a sibling's position within
    // it), so the projection must forward it byte-for-byte, not a
    // recomputed or alphabetized version.
    expect(body.mapping.root.children).toEqual(["reply-b", "reply-a"]);
    expect(body.mapping["reply-a"].children).toEqual([]);
  });

  it("preserves is_temporary/conversation_template_id/gizmo_id so session_kind and project scoping survive the bridge", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        is_temporary: true,
        conversation_template_id: "g-p-project123",
        gizmo_id: "g-custom-gpt-456",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "user" }, content: { parts: ["hi"] } } },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    // polylogue.sources.parsers.chatgpt.parse() reads these three fields to
    // set session_kind (is_temporary) and provider_project_ref
    // (conversation_template_id/gizmo_id) -- they must survive the bridge
    // now that the projection advertises itself as native_full.
    const body = JSON.parse(result.response.body);
    expect(body).toMatchObject({
      is_temporary: true,
      conversation_template_id: "g-p-project123",
      gizmo_id: "g-custom-gpt-456",
    });
  });

  it("preserves completion and asset descriptors in the bounded ChatGPT projection", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") {
        return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      }
      return new Response(JSON.stringify({
        id: "conversation-1",
        conversation_id: "conversation-1",
        title: "Completed package",
        current_node: "assistant-node",
        mapping: {
          "assistant-node": {
            id: "assistant-node",
            parent: null,
            message: {
              id: "assistant-message",
              author: { role: "assistant" },
              status: "finished_successfully",
              end_turn: true,
              recipient: "all",
              content: {
                content_type: "text",
                parts: [
                  "[Download](sandbox:/mnt/data/assistant-output.zip)",
                  { content_type: "image_asset_pointer", asset_pointer: "file-service://file-OUTPUT1" },
                ],
              },
              metadata: {
                model_slug: "gpt-5-6-pro",
                attachments: [{ id: "file-INPUT1", name: "context.tar.gz", mime_type: "application/gzip" }],
              },
            },
          },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({
      provider: "chatgpt",
      operation: "conversation",
      params: { nativeId: "conversation-1" },
      maxResponseBytes: 32 * 1024 * 1024,
    });

    const body = JSON.parse(result.response.body);
    expect(body.current_node).toBe("assistant-node");
    expect(body.mapping["assistant-node"].message).toMatchObject({
      status: "finished_successfully",
      end_turn: true,
      recipient: "all",
      content: {
        parts: [
          "[Download](sandbox:/mnt/data/assistant-output.zip)",
          { content_type: "image_asset_pointer", asset_pointer: "file-service://file-OUTPUT1" },
        ],
      },
      metadata: {
        model_slug: "gpt-5-6-pro",
        attachments: [{ id: "file-INPUT1", name: "context.tar.gz", mime_type: "application/gzip" }],
      },
    });
  });

  it("crosses a compact ChatGPT projection above 8 MiB within the bounded bridge", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "assistant" }, content: { parts: ["x".repeat(9 * 1024 * 1024)] } } },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    expect(result).toMatchObject({ ok: true, response: { ok: true } });
    expect(new globalThis.TextEncoder().encode(result.response.body).length).toBeGreaterThan(8 * 1024 * 1024);
    expect(new globalThis.TextEncoder().encode(result.response.body).length).toBeLessThan(24 * 1024 * 1024);
  });

  it("fails closed when a compact ChatGPT projection still exceeds its bridge limit", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "assistant" }, content: { parts: ["x".repeat(24 * 1024 * 1024)] } } },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    expect(result.error).toMatch(/^backfill_bridge_projection_too_large:observed_bytes=.+;limit_bytes=25165824$/);
  });

  it("accounts for escaping in the outer scripting-result bridge payload", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "assistant" }, content: { parts: ['"'.repeat(13 * 1024 * 1024)] } } },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    expect(result.error).toMatch(/^backfill_bridge_projection_too_large:observed_bytes=.+;limit_bytes=25165824$/);
  });

  it("returns malformed compact source as typed provider contract drift", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({ id: "conversation-1", mapping: [] }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    expect(result).toMatchObject({ ok: false, error: "provider_contract_drift:chatgpt_conversation.mapping_must_be_object" });
  });
  it("uses the exact Claude UI selector despite ambiguous per-organization keys", async () => {
    const selected = "22222222-2222-4222-8222-222222222222";
    const other = "11111111-1111-4111-8111-111111111111";
    installWindow("https://claude.ai/new", vi.fn(async () => new Response(JSON.stringify([{ uuid: other }, { uuid: selected }]), { headers: { "Content-Type": "application/json", "Retry-After": "60" } })), {
      [`claude-mcp-has-connectors:${other}`]: "true",
      [`claude-mcp-has-connectors:${selected}`]: "true",
      "omelette-org-settings-cache": JSON.stringify({ orgUuid: selected, settings: {} }),
    });

    const result = await executeProviderPageRequest({ provider: "claude-ai", operation: "organizations", params: {}, maxResponseBytes: 4096 });

    expect(JSON.parse(result.response.body).map((entry) => entry.uuid)).toEqual([selected, other]);
    expect(result.response.retryAfter).toBe("60");
    expect(result.response).not.toHaveProperty("headers");
  });
});
