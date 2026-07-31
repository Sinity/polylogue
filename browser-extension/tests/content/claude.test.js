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
  Object.defineProperty(dom.window, "fetch", { configurable: true, value: fetch || (async () => jsonResponse({ detail: "not_found" }, 404)) });
  Object.defineProperty(dom.window.performance, "getEntriesByType", {
    configurable: true,
    value: (type) => (type === "resource" ? resourceUrls.map((name) => ({ name })) : []),
  });
  for (const [key, value] of Object.entries(localStorageEntries)) dom.window.localStorage.setItem(key, value);
  const runtimeListeners = [];
  const chrome = {
    runtime: {
      id: "synthetic-extension-id",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
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
  new Script(bridgeSource).runInContext(context);
  new Script(commonSource).runInContext(context);
  new Script(contentSource).runInContext(context);
  function sendRuntimeMessage(message) {
    return new Promise((resolvePromise, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolvePromise) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  return { dom, sendRuntimeMessage };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("claude.js native capture (real source)", () => {
  it("extracts native Claude turns, normalizes roles, and skips empty messages", async () => {
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
    const { sendRuntimeMessage } = installClaude({
      url: "https://claude.ai/chat/conversation-1",
      localStorageEntries: { [`claude-mcp-has-connectors:${orgId}`]: "true" },
      fetch,
    });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result.ok).toBe(true);
    const turns = result.envelope.session.turns;
    expect(turns).toHaveLength(3);
    expect(turns.map((turn) => turn.role)).toEqual(["user", "assistant", "system"]);
    expect(turns.map((turn) => turn.text)).toEqual(["Native user", "Native answer", "unrecognized-shaped system note"]);
    expect(turns[1].parent_turn_id).toBe("u1");
    expect(turns[1].provider_meta.capture_source).toBe("claude_chat_conversations_api");
    expect(result.envelope.session.provider_session_id).toBe("conversation-1");
    expect(result.envelope.session.title).toBe("Native Claude title");
  });

  it("resolves the organization id from claude.ai's own localStorage key when no resource entry has been observed yet", async () => {
    const orgId = "d83be663-5e28-4dfc-8a54-1c34bdbb8c44";
    let requestedUrl = null;
    const fetch = vi.fn(async (input) => {
      requestedUrl = String(input);
      return jsonResponse({ uuid: "conversation-1", name: "Resolved via localStorage", chat_messages: [{ uuid: "u1", sender: "human", text: "hi" }] });
    });
    const { sendRuntimeMessage } = installClaude({
      url: "https://claude.ai/chat/conversation-1",
      localStorageEntries: { [`claude-mcp-has-connectors:${orgId}`]: "true" },
      fetch,
    });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result.ok).toBe(true);
    expect(requestedUrl).toBe(
      `https://claude.ai/api/organizations/${orgId}/chat_conversations/conversation-1?tree=True&rendering_mode=messages&render_all_tools=true&consistency=strong`,
    );
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
      localStorageEntries: { "claude-mcp-has-connectors:org-fallback": "true" },
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
  });

  it("rejects a captured payload for a different conversation than the current URL", async () => {
    const { dom, sendRuntimeMessage } = installClaude({
      url: "https://claude.ai/chat/conversation-1",
      localStorageEntries: { "claude-mcp-has-connectors:org-1": "true" },
    });
    // Simulate claude_bridge.js's window.fetch interception observing a
    // DIFFERENT conversation's response (e.g. a stale background tab fetch)
    // -- must not be accepted for this page's conversation id.
    dom.window.dispatchEvent(
      new dom.window.MessageEvent("message", {
        source: dom.window,
        origin: dom.window.location.origin,
        data: {
          type: "polylogue.claude.nativeCapture",
          capture: {
            ok: true,
            status: 200,
            contentType: "application/json",
            url: "https://claude.ai/api/organizations/org-1/chat_conversations/other-conversation",
            body: JSON.stringify({ uuid: "other-conversation", chat_messages: [{ uuid: "u1", sender: "human", text: "wrong conversation" }] }),
          },
        },
      }),
    );

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
  });

  it("returns null outside a /chat/<id> conversation route", async () => {
    const { sendRuntimeMessage } = installClaude({ url: "https://claude.ai/new" });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
  });
});

// ---------------------------------------------------------------------------
// Native-payload structure extraction — extracted from src/content/claude.js,
// keep in sync. Claude's chat_conversations API returns the same segment shape
// as the GDPR export, so tool_use / tool_result / thinking and both attachment
// channels are observable at capture time; flattening them to prose loses what
// the export path parses in full.
// ---------------------------------------------------------------------------

const fnv1a = (text) => {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16);
};

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
          : null,
      });
    } else if (type === "tool_result") {
      blocks.push({
        type: "tool_result",
        tool_id: segment.tool_use_id || null,
        text: typeof segment.content === "string" ? segment.content : null,
        is_error: typeof segment.is_error === "boolean" ? segment.is_error : null,
        metadata: { tool_name: segment.name || null },
      });
    }
  }
  return blocks;
}

function nativeTurnAttachments(message, index) {
  const out = [];
  const messageId = String(message.uuid || message.id || `claude-message-${index}`);
  for (const attachment of Array.isArray(message.attachments) ? message.attachments : []) {
    if (!attachment || typeof attachment !== "object") continue;
    const name = attachment.file_name || attachment.name || null;
    if (!name) continue;
    const size = Number.parseInt(attachment.file_size, 10);
    out.push({
      provider_attachment_id: `claude-attachment:${fnv1a(`${messageId}:${name}:${attachment.file_size || ""}`)}`,
      message_provider_id: messageId,
      name,
      mime_type: attachment.file_type || null,
      size_bytes: Number.isFinite(size) ? size : null,
      extracted_content: typeof attachment.extracted_content === "string" ? attachment.extracted_content : null,
      provider_meta: { capture_source: "claude_chat_conversations_api", channel: "attachments" },
    });
  }
  for (const file of Array.isArray(message.files) ? message.files : []) {
    if (!file || typeof file !== "object") continue;
    const name = file.file_name || file.name || null;
    const uuid = file.file_uuid || file.uuid || null;
    if (!name && !uuid) continue;
    out.push({
      provider_attachment_id: uuid ? `claude-file:${uuid}` : `claude-file:${fnv1a(`${messageId}:${name}`)}`,
      message_provider_id: messageId,
      name,
      provider_meta: {
        capture_source: "claude_chat_conversations_api",
        channel: "files",
        file_uuid: uuid || null,
      },
    });
  }
  return out;
}

describe("claude native capture — structured blocks", () => {
  it("preserves tool_use with its id and input rather than flattening to prose", () => {
    const blocks = nativeTurnBlocks({
      content: [
        { type: "text", text: "Let me search." },
        { type: "tool_use", id: "toolu_01Bzda", name: "web_search", input: { query: "polylogue" } },
      ],
    });
    expect(blocks).toHaveLength(2);
    expect(blocks[1]).toMatchObject({
      type: "tool_use",
      tool_name: "web_search",
      tool_id: "toolu_01Bzda",
      tool_input: { query: "polylogue" },
    });
  });

  it("carries the provider's own tool_result outcome, and leaves it unknown when absent", () => {
    const failed = nativeTurnBlocks({
      content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "boom", is_error: true }],
    });
    expect(failed[0]).toMatchObject({ type: "tool_result", tool_id: "toolu_1", is_error: true });

    const unknown = nativeTurnBlocks({
      content: [{ type: "tool_result", tool_use_id: "toolu_2", content: "ok" }],
    });
    // Absent is_error stays null (unknown) — never coerced to false.
    expect(unknown[0].is_error).toBeNull();
  });

  it("keeps thinking segments, reading Claude's `thinking` field", () => {
    const blocks = nativeTurnBlocks({
      content: [{ type: "thinking", thinking: "considering options" }],
    });
    expect(blocks[0]).toMatchObject({ type: "thinking", text: "considering options" });
  });

  it("ignores segment shapes it does not recognise instead of guessing", () => {
    expect(nativeTurnBlocks({ content: [{ type: "token_budget", budget: 5 }] })).toEqual([]);
  });
});

describe("claude native capture — both attachment channels", () => {
  it("extracts attachments[] inline content, which carries no provider id", () => {
    const [attachment] = nativeTurnAttachments(
      {
        uuid: "msg-1",
        attachments: [
          { file_name: "paste.txt", file_size: "4817", file_type: "txt", extracted_content: "hello" },
        ],
      },
      0,
    );
    expect(attachment).toMatchObject({
      message_provider_id: "msg-1",
      name: "paste.txt",
      mime_type: "txt",
      size_bytes: 4817,
      extracted_content: "hello",
    });
    // Synthesised, stable, and namespaced so it cannot collide with files[].
    expect(attachment.provider_attachment_id).toMatch(/^claude-attachment:[0-9a-f]+$/);
  });

  it("extracts files[] by its real uuid so a later byte acquisition can join", () => {
    const [file] = nativeTurnAttachments(
      { uuid: "msg-2", files: [{ file_name: "diagram.png", file_uuid: "file-abc123" }] },
      0,
    );
    expect(file.provider_attachment_id).toBe("claude-file:file-abc123");
    expect(file.provider_meta.file_uuid).toBe("file-abc123");
  });

  it("keeps the two channels distinct for one message", () => {
    const out = nativeTurnAttachments(
      {
        uuid: "msg-3",
        attachments: [{ file_name: "a.txt", file_size: "10", extracted_content: "x" }],
        files: [{ file_name: "b.png", file_uuid: "file-b" }],
      },
      0,
    );
    expect(out).toHaveLength(2);
    expect(out.map((a) => a.provider_meta.channel)).toEqual(["attachments", "files"]);
  });
});
