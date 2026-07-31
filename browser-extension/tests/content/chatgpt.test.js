/**
 * Tests for chatgpt.js content script id/URL parsing and the on-demand
 * native fetch contract.
 *
 * These functions are extracted from src/content/chatgpt.js and must stay
 * in sync with the source. The DOM-scrape fallback (roleFromNode,
 * attachmentNameFromNode, collectAttachments, chatgpt-dom-v1) was removed
 * from the source entirely -- native capture now covers the one case that
 * ever made it fire for real (a brand-new conversation's URL not yet
 * published, see tests/content/chatgpt_bridge.test.js's "waits for a
 * brand-new conversation's URL" and "captures a temporary chat" cases,
 * which exercise the real source through the integration harness) -- so
 * the tests for those functions were deleted here rather than updated.
 */

import { describe, it, expect } from "vitest";

function conversationIdFromUrl(url) {
  const parsed = new URL(url);
  const parts = parsed.pathname.split("/").filter(Boolean);
  const marker = parts.indexOf("c");
  return marker >= 0 && parts[marker + 1] ? parts[marker + 1] : null;
}

async function fetchNativePayloadOnDemand(pageUrl, fetchImpl) {
  const conversationId = conversationIdFromUrl(pageUrl);
  if (!conversationId) return null;
  try {
    const response = await fetchImpl(
      `/backend-api/conversation/${encodeURIComponent(conversationId)}`,
      {
        credentials: "include",
        cache: "no-store",
      },
    );
    const contentType = response.headers.get("content-type") || "";
    if (!response.ok || !contentType.includes("application/json")) return null;
    const payload = await response.clone().json();
    if (!payload || typeof payload !== "object" || !payload.mapping) return null;
    const payloadConversationId = payload.conversation_id || payload.id;
    if (payloadConversationId && String(payloadConversationId) !== conversationId)
      return null;
    return payload;
  } catch {
    return null;
  }
}

function makeFetchResponse(payload, options = {}) {
  const {
    ok = true,
    contentType = "application/json",
    throws = false,
  } = options;
  return {
    ok,
    headers: {
      get(name) {
        return name.toLowerCase() === "content-type" ? contentType : null;
      },
    },
    clone() {
      return {
        async json() {
          if (throws) throw new Error("bad json");
          return payload;
        },
      };
    },
  };
}

describe("chatgpt conversationIdFromUrl", () => {
  it("reads normal conversation routes", () => {
    expect(conversationIdFromUrl("https://chatgpt.com/c/abc-123")).toBe(
      "abc-123",
    );
  });

  it("reads custom GPT conversation routes", () => {
    expect(
      conversationIdFromUrl("https://chatgpt.com/g/g-p-abc/c/conv-123"),
    ).toBe("conv-123");
  });

  it("returns null outside conversation routes", () => {
    expect(conversationIdFromUrl("https://chatgpt.com/g/g-p-abc")).toBe(null);
  });

  it("returns null for a temporary chat's own route (it never gets /c/<id>)", () => {
    // src/content/chatgpt.js's isTemporaryChatUrl handles this case
    // separately (see tests/content/chatgpt_bridge.test.js); this local
    // conversationIdFromUrl copy stays a plain path parser.
    expect(
      conversationIdFromUrl("https://chatgpt.com/?temporary-chat=true"),
    ).toBe(null);
  });
});

describe("chatgpt fetchNativePayloadOnDemand", () => {
  it("fetches the current conversation JSON with credentials", async () => {
    const payload = {
      conversation_id: "conv-123",
      title: "Native ChatGPT title",
      mapping: { node: { message: { content: { parts: ["hello"] } } } },
    };
    const calls = [];
    const fetchImpl = async (...args) => {
      calls.push(args);
      return makeFetchResponse(payload);
    };

    await expect(
      fetchNativePayloadOnDemand("https://chatgpt.com/c/conv-123", fetchImpl),
    ).resolves.toEqual(payload);
    expect(calls).toEqual([
      [
        "/backend-api/conversation/conv-123",
        { credentials: "include", cache: "no-store" },
      ],
    ]);
  });

  it("supports custom GPT conversation routes", async () => {
    const payload = { id: "conv-123", mapping: { node: {} } };
    const calls = [];
    const fetchImpl = async (...args) => {
      calls.push(args);
      return makeFetchResponse(payload);
    };

    await expect(
      fetchNativePayloadOnDemand(
        "https://chatgpt.com/g/g-p-abc/c/conv-123",
        fetchImpl,
      ),
    ).resolves.toEqual(payload);
    expect(calls[0][0]).toBe("/backend-api/conversation/conv-123");
  });

  it("rejects mismatched, non-json, failed, malformed, and off-route payloads", async () => {
    await expect(
      fetchNativePayloadOnDemand("https://chatgpt.com/c/conv-123", async () =>
        makeFetchResponse({ conversation_id: "other", mapping: {} }),
      ),
    ).resolves.toBe(null);
    await expect(
      fetchNativePayloadOnDemand("https://chatgpt.com/c/conv-123", async () =>
        makeFetchResponse({ conversation_id: "conv-123", mapping: {} }, {
          contentType: "text/html",
        }),
      ),
    ).resolves.toBe(null);
    await expect(
      fetchNativePayloadOnDemand("https://chatgpt.com/c/conv-123", async () =>
        makeFetchResponse({ conversation_id: "conv-123", mapping: {} }, {
          ok: false,
        }),
      ),
    ).resolves.toBe(null);
    await expect(
      fetchNativePayloadOnDemand("https://chatgpt.com/c/conv-123", async () =>
        makeFetchResponse({ conversation_id: "conv-123" }),
      ),
    ).resolves.toBe(null);
    await expect(
      fetchNativePayloadOnDemand("https://chatgpt.com/c/conv-123", async () =>
        makeFetchResponse(null, { throws: true }),
      ),
    ).resolves.toBe(null);
    await expect(
      fetchNativePayloadOnDemand("https://chatgpt.com/", async () =>
        makeFetchResponse({ mapping: {} }),
      ),
    ).resolves.toBe(null);
  });
});

// ---------------------------------------------------------------------------
// Extracted from src/content/chatgpt.js — keep in sync (asset acquisition)
// ---------------------------------------------------------------------------

const sandboxLinkPattern = /sandbox:(\/mnt\/data\/[^\s)\]"'>]+)/g;

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

describe("asset descriptor collection", () => {
  it("collects sandbox links, upload ids, and asset pointers with parser-matching ids", () => {
    const payload = {
      mapping: {
        n1: {
          id: "n1",
          message: {
            id: "msg-a",
            author: { role: "assistant" },
            content: {
              parts: [
                "Kit ready: [zip](sandbox:/mnt/data/kit.zip) and [sum](sandbox:/mnt/data/kit.zip.sha256). Again sandbox:/mnt/data/kit.zip."
              ]
            },
            metadata: {}
          }
        },
        n2: {
          id: "n2",
          message: {
            id: "msg-b",
            author: { role: "user" },
            content: { parts: ["please use sandbox:/mnt/data/ignored.zip"] },
            metadata: { attachments: [{ id: "file-UP1", name: "input.csv", mime_type: "text/csv" }] }
          }
        },
        n3: {
          id: "n3",
          message: {
            id: "msg-c",
            author: { role: "assistant" },
            content: {
              parts: [{ content_type: "image_asset_pointer", asset_pointer: "file-service://file-IMG9" }]
            },
            metadata: {}
          }
        }
      }
    };

    const descriptors = collectAssetDescriptors(payload);
    const ids = descriptors.map((d) => d.provider_attachment_id);
    expect(ids).toEqual([
      "sandbox:msg-a:/mnt/data/kit.zip",
      "sandbox:msg-a:/mnt/data/kit.zip.sha256",
      "file-UP1",
      "file-service://file-IMG9"
    ]);
    const sandbox = descriptors[0];
    expect(sandbox.kind).toBe("sandbox");
    expect(sandbox.sandboxPath).toBe("/mnt/data/kit.zip");
    expect(sandbox.name).toBe("kit.zip");
    const pointer = descriptors[3];
    expect(pointer.kind).toBe("file");
    expect(pointer.fileId).toBe("file-IMG9");
  });

  it("strips trailing punctuation and dedupes sandbox paths", () => {
    expect(sandboxPathsFromText("see sandbox:/mnt/data/a.md. and sandbox:/mnt/data/a.md,")).toEqual([
      "/mnt/data/a.md"
    ]);
  });
});
