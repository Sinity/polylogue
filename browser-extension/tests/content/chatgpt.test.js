/**
 * Tests for chatgpt.js content script role detection.
 *
 * The roleFromNode function is extracted from src/content/chatgpt.js.
 * It must stay in sync with the source.
 *
 * Known gap (#622): the index-parity fallback at the end of roleFromNode
 * assigns "user"/"assistant" based on even/odd index when no DOM attribute
 * matches.  It should return "unknown" for ambiguous nodes per #622.
 */

import { describe, it, expect } from "vitest";
import { JSDOM } from "jsdom";

// ---------------------------------------------------------------------------
// Extracted from src/content/chatgpt.js — keep in sync
// ---------------------------------------------------------------------------

function roleFromNode(node, index) {
  const testId = node.getAttribute("data-testid") || "";
  if (testId.includes("user")) return "user";
  if (testId.includes("assistant")) return "assistant";
  const labelled = node.getAttribute("aria-label") || "";
  if (/you|user/i.test(labelled)) return "user";
  if (/chatgpt|assistant/i.test(labelled)) return "assistant";
  return index % 2 === 0 ? "user" : "assistant";
}

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function visibleText(node) {
  return (node?.innerText || node?.textContent || "").replace(/\s+\n/g, "\n").trim();
}

function attachmentNameFromNode(node) {
  const label = node.getAttribute("aria-label") || "";
  const download = node.getAttribute("download") || "";
  const alt = node.getAttribute("alt") || "";
  const text = visibleText(node);
  const href = node.getAttribute("href") || node.getAttribute("src") || "";
  const basename = href.split(/[/?#]/).filter(Boolean).at(-1) || "";
  const candidates = [label, download, alt, text, basename]
    .map((value) => String(value || "").trim())
    .filter(Boolean);
  const extensionPattern =
    "zip|tar|tgz|gz|bz2|xz|7z|rar|md|txt|pdf|doc|docx|json|jsonl|csv|tsv|py|js|ts|tsx|jsx|rs|go|java|c|cc|cpp|h|hpp|png|jpe?g|gif|webp|svg|mp3|mp4|wav|webm";
  const filePattern = new RegExp(`(?:^|\\s)([^\\s@/]+\\.(?:${extensionPattern}))(?:\\s|$)`, "i");
  for (const candidate of candidates) {
    const fileNameMatch = candidate.match(filePattern);
    if (fileNameMatch) return fileNameMatch[1].trim();
  }
  return null;
}

function collectAttachments(node, turnIndex) {
  const selector = [
    '[role="group"][aria-label]',
    "a[download]",
    "a[href][aria-label]",
    "img[alt]",
    "img[src]",
  ].join(",");
  const seen = new Set();
  const attachments = [];
  for (const candidate of node.querySelectorAll(selector)) {
    const name = attachmentNameFromNode(candidate);
    if (!name) continue;
    const rawHref = candidate.getAttribute("href") || candidate.getAttribute("src") || null;
    const url = rawHref && /^https?:\/\//i.test(rawHref) ? rawHref : null;
    const key = `${name}\n${url || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    attachments.push({
      provider_attachment_id: `dom:${fnv1a(`${turnIndex}:${key}`)}`,
      name,
      url,
      provider_meta: {
        dom_selector_index: turnIndex,
        dom_label: candidate.getAttribute("aria-label") || null,
        dom_text: visibleText(candidate) || null,
        capture_source: "chatgpt_dom_attachment",
      },
    });
  }
  return attachments;
}

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeNode(attrs = {}) {
  const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>");
  const div = dom.window.document.createElement("div");
  for (const [key, value] of Object.entries(attrs)) {
    div.setAttribute(key, value);
  }
  return div;
}

function makeNodes(attrLists) {
  return attrLists.map((attrs) => makeNode(attrs));
}

// ---------------------------------------------------------------------------
// Tests: known cases
// ---------------------------------------------------------------------------

describe("chatgpt roleFromNode — known cases", () => {
  it("detects user from data-testid containing 'user'", () => {
    const node = makeNode({ "data-testid": "conversation-turn-user-1" });
    expect(roleFromNode(node, 0)).toBe("user");
  });

  it("detects assistant from data-testid containing 'assistant'", () => {
    const node = makeNode({
      "data-testid": "conversation-turn-assistant-1",
    });
    expect(roleFromNode(node, 1)).toBe("assistant");
  });

  it("detects user from aria-label matching you/user", () => {
    expect(roleFromNode(makeNode({ "aria-label": "You said:" }), 0)).toBe(
      "user",
    );
    expect(roleFromNode(makeNode({ "aria-label": "User message" }), 2)).toBe(
      "user",
    );
  });

  it("detects assistant from aria-label matching chatgpt/assistant", () => {
    expect(
      roleFromNode(makeNode({ "aria-label": "ChatGPT said:" }), 0),
    ).toBe("assistant");
    expect(
      roleFromNode(makeNode({ "aria-label": "assistant response" }), 0),
    ).toBe("assistant");
  });

  it("prefers data-testid over aria-label", () => {
    // data-testid wins because it is checked first
    const node = makeNode({
      "data-testid": "conversation-turn-user-1",
      "aria-label": "ChatGPT said:",
    });
    expect(roleFromNode(node, 0)).toBe("user");
  });

  it("handles nodes with no role attributes", () => {
    const node = makeNode({});
    // Falls through to index parity
    expect(roleFromNode(node, 0)).toBe("user");
    expect(roleFromNode(node, 1)).toBe("assistant");
  });
});

// ---------------------------------------------------------------------------
// Tests: interleaved turns
// ---------------------------------------------------------------------------

describe("chatgpt roleFromNode — interleaved turns", () => {
  it("assigns alternating roles for bare nodes via index parity", () => {
    const nodes = makeNodes([{}, {}, {}, {}]);
    const roles = nodes.map((n, i) => roleFromNode(n, i));
    expect(roles).toEqual(["user", "assistant", "user", "assistant"]);
  });
});

// ---------------------------------------------------------------------------
// Tests: #622 gap — index-parity fallback
// ---------------------------------------------------------------------------

describe("chatgpt roleFromNode — #622 gap (index-parity fallback)", () => {
  it("returns user/assistant for ambiguous nodes instead of unknown", () => {
    // A node whose data-testid does NOT contain 'user' or 'assistant'
    // and with no aria-label — it should ideally be "unknown" (#622).
    const node = makeNode({ "data-testid": "conversation-turn-1" });
    const result = roleFromNode(node, 0);
    expect(result).toBe("user");
    // TODO(#622): expect(result).toBe("unknown");
  });

  it("index parity can mislabel roles when turns are missing", () => {
    // If turn 0 is deleted (user message gone), turn 1 (assistant)
    // would be labeled "user" by index parity because it's now at index 0.
    const node = makeNode({ "data-testid": "something-else" });
    expect(roleFromNode(node, 0)).toBe("user");
    expect(roleFromNode(node, 1)).toBe("assistant");
    // Both "something-else" nodes get different roles based solely on index.
    // #622 should fix this so ambiguous nodes return "unknown".
  });

  it("documents the exact fallback line for #622 reference", () => {
    // The fallback is:  return index % 2 === 0 ? "user" : "assistant";
    // This is the last line of roleFromNode in src/content/chatgpt.js
    const node = makeNode({});
    // Even index
    expect(roleFromNode(node, 100)).toBe("user");
    // Odd index
    expect(roleFromNode(node, 101)).toBe("assistant");
  });
});

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
});

describe("chatgpt collectAttachments", () => {
  it("captures ChatGPT file tiles by aria-label", () => {
    const dom = new JSDOM(
      '<!DOCTYPE html><article><div role="group" aria-label="polylogue-all.tar.gz"><span>polylogue-all.tar.gz</span></div></article>',
    );
    const article = dom.window.document.querySelector("article");

    expect(collectAttachments(article, 3)).toEqual([
      {
        provider_attachment_id: `dom:${fnv1a("3:polylogue-all.tar.gz\n")}`,
        name: "polylogue-all.tar.gz",
        url: null,
        provider_meta: {
          dom_selector_index: 3,
          dom_label: "polylogue-all.tar.gz",
          dom_text: "polylogue-all.tar.gz",
          capture_source: "chatgpt_dom_attachment",
        },
      },
    ]);
  });

  it("deduplicates repeated labels within a turn", () => {
    const dom = new JSDOM(
      '<!DOCTYPE html><article><div role="group" aria-label="career.md"></div><div role="group" aria-label="career.md"></div></article>',
    );
    const article = dom.window.document.querySelector("article");

    expect(collectAttachments(article, 0)).toHaveLength(1);
  });

  it("does not treat domains, emails, or version prose as file uploads", () => {
    const dom = new JSDOM(
      '<!DOCTYPE html><article><a aria-label="dbreunig.com" href="https://dbreunig.com">dbreunig.com</a><a aria-label="recruiting@nousresearch.com" href="mailto:recruiting@nousresearch.com">recruiting@nousresearch.com</a><div role="group" aria-label="After Opus 4.5">After Opus 4.5</div></article>',
    );
    const article = dom.window.document.querySelector("article");

    expect(collectAttachments(article, 0)).toEqual([]);
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
