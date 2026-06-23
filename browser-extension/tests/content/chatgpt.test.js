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

function conversationIdFromUrl(url) {
  const parsed = new URL(url);
  const parts = parsed.pathname.split("/").filter(Boolean);
  const marker = parts.indexOf("c");
  return marker >= 0 && parts[marker + 1] ? parts[marker + 1] : null;
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
