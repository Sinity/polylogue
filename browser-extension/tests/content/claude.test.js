/**
 * Tests for claude.js content script role detection.
 *
 * The roleFromNode function is extracted from src/content/claude.js.
 * It must stay in sync with the source.
 *
 * Known gap (#622): the index-parity fallback at the end of roleFromNode
 * assigns "user"/"assistant" based on even/odd index when no DOM attribute
 * matches.  It should return "unknown" for ambiguous nodes per #622.
 */

import { describe, it, expect } from "vitest";
import { JSDOM } from "jsdom";

// ---------------------------------------------------------------------------
// Extracted from src/content/claude.js — keep in sync
// ---------------------------------------------------------------------------

function roleFromNode(node, index) {
  const role =
    node.getAttribute("data-message-author-role") ||
    node.getAttribute("data-testid") ||
    "";
  if (/human|user/i.test(role)) return "user";
  if (/assistant|claude/i.test(role)) return "assistant";
  return index % 2 === 0 ? "user" : "assistant";
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

describe("claude roleFromNode — known cases", () => {
  it("detects user from data-message-author-role='human'", () => {
    const node = makeNode({ "data-message-author-role": "human" });
    expect(roleFromNode(node, 0)).toBe("user");
  });

  it("detects user from data-message-author-role='user'", () => {
    const node = makeNode({ "data-message-author-role": "user" });
    expect(roleFromNode(node, 0)).toBe("user");
  });

  it("detects assistant from data-message-author-role='assistant'", () => {
    const node = makeNode({ "data-message-author-role": "assistant" });
    expect(roleFromNode(node, 1)).toBe("assistant");
  });

  it("detects assistant from data-message-author-role='claude'", () => {
    const node = makeNode({ "data-message-author-role": "claude" });
    expect(roleFromNode(node, 1)).toBe("assistant");
  });

  it("detects user from data-testid containing 'user' (fallback attr)", () => {
    // data-message-author-role is checked first; if absent, data-testid is used
    const node = makeNode({ "data-testid": "user-message-1" });
    expect(roleFromNode(node, 0)).toBe("user");
  });

  it("detects assistant from data-testid containing 'assistant' (fallback attr)", () => {
    const node = makeNode({ "data-testid": "assistant-message-1" });
    expect(roleFromNode(node, 1)).toBe("assistant");
  });

  it("prefers data-message-author-role over data-testid", () => {
    const node = makeNode({
      "data-message-author-role": "human",
      "data-testid": "assistant-message-1",
    });
    expect(roleFromNode(node, 0)).toBe("user");
  });

  it("handles nodes with neither attribute", () => {
    const node = makeNode({});
    expect(roleFromNode(node, 0)).toBe("user");
    expect(roleFromNode(node, 1)).toBe("assistant");
  });
});

// ---------------------------------------------------------------------------
// Tests: interleaved turns
// ---------------------------------------------------------------------------

describe("claude roleFromNode — interleaved turns", () => {
  it("assigns alternating roles for bare nodes via index parity", () => {
    const nodes = makeNodes([{}, {}, {}, {}]);
    const roles = nodes.map((n, i) => roleFromNode(n, i));
    expect(roles).toEqual(["user", "assistant", "user", "assistant"]);
  });

  it("interleaves explicit and bare nodes correctly", () => {
    const nodes = [
      makeNode({ "data-message-author-role": "human" }), // user
      makeNode({}), // bare, odd -> assistant
      makeNode({ "data-message-author-role": "assistant" }), // assistant
      makeNode({}), // bare, odd -> assistant
    ];
    const roles = nodes.map((n, i) => roleFromNode(n, i));
    expect(roles).toEqual(["user", "assistant", "assistant", "assistant"]);
  });
});

// ---------------------------------------------------------------------------
// Tests: #622 gap — index-parity fallback
// ---------------------------------------------------------------------------

describe("claude roleFromNode — #622 gap (index-parity fallback)", () => {
  it("returns user/assistant for ambiguous nodes instead of unknown", () => {
    // A node with data-message-author-role set to something unrecognized
    const node = makeNode({ "data-message-author-role": "system" });
    const result = roleFromNode(node, 0);
    expect(result).toBe("user");
    // "system" is not "human"/"user"/"assistant"/"claude" — should be
    // "unknown" per #622.
    // TODO(#622): expect(result).toBe("unknown");
  });

  it("index parity can mislabel roles when turns are missing", () => {
    // If turn 0 is deleted, turn 1 (assistant) would be labeled "user"
    // by index parity because it's now at index 0.
    const node = makeNode({ "data-testid": "message-1" });
    expect(roleFromNode(node, 0)).toBe("user");
    expect(roleFromNode(node, 1)).toBe("assistant");
    // Both "message-1" nodes get different roles based solely on index.
    // #622 should fix this so ambiguous nodes return "unknown".
  });

  it("documents the exact fallback line for #622 reference", () => {
    // The fallback is:  return index % 2 === 0 ? "user" : "assistant";
    // This is the last line of roleFromNode in src/content/claude.js
    const node = makeNode({});
    expect(roleFromNode(node, 100)).toBe("user");
    expect(roleFromNode(node, 101)).toBe("assistant");
  });
});
