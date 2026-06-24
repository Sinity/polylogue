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

function conversationIdFromUrl(url) {
  const parsed = new URL(url);
  const parts = parsed.pathname.split("/").filter(Boolean);
  return parts[0] === "chat" && parts[1] ? parts[1] : null;
}

function textFromMessage(message) {
  if (!message || typeof message !== "object") return "";
  if (typeof message.text === "string" && message.text) return message.text;
  if (typeof message.content === "string" && message.content) return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && typeof part.text === "string")
          return part.text;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function roleFromNativeMessage(message) {
  const raw = message && (message.sender || message.role || message.author);
  if (raw === "human" || raw === "user") return "user";
  if (raw === "assistant" || raw === "claude") return "assistant";
  if (raw === "system" || raw === "tool") return raw;
  return "unknown";
}

function collectNativeTurns(payload) {
  const messages = payload && payload.chat_messages;
  if (!Array.isArray(messages)) return [];
  return messages
    .map((message, index) => {
      const text = textFromMessage(message);
      if (!text) return null;
      return {
        provider_turn_id: String(
          message.uuid || message.id || `claude-message-${index}`,
        ),
        role: roleFromNativeMessage(message),
        text,
        timestamp: message.created_at || message.updated_at || null,
        parent_turn_id: message.parent_message_uuid || message.parent_uuid || null,
        provider_meta: {
          model: message.model || null,
          sender: message.sender || message.role || null,
          capture_source: "claude_chat_conversations_api",
        },
      };
    })
    .filter(Boolean);
}

function parseNativeCapture(capture, pageUrl) {
  if (!capture || !capture.ok || typeof capture.body !== "string") return null;
  const currentConversationId = conversationIdFromUrl(pageUrl);
  if (
    !currentConversationId ||
    !String(capture.url || "").includes(
      `/chat_conversations/${currentConversationId}`,
    )
  ) {
    return null;
  }
  try {
    const payload = JSON.parse(capture.body);
    if (
      !payload ||
      typeof payload !== "object" ||
      !Array.isArray(payload.chat_messages)
    )
      return null;
    if (payload.uuid && String(payload.uuid) !== currentConversationId)
      return null;
    return payload;
  } catch {
    return null;
  }
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

describe("claude native capture helpers", () => {
  const pageUrl = "https://claude.ai/chat/conv-123";
  const apiUrl =
    "https://claude.ai/api/organizations/org-1/chat_conversations/conv-123?tree=True";

  it("reads Claude conversation routes", () => {
    expect(conversationIdFromUrl(pageUrl)).toBe("conv-123");
    expect(conversationIdFromUrl("https://claude.ai/new")).toBe(null);
  });

  it("accepts only current chat_conversations payloads with messages", () => {
    const payload = {
      uuid: "conv-123",
      name: "Native Claude title",
      chat_messages: [{ uuid: "m1", sender: "human", text: "hello" }],
    };
    expect(
      parseNativeCapture(
        { ok: true, url: apiUrl, body: JSON.stringify(payload) },
        pageUrl,
      ),
    ).toEqual(payload);
    expect(
      parseNativeCapture(
        {
          ok: true,
          url: "https://claude.ai/api/organizations/org-1/chat_conversations/other",
          body: JSON.stringify(payload),
        },
        pageUrl,
      ),
    ).toBe(null);
    expect(
      parseNativeCapture(
        {
          ok: true,
          url: apiUrl,
          body: JSON.stringify({ uuid: "other", chat_messages: [] }),
        },
        pageUrl,
      ),
    ).toBe(null);
  });

  it("extracts native Claude turns without DOM role parity fallback", () => {
    const turns = collectNativeTurns({
      chat_messages: [
        {
          uuid: "u1",
          sender: "human",
          text: "Native user",
          created_at: "2026-06-24T00:00:00Z",
        },
        {
          uuid: "a1",
          sender: "assistant",
          content: [{ text: "Native answer" }],
          model: "claude-native",
          parent_message_uuid: "u1",
        },
        { uuid: "empty", sender: "assistant", text: "" },
      ],
    });

    expect(turns).toHaveLength(2);
    expect(turns.map((turn) => turn.role)).toEqual(["user", "assistant"]);
    expect(turns.map((turn) => turn.text)).toEqual([
      "Native user",
      "Native answer",
    ]);
    expect(turns[1].parent_turn_id).toBe("u1");
    expect(turns[1].provider_meta.capture_source).toBe(
      "claude_chat_conversations_api",
    );
  });

  it("keeps unknown native roles explicit", () => {
    expect(roleFromNativeMessage({ sender: "system" })).toBe("system");
    expect(roleFromNativeMessage({ sender: "unexpected" })).toBe("unknown");
  });
});
