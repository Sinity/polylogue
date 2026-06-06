/**
 * Tests for common.js shared capture utilities.
 *
 * Functions are extracted from src/common.js and must stay in sync.
 */

import { describe, it, expect } from "vitest";
import { JSDOM } from "jsdom";

// ---------------------------------------------------------------------------
// Extracted from src/common.js — keep in sync
// ---------------------------------------------------------------------------

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function sessionIdFromUrl(provider, url) {
  const parsed = new URL(url);
  const parts = parsed.pathname.split("/").filter(Boolean);
  const sessionToken =
    parts.at(-1) || parsed.pathname || parsed.hostname;
  return `${provider}:${sessionToken}:${fnv1a(parsed.origin + parsed.pathname)}`;
}

function visibleText(node) {
  return (node?.innerText || node?.textContent || "")
    .replace(/\s+\n/g, "\n")
    .trim();
}

function buildEnvelope({ provider, adapterName, turns, model = null }) {
  const sourceUrl = "https://chatgpt.com/c/test-conv";
  const providerSessionId = sessionIdFromUrl(provider, sourceUrl);
  const now = new Date().toISOString();
  return {
    polylogue_capture_kind: "browser_llm_session",
    schema_version: 1,
    capture_id: `${provider}:${providerSessionId}`,
    source: "browser-extension",
    provenance: {
      source_url: sourceUrl,
      page_title: "Test Page",
      captured_at: now,
      extension_id: "test-ext-id",
      adapter_name: adapterName,
      adapter_version: "0.1.0",
      capture_mode: "snapshot",
    },
    session: {
      provider,
      provider_session_id: providerSessionId,
      title: "Test Page",
      updated_at: now,
      model,
      turns: turns.map((turn, ordinal) => ({
        provider_turn_id:
          turn.provider_turn_id ||
          `${providerSessionId}:turn:${ordinal}:${fnv1a(turn.role + ":" + turn.text)}`,
        role: turn.role,
        text: turn.text,
        ordinal,
        provider_meta: turn.provider_meta || {},
      })),
    },
  };
}

// ---------------------------------------------------------------------------
// Tests: fnv1a
// ---------------------------------------------------------------------------

describe("fnv1a", () => {
  it("produces deterministic hashes", () => {
    expect(fnv1a("hello")).toBe(fnv1a("hello"));
  });

  it("produces different hashes for different inputs", () => {
    expect(fnv1a("hello")).not.toBe(fnv1a("world"));
  });

  it("returns 8-char hex strings", () => {
    const h = fnv1a("test");
    expect(h).toHaveLength(8);
    expect(h).toMatch(/^[0-9a-f]{8}$/);
  });

  it("handles empty string", () => {
    const h = fnv1a("");
    expect(h).toHaveLength(8);
  });
});

// ---------------------------------------------------------------------------
// Tests: sessionIdFromUrl
// ---------------------------------------------------------------------------

describe("sessionIdFromUrl", () => {
  it("extracts last path segment as token", () => {
    const id = sessionIdFromUrl(
      "chatgpt",
      "https://chatgpt.com/c/abc-123",
    );
    expect(id).toMatch(/^chatgpt:abc-123:/);
  });

  it("includes provider prefix", () => {
    const id = sessionIdFromUrl(
      "claude-ai",
      "https://claude.ai/chat/test-conv",
    );
    expect(id).toMatch(/^claude-ai:test-conv:/);
  });

  it("handles URL with query params", () => {
    const id = sessionIdFromUrl(
      "chatgpt",
      "https://chatgpt.com/c/conv?q=1",
    );
    expect(id).toMatch(/^chatgpt:conv:/);
  });
});

// ---------------------------------------------------------------------------
// Tests: visibleText
// ---------------------------------------------------------------------------

describe("visibleText", () => {
  it("extracts text from a DOM node", () => {
    const dom = new JSDOM(
      "<!DOCTYPE html><div>Hello <span>World</span></div>",
    );
    const div = dom.window.document.querySelector("div");
    expect(visibleText(div)).toBe("Hello World");
  });

  it("returns empty string for empty node", () => {
    const dom = new JSDOM(
      "<!DOCTYPE html><div></div>",
    );
    const div = dom.window.document.querySelector("div");
    expect(visibleText(div)).toBe("");
  });

  it("returns empty string for null node", () => {
    expect(visibleText(null)).toBe("");
  });

  it("returns empty string for undefined node", () => {
    expect(visibleText(undefined)).toBe("");
  });

  it("collapses whitespace before newlines", () => {
    const dom = new JSDOM(
      "<!DOCTYPE html><div>line one  \nline two</div>",
    );
    const div = dom.window.document.querySelector("div");
    expect(visibleText(div)).toBe("line one\nline two");
  });
});

// ---------------------------------------------------------------------------
// Tests: buildEnvelope
// ---------------------------------------------------------------------------

describe("buildEnvelope", () => {
  it("produces a valid envelope shape", () => {
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [{ role: "user", text: "Hello" }],
    });

    expect(envelope.polylogue_capture_kind).toBe("browser_llm_session");
    expect(envelope.schema_version).toBe(1);
    expect(envelope.source).toBe("browser-extension");
    expect(envelope.provenance.adapter_name).toBe("chatgpt-dom-v1");
    expect(envelope.provenance.capture_mode).toBe("snapshot");
    expect(envelope.session.provider).toBe("chatgpt");
    expect(envelope.session.turns).toHaveLength(1);
  });

  it("assigns ordinals to turns", () => {
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [
        { role: "user", text: "Q1" },
        { role: "assistant", text: "A1" },
      ],
    });

    expect(envelope.session.turns[0].ordinal).toBe(0);
    expect(envelope.session.turns[1].ordinal).toBe(1);
  });

  it("generates provider_turn_id when not provided", () => {
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [{ role: "user", text: "Hello" }],
    });

    expect(envelope.session.turns[0].provider_turn_id).toBeTruthy();
    expect(envelope.session.turns[0].provider_turn_id).toContain(":turn:0:");
  });

  it("respects provided provider_turn_id", () => {
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [{ role: "user", text: "Hello", provider_turn_id: "custom-id" }],
    });

    expect(envelope.session.turns[0].provider_turn_id).toBe("custom-id");
  });

  it("passes through provider_meta on turns", () => {
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [
        {
          role: "user",
          text: "Hello",
          provider_meta: { selector_index: 0 },
        },
      ],
    });

    expect(envelope.session.turns[0].provider_meta).toEqual({
      selector_index: 0,
    });
  });

  it("accepts null model", () => {
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [{ role: "user", text: "Hello" }],
      model: null,
    });

    expect(envelope.session.model).toBeNull();
  });
});
