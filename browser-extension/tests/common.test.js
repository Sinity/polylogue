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

function buildEnvelope({
  provider,
  adapterName,
  turns,
  model = null,
  providerSessionId = null,
  title = null,
  createdAt = null,
  updatedAt = null,
  providerMeta = {},
  rawProviderPayload = null,
}) {
  const sourceUrl = "https://chatgpt.com/c/test-conv";
  const stableProviderSessionId =
    providerSessionId || sessionIdFromUrl(provider, sourceUrl);
  const stableCaptureId = stableProviderSessionId.startsWith(`${provider}:`)
    ? stableProviderSessionId
    : `${provider}:${stableProviderSessionId}`;
  const now = new Date().toISOString();
  const envelope = {
    polylogue_capture_kind: "browser_llm_session",
    schema_version: 1,
    capture_id: stableCaptureId,
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
      provider_session_id: stableProviderSessionId,
      title: title || "Test Page",
      created_at: createdAt,
      updated_at: updatedAt || now,
      model,
      provider_meta: providerMeta,
      turns: turns.map((turn, ordinal) => ({
        provider_turn_id:
          turn.provider_turn_id ||
          `${stableProviderSessionId}:turn:${ordinal}:${fnv1a(turn.role + ":" + turn.text)}`,
        role: turn.role,
        text: turn.text,
        timestamp: turn.timestamp || null,
        ordinal,
        parent_turn_id: turn.parent_turn_id || null,
        provider_meta: turn.provider_meta || {},
      })),
    },
  };
  if (rawProviderPayload && typeof rawProviderPayload === "object") {
    envelope.raw_provider_payload = rawProviderPayload;
  }
  return envelope;
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

  it("does not duplicate provider prefix in capture id", () => {
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [{ role: "user", text: "Hello" }],
    });
    expect(envelope.session.provider_session_id).toMatch(/^chatgpt:/);
    expect(envelope.capture_id).toBe(envelope.session.provider_session_id);
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

  it("uses native provider ids and carries raw provider payloads", () => {
    const rawProviderPayload = {
      id: "native-conv",
      mapping: {
        node: {
          message: {
            id: "native-message",
            author: { role: "user" },
            content: { parts: ["Hello"] },
          },
        },
      },
    };
    const envelope = buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-native-v1",
      providerSessionId: "native-conv",
      title: "Native title",
      createdAt: "2026-06-14T13:14:26.000Z",
      updatedAt: "2026-06-21T12:04:19.000Z",
      turns: [
        {
          provider_turn_id: "native-message",
          role: "user",
          text: "Hello",
          timestamp: "2026-06-14T13:14:30.000Z",
          parent_turn_id: "root",
        },
      ],
      providerMeta: { capture_source: "chatgpt_backend_api" },
      rawProviderPayload,
    });

    expect(envelope.capture_id).toBe("chatgpt:native-conv");
    expect(envelope.session.provider_session_id).toBe("native-conv");
    expect(envelope.session.title).toBe("Native title");
    expect(envelope.session.created_at).toBe("2026-06-14T13:14:26.000Z");
    expect(envelope.session.updated_at).toBe("2026-06-21T12:04:19.000Z");
    expect(envelope.session.provider_meta).toEqual({
      capture_source: "chatgpt_backend_api",
    });
    expect(envelope.session.turns[0].timestamp).toBe(
      "2026-06-14T13:14:30.000Z",
    );
    expect(envelope.session.turns[0].parent_turn_id).toBe("root");
    expect(envelope.raw_provider_payload).toBe(rawProviderPayload);
  });
});
