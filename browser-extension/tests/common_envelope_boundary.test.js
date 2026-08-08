/**
 * Boundary contract test for src/common.js's buildEnvelope: a capture
 * adapter (chatgpt.js/claude.js/grok.js) can populate every field the wire
 * format (polylogue/browser_capture/models.py's BrowserCaptureTurn /
 * BrowserCaptureBlock) defines, and buildEnvelope must carry every one of
 * them into the envelope unmodified. It must not re-introduce a field
 * allowlist that silently drops whatever it does not enumerate.
 *
 * This is a regression guard for exactly that failure mode: buildEnvelope's
 * turns.map projection used to hand-list a fixed set of output keys with no
 * `blocks` entry at all, so every structured block a content script built
 * (tool_use/tool_result/thinking/...) was silently discarded at the
 * envelope boundary while `text` alone kept looking like a complete,
 * successful capture (polylogue-ah21 regressed this once already).
 *
 * Loads the REAL src/common.js (via dom.window.eval, the same pattern
 * tests/content/grok.test.js already uses) rather than a hand-copied
 * function -- a local copy of buildEnvelope would test itself, not the
 * production code, which is exactly how the blocks drop went unnoticed for
 * as long as it did (tests/content/chatgpt.test.js and
 * tests/content/claude.test.js both had this problem too; see their
 * changes in this same commit).
 *
 * Derive-vs-hardcode: the ideal version of this test derives its expected
 * field set from polylogue/browser_capture/models.py's
 * BrowserCaptureTurn/BrowserCaptureBlock Pydantic models directly, so
 * adding a field on the Python side fails this JS test until the JS
 * projection carries it too. That was not done here: there is no existing
 * pattern anywhere in this repo for a vitest test shelling out to Python,
 * no CI job wires the Python venv into the browser-extension test run, and
 * `npx vitest run` is documented (README.md) as a plain `npm`-only command
 * a contributor can run with zero Python setup. Making this one test
 * conditionally require an importable `polylogue` package would make an
 * otherwise-unrelated JS change fail (or silently skip, which defeats the
 * point) in exactly that environment. Falling back to a hardcoded maximal
 * turn/block round-trip is a real regression guard for the allowlist-drift
 * class today; keeping the field lists below in sync with models.py when
 * either side changes is a known, accepted manual cost until that
 * cross-language plumbing is built as its own piece of work.
 */

import { readFileSync } from "node:fs";

import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const commonSource = readFileSync("src/common.js", "utf8");

// Mirrors polylogue/browser_capture/models.py::BrowserCaptureBlock field by
// field. `type` is required and typed (BlockType); everything else is
// optional. ordinal/id-shaped identity fields are not part of this model.
const BROWSER_CAPTURE_BLOCK_FIELDS = [
  "type",
  "text",
  "tool_name",
  "tool_id",
  "tool_input",
  "media_type",
  "metadata",
  "is_error",
  "exit_code",
];

// Mirrors polylogue/browser_capture/models.py::BrowserCaptureTurn field by
// field. `ordinal` is populated by buildEnvelope itself (the turn's index),
// not read from the caller's turn object, but is still part of the wire
// contract this test guards.
const BROWSER_CAPTURE_TURN_FIELDS = [
  "provider_turn_id",
  "role",
  "text",
  "timestamp",
  "ordinal",
  "parent_turn_id",
  "attachments",
  "blocks",
  "provider_meta",
];

function installCommon(url = "https://chatgpt.com/c/conversation-1") {
  const dom = new JSDOM("<title>Boundary contract fixture</title>", {
    runScripts: "outside-only",
    url,
  });
  dom.window.chrome = {
    runtime: {
      id: "synthetic-extension-id",
      getManifest: () => ({ version: "0.1.0" }),
    },
  };
  dom.window.eval(commonSource);
  return dom;
}

function maximalBlock() {
  return {
    type: "tool_use",
    text: "block prose rendering",
    tool_name: "shell",
    tool_id: "tool-call-1",
    tool_input: { command: "ls -la", cwd: "/tmp" },
    media_type: "application/json",
    metadata: { content_type: "code", extra_nested: { deep: true } },
    is_error: false,
    exit_code: 0,
  };
}

function maximalTurn() {
  return {
    provider_turn_id: "turn-1",
    role: "assistant",
    text: "turn prose",
    timestamp: "2026-01-01T00:00:00.000Z",
    parent_turn_id: "turn-0",
    attachments: [
      {
        provider_attachment_id: "attachment-1",
        name: "report.pdf",
        mime_type: "application/pdf",
        size_bytes: 4096,
        url: "https://files.example.test/report.pdf",
      },
    ],
    blocks: [maximalBlock()],
    provider_meta: { node_id: "node-1", content_type: "tool_use" },
  };
}

describe("common.js buildEnvelope boundary contract (real source, not a copy)", () => {
  it("types provider, page, and session-id title evidence", () => {
    const providerDom = installCommon();
    const providerEnvelope = providerDom.window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-native-v1",
      turns: [maximalTurn()],
      providerSessionId: "conversation-1",
      title: "Provider conversation title",
    });
    expect(providerEnvelope.session.title).toBe("Provider conversation title");
    expect(providerEnvelope.session.title_source).toBe("provider");

    const pageDom = installCommon();
    const pageEnvelope = pageDom.window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [maximalTurn()],
      providerSessionId: "conversation-1",
    });
    expect(pageEnvelope.session.title).toBe("Boundary contract fixture");
    expect(pageEnvelope.session.title_source).toBe("page");

    const sessionIdDom = installCommon();
    sessionIdDom.window.document.title = "";
    const sessionIdEnvelope = sessionIdDom.window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-dom-v1",
      turns: [maximalTurn()],
      providerSessionId: "conversation-1",
    });
    expect(sessionIdEnvelope.session.title).toBe("conversation-1");
    expect(sessionIdEnvelope.session.title_source).toBe("session-id");
  });

  it("carries every BrowserCaptureTurn field from a maximal turn into the envelope", () => {
    const dom = installCommon();
    const envelope = dom.window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-native-v1",
      turns: [maximalTurn()],
      providerSessionId: "conversation-1",
    });

    const [turn] = envelope.session.turns;
    expect(Object.keys(turn).sort()).toEqual([...BROWSER_CAPTURE_TURN_FIELDS].sort());
    expect(turn).toMatchObject({
      provider_turn_id: "turn-1",
      role: "assistant",
      text: "turn prose",
      timestamp: "2026-01-01T00:00:00.000Z",
      ordinal: 0,
      parent_turn_id: "turn-0",
      provider_meta: { node_id: "node-1", content_type: "tool_use" },
    });
    expect(turn.attachments).toEqual(maximalTurn().attachments);
    expect(turn.blocks).toEqual(maximalTurn().blocks);
  });

  it("carries every BrowserCaptureBlock field through unmodified", () => {
    const dom = installCommon();
    const envelope = dom.window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-native-v1",
      turns: [maximalTurn()],
      providerSessionId: "conversation-1",
    });

    const [block] = envelope.session.turns[0].blocks;
    expect(Object.keys(block).sort()).toEqual([...BROWSER_CAPTURE_BLOCK_FIELDS].sort());
    expect(block).toEqual(maximalBlock());
  });

  it("still defaults blocks/attachments to an empty array for a turn that observed neither", () => {
    const dom = installCommon();
    const envelope = dom.window.polylogueCapture.buildEnvelope({
      provider: "chatgpt",
      adapterName: "chatgpt-native-v1",
      turns: [{ provider_turn_id: "turn-text-only", role: "user", text: "hello" }],
      providerSessionId: "conversation-1",
    });

    const [turn] = envelope.session.turns;
    expect(turn.blocks).toEqual([]);
    expect(turn.attachments).toEqual([]);
  });
});
