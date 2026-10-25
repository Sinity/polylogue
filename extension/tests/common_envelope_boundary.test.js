/**
 * Boundary contract test for src/common.js's buildEnvelope: a capture
 * adapter (chatgpt.js/claude.js/grok.js) can populate every field of a
 * capture turn and block, and buildEnvelope must carry every one of
 * them into the envelope unmodified. It must not re-introduce a field
 * allowlist that silently drops whatever it does not enumerate.
 *
 * A projection that hand-lists output keys without `blocks` would silently
 * discard every structured block a content script built
 * (tool_use/tool_result/thinking/...) while `text` alone still looked like
 * a complete, successful capture.
 *
 * Loads the REAL src/common.js (via dom.window.eval, the same pattern
 * tests/content/grok.test.js already uses) rather than a hand-copied
 * function -- a local copy of buildEnvelope would test itself, not the
 * production code.
 *
 * The field lists below are written out by hand. The receiver's parser
 * (crates/sources/src/browser_capture.rs) reads these turn and block fields;
 * keep both sides in step when either changes.
 */

import { readFileSync } from "node:fs";

import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const commonSource = readFileSync("src/common.js", "utf8");

// Every block field a capture can carry. `type` is required; everything else
// is optional. ordinal/id-shaped identity fields are not block fields.
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

// Every turn field a capture can carry. `ordinal` is populated by buildEnvelope itself (the turn's index),
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
  it("anti-vacuity: emits a full SHA-256 fingerprint matching the digest", () => {
    const dom = installCommon();
    const observation = dom.window.polylogueCapture.identityObservation({
      provider: "chatgpt", conversationId: "c-1", messageId: "m-1", text: "abc",
      adapterName: "fixture", fidelity: "native",
    });
    expect(observation.content_fingerprint)
      .toBe("sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    dom.window.close();
  });

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

  it("anti-vacuity: fallback turn ids survive an inserted turn that differs only by timestamp", () => {
    // A digest over role and text alone puts both tool-only turns in one bucket,
    // so inserting the earlier one shifts the occurrence counter onto the later one.
    const dom = installCommon();
    const later = { role: "assistant", text: "", timestamp: "2026-01-01T00:00:02Z", blocks: [maximalBlock()] };
    const earlier = { ...later, timestamp: "2026-01-01T00:00:01Z" };
    const build = (turns) => dom.window.polylogueCapture.buildEnvelope({ provider: "chatgpt", adapterName: "chatgpt-dom-v1", turns })
      .session.turns.map((turn) => turn.provider_turn_id);
    const [laterAlone] = build([later]);
    const [earlierId, laterId] = build([earlier, later]);
    expect(laterId).toBe(laterAlone);
    expect(earlierId).not.toBe(laterId);
  });
});
