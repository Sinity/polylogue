import { Buffer } from "node:buffer";
import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { TextEncoder } from "node:util";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const testDirectory = dirname(fileURLToPath(import.meta.url));
const bridgeSource = readFileSync(resolve(testDirectory, "../../src/content/grok_bridge.js"), "utf8");
const commonSource = readFileSync(resolve(testDirectory, "../../src/common.js"), "utf8");
const contentSource = readFileSync(resolve(testDirectory, "../../src/content/grok.js"), "utf8");
const conversationId = "1f9de430-6505-4d43-935b-ec0dd1c13222";
const assetBytes = new TextEncoder().encode("polylogue grok attachment fixture\n");
const expectedSha256 = createHash("sha256").update(assetBytes).digest("hex");
const openDoms = [];

function jsonResponse(body, status = 200) {
  return new globalThis.Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function byteResponse(bytes, status = 200) {
  return new globalThis.Response(bytes, { status, headers: { "content-type": "text/markdown" } });
}

function conversationMetadata(overrides = {}) {
  return {
    conversationId,
    title: "Kane's insatiable seed: breeding legacy",
    temporary: false,
    createTime: "2026-06-27T12:39:08.985242Z",
    modifyTime: "2026-06-27T13:46:12.022Z",
    ...overrides,
  };
}

function humanResponse(overrides = {}) {
  return {
    responseId: "r-human-1",
    sender: "human",
    message: "Do write much better story",
    createTime: "2026-06-27T12:39:09.006Z",
    model: "grok-3",
    ...overrides,
  };
}

function assistantResponse(overrides = {}) {
  return {
    responseId: "r-assistant-1",
    parentResponseId: "r-human-1",
    sender: "ASSISTANT",
    message: "The Seed\n\nKane Voss was born restless.",
    createTime: "2026-06-27T12:39:59.733Z",
    model: "grok-3",
    steps: [
      { text: ["Thinking about your request"], tags: ["header", "thinking_start"] },
      { text: ["Writing the improved story"], tags: ["header"] },
    ],
    ...overrides,
  };
}

function makeDom(fetchImpl, url = `https://grok.com/c/${conversationId}`) {
  const dom = new JSDOM("<!doctype html><title>Grok fixture</title>", { url, runScripts: "outside-only" });
  openDoms.push(dom);
  const cryptoAdapter = {
    subtle: {
      digest(algorithm, data) {
        return webcrypto.subtle.digest(algorithm, Buffer.from(new dom.window.Uint8Array(data)));
      },
      getRandomValues: (array) => webcrypto.getRandomValues(array),
    },
    getRandomValues: (array) => webcrypto.getRandomValues(array),
  };
  Object.defineProperty(dom.window, "crypto", { configurable: true, value: cryptoAdapter });
  Object.defineProperty(dom.window, "fetch", { configurable: true, value: fetchImpl });
  return dom;
}

function installFullCapture(fetchImpl, { url } = {}) {
  const dom = makeDom(fetchImpl, url);
  const runtimeMessages = [];
  const runtimeListeners = [];
  const chrome = {
    runtime: {
      id: "synthetic-extension-id",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
        runtimeMessages.push(message);
        if (message.type === "polylogue.capture") {
          return { ok: true, provider: "grok", provider_session_id: conversationId, receiver_request_id: "synthetic-request" };
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
    return new Promise((resolve, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolve) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  return { dom, runtimeMessages, sendRuntimeMessage };
}

function conversationFetchImpl({ conversation = conversationMetadata(), responses = [humanResponse(), assistantResponse()], responsesStatus = 200, asset = null } = {}) {
  return vi.fn(async (input, options = {}) => {
    const url = new URL(String(input));
    if (url.hostname === "grok.com" && url.pathname === `/rest/app-chat/conversations/${conversationId}`) {
      return jsonResponse(conversation);
    }
    if (url.hostname === "grok.com" && url.pathname === `/rest/app-chat/conversations/${conversationId}/responses`) {
      return responsesStatus === 200 ? jsonResponse({ responses }) : jsonResponse({ code: 5 }, responsesStatus);
    }
    if (url.hostname === "grok.com" && url.pathname === `/rest/app-chat/conversations/${conversationId}/response-node`) {
      return jsonResponse({ responseNodes: [], inflightResponses: [] });
    }
    if (url.hostname === "assets.grok.com") {
      if (!asset) return new globalThis.Response("", { status: 404 });
      expect(options.credentials).toBe("include");
      return byteResponse(asset);
    }
    throw new Error(`unexpected request: ${url.href}`);
  });
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("Grok native capture end to end", () => {
  it("captures message text, reasoning steps as thinking blocks, and forwards the capture reason", async () => {
    const { runtimeMessages, sendRuntimeMessage } = installFullCapture(conversationFetchImpl());
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "auto_capture_missing" });

    expect(response.ok).toBe(true);
    expect(response.envelope.provenance.adapter_name).toBe("grok-native-v1");
    expect(response.envelope.session.provider_session_id).toBe(conversationId);
    expect(response.envelope.session.session_kind).toBe("standard");
    const turns = response.envelope.session.turns;
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({ role: "user", text: "Do write much better story" });
    expect(turns[1]).toMatchObject({ role: "assistant" });
    expect(turns[1].blocks).toContainEqual(
      expect.objectContaining({ type: "thinking", text: "Thinking about your request" }),
    );
    expect(runtimeMessages[0]).toMatchObject({ type: "polylogue.capture" });
    expect(runtimeMessages[0].reason).toBe("auto_capture_missing");
  });

  it("marks a temporary conversation's session_kind from the provider's own temporary flag", async () => {
    const { sendRuntimeMessage } = installFullCapture(
      conversationFetchImpl({ conversation: conversationMetadata({ temporary: true }) }),
    );
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(true);
    expect(response.envelope.session.session_kind).toBe("temporary");
    expect(response.envelope.session.provider_meta.conversation_temporary).toBe(true);
  });

  it("acquires a file attachment's bytes through assets.grok.com and verifies its sha256", async () => {
    const withAttachment = humanResponse({
      fileAttachments: ["asset-1"],
      fileUris: ["asset-1"],
      fileAttachmentsMetadata: [{ fileMetadataId: "asset-1", fileMimeType: "text/markdown", fileName: "notes.md", fileUri: "users/u1/asset-1/content", fileSource: "SELF_UPLOAD_FILE_SOURCE" }],
    });
    const { sendRuntimeMessage } = installFullCapture(
      conversationFetchImpl({ responses: [withAttachment, assistantResponse()], asset: assetBytes }),
    );
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(true);
    const attachment = response.envelope.session.attachments.find((entry) => entry.provider_attachment_id === "asset-1");
    expect(attachment).toBeTruthy();
    expect(attachment.name).toBe("notes.md");
    expect(attachment.provider_meta.content_sha256).toBe(expectedSha256);
    expect(Buffer.from(attachment.inline_base64, "base64").toString("utf8")).toBe("polylogue grok attachment fixture\n");
  });

  it("keeps an unrecognized toolResponses entry as a flagged diagnostic block instead of dropping it silently", async () => {
    const withOddTool = assistantResponse({ toolResponses: [{ weird_shape: true, payload: [1, 2, 3] }] });
    const { sendRuntimeMessage } = installFullCapture(conversationFetchImpl({ responses: [humanResponse(), withOddTool] }));
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(true);
    const assistantTurn = response.envelope.session.turns.find((turn) => turn.role === "assistant");
    expect(assistantTurn.blocks).toContainEqual(
      expect.objectContaining({ type: "tool_result", metadata: expect.objectContaining({ unrecognized_shape: true, source: "toolResponses" }) }),
    );
  });

  it("projects web search evidence into tool_use/tool_result blocks", async () => {
    const searchResponse = humanResponse({
      responseId: "r-search",
      query: "latest EU battery regulations",
      queryType: "web",
      webSearchResults: [{ url: "https://example.test/a", title: "A" }],
    });
    const { sendRuntimeMessage } = installFullCapture(conversationFetchImpl({ responses: [searchResponse, assistantResponse({ parentResponseId: "r-search" })] }));
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    const searchTurn = response.envelope.session.turns.find((turn) => turn.provider_turn_id === "r-search");
    expect(searchTurn.blocks).toContainEqual(
      expect.objectContaining({ type: "tool_use", tool_name: "web_search", tool_input: { query: "latest EU battery regulations", query_type: "web" } }),
    );
    expect(searchTurn.blocks).toContainEqual(
      expect.objectContaining({ type: "tool_result", tool_name: "web_search", metadata: expect.objectContaining({ field: "webSearchResults", count: 1 }) }),
    );
  });

  it("fails loud instead of sending an empty capture when the responses endpoint is unavailable", async () => {
    const { runtimeMessages, sendRuntimeMessage } = installFullCapture(conversationFetchImpl({ responsesStatus: 404 }));
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(false);
    expect(response.error).toBe("native_capture_unavailable");
    expect(response.native_attempts.length).toBeGreaterThan(0);
    expect(runtimeMessages).toHaveLength(0);
  });

  it("fails loud when no conversation id is present in the URL, without ever sending a capture", async () => {
    const { runtimeMessages, sendRuntimeMessage } = installFullCapture(conversationFetchImpl(), { url: "https://grok.com/" });
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(false);
    expect(response.error).toBe("native_capture_unavailable");
    expect(runtimeMessages).toHaveLength(0);
  });

  it("reports a rejected runtime capture without refreshing archive state", async () => {
    const dom = makeDom(conversationFetchImpl());
    const runtimeListeners = [];
    const chrome = {
      runtime: {
        getManifest: () => ({ version: "0.1.0" }),
        onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
        sendMessage: vi.fn(async (message) => (message.type === "polylogue.capture" ? { ok: false, error: "capture_rejected" } : { ok: true })),
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
    const listener = runtimeListeners[0];
    const response = await new Promise((resolve) => {
      listener({ type: "polylogue.capturePage" }, {}, resolve);
    });

    expect(response).toMatchObject({ ok: false, timelineRecorded: true, error: "capture_rejected" });
  });
});
