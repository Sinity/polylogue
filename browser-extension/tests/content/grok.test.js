import { stagingRuntime } from "../infra/capture-staging.js";
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
const assetStreamSource = readFileSync(resolve(testDirectory, "../../src/content/asset_stream.js"), "utf8");
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
    randomUUID: () => webcrypto.randomUUID(),
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

function installFullCapture(fetchImpl, { url, captureResult = null } = {}) {
  const dom = makeDom(fetchImpl, url);
  const runtimeMessages = [];
  const runtimeListeners = [];
  const chrome = {
    runtime: {
      id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
        const assetResult = await dom.__captureRuntime.sendMessage(message);
        if (assetResult !== undefined) return assetResult;
        runtimeMessages.push(message);
        if (message.type === "polylogue.capture") {
          return captureResult || { ok: true, provider: "grok", provider_session_id: conversationId, receiver_request_id: "synthetic-request" };
        }
        if (message.type === "polylogue.archiveState") return { captured: true, state: "archived" };
        return { ok: true };
      },
    },
  };
  dom.__captureRuntime = stagingRuntime(undefined, { tab_id: 42, document_id: "synthetic-document", provider: "grok" });
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
  new Script(assetStreamSource).runInContext(context);
  new Script(bridgeSource).runInContext(context);
  new Script(commonSource).runInContext(context);
  new Script(contentSource).runInContext(context);
  function sendRuntimeMessage(message) {
    return new Promise((resolve, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolve) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  dom.__captureRuntime.setDispatch(sendRuntimeMessage);
  return { dom, runtimeMessages, sendRuntimeMessage: async (message) => {
    const result = await sendRuntimeMessage(message);
    if (result?.envelope) result.envelope = await dom.__captureRuntime.materialize(result.envelope);
    return result;
  } };
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
  it("cancels and drains concurrent record asset acquisitions sharing a raw revision", async () => {
    const { dom, sendRuntimeMessage } = installFullCapture(conversationFetchImpl());
    let started; const ready = new Promise((resolve) => { started = resolve; });
    let requests = 0; let cancelled = 0;
    dom.window.polylogueAssetStream.request = ({ signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => { cancelled += 1; reject(signal.reason); }, { once: true });
      if (++requests === 2) started();
    });
    const request = { type: "polylogue.acquireRecordAssets", provider: "grok", nativeId: "session", capture_ref: "same-raw-revision", recordKey: "first", attachmentOrdinal: 0, attachments: [{ provider_attachment_id: "asset", url: "file/asset", name: "asset.txt" }] };
    const first = sendRuntimeMessage(request);
    const second = sendRuntimeMessage({ ...request, recordKey: "second" });
    await ready;
    expect(await sendRuntimeMessage({ type: "polylogue.cancelRecordAssets", capture_ref: request.capture_ref }))
      .toMatchObject({ ok: true, outcome: "cancelled" });
    expect(cancelled).toBe(2);
    expect(await first).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect(await second).toMatchObject({ ok: false, error: "capture_cancelled" });
  });

  it("retains original text and reasoning steps for canonical preparation and forwards capture reason", async () => {
    const { dom, runtimeMessages, sendRuntimeMessage } = installFullCapture(conversationFetchImpl());
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "auto_capture_missing" });

    expect(response.ok).toBe(true);
    expect(response.envelope.receiver_native).toBeTruthy();
    expect(response.envelope.session.provider_session_id).toBe(conversationId);
    expect(response.envelope.session.session_kind).toBe("standard");
    const retained = await dom.__captureRuntime.retainedNativeReplies(response.envelope);
    expect(retained.responses.responses).toEqual([humanResponse(), assistantResponse()]);
    expect(response.envelope.receiver_native).toBeDefined();
    expect(response.envelope.session.turns).toEqual([]);
    expect(runtimeMessages.filter((message) => message.type === "polylogue.capture")).toEqual([expect.objectContaining({ reason: "auto_capture_missing" })]);
  });

  it("uses the canonical temporary session kind while retaining the original provider flag", async () => {
    const { dom, sendRuntimeMessage } = installFullCapture(
      conversationFetchImpl({ conversation: conversationMetadata({ temporary: true }) }),
    );
    dom.__captureRuntime.nativeContract.summary = { session_kind: "temporary" };
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(true);
    expect(response.envelope.session.session_kind).toBe("temporary");
    expect((await dom.__captureRuntime.retainedNativeReplies(response.envelope)).conversation.temporary).toBe(true);
  });

  it("acquires a file attachment's bytes through assets.grok.com and verifies its sha256", async () => {
    const withAttachment = humanResponse({
      fileAttachments: ["asset-1"],
      fileUris: ["asset-1"],
      fileAttachmentsMetadata: [{ fileMetadataId: "asset-1", fileMimeType: "text/markdown", fileName: "notes.md", fileUri: "users/u1/asset-1/content", fileSource: "SELF_UPLOAD_FILE_SOURCE" }],
    });
    const { dom, sendRuntimeMessage } = installFullCapture(
      conversationFetchImpl({ responses: [withAttachment, assistantResponse()], asset: assetBytes }),
    );
    dom.__captureRuntime.nativeContract.plan = [{ ordinal: 0, descriptor: {
      provider_attachment_id: "asset-1", message_provider_id: "r-human-1", name: "notes.md", mime_type: "text/markdown",
      url: "users/u1/asset-1/content", original_record_ordinal: 0, provider_meta: { native_turn_ordinal: 0 },
    } }];
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(true);
    const acquired = dom.__captureRuntime.nativeContract.receipts[0].result.attachments[0];
    expect(acquired.provider_attachment_id).toBe("asset-1");
    expect(acquired.name).toBe("notes.md");
    expect(acquired.provider_meta.content_sha256).toBe(expectedSha256);
    expect(await (await dom.__captureRuntime.staging.file(acquired.staged_asset.id)).text()).toBe("polylogue grok attachment fixture\n");
  });

  it("retains unrecognized toolResponses verbatim for canonical receiver diagnostics", async () => {
    const withOddTool = assistantResponse({ toolResponses: [{ weird_shape: true, payload: [1, 2, 3] }] });
    const { dom, sendRuntimeMessage } = installFullCapture(conversationFetchImpl({ responses: [humanResponse(), withOddTool] }));
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(true);
    const retained = await dom.__captureRuntime.retainedNativeReplies(response.envelope);
    expect(retained.responses.responses[1].toolResponses).toEqual(withOddTool.toolResponses);
    expect(response.envelope.receiver_native).toBeDefined();
  });

  it("retains web search evidence verbatim for canonical receiver tool blocks", async () => {
    const searchResponse = humanResponse({
      responseId: "r-search",
      query: "latest EU battery regulations",
      queryType: "web",
      webSearchResults: [{ url: "https://example.test/a", title: "A" }],
    });
    const { dom, sendRuntimeMessage } = installFullCapture(conversationFetchImpl({ responses: [searchResponse, assistantResponse({ parentResponseId: "r-search" })] }));
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    const retained = await dom.__captureRuntime.retainedNativeReplies(response.envelope);
    expect(retained.responses.responses[0]).toEqual(searchResponse);
    expect(response.envelope.receiver_native).toBeDefined();
  });

  it("fails loud instead of sending an empty capture when the responses endpoint is unavailable", async () => {
    const { runtimeMessages, sendRuntimeMessage } = installFullCapture(conversationFetchImpl({ responsesStatus: 404 }));
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(false);
    expect(response.error).toBe("native_capture_unavailable");
    expect(response.native_attempts.length).toBeGreaterThan(0);
    let repeated = response;
    for (let attempt = 0; attempt < 8; attempt++) repeated = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    expect(repeated).toMatchObject({ ok: false, error: "native_capture_unavailable", native_attempts_dropped: 1 });
    expect(repeated.native_attempts).toHaveLength(8);
    expect(runtimeMessages.filter((message) => message.type === "polylogue.capture")).toHaveLength(0);
  });

  it("fails loud when no conversation id is present in the URL, without ever sending a capture", async () => {
    const { runtimeMessages, sendRuntimeMessage } = installFullCapture(conversationFetchImpl(), { url: "https://grok.com/" });
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });

    expect(response.ok).toBe(false);
    expect(response.error).toBe("native_capture_unavailable");
    expect(runtimeMessages.filter((message) => message.type === "polylogue.capture")).toHaveLength(0);
  });

  it("reports a rejected runtime capture without refreshing archive state", async () => {
    const { sendRuntimeMessage, runtimeMessages } = installFullCapture(conversationFetchImpl(), {
      captureResult: { ok: false, error: "capture_rejected" },
    });
    const response = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    expect(response).toMatchObject({ ok: false, timelineRecorded: true, error: "capture_rejected" });
    expect(runtimeMessages.filter((message) => message.type === "polylogue.capture")).toHaveLength(1);
    expect(runtimeMessages.some((message) => message.type === "polylogue.archiveState")).toBe(false);
  });
});
