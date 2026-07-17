import { readFileSync } from "node:fs";

import { JSDOM } from "jsdom";
import { expect, it, vi } from "vitest";

const commonSource = readFileSync("src/common.js", "utf8");
const grokSource = readFileSync("src/content/grok.js", "utf8");

it("forwards an automatic capture reason through the Grok content handler", async () => {
  const dom = new JSDOM('<article data-message-author-role="user">Capture this</article>', {
    runScripts: "outside-only",
    url: "https://grok.com/chat/conversation-1",
  });
  const runtimeListener = vi.fn();
  const sendMessage = vi.fn(async (message) => {
    if (message.type === "polylogue.capture") return { ok: true, state: "spooled_only" };
    if (message.type === "polylogue.archiveState") return { state: "spooled_only" };
    return null;
  });
  dom.window.chrome = {
    runtime: {
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: runtimeListener },
      sendMessage,
    },
  };

  dom.window.eval(commonSource);
  dom.window.eval(grokSource);
  const listener = runtimeListener.mock.calls[0][0];
  const response = await new Promise((resolve) => {
    expect(listener({ type: "polylogue.capturePage", reason: "auto_capture_missing" }, {}, resolve)).toBe(true);
  });

  expect(response).toEqual({ ok: true, envelope: expect.any(Object), captureResult: expect.any(Object), archiveState: expect.any(Object) });
  expect(sendMessage).toHaveBeenNthCalledWith(1, expect.objectContaining({
    type: "polylogue.capture",
    reason: "auto_capture_missing",
  }));
});

it("reports a rejected runtime capture without refreshing archive state", async () => {
  const dom = new JSDOM('<article data-message-author-role="user">Capture this</article>', {
    runScripts: "outside-only",
    url: "https://grok.com/chat/conversation-1",
  });
  const runtimeListener = vi.fn();
  const sendMessage = vi.fn(async () => ({ ok: false, error: "capture_rejected" }));
  dom.window.chrome = {
    runtime: {
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: runtimeListener },
      sendMessage,
    },
  };

  dom.window.eval(commonSource);
  dom.window.eval(grokSource);
  const listener = runtimeListener.mock.calls[0][0];
  const response = await new Promise((resolve) => {
    listener({ type: "polylogue.capturePage", reason: "auto_capture_missing" }, {}, resolve);
  });

  expect(response).toMatchObject({ ok: false, timelineRecorded: true, error: "capture_rejected" });
  expect(sendMessage).toHaveBeenCalledTimes(1);
});
