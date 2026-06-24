import { JSDOM } from "jsdom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const CHATGPT_TAB = {
  id: 42,
  title: "ChatGPT conversation",
  url: "https://chatgpt.com/c/test-conversation",
};

function installDom() {
  const dom = new JSDOM(`<!doctype html>
    <body>
      <span id="badge"></span>
      <span id="state"></span>
      <span id="receiver-request"></span>
      <span id="receiver"></span>
      <span id="page"></span>
      <span id="archive"></span>
      <input id="receiver-url" />
      <input id="receiver-token" />
      <button id="check"></button>
      <button id="save"></button>
      <button id="capture"></button>
    </body>`);
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
}

function installChromeMock() {
  let captureAttempts = 0;
  globalThis.chrome = {
    runtime: {
      sendMessage: vi.fn(async () => ({ ok: true })),
    },
    scripting: {
      executeScript: vi.fn(async () => undefined),
    },
    storage: {
      local: {
        get: vi.fn(async (defaults) => ({ ...defaults })),
        set: vi.fn(async () => undefined),
      },
    },
    tabs: {
      query: vi.fn(async () => [CHATGPT_TAB]),
      sendMessage: vi.fn(async () => {
        captureAttempts += 1;
        if (captureAttempts === 1) {
          throw new Error("Could not establish connection. Receiving end does not exist.");
        }
        return {
          ok: true,
          captureResult: { ok: true },
          archiveState: { captured: true },
        };
      }),
    },
  };
}

async function loadPopup() {
  vi.resetModules();
  installDom();
  installChromeMock();
  await import("../src/popup.js");
  await vi.waitFor(() => expect(globalThis.document.getElementById("page").textContent).toBe("ChatGPT"));
}

describe("popup capture", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("injects provider content scripts before retrying an already-open tab capture", async () => {
    await loadPopup();

    globalThis.document.getElementById("capture").click();

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(2));
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledTimes(2);
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenNthCalledWith(1, {
      target: { tabId: CHATGPT_TAB.id },
      files: ["src/common.js"],
    });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenNthCalledWith(2, {
      target: { tabId: CHATGPT_TAB.id },
      files: ["src/content/chatgpt.js"],
    });
    expect(globalThis.chrome.storage.local.set).not.toHaveBeenCalledWith(
      expect.objectContaining({
        polylogueState: expect.objectContaining({ online: false }),
      }),
    );
  });
});
