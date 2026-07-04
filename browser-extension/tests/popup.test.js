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
      <span id="updated"></span>
      <span id="receiver"></span>
      <span id="page"></span>
      <span id="archive"></span>
      <input id="receiver-url" />
      <input id="receiver-token" />
      <p id="state-detail"></p>
      <button id="check"><span class="button-status"></span></button>
      <button id="save"><span class="button-status"></span></button>
      <button id="capture"><span class="button-status"></span></button>
      <button id="sync-open-tabs"><span class="button-status"></span></button>
      <button id="copy-ref"><span class="button-status"></span></button>
      <button id="open-polylogue"><span class="button-status"></span></button>
      <button id="debug-toggle"><span class="button-status"></span></button>
      <button id="debug-export"><span class="button-status"></span></button>
      <span id="mode"></span>
      <span id="turns"></span>
      <span id="log-count"></span>
      <span id="debug-count"></span>
      <div id="log"></div>
      <div id="debug-panel" hidden><div id="debug-log"></div></div>
    </body>`);
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  globalThis.Blob = dom.window.Blob;
  globalThis.URL = dom.window.URL;
  globalThis.URL.createObjectURL = vi.fn(() => "blob:debug");
  globalThis.URL.revokeObjectURL = vi.fn();
  dom.window.HTMLAnchorElement.prototype.click = vi.fn();
}

function installChromeMock(storagePatch = {}) {
  let captureAttempts = 0;
  const defaults = {
    polylogueCaptureLog: [],
    polylogueDebugLog: [],
    polylogueState: null,
    receiverAuthToken: "",
    receiverBaseUrl: "http://127.0.0.1:8765",
    ...storagePatch,
  };
  globalThis.chrome = {
    runtime: {
      sendMessage: vi.fn(async () => ({ ok: true })),
    },
    scripting: {
      executeScript: vi.fn(async () => undefined),
    },
    storage: {
      local: {
        get: vi.fn(async (requestedDefaults) => ({ ...requestedDefaults, ...defaults })),
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

async function loadPopup(storagePatch = {}) {
  vi.resetModules();
  installDom();
  installChromeMock(storagePatch);
  await import("../src/popup.js");
  await vi.waitFor(() => expect(globalThis.document.getElementById("page").textContent).toContain("ChatGPT"));
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

  it("refreshes receiver status automatically on popup open", async () => {
    await loadPopup();

    await vi.waitFor(() => expect(globalThis.chrome.runtime.sendMessage).toHaveBeenCalledWith({
      type: "polylogue.status",
      reason: "popup_open",
    }));
  });

  it("renders stale archive state with operator-facing explanation", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        archive_state: { state: "stale", indexed_message_count: 12 },
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("badge").textContent).toBe("stale");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Stale");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("daemon has not caught up");
  });

  it("renders DOM fallback with concrete next action", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        capture_mode: "dom_degraded",
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("badge").textContent).toBe("dom");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("provider-native app data");
  });

  it("renders redacted debug log entries and export control", async () => {
    await loadPopup({
      polylogueDebugLog: [
        {
          at: new Date().toISOString(),
          stage: "receiver_response",
          method: "POST",
          path: "/v1/browser-captures",
          ok: true,
          status: 202,
          provider: "chatgpt",
          provider_session_id: "conv-123",
          receiver_request_id: "req-1",
        },
      ],
    });

    expect(globalThis.document.getElementById("debug-count").textContent).toBe("1");
    expect(globalThis.document.getElementById("debug-log").textContent).toContain("receiver_response POST /v1/browser-captures");

    globalThis.document.getElementById("debug-export").click();
    await vi.waitFor(() => expect(globalThis.URL.createObjectURL).toHaveBeenCalled());
  });
});
