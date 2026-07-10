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
      <button id="check-receiver"><span class="button-status"></span></button>
      <button id="debug-toggle"><span class="button-status"></span></button>
      <button id="debug-export"><span class="button-status"></span></button>
      <span id="mode"></span>
      <span id="fidelity"></span>
      <span id="turns"></span>
      <span id="assets"></span>
      <div id="asset-failures"></div>
      <span id="receiver-health"></span>
      <span id="queue-count"></span>
      <div id="queue-log"></div>
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

  it("renders an unauthorized receiver as a pairing prompt, not a generic offline state", async () => {
    await loadPopup({
      polylogueState: {
        online: false,
        captured: false,
        error: "unauthorized",
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("archive").textContent).toBe("Unauthorized");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("browser-capture token show");
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

  it("renders missing archive state as a capture prompt, not a receiver failure", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        active_page_state: "conversation",
        archive_state: { state: "missing", indexed_message_count: 0 },
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("badge").textContent).toBe("missing");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Not archived");
    expect(globalThis.document.getElementById("state").textContent).toContain("No capture exists");
  });

  it("renders supported pages without a conversation id without implying capture happened", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        active_page_state: "supported_no_session",
        provider: "chatgpt",
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("badge").textContent).toBe("ready");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Ready");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("does not read page content");
  });

  it("renders unsupported pages with a concrete next action", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        active_page_state: "unsupported",
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("badge").textContent).toBe("idle");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Unsupported");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("ChatGPT, Claude.ai, or Grok/X");
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

  it("renders capture fidelity and asset acquisition outcome from the last capture", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: true,
        capture_mode: "native_full",
        turn_count: 12,
        asset_acquisition: {
          attempted: 3,
          acquired: 2,
          failed: [{ provider_attachment_id: "file-abc", error: "timeout" }],
          skipped_over_budget: 0,
        },
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("fidelity").textContent).toBe("Native");
    expect(globalThis.document.getElementById("turns").textContent).toBe("12");
    expect(globalThis.document.getElementById("assets").textContent).toBe("2 acquired · 1 failed");
    expect(globalThis.document.getElementById("asset-failures").textContent).toContain("file-abc: timeout");
  });

  it("renders DOM-fallback fidelity and a no-assets state distinctly", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: true,
        capture_mode: "dom_degraded",
        asset_acquisition: { attempted: 0, acquired: 0, failed: [], skipped_over_budget: 0 },
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("fidelity").textContent).toBe("DOM fallback");
    expect(globalThis.document.getElementById("assets").textContent).toBe("none");
    expect(globalThis.document.getElementById("asset-failures").textContent).toBe("");
  });

  it("renders queued retry entries with attempt and backoff detail", async () => {
    await loadPopup({
      polylogueCaptureQueue: {
        entries: [
          {
            id: "polylogue-ext-1",
            envelope: { session: { provider: "chatgpt", provider_session_id: "conv-9" } },
            attempts: 2,
            enqueued_at: new Date(Date.now() - 60000).toISOString(),
            next_attempt_at: new Date(Date.now() + 120000).toISOString(),
            last_error: "HTTP 503",
          },
        ],
        dropped_count: 1,
      },
    });

    expect(globalThis.document.getElementById("queue-count").textContent).toBe("1 (+1 dropped)");
    expect(globalThis.document.getElementById("queue-log").textContent).toContain("chatgpt conv-9");
    expect(globalThis.document.getElementById("queue-log").textContent).toContain("attempt 2");
    expect(globalThis.document.getElementById("queue-log").textContent).toContain("HTTP 503");
  });

  it("renders an empty queue as an explicit no-op state", async () => {
    await loadPopup();

    expect(globalThis.document.getElementById("queue-count").textContent).toBe("0");
    expect(globalThis.document.getElementById("queue-log").textContent).toContain("No captures queued for retry.");
  });

  it("checks receiver health and shows a reachable-but-unauthorized result", async () => {
    await loadPopup();
    globalThis.chrome.runtime.sendMessage = vi.fn(async (message) => {
      if (message.type === "polylogue.checkReceiverHealth") {
        return { ok: true, status: "unauthorized", detail: "unauthorized" };
      }
      return { ok: true };
    });

    globalThis.document.getElementById("check-receiver").click();

    await vi.waitFor(() =>
      expect(globalThis.document.getElementById("receiver-health").textContent).toBe("Unauthorized"),
    );
    expect(globalThis.chrome.runtime.sendMessage).toHaveBeenCalledWith({ type: "polylogue.checkReceiverHealth" });
  });

  it("checks receiver health and shows an unreachable result as a failed action", async () => {
    await loadPopup();
    globalThis.chrome.runtime.sendMessage = vi.fn(async (message) => {
      if (message.type === "polylogue.checkReceiverHealth") {
        return { ok: false, status: "unreachable", detail: "Failed to fetch" };
      }
      return { ok: true };
    });

    globalThis.document.getElementById("check-receiver").click();

    await vi.waitFor(() =>
      expect(globalThis.document.getElementById("receiver-health").textContent).toBe("Unreachable"),
    );
    await vi.waitFor(() =>
      expect(globalThis.document.getElementById("check-receiver").dataset.state).toBe("bad"),
    );
  });
});
