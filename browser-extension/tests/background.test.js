import { beforeEach, describe, expect, it, vi } from "vitest";
import { IDBFactory } from "fake-indexeddb";

let messageListener;
let installedListener;
let activatedListener;
let updatedListener;
let alarmListener;
let stored;
let fetchCalls;
let tabs;

function installChromeMock() {
  stored = {
    receiverAuthToken: "token-1",
    receiverBaseUrl: "http://127.0.0.1:8875",
  };
  messageListener = null;
  installedListener = null;
  activatedListener = null;
  updatedListener = null;
  alarmListener = null;
  fetchCalls = [];
  tabs = [{ id: 42, url: "https://chatgpt.com/?temporary-chat=true", title: "ChatGPT" }];
  globalThis.chrome = {
    action: {
      setBadgeBackgroundColor: vi.fn(async () => undefined),
      setBadgeText: vi.fn(async () => undefined),
    },
    alarms: {
      create: vi.fn(async () => undefined),
      clear: vi.fn(async () => undefined),
      onAlarm: {
        addListener: vi.fn((fn) => {
          alarmListener = fn;
        }),
      },
    },
    runtime: {
      onInstalled: {
        addListener: vi.fn((fn) => {
          installedListener = fn;
        }),
      },
      onStartup: {
        addListener: vi.fn(),
      },
      onMessage: {
        addListener: vi.fn((fn) => {
          messageListener = fn;
        }),
      },
    },
    scripting: {
      executeScript: vi.fn(async (details) => {
        if (!details.func) return undefined;
        const request = details.args[0];
        const body = request.operation === "inventory"
          ? { items: [{ id: "backfill-1", update_time: 1780000000 }], total: 1 }
          : { id: "backfill-1", mapping: {} };
        return [{ result: { ok: true, response: { ok: true, status: 200, contentType: "application/json", body: JSON.stringify(body) } } }];
      }),
    },
    storage: {
      local: {
        get: vi.fn(async (defaults) => ({ ...defaults, ...stored })),
        set: vi.fn(async (patch) => {
          stored = { ...stored, ...patch };
        }),
      },
    },
    tabs: {
      create: vi.fn(async ({ url }) => ({ id: 99, url, status: "complete" })),
      get: vi.fn(async (tabId) => tabs.find((tab) => tab.id === tabId)),
      remove: vi.fn(async () => undefined),
      onActivated: {
        addListener: vi.fn((fn) => {
          activatedListener = fn;
        }),
      },
      onUpdated: {
        addListener: vi.fn((fn) => {
          updatedListener = fn;
        }),
      },
      query: vi.fn(async () => tabs),
      sendMessage: vi.fn(async (_tabId, message) => {
        if (message.type === "polylogue.backfill.pageRequest") {
          const body = message.operation === "inventory"
            ? { items: [{ id: "backfill-1", update_time: 1780000000 }], total: 1 }
            : { id: "backfill-1", mapping: {} };
          return { ok: true, response: { ok: true, status: 200, contentType: "application/json", body: JSON.stringify(body) } };
        }
        return {
          ok: true,
          captureResult: {
            receiver_request_id: "capture-request-1",
            provider: "chatgpt",
            provider_session_id: "temporary:abc",
          },
          archiveState: {
            receiver_request_id: "state-request-1",
            captured: true,
          },
        };
      }),
    },
  };
}

async function loadBackground() {
  vi.resetModules();
  globalThis.indexedDB = new IDBFactory();
  installChromeMock();
  await import("../src/background.js");
  expect(messageListener).toBeTypeOf("function");
}

async function sendRuntimeMessage(message) {
  let response;
  const keepAlive = messageListener(message, {}, (payload) => {
    response = payload;
  });
  await vi.waitFor(() => expect(response).toBeDefined());
  expect(keepAlive).toBe(true);
  return response;
}

function responseJson(body, { ok = true, status = 200, requestId = "receiver-request-1" } = {}) {
  return {
    headers: {
      get: vi.fn((name) => (name === "X-Request-ID" ? requestId : null)),
    },
    json: vi.fn(async () => body),
    ok,
    status,
  };
}

describe("background receiver diagnostics", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("sends a request id to the receiver and stores the echoed id", async () => {
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      return responseJson({
        ok: true,
        provider: "chatgpt",
        provider_session_id: "conv-123",
        artifact_ref: "chatgpt/conv-123.json",
      });
    });

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { polylogue_capture_kind: "browser_llm_session" },
    });

    expect(response.receiver_request_id).toBe("receiver-request-1");
    expect(fetchCalls).toHaveLength(1);
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/browser-captures");
    expect(fetchCalls[0].options.headers.Authorization).toBe("Bearer token-1");
    expect(fetchCalls[0].options.headers["Content-Type"]).toBe("application/json");
    expect(fetchCalls[0].options.headers["X-Request-ID"]).toMatch(/^polylogue-ext-/);
    expect(stored.polylogueState.last_receiver_request_id).toBe("receiver-request-1");
    expect(stored.polylogueState.last_capture.receiver_request_id).toBe("receiver-request-1");
  });

  it("coalesces concurrent capture attribution into one stable service-worker instance", async () => {
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      return responseJson({ ok: true, provider: "chatgpt", provider_session_id: "conv-123" });
    });

    await Promise.all([
      sendRuntimeMessage({
        type: "polylogue.capture",
        envelope: {
          provenance: { extension_instance_id: "untrusted-content-script" },
          session: { provider: "chatgpt", provider_session_id: "conv-123" },
        },
      }),
      sendRuntimeMessage({
        type: "polylogue.capture",
        envelope: { session: { provider: "chatgpt", provider_session_id: "conv-124" } },
      }),
    ]);

    const first = JSON.parse(fetchCalls[0].options.body);
    const second = JSON.parse(fetchCalls[1].options.body);
    expect(first.provenance.extension_instance_id).not.toBe("untrusted-content-script");
    expect(first.provenance.extension_instance_id).toBe(second.provenance.extension_instance_id);
    expect(stored.polylogueExtensionInstanceId).toBe(first.provenance.extension_instance_id);
  });

  it("keeps receiver request id on error state", async () => {
    globalThis.fetch = vi.fn(async () =>
      responseJson({ error: "unauthorized" }, { ok: false, status: 401, requestId: "reject-42" }),
    );

    const response = await sendRuntimeMessage({ type: "polylogue.status" });

    expect(response).toEqual({
      ok: false,
      error: "unauthorized",
      receiver_request_id: "reject-42",
    });
    expect(stored.polylogueState.online).toBe(false);
    expect(stored.polylogueState.last_receiver_request_id).toBe("reject-42");
  });

  it("does not capture existing provider tabs on extension update", async () => {
    expect(installedListener).toBeTypeOf("function");
    globalThis.fetch = vi.fn(async () => responseJson({ ok: true, active: true }));

    installedListener();

    await Promise.resolve();
    await Promise.resolve();
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("routes backfill inventory through an existing provider page instead of service-worker fetch", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "unexpected_service_worker_provider_fetch" }, { ok: false, status: 500 }));

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
      policy: { baseCadenceMs: 1000 },
    });

    expect(started.ok).toBe(true);
    await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith(expect.objectContaining({
      target: { tabId: 42 },
      world: "MAIN",
      func: expect.any(Function),
      args: [expect.objectContaining({ provider: "chatgpt", operation: "inventory" })],
    })));
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("does not reuse unsupported provider subdomains as authenticated transport roots", async () => {
    tabs = [{ id: 43, url: "https://help.chatgpt.com/article", title: "Help" }];

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    expect(started.ok).toBe(true);
    await vi.waitFor(() => expect(globalThis.chrome.tabs.create).toHaveBeenCalledWith({ url: "https://chatgpt.com/", active: false }));
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith(expect.objectContaining({ target: { tabId: 99 } }));
  });

  it("preserves page-bridge Retry-After through the adapter and coordinator", async () => {
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      if (details.func) {
        return [{ result: { ok: true, response: {
          ok: false,
          status: 429,
          contentType: "application/json",
          retryAfter: "60",
          body: JSON.stringify({ detail: "slow down" }),
        } } }];
      }
      return undefined;
    });

    await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });
    let status;
    await vi.waitFor(async () => {
      status = (await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs[0];
      expect(status.cooldown_reason).toBe("provider_rate_limited");
    });

    expect(status.cooldown_until_ms).toBe(Date.parse(status.updated_at) + 60000);
    expect(status.inventory_complete).toBe(false);
  });

  it("schedules cleanup before waiting and removes an inactive tab when readiness fails", async () => {
    tabs = [];
    globalThis.chrome.tabs.create = vi.fn(async ({ url, active }) => ({ id: 99, url, active, status: "loading" }));
    globalThis.chrome.tabs.get = vi.fn(async () => { throw new Error("synthetic_tab_load_failure"); });

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });
    expect(started.ok, started.error).toBe(true);
    await vi.waitFor(() => expect(globalThis.chrome.tabs.remove).toHaveBeenCalledWith(99));

    expect(globalThis.chrome.alarms.create).toHaveBeenCalledWith(
      "polylogueBackfillTransportCleanup:chatgpt:99",
      expect.objectContaining({ when: expect.any(Number) }),
    );
    expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith("polylogueBackfillTransportCleanup:chatgpt:99");
  });

  it("surfaces a stale Claude UI selection as a cancel-and-restart reason", async () => {
    tabs = [{ id: 52, url: "https://claude.ai/new", title: "Claude" }];
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => details.func
      ? [{ result: { ok: false, error: "backfill_bridge_selected_organization_stale" } }]
      : undefined);

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "claude-ai",
      cutoff: "2026-01-01T00:00:00Z",
    });
    expect(started.ok).toBe(true);
    let status;
    await vi.waitFor(async () => {
      status = (await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs[0];
      expect(status.cooldown_reason).toBe("backfill_bridge_selected_organization_stale");
    });
    expect(status.inventory_complete).toBe(false);
  });

  it("captures an automatically detected missing conversation and records the decision timeline", async () => {
    expect(activatedListener).toBeTypeOf("function");
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-123", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      return responseJson(
        {
          provider: "chatgpt",
          provider_session_id: "conv-123",
          state: "missing",
          lifecycle: "missing",
          captured: false,
          spooled: false,
          artifact_ref: "chatgpt/conv-123.json",
        },
        { requestId: "archive-state-1" },
      );
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(stored.polylogueState?.active_page_state).toBe("conversation"));
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/archive-state?provider=chatgpt&provider_session_id=conv-123");
    expect(stored.polylogueState.captured).toBe(true);
    expect(stored.polylogueState.last_receiver_request_id).toBe("capture-request-1");
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "auto_capture_missing",
    });
    const timeline = stored.polylogueConversationTimeline["chatgpt:conv-123"];
    expect(timeline.map((entry) => entry.event)).toEqual(["captured", "detected_new", "first_seen"]);
    expect(timeline[0].reason).toBe("auto_capture_missing");
  });

  it("refreshes receiver status for supported pages without a conversation id", async () => {
    expect(updatedListener).toBeTypeOf("function");
    tabs = [{ id: 42, url: "https://chatgpt.com/", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      return responseJson({ ok: true, active: true }, { requestId: "status-1" });
    });

    updatedListener(42, { status: "complete" }, tabs[0]);

    await vi.waitFor(() => expect(stored.polylogueState?.active_page_state).toBe("supported_no_session"));
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/status");
    expect(stored.polylogueState.provider).toBe("chatgpt");
    expect(stored.polylogueState.last_receiver_request_id).toBe("status-1");
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("injects capture scripts into existing provider tabs on explicit sync", async () => {
    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1));

    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: ["src/content/chatgpt_bridge.js"],
      world: "MAIN",
    });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: ["src/common.js", "src/content/chatgpt.js"],
    });
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "popup_sync_open_tabs",
    });
    expect(stored.polylogueState.online).toBe(true);
    expect(stored.polylogueState.captured).toBe(true);
    expect(stored.polylogueState.last_receiver_request_id).toBe("capture-request-1");
  });

  it("bounds explicit sync when a provider tab never answers capture", async () => {
    vi.useFakeTimers();
    globalThis.chrome.tabs.sendMessage = vi.fn(() => new Promise(() => {}));

    const responsePromise = sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    await vi.advanceTimersByTimeAsync(15000);
    const response = await responsePromise;

    expect(response).toEqual({ ok: true });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: ["src/content/chatgpt_bridge.js"],
      world: "MAIN",
    });
    expect(stored.polylogueCaptureLog[0].ok).toBe(false);
    expect(stored.polylogueCaptureLog[0].error).toContain("capture_message_timeout_after_15000ms");
    expect(stored.polylogueDebugLog[0].stage).toBe("capture_result");
    expect(stored.polylogueDebugLog[0].ok).toBe(false);
  });

  it("injects Grok DOM capture scripts for open Grok/X tabs", async () => {
    tabs = [{ id: 77, url: "https://x.com/i/grok", title: "Grok" }];

    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1));

    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 77 },
      files: ["src/common.js", "src/content/grok.js"],
    });
  });
});

describe("capture retry queue", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("queues a capture for retry when the receiver is unreachable, sets a badge, then drains on the next alarm", async () => {
    let callCount = 0;
    globalThis.fetch = vi.fn(async (url) => {
      callCount += 1;
      fetchCalls.push({ url });
      if (callCount === 1) throw new TypeError("Failed to fetch");
      return responseJson({
        ok: true,
        provider: "chatgpt",
        provider_session_id: "conv-9",
        artifact_ref: "chatgpt/conv-9.json",
      });
    });

    const envelope = {
      session: {
        provider: "chatgpt",
        provider_session_id: "conv-9",
        provider_meta: { capture_fidelity: "native_full" },
        turns: [{ role: "user" }, { role: "assistant" }],
      },
    };

    const response = await sendRuntimeMessage({ type: "polylogue.capture", envelope, reason: "content_script_capture" });

    expect(response).toEqual({ ok: false, queued: true, error: "Failed to fetch", receiver_request_id: null });
    expect(stored.polylogueCaptureQueue.entries).toHaveLength(1);
    expect(stored.polylogueCaptureQueue.entries[0].envelope.session.provider_session_id).toBe("conv-9");
    expect(stored.polylogueCaptureQueue.entries[0].attempts).toBe(0);
    expect(globalThis.chrome.alarms.create).toHaveBeenCalledWith(
      "polylogueCaptureRetry",
      expect.objectContaining({ periodInMinutes: 1 }),
    );
    const lastBadgeCall = globalThis.chrome.action.setBadgeText.mock.calls.at(-1);
    expect(lastBadgeCall[0]).toEqual({ text: "1" });

    // Force the queued entry's backoff window to be due, then simulate the
    // retry alarm firing (real Chrome would deliver this on its own timer).
    stored.polylogueCaptureQueue.entries[0].next_attempt_at = new Date(Date.now() - 1000).toISOString();
    expect(alarmListener).toBeTypeOf("function");
    alarmListener({ name: "polylogueCaptureRetry" });

    await vi.waitFor(() => expect(stored.polylogueCaptureQueue.entries).toHaveLength(0));
    expect(callCount).toBe(2);
    expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith("polylogueCaptureRetry");
    expect(stored.polylogueCaptureLog[0].reason).toBe("capture_retry_drained");
    expect(stored.polylogueState.captured).toBe(true);
    expect(stored.polylogueState.provider_session_id).toBe("conv-9");
  });

  it("does not queue a capture rejected with a client error", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "invalid_envelope" }, { ok: false, status: 400 }));

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-1" } },
    });

    expect(response).toEqual({ ok: false, error: "invalid_envelope", receiver_request_id: "receiver-request-1" });
    expect(stored.polylogueCaptureQueue).toBeUndefined();
    expect(globalThis.chrome.alarms.create).not.toHaveBeenCalled();
  });

  it("retries a 503 receiver response but bounds the queue at 20 entries with a drop counter", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "unavailable" }, { ok: false, status: 503 }));

    for (let i = 0; i < 22; i += 1) {
      await sendRuntimeMessage({
        type: "polylogue.capture",
        envelope: { session: { provider: "chatgpt", provider_session_id: `conv-${i}` } },
      });
    }

    expect(stored.polylogueCaptureQueue.entries).toHaveLength(20);
    expect(stored.polylogueCaptureQueue.dropped_count).toBe(2);
    expect(stored.polylogueCaptureQueue.entries[0].envelope.session.provider_session_id).toBe("conv-2");
    expect(stored.polylogueCaptureQueue.entries.at(-1).envelope.session.provider_session_id).toBe("conv-21");
  });

  it("summarizes the retry queue for the popup without leaking full envelope internals", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("offline");
    });
    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: {
        session: { provider: "chatgpt", provider_session_id: "conv-5", turns: [{ role: "user", text: "secret" }] },
      },
    });

    const response = await sendRuntimeMessage({ type: "polylogue.getCaptureQueue" });

    expect(response.ok).toBe(true);
    expect(response.dropped_count).toBe(0);
    expect(response.entries).toHaveLength(1);
    expect(response.entries[0]).toMatchObject({ provider: "chatgpt", provider_session_id: "conv-5", attempts: 0 });
    expect(response.entries[0].envelope).toBeUndefined();
  });

  it("drains the retry queue once a subsequent capture proves the receiver is reachable again", async () => {
    let callCount = 0;
    globalThis.fetch = vi.fn(async () => {
      callCount += 1;
      if (callCount === 1) throw new TypeError("Failed to fetch");
      return responseJson({ ok: true, provider: "chatgpt", provider_session_id: "conv-7" });
    });

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-7" } },
    });
    expect(stored.polylogueCaptureQueue.entries).toHaveLength(1);

    // Make the queued entry due, then drive a second capture that succeeds —
    // its success should trigger a queue drain as a side effect.
    stored.polylogueCaptureQueue.entries[0].next_attempt_at = new Date(Date.now() - 1000).toISOString();
    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-8" } },
    });

    await vi.waitFor(() => expect(stored.polylogueCaptureQueue.entries).toHaveLength(0));
  });
});

describe("receiver health probe", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("reports the receiver as reachable and authorized when /api/health returns ok:true", async () => {
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({ ok: true });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toEqual({ ok: true, status: "ok", detail: null });
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/api/health");
  });

  it("reports the receiver as reachable but unauthorized when no valid token is configured", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ ok: false, error: "unauthorized" }));

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toEqual({ ok: true, status: "unauthorized", detail: "unauthorized" });
  });

  it("reports the receiver as unreachable when the fetch itself fails", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toEqual({ ok: false, status: "unreachable", detail: "Failed to fetch" });
  });

  it("reports the receiver as unreachable when the response body is not JSON", async () => {
    globalThis.fetch = vi.fn(async () => ({
      headers: { get: vi.fn(() => null) },
      json: vi.fn(async () => {
        throw new Error("not json");
      }),
      ok: true,
      status: 200,
    }));

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toEqual({ ok: false, status: "unreachable", detail: "non_json_response" });
  });
});
