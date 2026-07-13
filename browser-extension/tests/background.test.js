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

function installChromeMock(storagePatch = {}) {
  stored = {
    receiverAuthToken: "token-1",
    receiverBaseUrl: "http://127.0.0.1:8875",
    ...storagePatch,
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
  globalThis.fetch = vi.fn(async (url, options = {}) => {
    fetchCalls.push({ url, options });
    if (String(url).endsWith("/v1/browser-captures/capabilities")) {
      return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] }, { requestId: "capability-1" });
    }
    return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
  });
}

async function loadBackground(storagePatch = {}) {
  vi.resetModules();
  globalThis.indexedDB = new IDBFactory();
  installChromeMock(storagePatch);
  await import("../src/background.js");
  expect(messageListener).toBeTypeOf("function");
}

async function sendRuntimeMessage(message, sender = {}) {
  let response;
  const keepAlive = messageListener(message, sender, (payload) => {
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

  it("records a direct manual capture as pending timeline evidence", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-manual",
      state: "spooled_only",
      artifact_ref: "chatgpt/conv-manual.json",
    }));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      reason: "content_script_capture",
      envelope: {
        session: {
          provider: "chatgpt",
          provider_session_id: "conv-manual",
          turns: [{ role: "user" }],
        },
      },
    });

    expect(stored.polylogueState.archive_state).toEqual({ state: "spooled_only" });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-manual"][0]).toMatchObject({
      event: "captured",
      reason: "content_script_capture",
      detail: "spooled_only",
    });
    expect(stored.polylogueSessionLedger["chatgpt:conv-manual"].archive_state).toEqual({ state: "spooled_only" });
  });

  it("records an inactive direct capture as catching up in its ledger", async () => {
    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-inactive", active: false },
    ];
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-inactive",
      state: "spooled_only",
    }));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-inactive", turns: [] } },
    }, { tab: tabs[1] });

    expect(stored.polylogueSessionLedger["chatgpt:conv-inactive"].archive_state).toEqual({ state: "spooled_only" });
    expect(stored.polylogueState).toBeUndefined();
  });

  it("records a non-retryable capture rejection as a held decision", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "invalid capture" }, { ok: false, status: 400 }));

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      reason: "auto_capture_missing",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-rejected", turns: [] } },
    });

    expect(response).toMatchObject({ ok: false, error: "invalid capture" });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-rejected"][0]).toMatchObject({
      event: "held_with_reason",
      reason: "auto_capture_missing",
      detail: "capture_rejected",
    });
    expect(stored.polylogueSessionLedger["chatgpt:conv-rejected"].last_error).toBe("invalid capture");
  });

  it("does not let an inactive rejection replace the active conversation card", async () => {
    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-rejected", active: false },
    ];
    stored.polylogueState = { online: true, provider: "chatgpt", provider_session_id: "conv-active", archive_state: { state: "archived" } };
    globalThis.fetch = vi.fn(async () => responseJson({ error: "invalid capture" }, { ok: false, status: 400 }));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-rejected", turns: [] } },
    }, { tab: tabs[1] });

    expect(stored.polylogueState.provider_session_id).toBe("conv-active");
    expect(stored.polylogueConversationTimeline["chatgpt:conv-rejected"][0].detail).toBe("capture_rejected");
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

  it("records a held decision when archive-state cannot be checked", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-offline", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(stored.polylogueConversationTimeline["chatgpt:conv-offline"]?.[0]).toMatchObject({
      event: "held_with_reason",
      reason: "tab_activated",
      detail: "archive_state_check_failed",
    }));
    expect(stored.polylogueState.active_page_state).toBe("receiver_error");
  });

  it("refreshes the active conversation instead of discarding its archive identity", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-status", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-status",
      state: "spooled_only",
      captured: false,
    }));

    const response = await sendRuntimeMessage({ type: "polylogue.status", reason: "popup_open" });

    expect(response.state).toBe("spooled_only");
    expect(stored.polylogueState).toMatchObject({
      provider: "chatgpt",
      provider_session_id: "conv-status",
      archive_state: { state: "spooled_only" },
    });
    expect(stored.polylogueSessionLedger["chatgpt:conv-status"].archive_state.state).toBe("spooled_only");
  });

  it("propagates content-script archive state into the multi-tab ledger", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-content",
      state: "stale",
      captured: false,
    }));

    await sendRuntimeMessage({
      type: "polylogue.archiveState",
      provider: "chatgpt",
      provider_session_id: "conv-content",
    });

    expect(stored.polylogueSessionLedger["chatgpt:conv-content"].archive_state.state).toBe("stale");
    expect(globalThis.chrome.action.setBadgeText.mock.calls.at(-1)[0]).toEqual({ text: "…" });
  });

  it("preserves capture metadata when content refreshes its archive state", async () => {
    let request = 0;
    globalThis.fetch = vi.fn(async () => {
      request += 1;
      return responseJson(request === 1
        ? { provider: "chatgpt", provider_session_id: "conv-metadata" }
        : { provider: "chatgpt", provider_session_id: "conv-metadata", state: "spooled_only", captured: false });
    });

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: {
        session: {
          provider: "chatgpt",
          provider_session_id: "conv-metadata",
          provider_meta: { capture_fidelity: "dom_degraded" },
          turns: [{ role: "user" }, { role: "assistant" }],
        },
      },
    });
    await sendRuntimeMessage({
      type: "polylogue.archiveState",
      provider: "chatgpt",
      provider_session_id: "conv-metadata",
    });

    expect(stored.polylogueState).toMatchObject({
      capture_mode: "dom_degraded",
      turn_count: 2,
      archive_state: { state: "spooled_only" },
    });
  });

  it("does not let an inactive tab update replace the active conversation card state", async () => {
    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", title: "Active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-inactive", title: "Inactive", active: false },
    ];
    stored.polylogueState = {
      online: true,
      provider: "chatgpt",
      provider_session_id: "conv-active",
      archive_state: { state: "archived" },
    };
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-inactive",
      state: "archived",
      captured: true,
    }));

    updatedListener(2, { status: "complete" }, tabs[1]);

    await vi.waitFor(() => expect(stored.polylogueSessionLedger["chatgpt:conv-inactive"]?.archive_state?.state).toBe("archived"));
    expect(stored.polylogueState.provider_session_id).toBe("conv-active");
  });

  it("does not let a delayed prior conversation overwrite same-tab navigation", async () => {
    tabs = [{ id: 1, url: "https://chatgpt.com/c/conv-a", title: "A", active: true }];
    let resolveA;
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).includes("conv-a")) return new Promise((resolve) => { resolveA = resolve; });
      return responseJson({ provider: "chatgpt", provider_session_id: "conv-b", state: "archived", captured: true });
    });

    updatedListener(1, { status: "complete" }, { id: 1, url: "https://chatgpt.com/c/conv-a", title: "A" });
    tabs[0] = { id: 1, url: "https://chatgpt.com/c/conv-b", title: "B", active: true };
    updatedListener(1, { url: "https://chatgpt.com/c/conv-b" }, tabs[0]);

    await vi.waitFor(() => expect(stored.polylogueState?.provider_session_id).toBe("conv-b"));
    resolveA(responseJson({ provider: "chatgpt", provider_session_id: "conv-a", state: "archived", captured: true }));
    await vi.waitFor(() => expect(stored.polylogueSessionLedger["chatgpt:conv-a"]?.archive_state?.state).toBe("archived"));
    expect(stored.polylogueState.provider_session_id).toBe("conv-b");
  });

  it("does not restore a delayed conversation after same-tab navigation to a new page", async () => {
    tabs = [{ id: 1, url: "https://chatgpt.com/c/conv-a", title: "A", active: true }];
    let resolveA;
    let fetchCount = 0;
    globalThis.fetch = vi.fn(async () => {
      fetchCount += 1;
      if (fetchCount === 1) return new Promise((resolve) => { resolveA = resolve; });
      return new Promise(() => {});
    });

    updatedListener(1, { status: "complete" }, tabs[0]);
    tabs[0] = { id: 1, url: "https://chatgpt.com/new", title: "New", active: true };
    updatedListener(1, { url: "https://chatgpt.com/new" }, tabs[0]);

    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(2));
    resolveA(responseJson({ provider: "chatgpt", provider_session_id: "conv-a", state: "archived", captured: true }));
    await vi.waitFor(() => expect(stored.polylogueSessionLedger["chatgpt:conv-a"]?.archive_state?.state).toBe("archived"));
    expect(stored.polylogueState?.provider_session_id).not.toBe("conv-a");
  });

  it("holds a missing conversation when the tab navigates before auto-capture", async () => {
    tabs = [{ id: 1, url: "https://chatgpt.com/c/conv-a", title: "A", active: true }];
    let resolveArchiveState;
    globalThis.fetch = vi.fn(async () => new Promise((resolve) => { resolveArchiveState = resolve; }));

    updatedListener(1, { status: "complete" }, tabs[0]);
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(1));
    tabs[0] = { id: 1, url: "https://chatgpt.com/c/conv-b", title: "B", active: true };
    resolveArchiveState(responseJson({ provider: "chatgpt", provider_session_id: "conv-a", state: "missing", captured: false }));

    await vi.waitFor(() => expect(stored.polylogueConversationTimeline["chatgpt:conv-a"]?.[0]).toMatchObject({
      event: "held_with_reason",
      detail: "tab_navigation_changed",
    }));
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("uses Grok query conversation identity for archive polling and the ledger", async () => {
    tabs = [{ id: 77, url: "https://grok.com/?conversation=query-77", title: "Grok", active: true }];
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({ provider: "grok", provider_session_id: "query-77", state: "archived", captured: true });
    });

    activatedListener({ tabId: 77 });

    await vi.waitFor(() => expect(stored.polylogueSessionLedger["grok:query-77"]?.archive_state?.state).toBe("archived"));
    expect(fetchCalls[0].url).toContain("provider=grok&provider_session_id=query-77");
  });

  it("does not invent a conversation identity for the X home timeline", async () => {
    tabs = [{ id: 78, url: "https://x.com/home", title: "Home", active: true }];
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({ ok: true });
    });

    activatedListener({ tabId: 78 });

    await vi.waitFor(() => expect(fetchCalls).toHaveLength(1));
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/status");
    expect(stored.polylogueSessionLedger).toBeUndefined();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("records a local popup capture failure as a held decision without offline state", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-local-failure", title: "ChatGPT", active: true }];

    await sendRuntimeMessage({
      type: "polylogue.capturePageFailed",
      tab_id: 42,
      tab_url: tabs[0].url,
      error: "no_turns",
    });

    expect(stored.polylogueConversationTimeline["chatgpt:conv-local-failure"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "content_capture_failed",
    });
    expect(stored.polylogueState).toMatchObject({ online: true, error: "no_turns" });
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
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] }, { requestId: "capability-1" });
      }
      return responseJson({ error: "unexpected_service_worker_provider_fetch" }, { ok: false, status: 500 });
    });

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
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(fetchCalls[0].url).toContain("/v1/browser-captures/capabilities");
    expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("retries coordinator initialization after recovery storage fails once", async () => {
    globalThis.chrome.storage.local.get = vi.fn()
      .mockRejectedValueOnce(new Error("synthetic_recovery_storage_failure"))
      .mockImplementation(async (defaults) => ({ ...defaults, ...stored }));
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: false, error: "synthetic_recovery_storage_failure" });
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: true, jobs: [] });
  });

  it("bounds receiver capability preflight with the provider request timeout", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] }, { requestId: "capability-1" });
    });
    await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    expect(fetchCalls[0]).toMatchObject({ url: expect.stringContaining("/v1/browser-captures/capabilities") });
    expect(fetchCalls[0].options.signal).toBeDefined();
  });

  it("classifies a reachable receiver missing the capability route as contract-incompatible", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "not_found" }, { ok: false, status: 404 }));
    const started = await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    expect(started.job).toMatchObject({ status: "paused", cooldown_reason: "receiver_contract_incompatible" });
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
  });

  it("restores a packaged recovery checkpoint as an actionable paused job and ignores its alarm", async () => {
    await loadBackground({
      polylogueBackfillRecoveryCheckpoint: {
        version: 1,
        jobs: [{
          id: "recovered-job", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
          inventory_cursor: "17", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
          learned_cadence_ms: 40000, daily_requests: 7, last_ack: { receiver_request_id: "ack-1", content_hash: "hash-1" },
        }],
        queue: [{ id: "recovered-item", job_id: "recovered-job", provider: "chatgpt", native_id: "one", state: "captured_waiting_receiver", content_hash: "hash-1" }],
        revisions: [],
      },
    });
    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });
    expect(status.jobs[0]).toMatchObject({
      id: "recovered-job", status: "paused", cooldown_reason: "browser_profile_recovery_required",
      inventory_cursor: "17", daily_requests: 7, last_ack: { receiver_request_id: "ack-1" },
      progress: { operator_action: 1 },
    });
    alarmListener({ name: "polylogueBackfillWake:recovered-job" });
    await Promise.resolve();
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
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
      if (String(url).endsWith("/v1/browser-captures")) {
        return responseJson({
          provider: "chatgpt",
          provider_session_id: "conv-123",
          state: "spooled_only",
          receiver_request_id: "capture-request-1",
        }, { requestId: "capture-request-1" });
      }
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
    globalThis.chrome.tabs.sendMessage = vi.fn(async (tabId, message) => {
      if (message.type !== "polylogue.capturePage") return null;
      const envelope = {
        session: {
          provider: "chatgpt",
          provider_session_id: "conv-123",
          turns: [{ role: "user" }],
        },
      };
      const captureResult = await new Promise((resolve) => {
        messageListener(
          { type: "polylogue.capture", envelope, reason: message.reason },
          { tab: { id: tabId, url: "https://chatgpt.com/c/conv-123" } },
          resolve,
        );
      });
      return { ok: true, envelope, captureResult, archiveState: { state: "spooled_only" } };
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(stored.polylogueState?.active_page_state).toBe("conversation"));
    await vi.waitFor(() => expect(stored.polylogueState?.captured).toBe(true));
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/archive-state?provider=chatgpt&provider_session_id=conv-123");
    expect(stored.polylogueState.captured).toBe(true);
    expect(stored.polylogueState.last_receiver_request_id).toBe("capture-request-1");
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "auto_capture_missing",
    });
    expect(fetchCalls.map((call) => call.url)).toContain("http://127.0.0.1:8875/v1/browser-captures");
    const timeline = stored.polylogueConversationTimeline["chatgpt:conv-123"];
    expect(timeline.map((entry) => entry.event)).toEqual(["captured", "detected_new", "first_seen"]);
    expect(timeline[0]).toMatchObject({ reason: "auto_capture_missing", detail: "spooled_only" });
  });

  it("serializes concurrent captures without losing either ledger or timeline entry", async () => {
    globalThis.fetch = vi.fn(async (_url, options) => {
      const session = JSON.parse(options.body).session;
      return responseJson({ provider: session.provider, provider_session_id: session.provider_session_id });
    });

    await Promise.all(["conv-a", "conv-b"].map((providerSessionId) => sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: providerSessionId } },
    })));

    expect(Object.keys(stored.polylogueSessionLedger).sort()).toEqual(["chatgpt:conv-a", "chatgpt:conv-b"]);
    expect(Object.keys(stored.polylogueConversationTimeline).sort()).toEqual(["chatgpt:conv-a", "chatgpt:conv-b"]);
  });

  it("records a held decision when automatic capture is throttled", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-123", title: "ChatGPT" }];
    const now = vi.spyOn(Date, "now");
    now.mockReturnValue(100000);
    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });
    now.mockReturnValue(105000);
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-123",
      state: "missing",
      captured: false,
    }));

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(stored.polylogueState?.active_page_state).toBe("conversation"));
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(stored.polylogueConversationTimeline["chatgpt:conv-123"]?.[0]).toMatchObject({
      event: "held_with_reason",
      reason: "auto_capture_missing",
      detail: "background_capture_throttled",
    }));
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
        state: "spooled_only",
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

    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-9", active: false },
    ];
    stored.polylogueState = { online: true, provider: "chatgpt", provider_session_id: "conv-active", archive_state: { state: "archived" } };
    const response = await sendRuntimeMessage(
      { type: "polylogue.capture", envelope, reason: "content_script_capture" },
      { tab: tabs[1] },
    );

    expect(response).toEqual({ ok: false, queued: true, error: "Failed to fetch", receiver_request_id: null });
    expect(stored.polylogueCaptureQueue.entries).toHaveLength(1);
    expect(stored.polylogueCaptureQueue.entries[0].envelope.session.provider_session_id).toBe("conv-9");
    expect(stored.polylogueCaptureQueue.entries[0].attempts).toBe(0);
    expect(stored.polylogueConversationTimeline["chatgpt:conv-9"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "capture_queued_for_retry",
    });
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
    expect(stored.polylogueState.captured).toBeUndefined();
    expect(stored.polylogueState.provider_session_id).toBe("conv-active");
    expect(stored.polylogueState.archive_state).toEqual({ state: "archived" });
    expect(stored.polylogueSessionLedger["chatgpt:conv-9"].archive_state).toEqual({ state: "spooled_only" });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-9"][0]).toMatchObject({
      event: "captured",
      reason: "capture_retry_drained",
      detail: "spooled_only",
    });
  });

  it("keeps concurrent retry enqueues instead of overwriting one capture", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const capture = (sessionId) => sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: sessionId, turns: [] } },
    });

    await Promise.all([capture("conv-concurrent-a"), capture("conv-concurrent-b")]);

    expect(stored.polylogueCaptureQueue.entries.map((entry) => entry.envelope.session.provider_session_id)).toEqual([
      "conv-concurrent-a",
      "conv-concurrent-b",
    ]);
  });

  it("reports an oversized retry capture as dropped instead of queued", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: {
        session: { provider: "chatgpt", provider_session_id: "conv-oversized", turns: [{ text: "x".repeat(43_000_000) }] },
      },
    });

    expect(response.queued).toBe(false);
    expect(stored.polylogueCaptureQueue.entries).toHaveLength(0);
    expect(stored.polylogueConversationTimeline["chatgpt:conv-oversized"][0].detail).toBe("capture_queue_entry_over_budget");
  });

  it("drops a retry after a later non-retryable receiver rejection", async () => {
    let callCount = 0;
    globalThis.fetch = vi.fn(async () => {
      callCount += 1;
      if (callCount === 1) throw new TypeError("Failed to fetch");
      return responseJson({ error: "invalid capture" }, { ok: false, status: 400 });
    });
    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-retry-rejected", turns: [] } },
    });
    stored.polylogueCaptureQueue.entries[0].next_attempt_at = new Date(Date.now() - 1000).toISOString();

    alarmListener({ name: "polylogueCaptureRetry" });

    await vi.waitFor(() => expect(stored.polylogueCaptureQueue.entries).toHaveLength(0));
    expect(stored.polylogueConversationTimeline["chatgpt:conv-retry-rejected"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "capture_rejected",
    });
    expect(stored.polylogueCaptureLog[0].reason).toBe("capture_retry_rejected");
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
