import { beforeEach, describe, expect, it, vi } from "vitest";
import { IDBFactory } from "fake-indexeddb";

let messageListener;
let installedListener;
let activatedListener;
let updatedListener;
let alarmListener;
let stored;
let sessionStored;
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
  sessionStored = {};
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
      id: "polylogue-test-extension",
      getManifest: vi.fn(() => ({ version: "0.1.0" })),
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
        if (request.operation === "identity") {
          return [{ result: { ok: true, response: { accountHandle: `test-account-${request.provider}` } } }];
        }
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
        remove: vi.fn(async (key) => {
          const keys = Array.isArray(key) ? key : [key];
          const next = { ...stored };
          for (const item of keys) delete next[item];
          stored = next;
        }),
      },
      session: {
        get: vi.fn(async (defaults) => ({ ...defaults, ...sessionStored })),
        set: vi.fn(async (patch) => {
          sessionStored = { ...sessionStored, ...patch };
        }),
        remove: vi.fn(async (key) => {
          const keys = Array.isArray(key) ? key : [key];
          for (const item of keys) delete sessionStored[item];
        }),
      },
    },
    tabs: {
      create: vi.fn(async ({ url, active = false }) => {
        const created = { id: 99, url, active, status: "complete" };
        tabs = [...tabs, created];
        return created;
      }),
      get: vi.fn(async (tabId) => tabs.find((tab) => tab.id === tabId)),
      update: vi.fn(async (tabId, patch) => {
        const current = tabs.find((tab) => tab.id === tabId);
        const updated = { ...current, ...patch, status: "complete" };
        tabs = tabs.map((tab) => (tab.id === tabId ? updated : tab));
        return updated;
      }),
      remove: vi.fn(async (tabId) => {
        tabs = tabs.filter((tab) => tab.id !== tabId);
      }),
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
    const captureJobResponse = captureJobFixtureResponse(url, options);
    if (captureJobResponse) return captureJobResponse;
    if (String(url).endsWith("/v1/browser-captures/capabilities")) {
      return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] }, { requestId: "capability-1" });
    }
    return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
  });
}

function captureJobFixtureResponse(url, options = {}) {
  const path = new URL(url).pathname;
  if (!path.startsWith("/v1/capture-jobs")) return null;
  const body = options.body ? JSON.parse(options.body) : {};
  if (path === "/v1/capture-jobs/capabilities") {
    return responseJson({
      schema: "polylogue.capture-jobs.capabilities.v1",
      protocol_min: 1,
      protocol_max: 1,
      scope_namespace: "cjs1:fixture-stable-namespace",
    });
  }
  if (path === "/v1/capture-jobs/discover") return responseJson({ jobs: [] });
  if (path === "/v1/capture-jobs") {
    return responseJson({ job: {
      job_id: "fixture-capture-job", provider: body.provider, account_scope: body.account_scope,
      intent_key: body.intent.intent_key, revision: 0, lease_generation: 0,
    } }, { status: 201 });
  }
  if (path.endsWith("/adopt")) {
    return responseJson({
      job: {
        job_id: "fixture-capture-job", provider: body.provider, account_scope: body.account_scope,
        intent_key: "fixture-intent", revision: 1, lease_generation: 1,
      },
      lease: { lease_id: "fixture-lease", generation: 1, proof: "fixture-proof", expires_at: "2099-01-01T00:00:00Z" },
    });
  }
  if (path.endsWith("/update")) {
    return responseJson({
      job: {
        job_id: "fixture-capture-job", provider: body.provider, account_scope: body.account_scope,
        intent_key: "fixture-intent", revision: 2, lease_generation: 1,
        lease_expires_at: "2099-01-01T00:00:00Z", checkpoint_sequence: null,
      },
      receipt: { kind: "capture_job_update", revision: 2 },
    });
  }
  if (path.endsWith("/checkpoint")) {
    return responseJson({ job: { job_id: "fixture-capture-job", revision: 3, checkpoint_sequence: 0 }, receipt: {} });
  }
  return responseJson({ error: "unexpected_capture_job_request" }, { ok: false, status: 500 });
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
    expect(fetchCalls[0].options.headers["X-Polylogue-Extension-Contract"])
      .toBe("canonical-capture-mission-control-v1");
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

    expect(response).toMatchObject({
      ok: false,
      error: "unauthorized",
      receiver_request_id: "reject-42",
    });
    expect(response.receiver_pairing).toBeNull();
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
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          api_schema: "polylogue-browser-capture/v1",
          receiver_id: "rx-status",
        }, { requestId: "status-1" });
      }
      return responseJson({
        provider: "chatgpt",
        provider_session_id: "conv-status",
        state: "spooled_only",
        captured: false,
      });
    });

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
    await vi.waitFor(() => expect(stored.polylogueSessionLedger?.["chatgpt:conv-a"]?.archive_state?.state).toBe("archived"));
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
    await new Promise((resolve) => globalThis.setTimeout(resolve, 20));
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

  it("automatically captures existing conversation tabs on extension update", async () => {
    expect(installedListener).toBeTypeOf("function");
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-installed", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => responseJson({ ok: true, active: true }));

    installedListener();

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "extension_installed_or_updated",
    }));
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled();
  });

  it("routes backfill inventory through a dedicated inactive provider page", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
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
    await vi.waitFor(() => expect(globalThis.chrome.tabs.create).toHaveBeenCalledWith({
      url: "https://chatgpt.com/",
      active: false,
    }));
    await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith(expect.objectContaining({
      target: { tabId: 99 },
      world: "MAIN",
      func: expect.any(Function),
      args: [expect.objectContaining({ provider: "chatgpt", operation: "inventory" })],
    })));
    // Provider inventory/conversation fetches route through the page bridge
    // (chrome.scripting.executeScript above), never through this
    // service-worker `fetch`. Receiver capability and CaptureJob durability
    // traffic are the only legitimate service-worker fetches for this flow.
    for (const call of fetchCalls) {
      expect(call.url).toMatch(/\/v1\/(browser-captures\/capabilities|capture-jobs|backfill-checkpoint)/);
    }
    expect(fetchCalls.filter((call) => String(call.url).includes("/v1/browser-captures/capabilities"))).toHaveLength(1);
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("shares one extension-owned provider transport across concurrent backfill jobs", async () => {
    tabs = [];
    globalThis.chrome.tabs.create.mockImplementation(async ({ url, active }) => {
      const tab = { id: 99, url, active, status: "complete" };
      tabs.push(tab);
      return tab;
    });
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const requests = ["2026-01-01T00:00:00Z", "2025-01-01T00:00:00Z"].map((cutoff) => sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff,
      policy: { baseCadenceMs: 1000 },
    }));
    await Promise.all(requests);

    await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled());
    const chatGptTabs = globalThis.chrome.tabs.create.mock.calls.filter(([request]) => request.url === "https://chatgpt.com/");
    expect(chatGptTabs).toHaveLength(1);
    expect(globalThis.chrome.tabs.create).toHaveBeenCalledWith({ url: "https://chatgpt.com/", active: false });
    expect(globalThis.chrome.scripting.executeScript.mock.calls.every(([details]) => details.target.tabId === 99)).toBe(true);
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
    const capabilityCall = fetchCalls.find((call) => String(call.url).includes("/v1/browser-captures/capabilities"));
    expect(capabilityCall).toBeDefined();
    expect(capabilityCall.options.signal).toBeDefined();
  });

  it("classifies a reachable receiver missing the capability route as contract-incompatible", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "not_found" }, { ok: false, status: 404 }));
    const started = await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    expect(started.job).toMatchObject({ status: "paused", cooldown_reason: "receiver_contract_incompatible" });
    expect(globalThis.chrome.scripting.executeScript.mock.calls.filter(
      ([details]) => details.args?.[0]?.operation !== "identity",
    )).toHaveLength(0);
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
    const pageWorkBeforeAlarm = globalThis.chrome.scripting.executeScript.mock.calls.filter(
      ([details]) => details.args?.[0]?.operation !== "identity",
    ).length;
    alarmListener({ name: "polylogueBackfillWake:recovered-job" });
    await Promise.resolve();
    expect(globalThis.chrome.scripting.executeScript.mock.calls.filter(
      ([details]) => details.args?.[0]?.operation !== "identity",
    )).toHaveLength(pageWorkBeforeAlarm);
  });

  it("commits CaptureJobs before updating the local checkpoint cache", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] });
      }
      if (String(url).includes("/v1/backfill-checkpoint")) {
        return responseJson(
          { extension_instance_id: "mirrored", stored_at: "2026-01-01T00:00:00Z", bytes_written: 42 },
          { status: 202 },
        );
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });

    const mirrorCall = fetchCalls.find(
      (call) => String(call.url).includes("/v1/backfill-checkpoint") && call.options.method === "POST",
    );
    expect(mirrorCall).toBeDefined();
    const body = JSON.parse(mirrorCall.options.body);
    expect(body.extension_instance_id).toBe(stored.polylogueExtensionInstanceId);
    expect(body.checkpoint).toMatchObject({ version: 1, jobs: [expect.objectContaining({ provider: "chatgpt" })] });
    const captureJobCommit = fetchCalls.findIndex((call) => new URL(call.url).pathname.endsWith("/checkpoint"));
    const localMirror = fetchCalls.findIndex((call) => new URL(call.url).pathname === "/v1/backfill-checkpoint" && call.options.method === "POST");
    expect(captureJobCommit).toBeGreaterThanOrEqual(0);
    expect(localMirror).toBeGreaterThan(captureJobCommit);
    expect(stored.polylogueBackfillRecoveryCheckpoint).toMatchObject({ version: 1 });
  });

  it("derives CaptureJob scope from live provider identity without disclosing the handle", async () => {
    const accountHandle = "stable-chatgpt-account-id";
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle } } }];
      }
      const body = request?.operation === "inventory"
        ? { items: [], total: 0 }
        : { id: "backfill-1", mapping: {} };
      return [{ result: { ok: true, response: {
        ok: true,
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      } } }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const path = new URL(url).pathname;
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1",
          protocol_min: 1,
          protocol_max: 1,
          scope_namespace: "cjs1:identity-test-namespace",
        });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] });
      }
      if (path === "/v1/backfill-checkpoint") {
        return responseJson({ stored_at: "2026-01-01T00:00:00Z" }, { status: 202 });
      }
      if (path === "/v1/capture-jobs/discover") return responseJson({ jobs: [] });
      if (path === "/v1/capture-jobs") {
        return responseJson({ job: {
          job_id: "capture-job-1", provider: "chatgpt", revision: 0, lease_generation: 0,
        } }, { status: 201 });
      }
      if (path.endsWith("/adopt")) {
        return responseJson({
          job: { job_id: "capture-job-1", provider: "chatgpt", revision: 1, lease_generation: 1 },
          lease: { lease_id: "lease-1", generation: 1, proof: "proof-1" },
        });
      }
      if (path.endsWith("/update")) {
        return responseJson({
          job: {
            job_id: "capture-job-1", provider: "chatgpt", revision: 2, lease_generation: 1,
            lease_expires_at: "2026-07-16T10:02:00Z",
          },
          receipt: { kind: "capture_job_update" },
        });
      }
      if (path.endsWith("/checkpoint")) {
        return responseJson({ job: { job_id: "capture-job-1", revision: 3 }, receipt: {} });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    await vi.waitFor(() => expect(fetchCalls.some(
      (call) => new URL(call.url).pathname.endsWith("/checkpoint") && call.options.method === "PUT",
    )).toBe(true));
    const captureJobCalls = fetchCalls.filter((call) => new URL(call.url).pathname.startsWith("/v1/capture-jobs"));
    expect(captureJobCalls.length).toBeGreaterThanOrEqual(5);
    for (const call of captureJobCalls) {
      if (!call.options.body) continue;
      expect(call.options.body).not.toContain(accountHandle);
      expect(call.options.body).not.toContain("paired:");
      const body = JSON.parse(call.options.body);
      if ("account_scope" in body) expect(body.account_scope).toMatch(/^h1:/);
    }
    expect(JSON.stringify(stored)).not.toContain(accountHandle);
  });

  it("fails CaptureJob mirroring closed when provider identity is unavailable", async () => {
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: false, error: "backfill_bridge_auth_context_unavailable" } }];
      }
      return [{ result: { ok: true, response: {
        ok: true,
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ items: [], total: 0 }),
      } } }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const path = new URL(url).pathname;
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1",
          protocol_min: 1,
          protocol_max: 1,
          scope_namespace: "cjs1:profile-recovery-namespace",
        });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] });
      }
      if (path === "/v1/backfill-checkpoint") return responseJson({ stored_at: "now" }, { status: 202 });
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith(
      expect.objectContaining({ args: [{ provider: "chatgpt", operation: "identity", params: {} }] }),
    ));
    expect(fetchCalls.some((call) => new URL(call.url).pathname.startsWith("/v1/capture-jobs"))).toBe(false);
    expect(JSON.stringify(stored)).not.toContain("paired:chatgpt");
  });

  it("holds the job visibly when the receiver authority cannot commit", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] });
      }
      if (String(url).includes("/v1/backfill-checkpoint")) {
        throw new Error("synthetic_receiver_unreachable");
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    expect(started.job.recovery_checkpoint_error).toBe("capture_job_request_failed");
    expect(started.job).toMatchObject({
      status: "paused",
      cooldown_reason: "receiver_capture_job_authority_unavailable",
    });
    expect(stored.polylogueBackfillRecoveryCheckpoint).toBeUndefined();
  });

  it("restores the backfill ledger from the receiver mirror when local checkpoint state is gone", async () => {
    const remoteCheckpoint = {
      version: 1,
      jobs: [{
        id: "remote-job", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
        inventory_cursor: "5", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
        learned_cadence_ms: 40000, daily_requests: 2, last_ack: null,
      }],
      queue: [],
      revisions: [],
    };
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).includes("/v1/backfill-checkpoint")) {
        return responseJson({
          extension_instance_id: "remote-instance",
          checkpoint: remoteCheckpoint,
          stored_at: "2026-01-01T00:00:00Z",
        });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(status.jobs[0]).toMatchObject({
      id: "remote-job",
      status: "paused",
      cooldown_reason: "browser_profile_recovery_required",
      inventory_cursor: "5",
    });
    const restoreCall = fetchCalls.find(
      (call) => String(call.url).includes("/v1/backfill-checkpoint") && (call.options.method || "GET") === "GET",
    );
    expect(restoreCall).toBeDefined();
  });

  it("rehydrates a new browser profile from exact-scope CaptureJobs", async () => {
    const accountHandle = "stable-account-after-profile-loss";
    const remoteCheckpoint = {
      version: 1,
      jobs: [{
        id: "capture-job-local-work", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
        inventory_cursor: "9", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 2,
        learned_cadence_ms: 40000, daily_requests: 3, last_ack: null,
      }],
      queue: [],
      revisions: [],
    };
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle } } }];
      }
      return [{ result: { ok: true, response: {
        ok: true, status: 200, contentType: "application/json", body: JSON.stringify({ items: [], total: 0 }),
      } } }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const path = new URL(url).pathname;
      if (path === "/v1/backfill-checkpoint") {
        return responseJson({ error: "checkpoint_not_found" }, { ok: false, status: 404 });
      }
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1",
          protocol_min: 1,
          protocol_max: 1,
          scope_namespace: "cjs1:new-profile-recovery-namespace",
        });
      }
      if (path === "/v1/capture-jobs/discover") {
        const body = JSON.parse(options.body);
        if (body.provider !== "chatgpt") return responseJson({ jobs: [] });
        return responseJson({ jobs: [{
          job_id: "receiver-capture-job",
          provider: "chatgpt",
          account_scope: "opaque-scope",
          intent_key: "receiver-intent",
          revision: 4,
          lease_generation: 1,
          updated_at: "2026-07-16T10:00:00Z",
          checkpoint: { payload: remoteCheckpoint },
        }] });
      }
      if (path.endsWith("/adopt")) {
        return responseJson({
          job: {
            job_id: "receiver-capture-job", provider: "chatgpt", intent_key: "receiver-intent",
            revision: 5, lease_generation: 2, checkpoint: { payload: remoteCheckpoint },
          },
          lease: { lease_id: "recovery-lease", generation: 2, proof: "recovery-proof" },
        });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(status.jobs[0]).toMatchObject({
      id: "capture-job-local-work",
      status: "paused",
      cooldown_reason: "browser_profile_recovery_required",
      inventory_cursor: "9",
    });
    const discoverCall = fetchCalls.find((call) => new URL(call.url).pathname === "/v1/capture-jobs/discover");
    expect(discoverCall).toBeDefined();
    expect(discoverCall.options.body).not.toContain(accountHandle);
    expect(JSON.parse(discoverCall.options.body).account_scope).toMatch(/^h1:/);
    expect(stored.polylogueExtensionInstanceId).toBeDefined();
    expect(JSON.stringify(stored)).not.toContain(accountHandle);
    expect(fetchCalls.some((call) => new URL(call.url).pathname.endsWith("/adopt"))).toBe(true);
  });

  it("adopts and merges every provider checkpoint without replaying acknowledged work", async () => {
    const checkpoints = Object.fromEntries(["chatgpt", "claude-ai"].map((provider) => [provider, {
      version: 1,
      jobs: [{
        id: `local-${provider}`, provider, cutoff: "2026-01-01T00:00:00Z", status: "running",
        inventory_cursor: "done", inventory_complete: true,
        policy: { leaseMs: 180000, maxDailyRequests: 10, maxCapturesPerWake: 1 },
        execution_generation: 2, learned_cadence_ms: 40000, daily_requests: 3,
        last_ack: { receiver_request_id: `ack-${provider}`, content_hash: `hash-${provider}` },
      }],
      queue: [{
        id: `queue-${provider}`, job_id: `local-${provider}`, provider, native_id: `native-${provider}`,
        state: "complete", content_hash: `hash-${provider}`, completed_at: "2026-07-16T10:00:00Z",
      }],
      revisions: [{
        id: `${provider}:native-${provider}`, provider, native_id: `native-${provider}`,
        provider_updated_at: "2026-07-16T09:00:00Z", content_hash: `hash-${provider}`,
      }],
    }]));
    let providerWorkCalls = 0;
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle: `${request.provider}-account` } } }];
      }
      providerWorkCalls += 1;
      return [{ result: { ok: false, error: "provider_work_must_not_replay" } }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const path = new URL(url).pathname;
      const body = options.body ? JSON.parse(options.body) : {};
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1",
          protocol_min: 1,
          protocol_max: 1,
          scope_namespace: "cjs1:multi-provider-namespace",
        });
      }
      if (path === "/v1/backfill-checkpoint") {
        return responseJson({ error: "checkpoint_not_found" }, { ok: false, status: 404 });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash"] });
      }
      if (path === "/v1/capture-jobs/discover") {
        const provider = body.provider;
        return responseJson({ jobs: checkpoints[provider] ? [{
          job_id: `receiver-${provider}`, provider, intent_key: `intent-${provider}`,
          revision: 4, lease_generation: 1, checkpoint_sequence: 1,
          updated_at: "2026-07-16T10:00:00Z", checkpoint: { payload: checkpoints[provider] },
        }] : [] });
      }
      if (path.endsWith("/adopt")) {
        const provider = body.provider;
        return responseJson({
          job: {
            job_id: `receiver-${provider}`, provider, intent_key: `intent-${provider}`,
            revision: 5, lease_generation: 2, checkpoint_sequence: 1,
            checkpoint: { payload: checkpoints[provider] },
          },
          lease: { lease_id: `lease-${provider}`, generation: 2, proof: `proof-${provider}` },
        });
      }
      if (path.endsWith("/update")) {
        return responseJson({
          job: {
            job_id: path.split("/")[3], provider: body.provider, intent_key: `intent-${body.provider}`,
            revision: 6, lease_generation: 2, lease_expires_at: "2099-01-01T00:00:00Z",
            checkpoint_sequence: 1,
          },
          receipt: { kind: "capture_job_update", revision: 6 },
        });
      }
      if (path.endsWith("/checkpoint")) {
        return responseJson({ job: { job_id: path.split("/")[3], revision: 7, checkpoint_sequence: 2 }, receipt: {} });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(status.jobs.map((job) => job.id).sort()).toEqual(["local-chatgpt", "local-claude-ai"]);
    expect(status.jobs.every((job) => job.status === "paused")).toBe(true);
    expect(status.jobs.map((job) => job.last_ack.receiver_request_id).sort()).toEqual([
      "ack-chatgpt", "ack-claude-ai",
    ]);
    const adoptedPaths = fetchCalls
      .filter((call) => new URL(call.url).pathname.endsWith("/adopt"))
      .map((call) => new URL(call.url).pathname);
    expect(new Set(adoptedPaths)).toEqual(new Set([
      "/v1/capture-jobs/receiver-chatgpt/adopt",
      "/v1/capture-jobs/receiver-claude-ai/adopt",
    ]));
    await sendRuntimeMessage({
      type: "polylogue.backfill.control", job_id: "local-chatgpt", action: "resume",
    });
    alarmListener({ name: "polylogueBackfillWake:local-chatgpt" });
    await vi.waitFor(async () => {
      const refreshed = await sendRuntimeMessage({ type: "polylogue.backfill.status" });
      expect(refreshed.jobs.find((job) => job.id === "local-chatgpt")?.status).toBe("complete");
    });
    expect(providerWorkCalls).toBe(0);
  });

  it("does not query the receiver mirror when a local recovery checkpoint already restored jobs", async () => {
    await loadBackground({
      polylogueBackfillRecoveryCheckpoint: {
        version: 1,
        jobs: [{
          id: "local-job", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
          inventory_cursor: "3", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
          learned_cadence_ms: 40000, daily_requests: 1, last_ack: null,
        }],
        queue: [],
        revisions: [],
      },
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      return responseJson({ error: "should_not_be_called" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(fetchCalls.some((call) => String(call.url).includes("/v1/backfill-checkpoint") && (call.options.method || "GET") === "GET")).toBe(false);
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
        const request = details.args?.[0];
        if (request?.operation === "identity") {
          return [{ result: { ok: true, response: { accountHandle: "retry-account" } } }];
        }
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

  it("holds a browser scripting-result size rejection without spending retry attempts", async () => {
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      if (details.args?.[0]?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle: "oversize-account" } } }];
      }
      throw new Error("The message length exceeded the maximum allowed size");
    });

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    expect(started.ok).toBe(true);
    let status;
    await vi.waitFor(async () => {
      status = (await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs[0];
      expect(status).toMatchObject({ status: "paused", cooldown_reason: "backfill_bridge_response_too_large" });
    });
    expect(status.last_error).toContain("backfill_bridge_projection_too_large:observed_bytes=unavailable;limit_bytes=25165824");
    // The failed page invocation was a real ChatGPT inventory request (cost
    // two); it is accounted once but does not enter transport retry backoff.
    expect(status.daily_requests).toBe(2);
    expect(status.transport_failures).toBe(0);
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
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      if (!details.func) return undefined;
      if (details.args?.[0]?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle: "claude-account" } } }];
      }
      return [{ result: { ok: false, error: "backfill_bridge_selected_organization_stale" } }];
    });

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

  it("does not hand a throttled archive-state refresh to automatic capture", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-throttle", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-throttle",
      state: "archived",
      captured: true,
    }));

    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1));
    activatedListener({ tabId: 42 });
    await new Promise((resolve) => globalThis.setTimeout(resolve, 20));

    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1);
    expect((stored.polylogueConversationTimeline?.["chatgpt:conv-throttle"] || [])
      .some((event) => event.detail === "background_capture_throttled")).toBe(false);
  });

  it("keeps the earliest freshness deadline when a later hint is queued", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "earliest",
      reason: "first",
      delay_ms: 1_000,
    });
    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "later",
      reason: "second",
      delay_ms: 60_000,
    });

    const freshnessAlarms = globalThis.chrome.alarms.create.mock.calls
      .filter(([name]) => name === "polylogueCaptureFreshnessWake");
    expect(freshnessAlarms.at(-1)[1]).toEqual({ when: 101_000 });
  });

  it("persists terminal lifecycle evidence with an immediate freshness deadline", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    const observation = {
      observation_id: "terminal:turn-2",
      state: "completed",
      observed_at: "2026-07-16T01:26:30Z",
      displayed_elapsed_ms: 5_190_000,
    };

    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "terminal-conversation",
      reason: "generation_completed",
      delay_ms: 0,
      generation_observations: [observation],
    });

    expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:terminal-conversation"])
      .toMatchObject({
        next_attempt_at_ms: 100_000,
        generation_observations: [observation],
      });
    const freshnessAlarms = globalThis.chrome.alarms.create.mock.calls
      .filter(([name]) => name === "polylogueCaptureFreshnessWake");
    expect(freshnessAlarms.at(-1)[1]).toEqual({ when: 101_000 });
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
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-sync", title: "ChatGPT" }];
    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1));

    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: ["src/content/chatgpt_bridge.js"],
      world: "MAIN",
    });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: [
        "src/common.js",
        "src/operator_status.js",
        "src/content/message_layer.js",
        "src/content/ambient_surface.js",
        "src/content/chatgpt.js",
      ],
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
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-timeout", title: "ChatGPT" }];
    globalThis.chrome.tabs.sendMessage = vi.fn(() => new Promise(() => {}));

    const responsePromise = sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    await vi.advanceTimersByTimeAsync(35000);
    const response = await responsePromise;

    expect(response).toEqual({ ok: true });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: ["src/content/chatgpt_bridge.js"],
      world: "MAIN",
    });
    expect(stored.polylogueCaptureLog[0].ok).toBe(false);
    expect(stored.polylogueCaptureLog[0].error).toContain("capture_message_timeout_after_35000ms");
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
    expect(globalThis.chrome.alarms.create.mock.calls.some(([name]) => name === "polylogueCaptureRetry")).toBe(false);
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

describe("provider-neutral browser action worker", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("submits in an inactive provider tab and records an owner-bound exact receipt", async () => {
    const action = {
      action_id: "action-1",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Describe the requested implementation.",
      attachments: [],
      presentation: {
        surface: "chat",
        model_slug: "gpt-5-6-pro",
        model_label: "GPT-5.6 Sol",
        effort_label: "Pro",
      },
      submit_policy: "submit_once",
      status: "leased",
    };
    let claimed = false;
    const updates = [];
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).endsWith("/v1/browser-actions/action-1/events")) {
        const update = JSON.parse(options.body);
        updates.push(update);
        return responseJson({ action: { ...action, ...update } });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });
    globalThis.chrome.scripting.executeScript = vi.fn(async () => [{ result: {
      ok: true,
      outcome: "submitted",
      provider_conversation_id: "conversation-1",
      provider_conversation_url: "https://chatgpt.com/c/conversation-1",
      provider_turn_id: "user-turn-1",
      observed_surface: "Chat",
      observed_model: "GPT-5.6 Sol",
      observed_effort: "Pro",
      observed_project_ref: null,
      provider_evidence: { current_node: "assistant-running" },
    } }]);

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("submitted"));

    const owner = new URL(fetchCalls.find((call) => String(call.url).includes("claim_by="))?.url)
      .searchParams.get("claim_by");
    expect(updates[0]).toMatchObject({ outcome: "progress", phase: "submit_intent" });
    expect(updates.at(-1).receipt).toMatchObject({
      extension_instance_id: owner,
      provider_conversation_id: "conversation-1",
      provider_turn_id: "user-turn-1",
    });
    expect(globalThis.chrome.tabs.create).toHaveBeenCalledWith({
      url: "https://chatgpt.com/",
      active: false,
    });
    expect(globalThis.chrome.tabs.remove).toHaveBeenCalledWith(99);
  });

  it("renews the receiver lease while provider execution remains in flight", async () => {
    const action = {
      action_id: "action-slow",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Perform a slow provider operation.",
      attachments: [],
      presentation: {
        surface: "chat",
        model_slug: "gpt-5-6-pro",
        model_label: "GPT-5.6 Sol",
        effort_label: "Pro",
      },
      submit_policy: "submit_once",
      status: "leased",
    };
    const updates = [];
    let claimed = false;
    let heartbeat = null;
    let finishExecution = null;
    vi.spyOn(globalThis, "setInterval").mockImplementation((callback) => {
      heartbeat = callback;
      return 123;
    });
    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval").mockImplementation(() => undefined);
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).endsWith("/v1/browser-actions/action-slow/events")) {
        const update = JSON.parse(options.body);
        updates.push(update);
        return responseJson({ action: { ...action, ...update } });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });
    globalThis.chrome.scripting.executeScript = vi.fn(async (details) => {
      if (!details.func) return undefined;
      return new Promise((resolve) => {
        finishExecution = () => resolve([{ result: {
          ok: true,
          outcome: "submitted",
          provider_conversation_id: "conversation-slow",
          provider_conversation_url: "https://chatgpt.com/c/conversation-slow",
          provider_turn_id: "user-turn-slow",
          observed_surface: "Chat",
          observed_model: "GPT-5.6 Sol",
          observed_effort: "Pro",
          observed_project_ref: null,
          provider_evidence: { current_node: "assistant-running" },
        } }]);
      });
    });

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(heartbeat).toBeTypeOf("function"));
    heartbeat();
    await vi.waitFor(() => expect(updates).toEqual(expect.arrayContaining([
      expect.objectContaining({
        outcome: "progress",
        phase: "submit_intent",
        detail: "renewed browser action lease during provider execution",
      }),
    ])));
    expect(finishExecution).toBeTypeOf("function");
    finishExecution();
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("submitted"));
    expect(clearIntervalSpy).toHaveBeenCalledWith(123);
  });

  it("honors structured provider Retry-After after the submit boundary fails closed", async () => {
    const action = {
      action_id: "action-rate",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Harmless rate-limit fixture.",
      attachments: [],
      presentation: { surface: "chat", model_slug: "gpt-5-6-pro", model_label: "GPT-5.6 Sol", effort_label: "Pro" },
      submit_policy: "submit_once",
      status: "leased",
    };
    const updates = [];
    let claimed = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).endsWith("/v1/browser-actions/action-rate/events")) {
        updates.push(JSON.parse(options.body));
        return responseJson({ action });
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });
    globalThis.chrome.scripting.executeScript = vi.fn(async () => [{ result: {
      ok: false,
      detail: "provider response http_429",
      retry_after_seconds: 75,
      submission_may_have_occurred: false,
    } }]);

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("rate_limited"));
    expect(updates.at(-1)).toMatchObject({ retry_after_seconds: 75, phase: "provider_action_failed" });
  });

  it("stops reading an attachment response at the extension transport limit", async () => {
    const action = {
      action_id: "action-oversized",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Harmless bounded-read fixture.",
      attachments: [{
        attachment_id: "attachment-1",
        name: "context.bin",
        mime_type: "application/octet-stream",
        size_bytes: 1,
        sha256: "00".repeat(32),
      }],
      presentation: { surface: "chat", model_slug: "gpt-5-6-pro", model_label: "GPT-5.6 Sol", effort_label: "Pro" },
      submit_policy: "stage_only",
      status: "leased",
    };
    const updates = [];
    let claimed = false;
    let cancelled = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).includes("/attachments/attachment-1")) {
        const body = new globalThis.ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(8 * 1024 * 1024));
            controller.enqueue(new Uint8Array(9 * 1024 * 1024));
          },
          cancel() {
            cancelled = true;
          },
        });
        return new globalThis.Response(body, { status: 200 });
      }
      if (String(url).endsWith("/v1/browser-actions/action-oversized/events")) {
        updates.push(JSON.parse(options.body));
        return responseJson({ action });
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("provider_drift"));
    expect(updates.at(-1).detail).toContain("protocol_attachment_transport_limit");
    expect(cancelled).toBe(true);
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
  });
});

describe("ambient capture status", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("records seeing an already-safe chat as an explicit deduplicated no-action decision", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-safe", title: "Safe", active: true }];
    let clock = Date.now();
    vi.spyOn(Date, "now").mockImplementation(() => clock);
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-safe",
      state: "archived",
      captured: true,
    }));

    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(stored.polylogueConversationTimeline?.["chatgpt:conv-safe"])
      .toEqual(expect.arrayContaining([expect.objectContaining({ event: "observed_no_action", detail: "already_safe" })])));
    clock += 5_000;
    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(2));

    const noActionEvents = stored.polylogueConversationTimeline["chatgpt:conv-safe"]
      .filter((event) => event.event === "observed_no_action");
    expect(noActionEvents).toHaveLength(1);
  });

  it("persists a site-level ambient disable without changing the global default", async () => {
    const response = await sendRuntimeMessage({
      type: "polylogue.ambient.configure",
      site_enabled: false,
    }, { tab: { id: 42, url: "https://chatgpt.com/c/conv-ambient" } });

    expect(response).toMatchObject({
      ok: true,
      ambient: { enabled: true, site_enabled: false, site: "chatgpt.com" },
    });
    expect(stored.polylogueAmbientSettings).toEqual({
      enabled: true,
      disabled_sites: { "chatgpt.com": true },
    });
  });
});

describe("receiver health probe", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("reports the receiver as reachable and authorized when /v1/status returns ok:true", async () => {
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-health",
      });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({
      ok: true,
      status: "ok",
      detail: null,
      endpoint: "http://127.0.0.1:8875",
      receiver_request_id: "receiver-request-1",
      pairing: {
        state: "online",
        receiver_id: "rx-health",
        api_schema: "polylogue-browser-capture/v1",
      },
    });
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/status");
  });

  it("preflights paired writes and reuses the short trusted-identity cache", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "rx-stable",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          api_schema: "polylogue-browser-capture/v1",
          receiver_id: "rx-stable",
        });
      }
      if (String(url).endsWith("/v1/browser-captures")) {
        return responseJson({ provider: "chatgpt", provider_session_id: "conv-trusted", state: "spooled_only" });
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });

    const message = {
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-trusted", turns: [{ role: "user" }] } },
    };
    expect(await sendRuntimeMessage(message)).toMatchObject({ ok: true });
    expect(await sendRuntimeMessage(message)).toMatchObject({ ok: true });

    expect(fetchCalls.filter((call) => String(call.url).endsWith("/v1/status"))).toHaveLength(1);
    expect(fetchCalls.filter((call) => String(call.url).endsWith("/v1/browser-captures"))).toHaveLength(2);
  });

  it("fails closed before a paired capture reaches a different receiver", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "rx-expected",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-replacement",
      });
    });

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      reason: "content_script_capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-mismatch", turns: [{ role: "user" }] } },
    });

    expect(response).toMatchObject({ ok: false, error: "receiver_pairing_mismatch" });
    expect(fetchCalls).toHaveLength(1);
    expect(String(fetchCalls[0].url)).toMatch(/\/v1\/status$/);
    expect(stored.polylogueCaptureQueue?.entries || []).toHaveLength(0);
    expect(stored.polylogueReceiverPairing).toMatchObject({
      state: "mismatch",
      receiver_id: "rx-expected",
      observed_receiver_id: "rx-replacement",
    });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-mismatch"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "receiver_pairing_mismatch",
    });
  });

  it("recovers only at the canonical endpoint when the paired identity matches", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "offline",
        receiver_id: "rx-recover",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).startsWith("http://127.0.0.1:8875")) throw new TypeError("old endpoint offline");
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-recover",
      });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({
      ok: true,
      status: "recovered",
      endpoint: "http://127.0.0.1:8765",
      recovered_from: "http://127.0.0.1:8875",
    });
    expect(stored.receiverBaseUrl).toBe("http://127.0.0.1:8765");
    expect(fetchCalls.map((call) => call.url)).toEqual([
      "http://127.0.0.1:8875/v1/status",
      "http://127.0.0.1:8765/v1/status",
    ]);
  });

  it("does not adopt the canonical endpoint when its receiver identity differs", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "offline",
        receiver_id: "rx-expected",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).startsWith("http://127.0.0.1:8875")) throw new TypeError("old endpoint offline");
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-other",
      });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({ ok: false, status: "unreachable", endpoint: "http://127.0.0.1:8875" });
    expect(stored.receiverBaseUrl).toBe("http://127.0.0.1:8875");
    expect(stored.polylogueReceiverPairing.receiver_id).toBe("rx-expected");
  });

  it("resets only the pairing key and preserves pending work", async () => {
    const queue = { entries: [{ id: "queued-capture" }], dropped_count: 0 };
    await loadBackground({
      polylogueCaptureQueue: queue,
      polylogueReceiverPairing: {
        state: "mismatch",
        receiver_id: "rx-old",
        api_schema: "polylogue-browser-capture/v1",
      },
    });
    globalThis.fetch = vi.fn(async () => responseJson({
      ok: true,
      api_schema: "polylogue-browser-capture/v1",
      receiver_id: "rx-new",
    }));

    const response = await sendRuntimeMessage({ type: "polylogue.receiverPairing.reset" });

    expect(response.pairing).toMatchObject({ state: "online", receiver_id: "rx-new" });
    expect(stored.polylogueCaptureQueue).toEqual(queue);
  });

  it("reports the receiver as reachable but unauthorized when no valid token is configured", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ ok: false, error: "unauthorized" }));

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({ ok: true, status: "unauthorized", detail: "unauthorized" });
  });

  it("reports the receiver as unreachable when the fetch itself fails", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({ ok: false, status: "unreachable", detail: "Failed to fetch" });
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

    expect(response).toMatchObject({ ok: false, status: "unreachable", detail: "non_json_response" });
  });
});
