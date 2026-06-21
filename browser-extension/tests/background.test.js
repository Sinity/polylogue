import { beforeEach, describe, expect, it, vi } from "vitest";

let messageListener;
let stored;
let fetchCalls;

function installChromeMock() {
  stored = {
    receiverAuthToken: "token-1",
    receiverBaseUrl: "http://127.0.0.1:8875",
  };
  messageListener = null;
  fetchCalls = [];
  globalThis.chrome = {
    action: {
      setBadgeBackgroundColor: vi.fn(async () => undefined),
      setBadgeText: vi.fn(async () => undefined),
    },
    runtime: {
      onMessage: {
        addListener: vi.fn((fn) => {
          messageListener = fn;
        }),
      },
    },
    storage: {
      local: {
        get: vi.fn(async (defaults) => ({ ...defaults, ...stored })),
        set: vi.fn(async (patch) => {
          stored = { ...stored, ...patch };
        }),
      },
    },
  };
}

async function loadBackground() {
  vi.resetModules();
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
});
