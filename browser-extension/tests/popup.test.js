import { JSDOM } from "jsdom";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { beforeEach, describe, expect, it, vi } from "vitest";

const TEST_DIR = dirname(fileURLToPath(import.meta.url));

const CHATGPT_TAB = {
  id: 42,
  title: "ChatGPT conversation",
  url: "https://chatgpt.com/c/test-conversation",
};
const CHATGPT_HOME_TAB = {
  id: 43,
  title: "ChatGPT",
  url: "https://chatgpt.com/",
};
const GROK_QUERY_TAB = {
  id: 77,
  title: "Grok query conversation",
  url: "https://grok.com/?conversation=query-77",
};
const ORDINARY_TAB = {
  id: 88,
  title: "Example",
  url: "https://example.com/article",
};

function installDom() {
  const dom = new JSDOM(`<!doctype html>
    <body>
      <span id="badge"></span>
      <span id="pending-action-count"></span>
      <section id="attention-section" hidden>
        <strong id="attention-heading"></strong>
        <p id="attention-detail"></p>
        <button id="attention-action" hidden><span class="button-label"></span><span class="button-status"></span></button>
      </section>
      <details id="diagnostics"></details>
      <span id="operator-state"></span>
      <strong id="active-heading"></strong>
      <span id="fidelity-flag" hidden></span>
      <span id="open-tab-count"></span>
      <span id="state"></span>
      <span id="receiver-request"></span>
      <span id="extension-build"></span>
      <span id="updated"></span>
      <span id="receiver"></span>
      <span id="page"></span>
      <span id="archive"></span>
      <input id="receiver-url" />
      <input id="receiver-token" />
      <p id="state-detail"></p>
      <button id="save"><span class="button-status"></span></button>
      <button id="copy-ref" class="conversation-only"><span class="button-status"></span></button>
      <button id="open-polylogue" class="conversation-only"><span class="button-status"></span></button>
      <button id="debug-toggle"><span class="button-status"></span></button>
      <button id="debug-export"><span class="button-status"></span></button>
      <select id="backfill-provider"><option value="chatgpt">ChatGPT</option></select>
      <select id="backfill-job"></select>
      <input id="backfill-cutoff" />
      <span id="backfill-status"></span>
      <span id="backfill-cursor"></span>
      <span id="backfill-progress"></span>
      <span id="backfill-rate"></span>
      <span id="backfill-last"></span>
      <button id="backfill-start"><span class="button-status"></span></button>
      <button id="backfill-pause"><span class="button-status"></span></button>
      <button id="backfill-resume"><span class="button-status"></span></button>
      <button id="backfill-cancel"><span class="button-status"></span></button>
      <button id="backfill-export"><span class="button-status"></span></button>
      <span id="mode"></span>
      <span id="fidelity" class="conversation-only"></span>
      <span id="turns" class="conversation-only"></span>
      <span id="cost-tokens" class="conversation-only"></span>
      <span id="assets" class="conversation-only"></span>
      <div id="asset-failures" class="conversation-only"></div>
      <span id="receiver-health"></span>
      <span id="receiver-pairing-status"></span>
      <span id="receiver-pairing-detail"></span>
      <button id="reset-pairing"><span class="button-status"></span></button>
      <span id="work-count"></span>
      <div id="work-queue"></div>
      <span id="queue-count"></span>
      <div id="queue-log"></div>
      <span id="log-count"></span>
      <span id="debug-count"></span>
      <div id="log"></div>
      <div id="timeline"></div>
      <div id="open-tabs"></div>
      <input id="ambient-enabled" type="checkbox" />
      <input id="ambient-site-enabled" type="checkbox" />
      <input id="automatic-capture-enabled" type="checkbox" />
      <span id="ambient-site"></span>
      <span id="assertion-status"></span>
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

function installChromeMock(storagePatch = {}, tabs = [CHATGPT_TAB], sendMessage = null) {
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
      sendMessage: vi.fn(async (message) => sendMessage ? sendMessage(message) : ({ ok: true })),
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
      create: vi.fn(async (details) => ({ id: 99, ...details })),
      get: vi.fn(async (tabId) => tabs.find((tab) => tab.id === tabId) || { id: tabId }),
      update: vi.fn(async (tabId, details) => ({ id: tabId, ...details })),
      query: vi.fn(async () => tabs),
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

async function loadPopup(storagePatch = {}, tabs = [CHATGPT_TAB], sendMessage = null) {
  vi.resetModules();
  installDom();
  installChromeMock(storagePatch, tabs, sendMessage);
  await import("../src/operator_status.js");
  await import("../src/popup.js");
  await vi.waitFor(() => expect(globalThis.document.getElementById("page").textContent).not.toContain("Unknown"));
}

describe("popup capture", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps automatic maintenance out of the operator control surface", () => {
    const markup = readFileSync(join(TEST_DIR, "../src/popup.html"), "utf8");
    for (const id of ["capture", "check", "sync-open-tabs", "check-receiver", "retry-captures"]) {
      expect(markup).not.toContain(`id="${id}"`);
    }
    expect(markup).toContain("Capture, freshness checks, open-tab convergence, and receiver health run automatically.");
  });

  it("exposes a persisted automatic-capture circuit breaker", async () => {
    await loadPopup({}, [CHATGPT_TAB], async (message) => {
      if (message.type === "polylogue.ambient.configure") {
        return { ok: true, ambient: { enabled: true, automatic_capture_enabled: false, site_enabled: true } };
      }
      return { ok: true };
    });

    const control = document.getElementById("automatic-capture-enabled");
    control.checked = false;
    control.dispatchEvent(new globalThis.window.Event("change"));

    await vi.waitFor(() => expect(globalThis.chrome.runtime.sendMessage).toHaveBeenCalledWith({
      type: "polylogue.ambient.configure",
      automatic_capture_enabled: false,
    }));
  });

  it("starts in the manifest classic-script load order without a global redeclaration", () => {
    const context = vm.createContext({});
    context.globalThis = context;
    const operatorSource = readFileSync(join(TEST_DIR, "../src/operator_status.js"), "utf8");
    const popupSource = readFileSync(join(TEST_DIR, "../src/popup.js"), "utf8");

    new vm.Script(operatorSource, { filename: "operator_status.js" }).runInContext(context);
    let startupError = null;
    try {
      new vm.Script(popupSource, { filename: "popup.js" }).runInContext(context);
    } catch (error) {
      startupError = error;
    }

    // The intentionally minimal context has no DOM, so execution reaches the
    // first document access and stops there.  A classic-script global binding
    // collision fails earlier as SyntaxError and made the real popup inert.
    expect(startupError?.name).toBe("ReferenceError");
    expect(startupError?.message).toContain("document is not defined");
  });

  it("refreshes receiver status automatically on popup open", async () => {
    await loadPopup();

    await vi.waitFor(() => expect(globalThis.chrome.runtime.sendMessage).toHaveBeenCalledWith({
      type: "polylogue.status",
      reason: "popup_open",
    }));
  });

  it("renders stale archive state as catching up", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        archive_state: { state: "stale", indexed_message_count: 12 },
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("badge").textContent).toBe("catching up");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Catching up");
    expect(globalThis.document.getElementById("operator-state").textContent).toBe("Catching up");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("catching up");
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

    expect(globalThis.document.getElementById("archive").textContent).toBe("Needs attention");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("browser-capture token show");

    const attentionSection = globalThis.document.getElementById("attention-section");
    await vi.waitFor(() => expect(attentionSection.hidden).toBe(false));
    expect(globalThis.document.getElementById("attention-heading").textContent)
      .toBe("Receiver requires its pairing token");
    const actionButton = globalThis.document.getElementById("attention-action");
    expect(actionButton.hidden).toBe(false);

    const diagnostics = globalThis.document.getElementById("diagnostics");
    diagnostics.open = false;
    actionButton.dispatchEvent(new globalThis.window.Event("click", { bubbles: true }));
    expect(diagnostics.open).toBe(true);
  });

  it("shows no attention item and a zero pending-action count when everything is healthy", async () => {
    await loadPopup({
      polylogueState: { online: true, captured: true, archive_state: { state: "archived" } },
    });

    await vi.waitFor(() => expect(globalThis.document.getElementById("pending-action-count").textContent).toBe("0"));
    expect(globalThis.document.getElementById("attention-section").hidden).toBe(true);
  });

  it("counts queued browser actions and never lets a stuck action-outcome item auto-clear", async () => {
    await loadPopup({}, [CHATGPT_TAB], async (message) => {
      if (message.type === "polylogue.browserActions.status") {
        return {
          ok: true,
          actions: [
            { action_id: "a1", status: "queued" },
            { action_id: "a2", status: "leased" },
            {
              action_id: "a3",
              status: "outcome_unknown",
              last_error: "submit channel ended without a receipt",
            },
          ],
        };
      }
      return { ok: true };
    });

    await vi.waitFor(() => expect(globalThis.document.getElementById("attention-section").hidden).toBe(false));
    expect(globalThis.document.getElementById("pending-action-count").textContent).toBe("2");
    expect(globalThis.document.getElementById("attention-heading").textContent)
      .toBe("A browser action's outcome could not be confirmed");
    expect(globalThis.document.getElementById("attention-detail").textContent)
      .toBe("submit channel ended without a receipt");
    // No actionId is offered for an outcome_unknown action -- there is no safe
    // one-click resolution, only "Details" (progressive disclosure, not a button).
    expect(globalThis.document.getElementById("attention-action").hidden).toBe(true);
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

    expect(globalThis.document.getElementById("badge").textContent).toBe("partial fidelity");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("provider-native data");
  });

  it("renders missing archive state as automatic catch-up, not a receiver failure", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        active_page_state: "conversation",
        archive_state: { state: "missing", indexed_message_count: 0 },
        updated_at: new Date().toISOString(),
      },
    });

    expect(globalThis.document.getElementById("badge").textContent).toBe("catching up");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Catching up");
    expect(globalThis.document.getElementById("state").textContent).toContain("not saved yet");
  });

  it("names receiver, recovery, and bridge-size holds as actionable backfill states", async () => {
    const job = (cooldown_reason) => ({
      id: `job-${cooldown_reason}`,
      provider: "chatgpt",
      status: "paused",
      cooldown_reason,
      inventory_cursor: "17",
      learned_cadence_ms: 40000,
      progress: { total: 1, complete: 0, retry: 1, no_turns: 0, error: 0, operator_action: 0 },
    });
    await loadPopup({}, [CHATGPT_TAB], async (message) => message.type === "polylogue.backfill.status"
      ? { ok: true, jobs: [job("receiver_contract_incompatible")] }
      : { ok: true });
    await vi.waitFor(() => expect(document.getElementById("backfill-status").textContent).toContain("receiver upgrade required"));
    expect(document.getElementById("backfill-last").textContent).toContain("Upgrade/restart receiver");

    await loadPopup({}, [CHATGPT_TAB], async (message) => message.type === "polylogue.backfill.status"
      ? { ok: true, jobs: [job("browser_profile_recovery_required")] }
      : { ok: true });
    await vi.waitFor(() => expect(document.getElementById("backfill-status").textContent).toContain("profile recovery required"));
    expect(document.getElementById("backfill-last").textContent).toContain("profile was replaced");
    expect(document.getElementById("backfill-resume").disabled).toBe(true);

    await loadPopup({}, [CHATGPT_TAB], async (message) => message.type === "polylogue.backfill.status"
      ? { ok: true, jobs: [job("backfill_bridge_response_too_large")] }
      : { ok: true });
    await vi.waitFor(() => expect(document.getElementById("backfill-status").textContent).toContain("bridge limit reached"));
    expect(document.getElementById("backfill-last").textContent).toContain("held; Resume explicitly retries");
    expect(document.getElementById("backfill-resume").disabled).toBe(false);
  });

  it("renders mission-control status, open tabs, and the active decision timeline", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: false,
        provider: "chatgpt",
        provider_session_id: "test-conversation",
        archive_state: { state: "missing" },
        updated_at: new Date().toISOString(),
      },
      polylogueSessionLedger: {
        "chatgpt:test-conversation": { archive_state: { state: "missing" } },
      },
      polylogueConversationTimeline: {
        "chatgpt:test-conversation": [
          { at: new Date().toISOString(), event: "held_with_reason", reason: "auto_capture_missing", detail: "background_capture_throttled" },
          { at: new Date().toISOString(), event: "detected_new", detail: "archive_state_missing" },
        ],
      },
    });

    expect(document.getElementById("operator-state").textContent).toBe("Catching up");
    expect(document.getElementById("open-tab-count").textContent).toBe("1");
    expect(document.getElementById("open-tabs").textContent).toContain("Catching up");
    expect(document.getElementById("timeline").textContent).toContain("Held");
    expect(document.getElementById("timeline").textContent).toContain("background_capture_throttled");
  });

  it("keeps assertion persistence disabled when only the receiver seam is advertised", async () => {
    await loadPopup({}, [CHATGPT_TAB], async (message) => {
      if (message.type === "polylogue.missionControl.status") return {
        ok: true,
        state: { online: true, archive_state: { state: "archived" } },
        receiver: { health: { status: "ok" }, pairing: null, configured_url: "http://127.0.0.1:8765" },
        work: { capture_queue: { entries: [] }, backfill_jobs: [] },
        timeline: [],
        ambient: { enabled: true, site_enabled: true, site: "chatgpt.com" },
        assertions: { persistence_supported: true },
      };
      return { ok: true };
    });

    expect(document.getElementById("assertion-status").textContent).toContain("no authenticated write handler");
    expect(document.getElementById("assertion-status").textContent).toContain("Save remains disabled");
  });

  it("binds the active card and timeline to the current tab instead of stale global state", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        provider: "chatgpt",
        provider_session_id: "previous-conversation",
        captured: true,
        archive_state: { state: "archived" },
        updated_at: new Date().toISOString(),
      },
      polylogueSessionLedger: {
        "chatgpt:test-conversation": {
          archive_state: { state: "spooled_only" },
          receiver_request_id: "pending-active",
        },
      },
      polylogueConversationTimeline: {
        "chatgpt:previous-conversation": [
          { at: new Date().toISOString(), event: "captured", detail: "old capture" },
        ],
        "chatgpt:test-conversation": [
          { at: new Date().toISOString(), event: "held_with_reason", detail: "active pending" },
        ],
      },
    });

    expect(document.getElementById("operator-state").textContent).toBe("Catching up");
    expect(document.getElementById("archive").textContent).toBe("Catching up");
    expect(document.getElementById("timeline").textContent).toContain("active pending");
    expect(document.getElementById("timeline").textContent).not.toContain("old capture");
    expect(document.getElementById("open-tabs").textContent).toContain("Catching up");
  });

  it("renders the ledger and timeline for a Grok query conversation", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        provider: "grok",
        provider_session_id: "query-77",
        archive_state: { state: "spooled_only" },
      },
      polylogueSessionLedger: {
        "grok:query-77": { archive_state: { state: "spooled_only" } },
      },
      polylogueConversationTimeline: {
        "grok:query-77": [{ at: new Date().toISOString(), event: "first_seen", detail: "query route" }],
      },
    }, [GROK_QUERY_TAB]);

    expect(document.getElementById("operator-state").textContent).toBe("Catching up");
    expect(document.getElementById("timeline").textContent).toContain("query route");
    expect(document.getElementById("open-tabs").textContent).toContain("Catching up");
  });

  it("does not list unrelated lookalike hosts as supported providers", async () => {
    await loadPopup({}, [{ id: 88, title: "Lookalike", url: "https://notx.com/chat/anything" }]);

    expect(document.getElementById("open-tab-count").textContent).toBe("0");
    expect(document.getElementById("open-tabs").textContent).toContain("No supported conversation tabs");
  });

  it("marks DOM-derived captures as partial fidelity in the operator vocabulary", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: true,
        capture_mode: "dom_degraded",
        archive_state: { state: "archived" },
        updated_at: new Date().toISOString(),
      },
    });

    expect(document.getElementById("operator-state").textContent).toBe("Safe / current");
    expect(document.getElementById("fidelity-flag").hidden).toBe(false);
  });

  it("maps every archive pipeline state through the shared operator vocabulary", async () => {
    await loadPopup();

    const { operatorStatusForState } = globalThis.PolylogueOperatorStatus;
    expect(operatorStatusForState({ online: true, archive_state: { state: "missing" } }).label).toBe("Catching up");
    expect(operatorStatusForState({ online: true, archive_state: { state: "spooled_only" } }).label).toBe("Catching up");
    expect(operatorStatusForState({ online: true, archive_state: { state: "ingest_pending" } }).label).toBe("Catching up");
    expect(operatorStatusForState({ online: true, archive_state: { state: "stale" } }).label).toBe("Catching up");
    expect(operatorStatusForState({ online: true, archive_state: { state: "archived" } }).label).toBe("Safe / current");
    expect(operatorStatusForState({ online: true, archive_state: { state: "failed" } }).label).toBe("Failed");
    expect(operatorStatusForState({ online: true, captured: true, archive_state: { state: "spooled_only" } }).label).toBe("Catching up");
    expect(operatorStatusForState({ online: true, captured: true, archive_state: { state: "missing" } }).label).toBe("Catching up");
    expect(operatorStatusForState({ online: true }).label).toBe("Catching up");
    expect(operatorStatusForState({ online: true, capture_mode: "dom_degraded" }).partialFidelity).toBe(true);
  });

  it("keeps a pending capture out of the safe operator state", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: true,
        archive_state: { state: "ingest_pending" },
        updated_at: new Date().toISOString(),
      },
    });

    expect(document.getElementById("operator-state").textContent).toBe("Catching up");
    expect(document.getElementById("archive").textContent).toBe("Catching up");
  });

  it("labels envelope turns as captured and indexed messages as visible", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        captured: true,
        turn_count: 12,
        archive_state: { state: "archived", indexed_message_count: 5 },
        updated_at: new Date().toISOString(),
      },
    });

    expect(document.getElementById("turns").textContent).toBe("12 captured / 5 visible");
  });

  it("renders known cost and token data or an explicit unavailable state", async () => {
    await loadPopup({
      polylogueState: {
        online: true,
        usage: { total_tokens: 17 },
        cost_usd: 0.012,
        updated_at: new Date().toISOString(),
      },
    });

    expect(document.getElementById("cost-tokens").textContent).toBe("$0.012 · 17 tokens");
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
    }, [CHATGPT_HOME_TAB]);

    expect(globalThis.document.getElementById("badge").textContent).toBe("idle");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Not applicable");
    expect(globalThis.document.getElementById("operator-state").textContent).toBe("No conversation");
    expect(globalThis.document.getElementById("state-detail").textContent).toContain("automatically");
  });

  it("renders ordinary webpages as neutral non-conversations without stale fidelity", async () => {
    await loadPopup({
      polylogueState: {
        online: false,
        captured: true,
        provider: "chatgpt",
        provider_session_id: "stale-conversation",
        active_page_state: "conversation",
        capture_mode: "dom_degraded",
        updated_at: new Date().toISOString(),
      },
    }, [ORDINARY_TAB]);

    expect(globalThis.document.getElementById("badge").textContent).toBe("idle");
    expect(globalThis.document.getElementById("archive").textContent).toBe("Not applicable");
    expect(globalThis.document.getElementById("operator-state").textContent).toBe("No conversation");
    expect(globalThis.document.getElementById("state").textContent).toBe("Ordinary webpage");
    expect(globalThis.document.getElementById("fidelity-flag").hidden).toBe(true);
    expect(globalThis.document.getElementById("fidelity").hidden).toBe(true);
    expect(globalThis.document.getElementById("copy-ref").disabled).toBe(true);
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
    expect(globalThis.document.getElementById("turns").textContent).toBe("12 captured / -- visible");
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

  it("starts and controls a background backfill while rendering rate and progress", async () => {
    let status = "running";
    const job = () => ({
      id: "backfill-1",
      provider: "chatgpt",
      status,
      inventory_cursor: "144",
      inventory_complete: false,
      learned_cadence_ms: 30000,
      cooldown_until_ms: Date.now() + 60000,
      last_error: "provider_http_429",
      progress: { total: 462, complete: 144, retry: 1, no_turns: 0, error: 0, operator_action: 0 },
    });
    globalThis.chrome.runtime.sendMessage.mockImplementation(async (message) => {
      if (message.type === "polylogue.backfill.status") return { ok: true, jobs: [job()] };
      if (message.type === "polylogue.backfill.start") return { ok: true, job: job() };
      if (message.type === "polylogue.backfill.control") { status = message.action === "pause" ? "paused" : message.action; return { ok: true, job: job() }; }
      return { ok: true };
    });
    document.getElementById("backfill-cutoff").value = "2026-04-23";

    document.getElementById("backfill-start").click();
    await vi.waitFor(() => expect(document.getElementById("backfill-progress").textContent).toContain("144/462"));
    expect(document.getElementById("backfill-rate").textContent).toContain("30s");
    expect(globalThis.chrome.runtime.sendMessage).toHaveBeenCalledWith(expect.objectContaining({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-04-23T00:00:00.000Z",
    }));

    document.getElementById("backfill-pause").click();
    await vi.waitFor(() => expect(document.getElementById("backfill-status").textContent).toContain("paused"));
  });

});
