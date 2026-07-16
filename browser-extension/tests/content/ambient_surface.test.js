import { readFileSync } from "node:fs";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const operatorSource = readFileSync("src/operator_status.js", "utf8");
const ambientSource = readFileSync("src/content/ambient_surface.js", "utf8");
const openDoms = [];
const mounted = [];

function missionFixture(overrides = {}) {
  return {
    ok: true,
    state: {
      online: true,
      captured: true,
      provider: "chatgpt",
      provider_session_id: "conversation-1",
      archive_state: { state: "archived" },
      ...overrides.state,
    },
    receiver: {
      configured_url: "http://127.0.0.1:8765",
      health: { status: "ok", endpoint: "http://127.0.0.1:8765" },
      pairing: {
        state: "online",
        receiver_id: "rx-ambient",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8765",
      },
      ...overrides.receiver,
    },
    timeline: [{
      at: "2026-07-16T12:00:00Z",
      event: "observed_no_action",
      reason: "tab_activated",
      detail: "already_safe",
    }],
    work: {
      capture_queue: {
        entries: [{
          id: "capture-1",
          next_attempt_at: "2026-07-16T12:01:00Z",
          envelope: { session: { provider: "chatgpt", provider_session_id: "conversation-2", title: "Queued conversation" } },
        }],
      },
      backfill_jobs: [{
        id: "backfill-1",
        provider: "chatgpt",
        status: "running",
        phase: "conversation_capture",
        learned_cadence_ms: 15000,
      }],
      launch_jobs: [{
        job_id: "launch-1",
        job_title: "Ambient mission control implementation",
        status: "completed",
        phase: "handoff_validated",
        cadence_minutes: 5,
        handoff_validated_at: "2026-07-16T11:59:00Z",
        handoff_file_count: 12,
        handoff_size_bytes: 18000,
      }],
      launch_owner_instance_id: "executor-this",
      ...overrides.work,
    },
    assertions: { selection_candidate_supported: true, persistence_supported: false },
    reverse: { enabled: false, default_state: "off", receiver_gate_required: true },
    ambient: { enabled: true, site_enabled: true, site: "chatgpt.com" },
    ...overrides,
  };
}

function freshDom(html = "<!doctype html><html><body><main>Provider content</main></body></html>") {
  const dom = new JSDOM(html, {
    url: "https://chatgpt.com/c/conversation-1",
    runScripts: "outside-only",
    pretendToBeVisual: true,
  });
  openDoms.push(dom);
  dom.window.eval(operatorSource);
  // Auto-mount sees no chrome.runtime and exits. Tests then supply the exact
  // runtime contract explicitly, which keeps network and browser globals out.
  dom.window.eval(ambientSource);
  return dom;
}

function mount(dom, response = missionFixture()) {
  const runtime = {
    sendMessage: vi.fn(async (message) => {
      if (message.type === "polylogue.missionControl.status") return response;
      return { ok: true };
    }),
  };
  const api = dom.window.PolylogueAmbientSurface.mount({
    doc: dom.window.document,
    runtime,
    selectionSource: dom.window,
    locationSource: dom.window.location,
  });
  mounted.push(api);
  return { api, runtime };
}

afterEach(() => {
  while (mounted.length) mounted.pop()?.stop?.();
  while (openDoms.length) openDoms.pop().window.close();
  vi.restoreAllMocks();
});

describe("ambient mission-control surface", () => {
  it("mounts one zero-layout-shift closed shadow surface with no remote assets", async () => {
    const dom = freshDom();
    const bodyChildrenBefore = dom.window.document.body.children.length;
    const { api, runtime } = mount(dom);

    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    expect(dom.window.document.body.children.length).toBe(bodyChildrenBefore);
    expect(api.host.parentElement).toBe(dom.window.document.documentElement);
    expect(api.host.style.position).toBe("fixed");
    expect(api.host.style.width).toBe("0px");
    expect(api.host.style.height).toBe("0px");
    expect(api.host.style.overflow).toBe("visible");
    expect(api.host.style.contain).toBe("style");
    expect(api.host.style.contain).not.toContain("paint");
    expect(api.shadow.querySelector(".root").hidden).toBe(false);
    expect(api.host.shadowRoot).toBeNull();
    expect(dom.window.document.querySelectorAll("#polylogue-ambient-surface")).toHaveLength(1);

    const style = api.shadow.querySelector("style").textContent;
    expect(style).toContain("prefers-color-scheme: dark");
    expect(style).toContain("prefers-reduced-motion: reduce");
    expect(style).not.toMatch(/https?:|@import|url\s*\(/i);

    const panel = api.shadow.querySelector("[role='dialog']");
    expect(panel.getAttribute("aria-labelledby")).toBe("polylogue-ambient-title");
    expect(panel.getAttribute("aria-modal")).toBe("false");
    expect(runtime.sendMessage).toHaveBeenCalledWith({
      type: "polylogue.missionControl.status",
      refresh: true,
    });
  });

  it("renders the same conversation, receiver, event, queue, assertion, and reverse contracts as the popup", async () => {
    const dom = freshDom();
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    const text = api.shadow.textContent;
    expect(text).toContain("Safe / current");
    expect(text).toContain("Paired receiver rx-ambient");
    expect(text).toContain("Observed; no action needed");
    expect(text).toContain("Archive was already current");
    expect(text).toContain("Queued conversation");
    expect(text).toContain("Ambient mission control implementation");
    expect(text).toContain("Handoff: Validated · 12 files · 18000 bytes");
    expect(text).toContain("Off — safe default");
    expect(text).toContain("Save assertion — receiver API unavailable");

    const chip = api.shadow.querySelector(".chip");
    expect(chip.getAttribute("aria-label")).toContain("Safe / current");
    expect(api.shadow.querySelector(".count").textContent).toBe("3");
    const assertion = [...api.shadow.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Save assertion"));
    expect(assertion.disabled).toBe(true);
  });

  it("marks cached Sol Pro work as last-known while the paired receiver is offline", async () => {
    const dom = freshDom();
    const cachedAt = "2026-07-16T11:55:00Z";
    const { api } = mount(dom, missionFixture({
      state: { online: false, error: "receiver_unavailable" },
      receiver: {
        health: { status: "unreachable", detail: "receiver offline" },
        pairing: {
          state: "offline",
          receiver_id: "rx-ambient",
          api_schema: "polylogue-browser-capture/v1",
          endpoint: "http://127.0.0.1:8765",
        },
      },
      work: {
        capture_queue: { entries: [] },
        backfill_jobs: [],
        launch_jobs: [{
          job_id: "launch-cached",
          job_title: "Last-known external analysis",
          status: "submitted",
          phase: "provider_running",
          cadence_minutes: 5,
        }],
        launch_source: "cached",
        launch_cached_at: cachedAt,
      },
    }));
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    const text = api.shadow.textContent;
    expect(text).toContain("Receiver offline");
    expect(text).toContain("Sol Pro state is last known from");
    expect(text).toContain("Last-known external analysis");
    expect(text).toContain("Receiver offline");
  });

  it("opens non-modally, closes on Escape, and restores focus to the chip", async () => {
    const dom = freshDom();
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    const panel = api.shadow.querySelector(".panel");
    const chip = api.shadow.querySelector(".chip");
    expect(panel.hidden).toBe(true);

    chip.click();
    expect(panel.hidden).toBe(false);
    expect(chip.getAttribute("aria-expanded")).toBe("true");
    expect(api.shadow.activeElement?.getAttribute("aria-label")).toBe("Close Polylogue mission control");

    panel.dispatchEvent(new dom.window.KeyboardEvent("keydown", {
      key: "Escape",
      bubbles: true,
      composed: true,
    }));
    expect(panel.hidden).toBe(true);
    expect(chip.getAttribute("aria-expanded")).toBe("false");
    expect(api.shadow.activeElement).toBe(chip);
  });

  it("creates an ephemeral assertion candidate only for text selected inside a supported message", async () => {
    const dom = freshDom(`<!doctype html><html><body>
      <article data-message-author-role="assistant"><span id="inside">A supported message selection</span></article>
      <p id="outside">Page chrome selection</p>
    </body></html>`);
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    const selectNode = (node) => {
      const range = dom.window.document.createRange();
      range.selectNodeContents(node);
      const selection = dom.window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      dom.window.document.dispatchEvent(new dom.window.Event("selectionchange"));
      return selection;
    };

    const insideSelection = selectNode(dom.window.document.getElementById("inside"));
    expect(api.getSelectionCandidate()).toEqual({
      kind: "selection_assertion_candidate",
      provider: "chatgpt",
      source_url: "https://chatgpt.com/c/conversation-1",
      captured_at: expect.any(String),
      text: "A supported message selection",
      truncated: false,
      persistence: "not_supported",
    });
    expect(dom.window.PolylogueAmbientSurface.deriveSelectionCandidate(insideSelection)?.text)
      .toBe("A supported message selection");

    const outsideSelection = selectNode(dom.window.document.getElementById("outside"));
    expect(api.getSelectionCandidate()).toBeNull();
    expect(dom.window.PolylogueAmbientSurface.deriveSelectionCandidate(outsideSelection)).toBeNull();

    const crossMessageRange = dom.window.document.createRange();
    crossMessageRange.setStart(dom.window.document.getElementById("inside").firstChild, 2);
    crossMessageRange.setEnd(dom.window.document.getElementById("outside").firstChild, 4);
    const crossMessageSelection = dom.window.getSelection();
    crossMessageSelection.removeAllRanges();
    crossMessageSelection.addRange(crossMessageRange);
    dom.window.document.dispatchEvent(new dom.window.Event("selectionchange"));
    expect(api.getSelectionCandidate()).toBeNull();
    expect(dom.window.PolylogueAmbientSurface.deriveSelectionCandidate(crossMessageSelection)).toBeNull();
  });

  it("removes itself calmly when globally disabled or hidden for the current site", async () => {
    const disabledDom = freshDom();
    mount(disabledDom, missionFixture({ ambient: { enabled: false, site_enabled: true } }));
    await vi.waitFor(() => expect(disabledDom.window.document.getElementById("polylogue-ambient-surface")).toBeNull());

    const hiddenDom = freshDom();
    const hidden = mount(hiddenDom);
    await vi.waitFor(() => expect(hidden.api.getSnapshot()?.ok).toBe(true));
    const hideButton = [...hidden.api.shadow.querySelectorAll("button")]
      .find((button) => button.textContent === "Hide on this site");
    hideButton.click();
    await vi.waitFor(() => expect(hiddenDom.window.document.getElementById("polylogue-ambient-surface")).toBeNull());
    expect(hidden.runtime.sendMessage).toHaveBeenCalledWith({
      type: "polylogue.ambient.configure",
      hostname: "chatgpt.com",
      site_enabled: false,
    });
  });
});
