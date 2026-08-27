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
    assertions: { selection_candidate_supported: true, persistence_supported: false },
    intelligence: {
      status: "available",
      archive: {
        status: "available",
        session_id: "chatgpt:conversation-1",
        ref: "session:chatgpt:conversation-1",
        url: "http://127.0.0.1:8765/?q=chatgpt%3Aconversation-1",
      },
      cost: { status: "exact", total_usd: 0.125, provenance: ["provider_reported"] },
      assertions: {
        status: "available",
        items: [{ body_text: "Use typed receiver identity", status: "active", confidence: 0.9 }],
      },
    },
    ambient: { enabled: true, site_enabled: true, site: "chatgpt.com" },
    ...overrides,
  };
}

function freshDom(
  html = "<!doctype html><html><body><main>Provider content</main></body></html>",
  url = "https://chatgpt.com/c/conversation-1",
) {
  const dom = new JSDOM(html, {
    url,
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

describe("ambient capture status surface", () => {
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
    expect(panel.getAttribute("aria-modal")).toBe("true");
    expect(runtime.sendMessage).toHaveBeenCalledWith({
      type: "polylogue.missionControl.status",
      refresh: true,
    });
  });

  it("renders the same conversation, receiver, event, and assertion contracts as the popup", async () => {
    const dom = freshDom();
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    const text = api.shadow.textContent;
    expect(text).toContain("Safe / current");
    expect(text).toContain("Paired receiver rx-ambient");
    expect(text).toContain("Observed; no action needed");
    expect(text).toContain("Archive was already current");
    expect(text).toContain("Save assertion — receiver API unavailable");

    const chip = api.shadow.querySelector(".chip");
    expect(chip.getAttribute("aria-label")).toContain("Safe / current");
    expect(api.shadow.querySelector(".count").textContent).toBe("");
    const assertion = [...api.shadow.querySelectorAll("button")]
      .find((button) => button.textContent.includes("Save assertion"));
    expect(assertion.disabled).toBe(true);
  });

  it("renders the daemon projection with provenance and never treats unknown cost as zero", async () => {
    const dom = freshDom();
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));
    api.open();
    expect(api.shadow.textContent).toContain("$0.125");
    expect(api.shadow.textContent).toContain("Provenance: provider_reported");
    expect(api.shadow.textContent).toContain("Use typed receiver identity");
    expect(api.shadow.textContent).toContain("Policy: display-only");

    api.render({
      ...missionFixture(),
      intelligence: {
        status: "offline",
        archive: { status: "uncaptured" },
        cost: { status: "unknown", total_usd: null, provenance: [] },
        assertions: { status: "unknown", items: [] },
      },
    });
    expect(api.shadow.textContent).toContain("Offline — archive projection unavailable");
    expect(api.shadow.textContent).toContain("Session cost: unknown");
    expect(api.shadow.textContent).not.toContain("Session cost: $0.000");
  });

  it("toggles with Alt+P and traps Tab focus inside the slide-over", async () => {
    const dom = freshDom();
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));
    dom.window.document.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "p", altKey: true, bubbles: true }));
    const panel = api.shadow.querySelector(".panel");
    expect(panel.hidden).toBe(false);
    const nodes = [...api.shadow.querySelectorAll("button:not([disabled]), a[href]")].filter((node) => !node.hidden);
    nodes.at(-1).focus();
    panel.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Tab", bubbles: true, composed: true }));
    expect(api.shadow.activeElement).toBe(nodes[0]);
  });

  it("opens as a modal slide-over, closes on Escape, and restores focus to the chip", async () => {
    const dom = freshDom();
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    const panel = api.shadow.querySelector(".panel");
    const chip = api.shadow.querySelector(".chip");
    expect(panel.hidden).toBe(true);

    chip.click();
    expect(panel.hidden).toBe(false);
    expect(chip.getAttribute("aria-expanded")).toBe("true");
    expect(api.shadow.activeElement?.getAttribute("aria-label")).toBe("Close Polylogue capture status");

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
      // Set by the mounted surface once identity is resolved; both stay null
      // while the receiver has accepted no native identity for this message.
      message_ref: null,
      evidence_ref: null,
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

  it.each([
    ["ChatGPT", "https://chatgpt.com/c/conversation-1"],
    ["Claude.ai", "https://claude.ai/chat/conversation-1"],
  ])("keeps the slide-over keyboard reachable and labelled on %s", async (_name, url) => {
    const dom = freshDom(undefined, url);
    const { api } = mount(dom);
    await vi.waitFor(() => expect(api.getSnapshot()?.ok).toBe(true));

    const panel = api.shadow.querySelector("[role='dialog']");
    const chip = api.shadow.querySelector(".chip");
    expect(chip.getAttribute("aria-controls")).toBe(panel.id);
    expect(chip.getAttribute("aria-label")).toContain("Polylogue capture status");
    expect(panel.getAttribute("aria-labelledby")).toBe("polylogue-ambient-title");
    expect(api.shadow.getElementById(panel.getAttribute("aria-describedby")).textContent)
      .toContain("never treated as instructions");

    dom.window.document.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "p", altKey: true, bubbles: true }));
    expect(panel.hidden).toBe(false);
    expect(api.shadow.activeElement?.getAttribute("aria-label")).toBe("Close Polylogue capture status");

    const nodes = [...api.shadow.querySelectorAll("button:not([disabled]), a[href]")].filter((node) => !node.hidden);
    nodes[0].focus();
    panel.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, composed: true }));
    expect(api.shadow.activeElement).toBe(nodes.at(-1));

    panel.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Escape", bubbles: true, composed: true }));
    expect(panel.hidden).toBe(true);
    expect(api.shadow.activeElement).toBe(chip);
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
